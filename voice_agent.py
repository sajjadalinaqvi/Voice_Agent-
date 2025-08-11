#!/usr/bin/env python3
"""
Voice Agent for VoIP Project

Features:
- Silero VAD via torch.hub when available, fallback to webrtcvad.
- RealtimeSTT (AudioToTextRecorder) usage with guarded calls.
- gTTS for speech synthesis, fallback to placeholder tone.
- Flask web UI endpoints to start/stop and check status.
NOTE: This file avoids hard-coded secrets. Set GROQ_API_KEY via environment or the UI.
"""

import os
import time
import threading
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

# Keep the project's logger import name (if exists)
try:
    import logger as logger_module
except Exception:
    logger_module = None

# Core dependencies
import torch
import numpy as np
from groq import Groq
from RealtimeSTT import AudioToTextRecorder
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try to load Silero VAD via torch.hub; fall back to webrtcvad
USE_SILERO = False
_silero_model = None
_get_speech_timestamps = None
_read_audio = None
_vad_webrtc = None

try:
    hub_load = torch.hub.load
    _silero_model, silero_utils = hub_load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    try:
        _get_speech_timestamps = silero_utils[0]
        _read_audio = silero_utils[2]
    except Exception:
        _get_speech_timestamps = silero_utils.get("get_speech_timestamps", None)
        _read_audio = silero_utils.get("read_audio", None)
    if _silero_model is not None and _get_speech_timestamps is not None:
        USE_SILERO = True
        print("[INFO] Silero VAD loaded via torch.hub")
except Exception as e:
    print(f"[WARN] Silero VAD not available via torch.hub: {e}")
    try:
        import webrtcvad

        _vad_webrtc = webrtcvad.Vad(3)  # aggressive
        print("[INFO] Using webrtcvad fallback")
    except Exception as e2:
        print(f"[WARN] webrtcvad also not available: {e2}")
        _vad_webrtc = None

# Audio I/O
import pyaudio

# Web server for UI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import socketio

# Logging (avoid shadowing user's logger module)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice_agent")


@dataclass
class VoiceAgentConfig:
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"
    sample_rate: int = 16000
    chunk_size: int = 1024
    vad_threshold: float = 0.5
    silence_timeout: float = 2.0
    max_response_length: int = 500
    consultant_personality: str = (
        "You are a helpful and confident consultant. Provide guidance and support to users "
        "with a warm, professional tone. Keep responses concise and actionable."
    )


class VAD:
    """Unified interface: Silero via torch.hub if available, else webrtcvad, else energy VAD."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.use_silero = USE_SILERO and _silero_model is not None and _get_speech_timestamps is not None
        self.webrtc = _vad_webrtc

    def is_speech(self, audio_float32: np.ndarray) -> bool:
        """audio_float32 should be float32 in range [-1, 1]"""
        try:
            if audio_float32 is None or audio_float32.size == 0:
                return False

            # Normalize if needed
            if audio_float32.dtype != np.float32:
                audio_float32 = audio_float32.astype(np.float32)
            max_abs = float(np.max(np.abs(audio_float32))) if audio_float32.size > 0 else 1.0
            if max_abs > 1.0:
                audio_float32 = audio_float32 / max_abs

            # Silero path
            if self.use_silero:
                try:
                    tensor = torch.from_numpy(audio_float32)
                    pred = _silero_model
                    prob = float(pred.item()) if hasattr(pred, "item") else float(pred)
                    return prob > self.threshold
                except Exception as e:
                    log.debug(f"Silero VAD runtime failed, falling back: {e}")

            # WebRTC path
            if self.webrtc is not None:
                try:
                    frame_ms = 30
                    frame_len = int(self.sample_rate * frame_ms / 1000)
                    total_frames = (len(audio_float32) // frame_len) * frame_len
                    if total_frames <= 0:
                        pcm = (audio_float32 * 32767).astype(np.int16).tobytes()
                        return self.webrtc.is_speech(pcm, self.sample_rate)
                    for i in range(0, total_frames, frame_len):
                        frame = audio_float32[i: i + frame_len]
                        pcm = (frame * 32767).astype(np.int16).tobytes()
                        if self.webrtc.is_speech(pcm, self.sample_rate):
                            return True
                    return False
                except Exception as e:
                    log.debug(f"webrtcvad call failed: {e}")

            # Fallback energy VAD
            energy = float(np.mean(audio_float32 ** 2))
            return energy > (self.threshold * 1e-4)

        except Exception as e:
            log.error(f"VAD.is_speech error: {e}")
            return False


class GroqLLMClient:
    """Minimal Groq wrapper with graceful fallback"""

    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = None
        self.model = model
        self.history = []
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as e:
                log.warning(f"Could not init Groq client: {e}")
                self.client = None

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        self.history.append({"role": "user", "content": prompt})
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.history[-20:])

        if not self.client:
            fallback = "(local fallback) " + prompt[:500]
            self.history.append({"role": "assistant", "content": fallback})
            return fallback

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=False,
            )
            text = r.choices[0].message.content
            self.history.append({"role": "assistant", "content": text})
            return text
        except Exception as e:
            log.error(f"Groq request error: {e}")
            return "(error) I'm unable to generate a response right now."


class GoogleTTS:
    """gTTS implementation with fallback to simple beep"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.lang = 'en'
        self.slow = False

    def synthesize(self, text: str) -> bytes:
        try:
            # Create gTTS object and save to bytes buffer
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            mp3_buffer = BytesIO()
            tts.write_to_fp(mp3_buffer)
            mp3_buffer.seek(0)

            # Convert MP3 to PCM
            audio = AudioSegment.from_mp3(mp3_buffer)
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(1)

            # Convert to 16-bit PCM
            pcm = audio.raw_data
            return pcm
        except Exception as e:
            print(f"[WARN] gTTS synthesis failed: {e}")
            return self._fallback_beep(text)

    def _fallback_beep(self, text: str) -> bytes:
        duration = max(0.2, min(len(text) * 0.04, 3.0))  # Limit duration
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        audio = np.sin(2 * np.pi * 440 * t) * 0.15
        return (audio * 32767).astype(np.int16).tobytes()


class VoiceAgent:
    """Main voice agent with speech queue to prevent overlapping"""

    def __init__(self, config: VoiceAgentConfig):
        self.config = config
        self.is_running = False
        self.is_speaking = False
        self.speech_queue = []
        self.current_speech_thread = None
        self.speech_lock = threading.Lock()
        self.playback_lock = threading.Lock()

        # Initialize components
        self.vad = VAD(threshold=config.vad_threshold, sample_rate=config.sample_rate)
        self.llm = GroqLLMClient(api_key=config.groq_api_key, model=config.groq_model)
        self.tts = GoogleTTS(sample_rate=24000)

        # Audio setup
        self.pyaudio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None

        # STT setup - fixed to use correct parameters
        try:
            self.stt_recorder = AudioToTextRecorder(
                model="tiny.en",
                language="en",
                spinner=False,
                use_microphone=True,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop
            )
        except Exception as e:
            log.error(f"Could not initialize STT recorder: {e}")
            self.stt_recorder = None

    def _on_recording_start(self):
        """Callback when recording starts"""
        log.info("Recording started")

    def _on_recording_stop(self):
        """Callback when recording stops"""
        log.info("Recording stopped")

    def speak_response(self, text: str):
        """Add speech to queue and process sequentially"""
        with self.speech_lock:
            self.speech_queue.append(text)

        if not self.is_speaking:
            self._process_speech_queue()

    def _process_speech_queue(self):
        """Process speech items from queue one at a time"""
        with self.speech_lock:
            if not self.speech_queue or self.is_speaking:
                return
            text = self.speech_queue.pop(0)
            self.is_speaking = True

        try:
            pcm = self.tts.synthesize(text)
            if pcm:
                log.info(f"Speaking: {text}")
                self.play_pcm(pcm, sample_rate=self.tts.sample_rate)
        except Exception as e:
            log.error(f"speak_response error: {e}")
        finally:
            self.is_speaking = False
            # Process next item in queue if any
            self._process_speech_queue()

    def stop_speaking(self):
        """Stop current speech and clear queue"""
        with self.speech_lock:
            self.speech_queue.clear()
            self.is_speaking = False

    def start_listening_loop(self):
        try:
            if not self.stt_recorder:
                log.warning("STT recorder not available; listening loop will idle.")
            while self.is_running:
                try:
                    if not self.stt_recorder:
                        time.sleep(0.1)
                        continue
                    try:
                        text = self.stt_recorder.text()
                    except Exception as stt_err:
                        log.debug(f"RealtimeSTT.text() raised: {stt_err}")
                        text = None

                    if text and text.strip():
                        log.info(f"Transcribed: {text.strip()}")
                        resp = self.process_speech(text.strip())
                        if resp:
                            self.speak_response(resp)
                    else:
                        time.sleep(0.05)
                except Exception as e:
                    log.error(f"Listening loop error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            log.error(f"start_listening_loop top-level error: {e}")

    def start_input_stream(self):
        try:
            if self.input_stream is not None:
                return
            self.input_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._pyaudio_input_callback,
            )
            self.input_stream.start_stream()
            log.info("PyAudio input stream started")
        except Exception as e:
            log.error(f"Failed to start PyAudio input stream: {e}")
            self.input_stream = None

    def _pyaudio_input_callback(self, in_data, frame_count, time_info, status):
        try:
            audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if self.is_speaking:
                if self.vad.is_speech(audio):
                    log.info("Barge-in detected -> stopping TTS")
                    self.stop_speaking()
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            log.error(f"Input callback error: {e}")
            return (in_data, pyaudio.paContinue)

    def stop_input_stream(self):
        try:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
        except Exception as e:
            log.debug(f"Error stopping input stream: {e}")

    def play_pcm(self, pcm_bytes: bytes, sample_rate: int):
        try:
            with self.playback_lock:
                if self.output_stream:
                    try:
                        self.output_stream.stop_stream()
                        self.output_stream.close()
                    except Exception:
                        pass
                    self.output_stream = None

                self.output_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=1024,
                )

                chunk_bytes = 1024 * 2
                idx = 0
                total = len(pcm_bytes)
                while idx < total and self.is_speaking:
                    end = min(idx + chunk_bytes, total)
                    self.output_stream.write(pcm_bytes[idx:end])
                    idx = end

                try:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                except Exception:
                    pass
                self.output_stream = None
        except Exception as e:
            log.error(f"Playback error: {e}")

    def process_speech(self, text: str) -> str:
        try:
            log.info(f"Processing transcribed text: {text}")
            return self.llm.generate_response(text, self.config.consultant_personality)
        except Exception as e:
            log.error(f"process_speech error: {e}")
            return "Sorry, I couldn't process that."

    def start(self):
        try:
            self.is_running = True
            self.start_input_stream()
            self.processing_thread = threading.Thread(target=self.start_listening_loop, daemon=True)
            self.processing_thread.start()
            log.info("VoiceAgent started")
        except Exception as e:
            log.error(f"start() error: {e}")

    def stop(self):
        try:
            self.is_running = False
            self.is_speaking = False
            self.stop_input_stream()
            try:
                if self.stt_recorder:
                    self.stt_recorder.stop()
            except Exception:
                pass
            try:
                if self.pyaudio:
                    self.pyaudio.terminate()
            except Exception:
                pass
            log.info("VoiceAgent stopped")
        except Exception as e:
            log.error(f"stop() error: {e}")


# Flask app
app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

voice_agent: VoiceAgent | None = None


@app.route("/")
def index():
    return render_template("templates/index.html")


@app.route("/start", methods=["POST"])
def start_agent():
    global voice_agent
    try:
        data = request.get_json() or {}
        groq_api_key = data.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
        config = VoiceAgentConfig(groq_api_key=groq_api_key)
        voice_agent = VoiceAgent(config)
        voice_agent.start()
        return jsonify({"status": "started"})
    except Exception as e:
        log.error(f"/start error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/stop", methods=["POST"])
def stop_agent():
    global voice_agent
    try:
        if voice_agent:
            voice_agent.stop()
            voice_agent = None
        return jsonify({"status": "stopped"})
    except Exception as e:
        log.error(f"/stop error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    global voice_agent
    running = bool(voice_agent and voice_agent.is_running)
    speaking = bool(voice_agent and voice_agent.is_speaking)
    return jsonify({"running": running, "speaking": speaking, "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    print("Starting Voice Agent server on http://0.0.0.0:5003")
    app.run(host="0.0.0.0", port=5003, debug=False, threaded=True)