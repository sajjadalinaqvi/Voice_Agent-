# Voice Agent for VoIP Project

## Overview

This project implements a fully functional voice agent designed for VoIP applications with zero-latency performance, utilizing cutting-edge technologies including Groq LLM, RealtimeSTT, and Silero VAD for advanced barge-in functionality. The system is architected to provide seamless real-time voice interactions with consultant-like guidance capabilities, making it ideal for customer service, technical support, and advisory applications.

## Key Features

### Core Capabilities
- **Zero Latency Performance**: Optimized for real-time voice interactions with minimal delay
- **Advanced Speech Recognition**: Powered by RealtimeSTT with Whisper models for accurate transcription
- **Intelligent Response Generation**: Groq LLM integration for contextual and helpful responses
- **Barge-in Detection**: Silero VAD implementation for natural conversation flow
- **Professional Web Interface**: Modern HTML/CSS/JavaScript UI with real-time status updates
- **FreeSwitch Compatibility**: Designed for integration with VoIP infrastructure

### Technical Architecture
- **Backend**: Python Flask application with WebSocket support
- **Frontend**: Responsive HTML interface with real-time communication
- **Speech Processing**: RealtimeSTT for speech-to-text conversion
- **AI Processing**: Groq API for language model responses
- **Voice Activity Detection**: Silero VAD with fallback to energy-based detection
- **Audio Processing**: PyAudio for real-time audio stream handling

## Installation and Setup

### Prerequisites
- Python 3.11 or higher
- Ubuntu 22.04 or compatible Linux distribution
- Groq API key (obtain from https://console.groq.com/)
- Audio input device (microphone)

### System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev portaudio19-dev
```

### Python Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install required packages
pip install RealtimeSTT silero-vad groq Flask Flask-CORS python-socketio PyAudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Configuration

### Environment Variables
Set your Groq API key as an environment variable:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Alternatively, you can provide the API key through the web interface.

### Voice Agent Configuration
The system uses a configuration class that can be customized:

```python
@dataclass
class VoiceAgentConfig:
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"
    sample_rate: int = 16000
    chunk_size: int = 1024
    vad_threshold: float = 0.5
    silence_timeout: float = 2.0
    max_response_length: int = 500
    consultant_personality: str = "You are a helpful and confident consultant..."
```

## Usage

### Starting the Application
1. Navigate to the project directory
2. Activate the virtual environment: `source venv/bin/activate`
3. Run the application: `python voice_agent.py`
4. Open your browser to `http://localhost:5000`

### Web Interface Operations
1. Enter your Groq API key in the provided field
2. Click "Start Agent" to initialize the voice agent
3. Begin speaking - the system will automatically detect speech and respond
4. The interface shows real-time status including listening/speaking states
5. Click "Stop Agent" to terminate the session

### API Endpoints
- `GET /`: Main web interface
- `POST /start`: Start the voice agent with API key
- `POST /stop`: Stop the voice agent
- `GET /status`: Get current agent status

## FreeSwitch Integration

### Overview
This voice agent is designed to be compatible with FreeSwitch, a popular open-source telephony platform. The integration allows the voice agent to handle incoming and outgoing calls through VoIP infrastructure.

### Integration Architecture
The voice agent can be integrated with FreeSwitch using several approaches:

1. **ESL (Event Socket Library) Integration**: Connect directly to FreeSwitch using the Event Socket interface
2. **Media Server Integration**: Use the voice agent as a media processing endpoint
3. **WebRTC Gateway**: Leverage WebRTC capabilities for browser-based calling

### FreeSwitch Configuration
To integrate with FreeSwitch, you'll need to configure the following components:

#### Dialplan Configuration
```xml
<extension name="voice_agent">
  <condition field="destination_number" expression="^(voice_agent)$">
    <action application="answer"/>
    <action application="socket" data="127.0.0.1:8084 async full"/>
  </condition>
</extension>
```

#### ESL Socket Configuration
The voice agent can be extended to listen on an ESL socket for FreeSwitch integration:

```python
# Additional configuration for FreeSwitch integration
FREESWITCH_ESL_HOST = "127.0.0.1"
FREESWITCH_ESL_PORT = 8084
FREESWITCH_ESL_PASSWORD = "ClueCon"
```

### Deployment Considerations
When deploying with FreeSwitch:

1. **Network Configuration**: Ensure proper firewall rules for RTP/SIP traffic
2. **Audio Codec Support**: Configure compatible audio codecs (G.711, G.722, Opus)
3. **Latency Optimization**: Minimize network hops and processing delays
4. **Scalability**: Consider load balancing for multiple concurrent calls
5. **Security**: Implement proper authentication and encryption

## Technical Implementation Details

### Speech Recognition Pipeline
The speech recognition system uses RealtimeSTT with the following configuration:
- Model: Whisper "tiny" for optimal speed
- Language: English (configurable)
- Real-time processing with minimal latency
- Voice activity detection for automatic start/stop

### Language Model Integration
Groq LLM integration provides:
- Fast inference using Groq's optimized hardware
- Conversation context management
- Configurable personality and response style
- Token limit management for cost optimization

### Voice Activity Detection
The system implements a dual-approach VAD:
1. **Primary**: Silero VAD for accurate speech detection
2. **Fallback**: Energy-based detection for compatibility

### Barge-in Functionality
The barge-in system monitors audio input during TTS playback:
- Continuous audio stream monitoring
- Real-time speech detection
- Immediate TTS interruption on user speech
- Seamless conversation flow

## Performance Optimization

### Latency Reduction Strategies
1. **Model Selection**: Use smaller, faster models where possible
2. **Preprocessing**: Minimize audio preprocessing overhead
3. **Caching**: Cache frequently used responses
4. **Parallel Processing**: Concurrent audio processing and response generation

### Resource Management
- Memory-efficient audio buffering
- CPU optimization for real-time processing
- GPU utilization when available
- Network optimization for API calls

## Troubleshooting

### Common Issues

#### Audio Device Problems
- Ensure microphone permissions are granted
- Check audio device availability with `arecord -l`
- Verify PyAudio installation and compatibility

#### API Connection Issues
- Verify Groq API key validity
- Check network connectivity
- Monitor API rate limits and quotas

#### Performance Issues
- Monitor CPU and memory usage
- Adjust model sizes for available resources
- Optimize audio buffer sizes

### Debugging
Enable detailed logging by setting the log level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Security Considerations

### API Key Management
- Store API keys securely using environment variables
- Implement key rotation policies
- Monitor API usage and costs

### Network Security
- Use HTTPS in production deployments
- Implement proper CORS policies
- Secure WebSocket connections

### Audio Privacy
- Implement audio data encryption
- Ensure compliance with privacy regulations
- Provide clear user consent mechanisms

## Deployment

### Production Deployment
For production deployment, consider:

1. **WSGI Server**: Use Gunicorn or uWSGI instead of Flask development server
2. **Reverse Proxy**: Implement Nginx for load balancing and SSL termination
3. **Process Management**: Use systemd or supervisor for service management
4. **Monitoring**: Implement logging and monitoring solutions
5. **Scaling**: Consider containerization with Docker/Kubernetes

### Docker Deployment
A Dockerfile is provided for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "voice_agent.py"]
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests before submitting pull requests

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add unit tests for new functionality
- Update documentation for changes

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For technical support and questions:
- Create an issue in the project repository
- Review the troubleshooting section
- Check the documentation for configuration details

## Acknowledgments

This project leverages several open-source technologies:
- RealtimeSTT for speech recognition
- Silero VAD for voice activity detection
- Groq for language model inference
- Flask for web framework
- PyAudio for audio processing


