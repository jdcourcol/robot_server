# 🤖 Robot Control Server

An advanced AI-powered robot control system that uses Qwen3-VL for multi-image analysis and agentic decision-making. The system provides intelligent robot control with comprehensive logging, Bluetooth communication, and real-time web interface.

## ✨ Features

### 🧠 AI-Powered Analysis
- **Multi-Image Context**: Qwen3-VL analyzes sequences of images for temporal understanding
- **Agentic Behavior**: AI follows complex multi-step instructions across multiple images
- **Auto-Execution**: Automatically parses and executes AI-generated robot commands
- **Safety Mode**: Built-in safety checks to prevent dangerous commands

### 📡 Robot Communication
- **Bluetooth Low Energy (BLE)**: Real-time communication with robot via Bleak library
- **Instruction Format**: Supports both JSON and simple text command formats
- **Device Discovery**: Automatic discovery and connection to nearby robots
- **Queue Management**: Instruction queuing for reliable command delivery

### 🌐 Web Interface
- **Real-time Chat**: Interactive chat with AI assistant
- **Live Image Display**: Shows current robot camera feed
- **Bluetooth Management**: Connect/disconnect and device discovery
- **Manual Control**: Send preset or custom robot instructions
- **Settings Panel**: Toggle auto-execution and safety modes

### 📊 Comprehensive Logging
- **AI Analysis Logs**: Complete reasoning and decision-making process
- **Robot Instructions**: All commands sent to robot with execution status
- **System Events**: Connection status, image uploads, and errors
- **Log Rotation**: Automatic log file management (10MB max, 5 backups)
- **Log Viewer**: Built-in script to view and follow logs in real-time

## Setup

### 1. Install Dependencies

```bash
# Run the setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set API Key

Create a `key.txt` file with your OpenRouter API key:

```bash
echo "your-openrouter-api-key-here" > key.txt
```

Get your API key from: https://openrouter.ai/keys

### 3. Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8040`

### 4. Access Web Interface

Open your browser and go to: `http://localhost:8040`

## 🚀 API Endpoints

### 🤖 Robot Communication
- `POST /image/` - Receive images from robot (saves to `images/` directory)
- `POST /send_instruction/` - Send instructions to robot via Bluetooth
- `GET /instructions/` - Get all robot instructions history
- `GET /image/{filename}` - Serve robot images

### 📡 Bluetooth Management
- `GET /bluetooth/status/` - Get Bluetooth connection status
- `POST /bluetooth/connect/` - Connect to robot via Bluetooth
- `POST /bluetooth/disconnect/` - Disconnect from robot
- `GET /bluetooth/discover/` - Discover nearby Bluetooth devices
- `POST /bluetooth/queue_instruction/` - Queue instruction for Bluetooth sending

### 🧠 AI & Analysis
- `POST /chat/` - Send chat messages to AI (uses multi-image context)
- `GET /chat_history/` - Get chat history
- `POST /complex_instruction/` - Start complex multi-step instructions
- `GET /instruction_status/` - Get current complex instruction status
- `WebSocket /ws` - Real-time communication

### ⚙️ Settings & Control
- `GET /settings/` - Get current auto-execution and safety settings
- `POST /settings/auto_execute/` - Toggle auto-execution of AI instructions
- `POST /settings/safety_mode/` - Toggle safety mode for instruction validation

### 🌐 Web Interface
- `GET /` - Main web interface
- `GET /docs` - API documentation (FastAPI auto-generated)

## 🎯 Usage

### Basic Workflow
1. **Start Server**: Run `python main.py` to start the server
2. **Connect to Robot**: Use the web interface to discover and connect to your robot via Bluetooth
3. **Robot sends images**: POST image data to `/image/` (images saved to `images/` directory)
4. **AI analyzes**: Qwen3-VL automatically analyzes image sequences and suggests actions
5. **Auto-execution**: AI-generated instructions are automatically sent to robot (if enabled)
6. **Chat with AI**: Use the web interface to ask questions or give complex instructions
7. **Manual control**: Send specific instructions to the robot via the control panel

### Advanced Features
- **Multi-image Analysis**: AI tracks changes across multiple images for temporal understanding
- **Complex Instructions**: Give multi-step instructions that AI follows across image sequences
- **Safety Mode**: Built-in safety checks prevent dangerous commands
- **Logging**: All AI reasoning and robot commands are logged to files for analysis

## 📡 Bluetooth Setup

### Prerequisites
- Bluetooth Low Energy (BLE) adapter on your computer
- Robot with BLE capability
- Robot's Bluetooth MAC address

### Connection Process
1. **Discover Devices**: Click "Discover Devices" in the web interface
2. **Connect**: Enter the robot's MAC address and click "Connect"
3. **Verify**: Check the Bluetooth status indicator shows "Connected"

### Supported Platforms
- **Linux**: Full BLE support with Bleak library
- **macOS**: Full BLE support with Bleak library
- **Windows**: Full BLE support with Bleak library

### Instruction Format
The robot expects instructions in this format:
```
left: 20 right: 30 duration: 2
```
Where:
- `left`: Left wheel power (-100 to 100, negative = reverse)
- `right`: Right wheel power (-100 to 100, negative = reverse)
- `duration`: Movement duration in seconds

## 🌐 Web Interface Features

- **Real-time Chat**: Interactive chat with AI assistant using multi-image context
- **Live Image Display**: Shows current robot camera feed with automatic updates
- **AI Analysis**: Real-time analysis of image sequences with reasoning
- **Bluetooth Management**: Device discovery, connection, and status monitoring
- **Manual Control**: Send preset or custom robot instructions
- **Settings Panel**: Toggle auto-execution and safety modes
- **Live Updates**: Real-time updates via WebSocket connection

## 📊 Logging System

### Log Files
- `logs/ai_analysis.log` - AI reasoning and decision-making process
- `logs/robot_instructions.log` - All robot commands and execution status
- `logs/system_events.log` - System events, connections, and errors

### Log Viewer
Use the built-in log viewer to monitor system activity:

```bash
# View all logs (last 50 lines)
python view_logs.py all

# View specific log type
python view_logs.py ai 100
python view_logs.py robot 50
python view_logs.py system 25

# Follow logs in real-time
python view_logs.py tail ai
python view_logs.py tail robot
```

### Log Rotation
- **Max file size**: 10MB per log file
- **Backup count**: 5 backup files (50MB total per log type)
- **Automatic rotation**: When files exceed size limit

## 🔑 API Key Setup

- `key.txt` - Required: Your OpenRouter API key file
  - Create a file named `key.txt` with your API key
  - Get your API key from: https://openrouter.ai/keys
  - Example: `echo "your-api-key-here" > key.txt`

## 💻 Example Robot Integration

```python
import requests

# Send image to server
with open("robot_image.jpg", "rb") as f:
    response = requests.post("http://localhost:8040/image/", data=f.read())

# Get AI analysis (automatic with multi-image context)
analysis = response.json()
print(f"AI suggests: {analysis['analysis']}")

# Connect to robot via Bluetooth
requests.post("http://localhost:8040/bluetooth/connect/", 
              json={"mac_address": "00:11:22:33:44:55"})

# Send instruction to robot via Bluetooth (JSON format)
requests.post("http://localhost:8040/send_instruction/", 
              json={"instruction": {"left_wheel": 30, "right_wheel": 20, "duration": 2}})

# Chat with AI for complex instructions
requests.post("http://localhost:8040/chat/", 
              json={"message": "Navigate around the obstacle and find the red object"})

# Start complex multi-step instruction
requests.post("http://localhost:8040/complex_instruction/", 
              json={"instruction": "Explore the room and map all objects"})
```

## 🛠️ Development

### Technology Stack
- **FastAPI**: Modern web framework for APIs
- **Qwen3-VL**: Multi-image analysis and agentic behavior
- **Bleak**: Bluetooth Low Energy communication
- **WebSocket**: Real-time communication
- **Background Tasks**: Asynchronous image processing
- **Logging**: Comprehensive file-based logging with rotation

### Project Structure
```
robot_server/
├── main.py                 # Main FastAPI application
├── bluetooth.py           # Bluetooth communication module
├── bluetooth_mock.py      # Mock Bluetooth for testing
├── view_logs.py           # Log viewer script
├── requirements.txt       # Python dependencies
├── key.txt               # OpenRouter API key
├── index.html            # Web interface
├── images/               # Robot images directory
├── logs/                 # Log files directory
│   ├── ai_analysis.log
│   ├── robot_instructions.log
│   └── system_events.log
└── README.md
```

## 🔧 Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure `key.txt` exists with valid OpenRouter API key
   - Check API key has sufficient credits
   - Verify key format: `echo "sk-or-v1-..." > key.txt`

2. **Dependencies**:
   - Run `pip install -r requirements.txt` to install all packages
   - For Bleak: `pip install bleak>=0.21.0`
   - For OpenAI: `pip install openai>=1.0.0`

3. **Bluetooth Issues**:
   - Ensure Bluetooth is enabled on your system
   - Check robot is in pairing/discoverable mode
   - Verify MAC address is correct (format: `XX:XX:XX:XX:XX:XX`)
   - On Linux: `sudo apt-get install bluetooth bluez libbluetooth-dev`
   - Check logs: `python view_logs.py system`

4. **Port Conflicts**:
   - Default port is 8040, change in `main.py` if needed
   - Check if port is already in use: `lsof -i :8040`

5. **WebSocket Issues**:
   - Check browser console for connection errors
   - Ensure WebSocket is supported in your browser
   - Try refreshing the page

6. **Image Display Issues**:
   - Check if `images/` directory exists
   - Verify image files are being saved correctly
   - Check logs: `python view_logs.py system`

7. **AI Analysis Issues**:
   - Check API key and credits
   - Verify image format (JPEG recommended)
   - Check logs: `python view_logs.py ai`

8. **Robot Instruction Issues**:
   - Verify Bluetooth connection status
   - Check instruction format (JSON or text)
   - Check logs: `python view_logs.py robot`

### Debug Commands

```bash
# Check server status
curl http://localhost:8040/settings/

# Check Bluetooth status
curl http://localhost:8040/bluetooth/status/

# View recent logs
python view_logs.py all 100

# Follow logs in real-time
python view_logs.py tail system
```

### Getting Help

- Check the logs first: `python view_logs.py all`
- Verify all dependencies are installed
- Ensure API key is valid and has credits
- Check Bluetooth connection and robot status
