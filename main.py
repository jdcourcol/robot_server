import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from typing import List, Optional
from logging.handlers import RotatingFileHandler

import openai
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

# Use real Bluetooth with Bleak
from bluetooth import (
    bluetooth_controller,
    connect_to_robot,
    send_robot_instruction,
    queue_robot_instruction,
    disconnect_from_robot,
    discover_robot_devices,
    is_bluetooth_available,
    is_connected,
)

# Uncomment below for mock Bluetooth (for testing without hardware)
# from bluetooth_mock import (
#     bluetooth_controller,
#     connect_to_robot,
#     disconnect_from_robot,
#     discover_robot_devices,
#     is_bluetooth_available,
#     is_connected,
#     queue_robot_instruction,
#     send_robot_instruction,
# )

app = FastAPI(
    title="Robot Control Server",
    description="AI-powered robot control with OpenAI integration",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load OpenAI API key from key.txt file
def load_api_key():
    """Load API key from key.txt file"""
    try:
        with open("key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ key.txt file not found. Please create it with your API key.")
        return None


API_KEY = load_api_key()


# Setup logging to file
def setup_file_logging():
    """Setup file logging for AI reasoning and robot instructions"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # AI Analysis Logger
    ai_logger = logging.getLogger("ai_analysis")
    ai_logger.setLevel(logging.INFO)
    ai_handler = RotatingFileHandler(
        "logs/ai_analysis.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    ai_handler.setFormatter(detailed_formatter)
    ai_logger.addHandler(ai_handler)

    # Robot Instructions Logger
    robot_logger = logging.getLogger("robot_instructions")
    robot_logger.setLevel(logging.INFO)
    robot_handler = RotatingFileHandler(
        "logs/robot_instructions.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    robot_handler.setFormatter(detailed_formatter)
    robot_logger.addHandler(robot_handler)

    # System Events Logger
    system_logger = logging.getLogger("system_events")
    system_logger.setLevel(logging.INFO)
    system_handler = RotatingFileHandler(
        "logs/system_events.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    system_handler.setFormatter(detailed_formatter)
    system_logger.addHandler(system_handler)

    return ai_logger, robot_logger, system_logger


# Initialize file loggers
ai_logger, robot_logger, system_logger = setup_file_logging()

# Global variables for robot state
current_image_path = None
robot_instructions = []
chat_history = []
active_connections: List[WebSocket] = []

# Multi-image context for Qwen3-VL
image_sequence = []  # Store recent images for multi-image analysis
max_image_context = 5  # Keep last 5 images for context

# Robot instruction queue
instruction_queue = asyncio.Queue()

# Auto-execution settings
auto_execute_instructions = True  # Set to False to require manual approval
safety_mode = True  # Add safety checks before executing commands

# Agentic instruction tracking
active_instruction_chain = None  # Current complex instruction being followed
instruction_step = 0  # Current step in multi-step instruction


# Instruction parsing and validation


def is_safe_instruction(instruction: str) -> bool:
    """Check if instruction is safe to execute automatically"""
    if not safety_mode:
        return True

    dangerous_commands = [
        "destroy",
        "break",
        "damage",
        "harm",
        "hurt",
        "delete",
        "remove permanently",
        "force",
        "push hard",
    ]

    instruction_lower = instruction.lower()

    for dangerous in dangerous_commands:
        if dangerous in instruction_lower:
            return False

    return True


def parse_robot_instruction(analysis: str) -> Optional[dict]:
    """Extract robot instruction in the specific JSON format: {'left_wheel': 30, 'right_wheel': 20, 'duration': 2}"""
    import re
    import json

    # Look for JSON-like patterns in the analysis
    json_patterns = [
        r"\{[^}]*left_wheel[^}]*right_wheel[^}]*duration[^}]*\}",
        r'\{[^}]*"left_wheel"[^}]*"right_wheel"[^}]*"duration"[^}]*\}',
        r"left_wheel[:\s]*(-?\d+)[,\s]*right_wheel[:\s]*(-?\d+)[,\s]*duration[:\s]*(\d+)",
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, analysis, re.IGNORECASE)
        for match in matches:
            try:
                if isinstance(match, tuple):
                    # Extract numbers from tuple match
                    left_wheel = int(match[0])
                    right_wheel = int(match[1])
                    duration = int(match[2])
                    return {
                        "left_wheel": left_wheel,
                        "right_wheel": right_wheel,
                        "duration": duration,
                    }
                else:
                    # Try to parse as JSON
                    instruction = json.loads(match)
                    if all(
                        key in instruction
                        for key in ["left_wheel", "right_wheel", "duration"]
                    ):
                        return {
                            "left_wheel": int(instruction["left_wheel"]),
                            "right_wheel": int(instruction["right_wheel"]),
                            "duration": int(instruction["duration"]),
                        }
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    # Look for simple movement commands and convert to JSON format
    analysis_lower = analysis.lower()

    # Forward movement
    if any(
        word in analysis_lower for word in ["move forward", "go forward", "advance"]
    ):
        return {"left_wheel": 50, "right_wheel": 50, "duration": 2}

    # Backward movement
    if any(
        word in analysis_lower
        for word in ["move back", "go back", "reverse", "backward"]
    ):
        return {"left_wheel": -50, "right_wheel": -50, "duration": 2}

    # Turn left
    if any(word in analysis_lower for word in ["turn left", "go left", "left"]):
        return {"left_wheel": -30, "right_wheel": 30, "duration": 2}

    # Turn right
    if any(word in analysis_lower for word in ["turn right", "go right", "right"]):
        return {"left_wheel": 30, "right_wheel": -30, "duration": 2}

    # Stop
    if any(word in analysis_lower for word in ["stop", "halt", "pause"]):
        return {"left_wheel": 0, "right_wheel": 0, "duration": 1}

    return None


async def execute_robot_instruction(
    instruction_dict: dict, source: str = "ai_analysis"
) -> bool:
    """Execute robot instruction in JSON format: {'left_wheel': 30, 'right_wheel': 20, 'duration': 2}"""
    if not instruction_dict:
        return False

    # Check if connected to robot
    if not is_connected():
        logging.warning("Not connected to robot - cannot execute instruction")
        return False

    # Validate instruction format
    required_keys = ["left_wheel", "right_wheel", "duration"]
    if not all(key in instruction_dict for key in required_keys):
        logging.warning(f"Invalid instruction format: {instruction_dict}")
        return False

    # Validate instruction safety
    instruction_str = f"left_wheel: {instruction_dict['left_wheel']}, right_wheel: {instruction_dict['right_wheel']}, duration: {instruction_dict['duration']}"
    if not is_safe_instruction(instruction_str):
        logging.warning(f"Unsafe instruction blocked: {instruction_str}")
        await broadcast_to_clients(
            {
                "type": "instruction_blocked",
                "instruction": instruction_str,
                "reason": "safety_check_failed",
            }
        )
        return False

    # Send instruction to robot as JSON string
    instruction_json = json.dumps(instruction_dict)
    success = await send_robot_instruction(instruction_json, "auto_command")

    if success:
        # Log the execution to file
        robot_logger.info(f"=== ROBOT INSTRUCTION EXECUTED ===")
        robot_logger.info(f"Source: {source}")
        robot_logger.info(f"Instruction JSON: {instruction_json}")
        robot_logger.info(f"Parsed Instruction: {instruction_dict}")
        robot_logger.info(f"Status: SUCCESS")
        robot_logger.info(f"Method: Bluetooth")
        robot_logger.info(f"================================")

        # Log the execution
        execution_log = {
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction_dict,
            "source": source,
            "status": "executed",
            "method": "bluetooth",
        }
        robot_instructions.append(execution_log)

        # Broadcast to clients
        await broadcast_to_clients(
            {
                "type": "instruction_executed",
                "instruction": instruction_dict,
                "source": source,
            }
        )

        logging.info(f"Executed robot instruction: {instruction_dict} (from {source})")
        return True
    else:
        # Log failed execution
        robot_logger.error(f"=== ROBOT INSTRUCTION FAILED ===")
        robot_logger.error(f"Source: {source}")
        robot_logger.error(f"Instruction JSON: {instruction_json}")
        robot_logger.error(f"Parsed Instruction: {instruction_dict}")
        robot_logger.error(f"Status: FAILED")
        robot_logger.error(f"Method: Bluetooth")
        robot_logger.error(f"=============================")

        logging.error(f"Failed to execute robot instruction: {instruction_dict}")
        return False


# OpenAI Functions
async def analyze_multi_image_context(user_message: str = None) -> str:
    """Analyze multiple recent images with Qwen3-VL for complex agentic interactions"""
    try:
        if not image_sequence:
            return "No images available for analysis"

        # Prepare multi-image messages for Qwen3-VL
        messages = [
            {
                "role": "system",
                "content": """You are an advanced robot control AI with multi-image analysis capabilities. You can:
                1. Analyze sequences of images to understand movement and changes over time
                2. Follow complex multi-step instructions across multiple images
                3. Track objects and changes between images
                4. Provide temporal analysis of what happened between images
                5. Execute complex workflows that require multiple observations
                
                You have access to a sequence of recent camera images. Analyze them to understand:
                - What has changed between images
                - Where the robot should move next
                - What objects or obstacles are present
                - The temporal progression of events

                The image is from a camera mounted on the robot and generated every 5 seconds.
                
                IMPORTANT: When providing robot movement instructions, use this EXACT JSON format:
                {"left_wheel": 30, "right_wheel": 20, "duration": 2}
                
                Where:
                - left_wheel: Power percentage for left wheel (-100 to 100, negative = reverse)
                - right_wheel: Power percentage for right wheel (-100 to 100, negative = reverse)  
                - duration: Duration of movement in seconds
                
                Examples:
                - Move forward: {"left_wheel": 50, "right_wheel": 50, "duration": 2}
                - Turn left: {"left_wheel": -30, "right_wheel": 30, "duration": 2}
                - Turn right: {"left_wheel": 30, "right_wheel": -30, "duration": 2}
                - Move backward: {"left_wheel": -50, "right_wheel": -50, "duration": 2}
                - Stop: {"left_wheel": 0, "right_wheel": 0, "duration": 1}
                
                Provide detailed analysis and include the robot instruction JSON when movement is needed.""",
            }
        ]

        # Add user message if provided
        if user_message:
            messages.append({"role": "user", "content": user_message})

        # Add multi-image content
        image_content = []
        for i, img_path in enumerate(image_sequence):
            if os.path.exists(img_path):
                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    )

        if image_content:
            messages.append({"role": "user", "content": image_content})

        # Add chat history context
        for msg in chat_history[-6:]:
            messages.insert(-1, msg)

        client = openai.AsyncOpenAI(
            api_key=API_KEY, base_url="https://openrouter.ai/api/v1"
        )
        response = await client.chat.completions.create(
            model="qwen/qwen3-vl-235b-a22b-thinking", messages=messages, max_tokens=800
        )

        analysis = response.choices[0].message.content

        # Log AI analysis to file
        ai_logger.info(f"=== AI ANALYSIS ===")
        ai_logger.info(f"User Message: {user_message or 'Image analysis request'}")
        ai_logger.info(f"Images in sequence: {len(image_sequence)}")
        ai_logger.info(f"Image paths: {image_sequence}")
        ai_logger.info(f"AI Response: {analysis}")
        ai_logger.info(f"==================")

        # Add analysis to chat history
        chat_history.append(
            {"role": "assistant", "content": f"Multi-image analysis: {analysis}"}
        )

        return analysis
    except Exception as e:
        return f"Error in multi-image analysis: {str(e)}"


async def analyze_image_with_openai(image_path: str, user_message: str = None) -> str:
    """Analyze image using multi-image context analysis"""
    try:
        # Use multi-image context analysis for all image analysis
        analysis = await analyze_multi_image_context(
            user_message or "Analyze the latest image and provide robot instructions"
        )

        # Auto-execute instruction if enabled
        if auto_execute_instructions:
            instruction_dict = parse_robot_instruction(analysis)
            if instruction_dict:
                await execute_robot_instruction(instruction_dict, "ai_image_analysis")

        return analysis

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


async def chat_with_openai(message: str) -> str:
    """Handle chat messages using multi-image context analysis"""
    try:
        # Add user message to chat history
        chat_history.append({"role": "user", "content": message})

        # Use multi-image context analysis for all chat interactions
        analysis = await analyze_multi_image_context(message)

        # Auto-execute instruction if enabled and it's a command
        if auto_execute_instructions:
            instruction_dict = parse_robot_instruction(analysis)
            if instruction_dict:
                await execute_robot_instruction(instruction_dict, "ai_chat")

        return analysis

    except Exception as e:
        return f"Error in chat: {str(e)}"


async def broadcast_to_clients(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            active_connections.remove(connection)


from logging.config import dictConfig

sample_logger = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s :: %(client_addr)s - "%(request_line)s" %(status_code)s',
            "use_colors": True,
        },
    },
    "handlers": {
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

dictConfig(sample_logger)


@app.post("/image/")
async def upload_image(request: Request, background_tasks: BackgroundTasks):
    """Receive image from robot and analyze it with OpenAI"""
    global current_image_path

    data = await request.body()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"robot_image_{timestamp}.jpg"

    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Save the image in the images subdirectory
    image_path = os.path.join("images", filename)
    with open(image_path, "wb") as buffer:
        buffer.write(data)

    current_image_path = image_path

    # Add to image sequence for multi-image analysis
    image_sequence.append(image_path)
    if len(image_sequence) > max_image_context:
        image_sequence.pop(0)  # Remove oldest image

    # Log image received event
    system_logger.info(f"=== IMAGE RECEIVED ===")
    system_logger.info(f"Filename: {filename}")
    system_logger.info(f"Path: {image_path}")
    system_logger.info(f"Image sequence length: {len(image_sequence)}")
    system_logger.info(f"====================")

    # Analyze image with OpenAI in background
    background_tasks.add_task(process_image_with_ai, image_path)

    # Broadcast to WebSocket clients
    await broadcast_to_clients(
        {"type": "image_received", "filename": filename, "timestamp": timestamp}
    )

    return {"status": "ok", "filename": filename}


async def process_image_with_ai(image_path: str):
    """Process image with OpenAI and broadcast results"""
    try:
        analysis = await analyze_image_with_openai(image_path)

        # Add to robot instructions
        instruction = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "analysis": analysis,
            "status": "pending",
        }
        robot_instructions.append(instruction)

        # Broadcast analysis to clients
        await broadcast_to_clients(
            {"type": "image_analysis", "analysis": analysis, "image_path": image_path}
        )

    except Exception as e:
        await broadcast_to_clients(
            {"type": "error", "message": f"Error processing image: {str(e)}"}
        )


@app.post("/chat/")
async def chat_endpoint(request: Request):
    """Handle chat messages"""
    data = await request.json()
    message = data.get("message", "")

    if not message:
        return {"error": "No message provided"}

    # Get response from OpenAI
    response = await chat_with_openai(message)

    # Broadcast to WebSocket clients
    await broadcast_to_clients(
        {"type": "chat_response", "user_message": message, "ai_response": response}
    )

    return {"response": response}


@app.post("/send_instruction/")
async def send_instruction_to_robot(request: Request):
    """Send instruction to robot via Bluetooth in JSON format"""
    data = await request.json()
    instruction = data.get("instruction", {})
    instruction_type = data.get("type", "command")

    if not instruction:
        return {"error": "No instruction provided"}

    # Check if instruction is in correct format
    # if not isinstance(instruction, dict) or not all(key in instruction for key in ['left_wheel', 'right_wheel', 'duration']):
    #     return {"error": "Instruction must be in format: {'left_wheel': 30, 'right_wheel': 20, 'duration': 2}"}

    # Check Bluetooth connection
    if not is_connected():
        return {"error": "Not connected to robot via Bluetooth"}

    # Send instruction via Bluetooth as JSON string
    # instruction_json = json.dumps(instruction)
    success = await send_robot_instruction(
        instruction=instruction, instruction_type=instruction_type
    )

    if success:
        # Log manual instruction to file
        robot_logger.info(f"=== MANUAL ROBOT INSTRUCTION ===")
        robot_logger.info(f"Type: {instruction_type}")
        robot_logger.info(f"Instruction: {instruction}")
        robot_logger.info(f"Status: SUCCESS")
        robot_logger.info(f"Method: Bluetooth")
        robot_logger.info(f"==============================")

        # Add to instruction queue for tracking
        await instruction_queue.put(
            {
                "instruction": instruction,
                "type": instruction_type,
                "timestamp": datetime.now().isoformat(),
                "status": "sent_via_bluetooth",
            }
        )

        # Broadcast to WebSocket clients
        await broadcast_to_clients(
            {
                "type": "instruction_sent",
                "instruction": instruction,
                "method": "bluetooth",
            }
        )

        return {
            "status": "instruction_sent",
            "instruction": instruction,
            "method": "bluetooth",
        }
    else:
        # Log failed manual instruction
        robot_logger.error(f"=== MANUAL ROBOT INSTRUCTION FAILED ===")
        robot_logger.error(f"Type: {instruction_type}")
        robot_logger.error(f"Instruction: {instruction}")
        robot_logger.error(f"Status: FAILED")
        robot_logger.error(f"Method: Bluetooth")
        robot_logger.error(f"===================================")

        return {"error": "Failed to send instruction via Bluetooth"}


@app.get("/instructions/")
async def get_instructions():
    """Get all robot instructions"""
    return {"instructions": robot_instructions}


@app.get("/chat_history/")
async def get_chat_history():
    """Get chat history"""
    return {"chat_history": chat_history}


@app.post("/complex_instruction/")
async def start_complex_instruction(request: Request):
    """Start a complex multi-step instruction that requires agentic behavior"""
    global active_instruction_chain, instruction_step

    data = await request.json()
    instruction = data.get("instruction", "")

    if not instruction:
        return {"error": "No instruction provided"}

    # Set up complex instruction tracking
    active_instruction_chain = {
        "instruction": instruction,
        "start_time": datetime.now().isoformat(),
        "steps": [],
        "status": "active",
    }
    instruction_step = 0

    # Analyze current context with the complex instruction
    analysis = await analyze_multi_image_context(
        f"Complex instruction: {instruction}. What should be the first step?"
    )

    # Broadcast to clients
    await broadcast_to_clients(
        {
            "type": "complex_instruction_started",
            "instruction": instruction,
            "first_analysis": analysis,
        }
    )

    return {
        "status": "complex_instruction_started",
        "instruction": instruction,
        "analysis": analysis,
    }


@app.get("/instruction_status/")
async def get_instruction_status():
    """Get current complex instruction status"""
    return {
        "active_instruction": active_instruction_chain,
        "current_step": instruction_step,
        "image_sequence_length": len(image_sequence),
    }


# Auto-execution control endpoints
@app.get("/settings/")
async def get_settings():
    """Get current auto-execution settings"""
    return {
        "auto_execute_instructions": auto_execute_instructions,
        "safety_mode": safety_mode,
        "bluetooth_connected": is_connected(),
    }


@app.post("/settings/auto_execute/")
async def toggle_auto_execute(request: Request):
    """Toggle auto-execution of AI instructions"""
    global auto_execute_instructions
    data = await request.json()
    auto_execute_instructions = data.get("enabled", True)

    await broadcast_to_clients(
        {"type": "settings_updated", "auto_execute": auto_execute_instructions}
    )

    return {"auto_execute_instructions": auto_execute_instructions}


@app.post("/settings/safety_mode/")
async def toggle_safety_mode(request: Request):
    """Toggle safety mode for instruction validation"""
    global safety_mode
    data = await request.json()
    safety_mode = data.get("enabled", True)

    await broadcast_to_clients({"type": "settings_updated", "safety_mode": safety_mode})

    return {"safety_mode": safety_mode}


# Bluetooth Management Endpoints
@app.get("/bluetooth/status/")
async def get_bluetooth_status():
    """Get Bluetooth connection status"""
    return {
        "bluetooth_available": is_bluetooth_available(),
        "connected": is_connected(),
        "robot_address": bluetooth_controller.robot_mac_address
        if is_connected()
        else None,
    }


@app.post("/bluetooth/connect/")
async def connect_bluetooth(request: Request):
    """Connect to robot via Bluetooth"""
    data = await request.json()
    mac_address = data.get("mac_address")

    if not mac_address:
        return {"error": "MAC address required"}

    success = await connect_to_robot(mac_address)

    if success:
        system_logger.info(f"=== BLUETOOTH CONNECTED ===")
        system_logger.info(f"MAC Address: {mac_address}")
        system_logger.info(f"Status: SUCCESS")
        system_logger.info(f"========================")

        await broadcast_to_clients(
            {"type": "bluetooth_connected", "mac_address": mac_address}
        )
        return {"status": "connected", "mac_address": mac_address}
    else:
        system_logger.error(f"=== BLUETOOTH CONNECTION FAILED ===")
        system_logger.error(f"MAC Address: {mac_address}")
        system_logger.error(f"Status: FAILED")
        system_logger.error(f"===============================")

        return {"error": "Failed to connect to robot"}


@app.post("/bluetooth/disconnect/")
async def disconnect_bluetooth():
    """Disconnect from robot"""
    await disconnect_from_robot()

    system_logger.info(f"=== BLUETOOTH DISCONNECTED ===")
    system_logger.info(f"Status: SUCCESS")
    system_logger.info(f"============================")

    await broadcast_to_clients({"type": "bluetooth_disconnected"})

    return {"status": "disconnected"}


@app.get("/bluetooth/discover/")
async def discover_bluetooth_devices():
    """Discover nearby Bluetooth devices"""
    devices = await discover_robot_devices()
    return {"devices": devices}


@app.post("/bluetooth/queue_instruction/")
async def queue_bluetooth_instruction(request: Request):
    """Queue instruction for Bluetooth sending"""
    data = await request.json()
    instruction = data.get("instruction", "")
    instruction_type = data.get("type", "command")

    if not instruction:
        return {"error": "No instruction provided"}

    await queue_robot_instruction(instruction, instruction_type)

    await broadcast_to_clients(
        {"type": "instruction_queued", "instruction": instruction}
    )

    return {"status": "instruction_queued", "instruction": instruction}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "chat":
                response = await chat_with_openai(message.get("message", ""))
                await websocket.send_text(
                    json.dumps({"type": "chat_response", "response": response})
                )

            elif message.get("type") == "analyze_image" and current_image_path:
                analysis = await analyze_image_with_openai(
                    current_image_path, message.get("message", "")
                )
                await websocket.send_text(
                    json.dumps({"type": "image_analysis", "analysis": analysis})
                )

            elif message.get("type") == "complex_instruction":
                instruction = message.get("instruction", "")
                if instruction:
                    # Start complex instruction
                    global active_instruction_chain, instruction_step
                    active_instruction_chain = {
                        "instruction": instruction,
                        "start_time": datetime.now().isoformat(),
                        "steps": [],
                        "status": "active",
                    }
                    instruction_step = 0

                    analysis = await analyze_multi_image_context(
                        f"Complex instruction: {instruction}. What should be the first step?"
                    )
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "complex_instruction_started",
                                "instruction": instruction,
                                "analysis": analysis,
                            }
                        )
                    )

            elif message.get("type") == "bluetooth_status":
                status = {
                    "bluetooth_available": is_bluetooth_available(),
                    "connected": is_connected(),
                    "robot_address": bluetooth_controller.robot_mac_address
                    if is_connected()
                    else None,
                }
                await websocket.send_text(
                    json.dumps({"type": "bluetooth_status", "status": status})
                )

            elif message.get("type") == "discover_devices":
                devices = await discover_robot_devices()
                await websocket.send_text(
                    json.dumps({"type": "bluetooth_devices", "devices": devices})
                )

    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.get("/image/{filename}")
async def get_image(filename: str):
    """Serve robot images"""
    # Check in images directory first
    image_path = os.path.join("images", filename)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/jpeg")
    # Fallback to root directory for backward compatibility
    elif os.path.exists(filename):
        return FileResponse(filename, media_type="image/jpeg")
    else:
        return {"error": "Image not found"}


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main interface"""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8040)
