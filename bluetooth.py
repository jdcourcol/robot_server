import asyncio
import logging
from datetime import datetime
from typing import Callable, Optional

try:
    from bleak import BleakClient, BleakScanner
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False
    logging.warning("Bleak not available. Install with: pip install bleak")

class BluetoothRobotController:
    """Bluetooth controller for robot communication using Bleak"""
    
    def __init__(self, robot_mac_address: Optional[str] = None, service_uuid: str = "0000180D-0000-1000-8000-00805F9B34FB"):
        self.robot_mac_address = robot_mac_address
        self.service_uuid = service_uuid
        self.client = None
        self.connected = False
        self.message_queue = asyncio.Queue()
        self.callbacks = []
        
    async def discover_devices(self) -> list:
        """Discover nearby Bluetooth devices"""
        if not BLUETOOTH_AVAILABLE:
            return []
            
        try:
            devices = await BleakScanner.discover(timeout=10.0)
            return [{"address": device.address, "name": device.name or "Unknown"} for device in devices]
        except Exception as e:
            logging.error(f"Error discovering devices: {e}")
            return []
    
    async def connect(self, mac_address: Optional[str] = None) -> bool:
        """Connect to robot via Bluetooth"""
        if not BLUETOOTH_AVAILABLE:
            logging.error("Bluetooth not available. Install Bleak.")
            return False
            
        target_address = mac_address or self.robot_mac_address
        if not target_address:
            logging.error("No MAC address provided for robot connection")
            return False
            
        try:
            self.client = BleakClient(target_address)
            await self.client.connect()
            
            # Services are automatically discovered in modern Bleak
            logging.info("Connected successfully, services will be available after connection")
            
            self.connected = True
            self.robot_mac_address = target_address
            
            logging.info(f"Connected to robot at {target_address}")
            
            # Start message processing loop
            asyncio.create_task(self._process_message_queue())
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to robot: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from robot"""
        if self.client:
            try:
                await self.client.disconnect()
                self.connected = False
                logging.info("Disconnected from robot")
            except Exception as e:
                logging.error(f"Error disconnecting: {e}")
    
    async def send_instruction(self, instruction, instruction_type: str = "command") -> bool:
        """Send instruction to robot via BLE characteristic in format: left: 20 right: 30 duration: 2"""
        if not self.connected or not self.client:
            logging.error("Not connected to robot")
            return False
            
        try:
            # Convert instruction to simple text format
            if isinstance(instruction, dict):
                left = instruction.get('left_wheel', 0)
                right = instruction.get('right_wheel', 0)
                duration = instruction.get('duration', 1)
                message_str = f"left: {left} right: {right} duration: {duration}\n"
            elif isinstance(instruction, str):
                # Try to parse JSON string first
                try:
                    import json
                    parsed = json.loads(instruction)
                    if isinstance(parsed, dict):
                        left = parsed.get('left_wheel', 0)
                        right = parsed.get('right_wheel', 0)
                        duration = parsed.get('duration', 1)
                        message_str = f"left: {left} right: {right} duration: {duration}\n"
                    else:
                        message_str = str(instruction) + "\n"
                except json.JSONDecodeError:
                    # If it's not JSON, use as is
                    message_str = str(instruction) + "\n"
            else:
                # If it's already a string, use it as is
                message_str = str(instruction) + "\n"
            print(f"Message string: {message_str}")
            # Wait a moment for services to be available (they're auto-discovered in modern Bleak)
            import asyncio
            max_retries = 5
            for attempt in range(max_retries):
                if hasattr(self.client, 'services') and self.client.services:
                    break
                logging.info(f"Waiting for services to be available... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(0.5)
            
            # Try to find a writable characteristic using the correct Bleak API
            services = self.client.services
            for service in services:
                for char in service.characteristics:
                    if "write" in char.properties:
                        await self.client.write_gatt_char(char.uuid, message_str.encode())
                        logging.info(f"Sent instruction to robot: {message_str.strip()}")
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            try:
                                await callback("instruction_sent", message_str.strip())
                            except Exception as e:
                                logging.error(f"Callback error: {e}")
                        
                        return True
            
            logging.error("No writable characteristic found")
            return False
            
        except Exception as e:
            logging.error(f"Failed to send instruction: {e}")
            return False
    
    async def send_raw_command(self, command: str) -> bool:
        """Send raw command string to robot"""
        if not self.connected or not self.client:
            logging.error("Not connected to robot")
            return False
            
        try:
            # Wait a moment for services to be available (they're auto-discovered in modern Bleak)
            import asyncio
            max_retries = 5
            for attempt in range(max_retries):
                if hasattr(self.client, 'services') and self.client.services:
                    break
                logging.info(f"Waiting for services to be available... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(0.5)
            
            # Try to find a writable characteristic
            services = self.client.services
            for service in services:
                for char in service.characteristics:
                    if "write" in char.properties:
                        await self.client.write_gatt_char(char.uuid, command.encode() + b"\n")
                        logging.info(f"Sent raw command: {command}")
                        return True
            
            logging.error("No writable characteristic found")
            return False
        except Exception as e:
            logging.error(f"Failed to send raw command: {e}")
            return False
    
    async def queue_instruction(self, instruction: str, instruction_type: str = "command"):
        """Queue instruction for sending"""
        await self.message_queue.put({
            "instruction": instruction,
            "type": instruction_type,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _process_message_queue(self):
        """Process queued messages"""
        while self.connected:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self.send_instruction(message["instruction"], message["type"])
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error processing message queue: {e}")
    
    def add_callback(self, callback: Callable):
        """Add callback for robot events"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def get_robot_status(self) -> dict:
        """Request robot status"""
        if not self.connected or not self.client:
            return {"status": "disconnected"}
            
        try:
            # For BLE, we'll return basic connection status
            # In a real implementation, you'd read from a specific characteristic
            return {
                "status": "connected",
                "address": self.robot_mac_address,
                "services": len(self.client.services)
            }
                
        except Exception as e:
            logging.error(f"Error getting robot status: {e}")
            return {"status": "error", "error": str(e)}

# Global Bluetooth controller instance
bluetooth_controller = BluetoothRobotController()

# Convenience functions
async def connect_to_robot(mac_address: str) -> bool:
    """Connect to robot with given MAC address"""
    return await bluetooth_controller.connect(mac_address)

async def send_robot_instruction(instruction: str, instruction_type: str = "command") -> bool:
    """Send instruction to robot"""
    return await bluetooth_controller.send_instruction(instruction, instruction_type)

async def queue_robot_instruction(instruction: str, instruction_type: str = "command"):
    """Queue instruction for robot"""
    await bluetooth_controller.queue_instruction(instruction, instruction_type)

async def disconnect_from_robot():
    """Disconnect from robot"""
    await bluetooth_controller.disconnect()

async def discover_robot_devices() -> list:
    """Discover available robot devices"""
    return await bluetooth_controller.discover_devices()

def is_bluetooth_available() -> bool:
    """Check if Bluetooth is available"""
    return BLUETOOTH_AVAILABLE

def is_connected() -> bool:
    """Check if connected to robot"""
    return bluetooth_controller.connected