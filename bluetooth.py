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
        self.writable_char_uuid = None  # Cache the writable characteristic UUID
        
    def _on_disconnect(self, client):
        """Callback when BLE device disconnects"""
        logging.warning(f"⚠️  Robot disconnected unexpectedly! Client: {client.address if hasattr(client, 'address') else 'unknown'}")
        self.connected = False
        self.writable_char_uuid = None
    
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
            # Create client with disconnect callback
            self.client = BleakClient(target_address, disconnected_callback=self._on_disconnect)
            await self.client.connect()
            
            # Explicitly get services (required in some Bleak versions)
            logging.info("Connected, discovering services...")
            try:
                # In modern Bleak, services are auto-discovered, but we wait for them
                await asyncio.sleep(1)  # Give time for auto-discovery
                
                # Access services to ensure they're loaded
                services = self.client.services
                if not services:
                    logging.error("No services found on device")
                    await self.client.disconnect()
                    return False
                
                # BleakGATTServiceCollection is iterable but not a list
                service_list = list(services)
                logging.info(f"Found {len(service_list)} services")
                
                # Cache the writable characteristic UUID for later use
                self.writable_char_uuid = None
                for service in service_list:
                    logging.info(f"Service: {service.uuid}")
                    for char in service.characteristics:
                        logging.info(f"  Characteristic: {char.uuid}, Properties: {char.properties}")
                        if "write" in char.properties or "write-without-response" in char.properties:
                            self.writable_char_uuid = char.uuid
                            logging.info(f"✓ Cached writable characteristic: {self.writable_char_uuid}")
                            break
                    if self.writable_char_uuid:
                        break
                
                if not self.writable_char_uuid:
                    logging.error("No writable characteristic found on device")
                    await self.client.disconnect()
                    return False
                    
            except Exception as e:
                logging.error(f"Error during service discovery: {e}")
                import traceback
                logging.error(traceback.format_exc())
                await self.client.disconnect()
                return False
            
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
                self.writable_char_uuid = None  # Clear cached characteristic
                logging.info("Disconnected from robot")
            except Exception as e:
                logging.error(f"Error disconnecting: {e}")
    
    async def send_instruction(self, instruction, instruction_type: str = "command") -> bool:
        """Send instruction to robot via BLE characteristic in format: left: 20 right: 30 duration: 2"""
        # Always check actual BLE connection state, not just our flag
        if not self.client or not self.client.is_connected:
            logging.warning(f"Not connected (client exists: {self.client is not None}, connected: {self.client.is_connected if self.client else False})")
            
            # Try to reconnect if we have a MAC address
            if self.robot_mac_address:
                logging.info(f"🔄 Attempting to reconnect to {self.robot_mac_address}...")
                self.connected = False
                self.writable_char_uuid = None
                
                success = await self.connect(self.robot_mac_address)
                if not success:
                    logging.error("❌ Failed to reconnect to robot")
                    return False
                logging.info("✅ Successfully reconnected to robot")
            else:
                logging.error("❌ Cannot reconnect - no MAC address stored")
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
            logging.info(f"Preparing to send: {message_str.strip()}")
            logging.info(f"Connection status - connected flag: {self.connected}, client.is_connected: {self.client.is_connected if self.client else 'N/A'}")
            
            # Use cached characteristic if available
            if self.writable_char_uuid:
                try:
                    logging.info(f"Using cached characteristic: {self.writable_char_uuid}")
                    await self.client.write_gatt_char(self.writable_char_uuid, message_str.encode(), response=False)
                    logging.info(f"✓ Successfully sent instruction: {message_str.strip()}")
                    
                    # Verify connection is still alive after send
                    if not self.client.is_connected:
                        logging.warning("⚠️  Connection dropped after sending instruction")
                        self.connected = False
                        self.writable_char_uuid = None
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        try:
                            await callback("instruction_sent", message_str.strip())
                        except Exception as e:
                            logging.error(f"Callback error: {e}")
                    
                    return True
                except Exception as e:
                    logging.error(f"Failed to send using cached characteristic: {e}")
                    logging.error(f"Characteristic UUID was: {self.writable_char_uuid}")
                    logging.error(f"Client connected: {self.client.is_connected if self.client else 'No client'}")
                    
                    # If connection was lost, mark as disconnected
                    if self.client and not self.client.is_connected:
                        logging.error("Connection was lost during send attempt")
                        self.connected = False
                        self.writable_char_uuid = None
                    
                    # Fall through to error
                    return False
            
            # Fallback: characteristic cache failed, should not happen
            logging.error("Characteristic cache was not available or failed")
            logging.error("This should not happen - please reconnect to the robot")
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
            # Use cached characteristic if available
            if self.writable_char_uuid:
                try:
                    logging.info(f"Sending raw command using cached characteristic: {self.writable_char_uuid}")
                    await self.client.write_gatt_char(self.writable_char_uuid, command.encode() + b"\n", response=False)
                    logging.info(f"✓ Successfully sent raw command: {command}")
                    return True
                except Exception as e:
                    logging.error(f"Failed to send raw command using cached characteristic: {e}")
                    logging.error(f"Characteristic UUID was: {self.writable_char_uuid}")
                    return False
            
            # Fallback: characteristic cache not available
            logging.error("Characteristic cache was not available")
            logging.error("This should not happen - please reconnect to the robot")
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