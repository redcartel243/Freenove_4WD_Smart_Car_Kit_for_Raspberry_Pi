"""
Robot Connection Module
Handles TCP communication with the Raspberry Pi robot
"""

import socket
import struct
import threading
import time
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from typing import Optional, Callable
from dataclasses import dataclass
from queue import Queue


@dataclass
class SensorData:
    """Current sensor readings from the robot"""
    ultrasonic_distance: float = 0.0  # cm
    light_left: float = 0.0  # voltage
    light_right: float = 0.0  # voltage
    line_sensors: int = 0  # 3-bit value (0-7)
    battery_voltage: float = 0.0


class RobotConnection:
    """Manages connection to the Freenove 4WD robot"""

    # Command constants (matching the robot's protocol)
    CMD_MOTOR = "CMD_MOTOR"
    CMD_M_MOTOR = "CMD_M_MOTOR"
    CMD_CAR_ROTATE = "CMD_CAR_ROTATE"
    CMD_SERVO = "CMD_SERVO"
    CMD_LED = "CMD_LED"
    CMD_LED_MOD = "CMD_LED_MOD"
    CMD_BUZZER = "CMD_BUZZER"
    CMD_SONIC = "CMD_SONIC"
    CMD_LIGHT = "CMD_LIGHT"
    CMD_POWER = "CMD_POWER"
    CMD_MODE = "CMD_MODE"
    CMD_LINE = "CMD_LINE"

    def __init__(self, ip: str, command_port: int = 5000, video_port: int = 8000):
        self.ip = ip
        self.command_port = command_port
        self.video_port = video_port

        self.command_socket: Optional[socket.socket] = None
        self.video_socket: Optional[socket.socket] = None

        self.connected = False
        self.video_streaming = False

        self.sensor_data = SensorData()
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.sensor_lock = threading.Lock()

        self._recv_thread: Optional[threading.Thread] = None
        self._video_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.on_frame_callback: Optional[Callable[[np.ndarray], None]] = None

    def connect(self) -> bool:
        """Connect to the robot's command and video servers"""
        try:
            # Connect command socket
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.command_socket.settimeout(5)
            self.command_socket.connect((self.ip, self.command_port))
            self.command_socket.settimeout(None)

            # Connect video socket
            self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_socket.settimeout(5)
            self.video_socket.connect((self.ip, self.video_port))
            self.video_socket.settimeout(None)

            self.connected = True
            self._stop_event.clear()

            # Start receiver threads
            self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._recv_thread.start()

            self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self._video_thread.start()

            self.video_streaming = True
            print(f"Connected to robot at {self.ip}")
            return True

        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Disconnect from the robot"""
        self._stop_event.set()
        self.connected = False
        self.video_streaming = False

        # Stop motors before disconnecting
        try:
            self.stop()
        except:
            pass

        if self.command_socket:
            try:
                self.command_socket.close()
            except:
                pass
            self.command_socket = None

        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass
            self.video_socket = None

        print("Disconnected from robot")

    def _send_command(self, command: str):
        """Send a command to the robot"""
        if not self.connected or not self.command_socket:
            return False
        try:
            self.command_socket.send(command.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False

    def _receive_loop(self):
        """Background thread to receive sensor data from robot"""
        buffer = ""
        while not self._stop_event.is_set() and self.connected:
            try:
                data = self.command_socket.recv(1024).decode('utf-8')
                if not data:
                    break

                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    self._parse_response(line)

            except socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Receive error: {e}")
                break

    def _parse_response(self, response: str):
        """Parse sensor data responses from robot"""
        parts = response.strip().split('#')
        if len(parts) < 2:
            return

        cmd = parts[0]
        with self.sensor_lock:
            if cmd == self.CMD_SONIC and len(parts) >= 2:
                try:
                    self.sensor_data.ultrasonic_distance = float(parts[1])
                except ValueError:
                    pass
            elif cmd == self.CMD_LIGHT and len(parts) >= 3:
                try:
                    self.sensor_data.light_left = float(parts[1])
                    self.sensor_data.light_right = float(parts[2])
                except ValueError:
                    pass
            elif cmd == self.CMD_POWER and len(parts) >= 2:
                try:
                    self.sensor_data.battery_voltage = float(parts[1])
                except ValueError:
                    pass
            elif cmd == self.CMD_LINE and len(parts) >= 2:
                try:
                    self.sensor_data.line_sensors = int(parts[1])
                except ValueError:
                    pass

    def _video_loop(self):
        """Background thread to receive video frames"""
        connection = self.video_socket.makefile('rb')

        while not self._stop_event.is_set() and self.video_streaming:
            try:
                # Read 4-byte length header
                header = connection.read(4)
                if len(header) < 4:
                    continue

                length = struct.unpack('<L', header)[0]

                # Read JPEG data
                jpg_data = connection.read(length)
                if len(jpg_data) < length:
                    continue

                # Decode frame
                frame = cv2.imdecode(
                    np.frombuffer(jpg_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame

                    if self.on_frame_callback:
                        self.on_frame_callback(frame)

            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"Video error: {e}")
                break

    # ==================== Motor Control ====================

    def set_motors(self, left_upper: int, left_lower: int, right_upper: int, right_lower: int):
        """
        Set motor speeds directly.
        Values range from -4095 to 4095.
        Positive = forward, Negative = backward
        """
        cmd = f"{self.CMD_MOTOR}#{left_upper}#{left_lower}#{right_upper}#{right_lower}\n"
        self._send_command(cmd)

    def move_forward(self, speed: int = 1500):
        """Move forward at specified speed (0-4095)"""
        self.set_motors(speed, speed, speed, speed)

    def move_backward(self, speed: int = 1500):
        """Move backward at specified speed (0-4095)"""
        self.set_motors(-speed, -speed, -speed, -speed)

    def turn_left(self, speed: int = 1500):
        """Turn left in place"""
        self.set_motors(-speed, -speed, speed, speed)

    def turn_right(self, speed: int = 1500):
        """Turn right in place"""
        self.set_motors(speed, speed, -speed, -speed)

    def strafe_left(self, speed: int = 1500):
        """Strafe left (mecanum wheels)"""
        cmd = f"{self.CMD_M_MOTOR}#90#{speed}#0#0\n"
        self._send_command(cmd)

    def strafe_right(self, speed: int = 1500):
        """Strafe right (mecanum wheels)"""
        cmd = f"{self.CMD_M_MOTOR}#-90#{speed}#0#0\n"
        self._send_command(cmd)

    def stop(self):
        """Stop all motors"""
        self.set_motors(0, 0, 0, 0)

    # ==================== Servo Control ====================

    def set_servo(self, channel: int, angle: int):
        """
        Set servo position.
        Channel 0 = pan (left/right), Channel 1 = tilt (up/down)
        Angle: 0-180 degrees
        """
        angle = max(0, min(180, angle))
        cmd = f"{self.CMD_SERVO}#{channel}#{angle}\n"
        self._send_command(cmd)

    def set_camera_pan(self, angle: int):
        """Set camera pan angle (0-180, 90 = center)"""
        self.set_servo(0, angle)

    def set_camera_tilt(self, angle: int):
        """Set camera tilt angle (80-180, 90 = center)"""
        self.set_servo(1, angle)

    def center_camera(self):
        """Center the camera"""
        self.set_camera_pan(90)
        self.set_camera_tilt(90)

    # ==================== LED Control ====================

    def set_led(self, led_index: int, r: int, g: int, b: int):
        """
        Set LED color.
        led_index: bitmask (0x01=LED1, 0x02=LED2, ... 0xFF=all)
        r, g, b: 0-255
        """
        cmd = f"{self.CMD_LED}#{led_index}#{r}#{g}#{b}\n"
        self._send_command(cmd)

    def set_all_leds(self, r: int, g: int, b: int):
        """Set all LEDs to same color"""
        self.set_led(0xFF, r, g, b)

    def set_led_mode(self, mode: int):
        """
        Set LED animation mode.
        0=off, 1=follow, 2=blink, 3=rainbow breathe, 4=rainbow cycle
        """
        cmd = f"{self.CMD_LED_MOD}#{mode}\n"
        self._send_command(cmd)

    # ==================== Buzzer ====================

    def buzzer_on(self):
        """Turn buzzer on"""
        cmd = f"{self.CMD_BUZZER}#1\n"
        self._send_command(cmd)

    def buzzer_off(self):
        """Turn buzzer off"""
        cmd = f"{self.CMD_BUZZER}#0\n"
        self._send_command(cmd)

    def beep(self, duration_ms: int = 200):
        """Beep for specified duration"""
        self.buzzer_on()
        time.sleep(duration_ms / 1000)
        self.buzzer_off()

    # ==================== Sensor Requests ====================

    def request_distance(self):
        """Request ultrasonic distance reading"""
        cmd = f"{self.CMD_SONIC}#1\n"
        self._send_command(cmd)

    def request_light(self):
        """Request light sensor readings"""
        cmd = f"{self.CMD_LIGHT}#1\n"
        self._send_command(cmd)

    def request_power(self):
        """Request battery voltage"""
        cmd = f"{self.CMD_POWER}\n"
        self._send_command(cmd)

    def request_line_sensors(self):
        """Request infrared line sensor readings"""
        cmd = f"{self.CMD_LINE}#1\n"
        self._send_command(cmd)

    # ==================== Getters ====================

    def get_distance(self) -> float:
        """Get last ultrasonic distance reading (cm)"""
        with self.sensor_lock:
            return self.sensor_data.ultrasonic_distance

    def get_light_sensors(self) -> tuple:
        """Get light sensor readings (left_voltage, right_voltage)"""
        with self.sensor_lock:
            return (self.sensor_data.light_left, self.sensor_data.light_right)

    def get_battery_voltage(self) -> float:
        """Get battery voltage"""
        with self.sensor_lock:
            return self.sensor_data.battery_voltage

    def get_line_sensors(self) -> int:
        """Get line sensor reading (3-bit value)"""
        with self.sensor_lock:
            return self.sensor_data.line_sensors

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def capture_image(self, path: str = None) -> Optional[np.ndarray]:
        """Capture current frame and optionally save to file"""
        frame = self.get_frame()
        if frame is not None and path:
            cv2.imwrite(path, frame)
        return frame


# Test connection
if __name__ == "__main__":
    robot = RobotConnection("192.168.1.100")

    if robot.connect():
        print("Connected!")

        # Test motors
        robot.move_forward(1000)
        time.sleep(1)
        robot.stop()

        # Test sensors
        robot.request_distance()
        time.sleep(0.5)
        print(f"Distance: {robot.get_distance()} cm")

        robot.disconnect()
    else:
        print("Failed to connect")
