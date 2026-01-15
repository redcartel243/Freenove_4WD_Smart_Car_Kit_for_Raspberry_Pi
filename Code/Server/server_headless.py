#!/usr/bin/env python3
"""
Headless Server for Freenove 4WD Smart Car
Runs without GUI - perfect for AI control

Usage:
    python server_headless.py
"""

import sys
import struct
import time
import signal
import math
import threading
import multiprocessing
from server import Server
from message import Message_Parse
from command import Command
from led import Led
from camera import Camera
from car import Car
from buzzer import Buzzer


class HeadlessServer:
    """Headless server that runs all robot functionality without PyQt5 GUI"""

    def __init__(self):
        print("Initializing headless server...")

        self.tcp_server = Server()
        self.command = Command()
        self.led = Led()
        self.car = Car()
        self.buzzer = Buzzer()

        # Camera might fail on some setups
        try:
            self.camera = Camera(stream_size=(400, 300))
            self.camera_available = True
            print("Camera initialized")
        except Exception as e:
            print(f"Camera not available: {e}")
            self.camera = None
            self.camera_available = False

        self.queue_cmd = multiprocessing.Queue()
        self.cmd_parse = Message_Parse()
        self.queue_led = multiprocessing.Queue()
        self.led_parse = Message_Parse()

        self.cmd_thread = None
        self.video_thread = None
        self.car_thread = None
        self.led_process = None

        self.running = False
        self.car_mode = 1
        self.rotation_flag = False
        self.led_mode = 0

        self.send_sonic_data_time = time.time()
        self.send_light_data_time = time.time()
        self.send_line_data_time = time.time()

        # Handle Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nShutting down...")
        self.stop()
        sys.exit(0)

    def start(self):
        """Start the server and all threads"""
        print(f"Starting server on {self.tcp_server.ip_address}...")

        self.running = True
        self.tcp_server.start_tcp_servers()

        # Start threads
        self.cmd_thread = threading.Thread(target=self.cmd_receive_loop, daemon=True)
        self.cmd_thread.start()

        if self.camera_available:
            self.video_thread = threading.Thread(target=self.video_send_loop, daemon=True)
            self.video_thread.start()

        self.car_thread = threading.Thread(target=self.car_task_loop, daemon=True)
        self.car_thread.start()

        self.led_process = multiprocessing.Process(target=self.led_loop, args=(self.queue_led,))
        self.led_process.start()

        print(f"Server running!")
        print(f"  Command port: 5000")
        print(f"  Video port:   8000")
        print(f"  Camera:       {'Available' if self.camera_available else 'Not available'}")
        print("Press Ctrl+C to stop")

        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the server and cleanup"""
        print("Stopping server...")
        self.running = False

        # Stop LED
        try:
            self.led.colorBlink(0)
        except:
            pass

        # Stop camera
        if self.camera:
            try:
                self.camera.stop_stream()
                self.camera.close()
            except:
                pass

        # Stop car
        try:
            self.car.motor.set_motor_model(0, 0, 0, 0)
            self.car.close()
        except:
            pass

        # Stop TCP servers
        try:
            self.tcp_server.stop_tcp_servers()
        except:
            pass

        # Terminate LED process
        if self.led_process and self.led_process.is_alive():
            self.led_process.terminate()
            self.led_process.join(0.5)

        print("Server stopped")

    def send_sonic_data(self):
        """Send ultrasonic distance to client"""
        if time.time() - self.send_sonic_data_time > 0.5:
            self.send_sonic_data_time = time.time()
            if not self.tcp_server.get_command_server_busy():
                try:
                    distance = self.car.sonic.get_distance()
                    cmd = f"{self.command.CMD_SONIC}#{distance:.2f}\n"
                    self.tcp_server.send_data_to_command_client(cmd)
                except Exception as e:
                    pass

    def send_light_data(self):
        """Send light sensor data to client"""
        if time.time() - self.send_light_data_time > 0.3:
            self.send_light_data_time = time.time()
            if not self.tcp_server.get_command_server_busy():
                try:
                    adc_light_1 = self.car.adc.read_adc(0)
                    adc_light_2 = self.car.adc.read_adc(1)
                    cmd = f"{self.command.CMD_LIGHT}#{adc_light_1:.2f}#{adc_light_2:.2f}\n"
                    self.tcp_server.send_data_to_command_client(cmd)
                except Exception as e:
                    pass

    def send_line_data(self):
        """Send infrared line sensor data to client"""
        if time.time() - self.send_line_data_time > 0.3:
            self.send_line_data_time = time.time()
            if not self.tcp_server.get_command_server_busy():
                try:
                    ir_value_1 = self.car.infrared.read_one_infrared(1)
                    ir_value_2 = self.car.infrared.read_one_infrared(2)
                    ir_value_3 = self.car.infrared.read_one_infrared(3)
                    cmd = f"{self.command.CMD_LINE}#{ir_value_1}#{ir_value_2}#{ir_value_3}\n"
                    self.tcp_server.send_data_to_command_client(cmd)
                except Exception as e:
                    pass

    def send_power_data(self):
        """Send battery power data to client"""
        if not self.tcp_server.get_command_server_busy():
            try:
                power = self.car.adc.read_adc(2) * (3 if self.car.adc.pcb_version == 1 else 2)
                cmd = f"{self.command.CMD_POWER}#{power}\n"
                self.tcp_server.send_data_to_command_client(cmd)
            except Exception as e:
                pass

    def cmd_receive_loop(self):
        """Thread: Receive and process commands"""
        while self.running:
            try:
                cmd_queue = self.tcp_server.read_data_from_command_server()
                if cmd_queue.qsize() > 0:
                    client_address, all_message = cmd_queue.get()
                    main_message = all_message.strip()

                    messages = main_message.split("\n") if "\n" in main_message else [main_message]
                    for msg in messages:
                        if msg:
                            self.queue_cmd.put(msg)

                while not self.queue_cmd.empty():
                    msg = self.queue_cmd.get()
                    self.process_command(msg)

                time.sleep(0.001)
            except Exception as e:
                print(f"Command error: {e}")
                time.sleep(0.1)

    def process_command(self, msg):
        """Process a single command"""
        self.cmd_parse.clear_parameters()
        self.cmd_parse.parse(msg)
        cmd = self.cmd_parse.command_string

        print(f"CMD: {msg}")

        try:
            if cmd == self.command.CMD_LED:
                self.queue_led.put(msg)
            elif cmd == self.command.CMD_LED_MOD:
                self.queue_led.put(msg)
            elif cmd == self.command.CMD_SONIC:
                self.send_sonic_data()
            elif cmd == self.command.CMD_LIGHT:
                self.send_light_data()
            elif cmd == self.command.CMD_LINE:
                self.send_line_data()
            elif cmd == self.command.CMD_POWER:
                self.send_power_data()
            elif cmd == self.command.CMD_BUZZER:
                self.buzzer.set_state(self.cmd_parse.int_parameter[0])
            elif cmd == self.command.CMD_SERVO:
                channel = int(self.cmd_parse.int_parameter[0])
                angle = int(self.cmd_parse.int_parameter[1])
                self.car.servo.set_servo_pwm(str(channel), angle)
            elif cmd == self.command.CMD_MOTOR:
                self.car_mode = 1
                duty = [int(self.cmd_parse.int_parameter[i]) for i in range(4)]
                scale_factor = 0.8
                scaled_duty = [int(round(d * scale_factor)) for d in duty]
                self.car.motor.set_motor_model(scaled_duty[0], scaled_duty[1], scaled_duty[2], scaled_duty[3])
            elif cmd == self.command.CMD_M_MOTOR:
                self.car_mode = 1
                duty = [int(self.cmd_parse.int_parameter[i]) for i in range(4)]
                LX = -int((duty[1] * math.sin(math.radians(duty[0]))))
                LY = int(duty[1] * math.cos(math.radians(duty[0])))
                RX = int(duty[3] * math.sin(math.radians(duty[2])))
                RY = int(duty[3] * math.cos(math.radians(duty[2])))
                FR = LY - LX + RX
                FL = LY + LX - RX
                BL = LY - LX - RX
                BR = LY + LX + RX
                self.car.motor.set_motor_model(FL, BL, FR, BR)
            elif cmd == self.command.CMD_MODE:
                mode = self.cmd_parse.int_parameter[0]
                if mode == 0:
                    self.car_mode = 1
                    self.car.motor.set_motor_model(0, 0, 0, 0)
                    print("Mode: Manual")
                elif mode == 1:
                    self.car_mode = 2
                    print("Mode: Light Following")
                elif mode == 2:
                    self.car_mode = 3
                    print("Mode: Infrared")
                elif mode == 3:
                    self.car_mode = 4
                    print("Mode: Ultrasonic")
        except Exception as e:
            print(f"Process command error: {e}")

    def video_send_loop(self):
        """Thread: Send video frames to client"""
        print("Video send loop started")
        frame_count = 0
        while self.running:
            try:
                connected = self.tcp_server.is_video_server_connected()
                if connected:
                    print(f"Video client connected, starting stream...")
                    self.camera.start_stream()
                    while self.tcp_server.is_video_server_connected() and self.running:
                        frame = self.camera.get_frame()
                        lenFrame = len(frame)
                        lengthBin = struct.pack('<I', lenFrame)
                        try:
                            self.tcp_server.send_data_to_video_client(lengthBin)
                            self.tcp_server.send_data_to_video_client(frame)
                            frame_count += 1
                            if frame_count % 30 == 1:
                                print(f"Sent frame {frame_count}, size={lenFrame}")
                        except Exception as e:
                            print(f"Send error: {e}")
                            break
                    print("Video client disconnected, stopping stream")
                    self.camera.stop_stream()
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Video error: {e}")
                time.sleep(0.5)

    def car_task_loop(self):
        """Thread: Run autonomous car modes"""
        while self.running:
            try:
                if self.car_mode == 1:
                    pass  # Manual mode
                elif self.car_mode == 2:
                    self.car.mode_light()
                    self.send_light_data()
                elif self.car_mode == 3:
                    self.car.mode_infrared()
                elif self.car_mode == 4:
                    self.car.mode_ultrasonic()
                    self.send_sonic_data()
                time.sleep(0.01)
            except Exception as e:
                print(f"Car task error: {e}")
                time.sleep(0.1)

    def led_loop(self, queue_led):
        """Process: Handle LED animations"""
        led = Led()
        led_parse = Message_Parse()
        led_mode = 0

        try:
            while True:
                if not queue_led.empty():
                    msg = queue_led.get()
                    led_parse.clear_parameters()
                    led_parse.parse(msg)

                    if led_parse.command_string == "CMD_LED" and led_mode == 1:
                        try:
                            data1 = int(led_parse.int_parameter[0])
                            data2 = int(led_parse.int_parameter[1])
                            data3 = int(led_parse.int_parameter[2])
                            data4 = int(led_parse.int_parameter[3])
                            led.ledIndex(data1, data2, data3, data4)
                        except:
                            pass
                    elif led_parse.command_string == "CMD_LED_MOD":
                        led_mode = led_parse.int_parameter[0]

                # Run LED mode
                if led_mode == 0:
                    led.colorBlink(0)
                elif led_mode == 2:
                    led.following()
                elif led_mode == 3:
                    led.colorBlink(1)
                elif led_mode == 4:
                    led.rainbowbreathing()
                elif led_mode == 5:
                    led.rainbowCycle()

                time.sleep(0.01)
        except:
            led.colorBlink(0)


if __name__ == '__main__':
    server = HeadlessServer()
    server.start()
