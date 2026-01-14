"""
Quick test to verify robot connection works
"""
import time
from robot_connection import RobotConnection

# Change this to your Pi's IP
ROBOT_IP = "192.168.0.105"

def main():
    print(f"Connecting to robot at {ROBOT_IP}...")
    robot = RobotConnection(ROBOT_IP)

    if not robot.connect():
        print("Failed to connect!")
        return

    print("Connected! Testing motors...")

    # Test forward
    print("Moving forward...")
    robot.move_forward(1000)
    time.sleep(1)

    print("Stopping...")
    robot.stop()
    time.sleep(0.5)

    # Test backward
    print("Moving backward...")
    robot.move_backward(1000)
    time.sleep(1)

    print("Stopping...")
    robot.stop()
    time.sleep(0.5)

    # Test buzzer
    print("Testing buzzer...")
    robot.beep(500)

    # Test distance sensor
    print("Testing ultrasonic...")
    robot.request_distance()
    time.sleep(0.5)
    print(f"Distance: {robot.get_distance()} cm")

    print("Disconnecting...")
    robot.disconnect()
    print("Done!")

if __name__ == "__main__":
    main()
