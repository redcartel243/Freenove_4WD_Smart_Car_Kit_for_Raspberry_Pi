"""
AI Robot Control - Main Entry Point

This server:
1. Connects to the robot (Raspberry Pi)
2. Processes video with YOLO for object detection
3. Runs an LLM agent with tool calling for autonomous control
4. Provides a web interface for monitoring and manual control
"""

import asyncio
import json
import time
import cv2
import yaml
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from robot_connection import RobotConnection
from vision import YOLOProcessor
from ai import RobotAgent, AgentConfig
from autonomous import AutonomousSystem, CameraCalibration


# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Global instances
config = load_config()
robot: Optional[RobotConnection] = None
yolo: Optional[YOLOProcessor] = None
agent: Optional[RobotAgent] = None
autonomous: Optional[AutonomousSystem] = None

# State
is_autonomous = False
autonomous_task: Optional[asyncio.Task] = None
connected_websockets: list = []
main_event_loop: Optional[asyncio.AbstractEventLoop] = None  # For thread-safe broadcasts

# AI Mode State (kept for backward compatibility with endpoints)
ai_follow_mode = False
ai_follow_target = "person"
ai_follow_speed = 1200
ai_patrol_mode = False
ai_mode_lock = threading.Lock()


def create_tool_handlers():
    """Create handler functions for all robot tools"""

    def move(direction: str, speed: int = 1200, duration_ms: int = 500):
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        speed = min(2000, max(500, speed))

        if direction == "forward":
            robot.move_forward(speed)
        elif direction == "backward":
            robot.move_backward(speed)
        elif direction == "left":
            robot.turn_left(speed)
        elif direction == "right":
            robot.turn_right(speed)
        else:
            return {"error": f"Unknown direction: {direction}"}

        time.sleep(duration_ms / 1000)
        robot.stop()

        return {"success": True, "action": f"Moved {direction} at speed {speed} for {duration_ms}ms"}

    def rotate(degrees: int, speed: int = 1000):
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        # Approximate rotation time (calibrate based on your robot)
        # This is a rough estimate - you may need to adjust
        rotation_time = abs(degrees) / 90 * 0.5  # ~0.5s for 90 degrees

        if degrees > 0:
            robot.turn_right(speed)
        else:
            robot.turn_left(speed)

        time.sleep(rotation_time)
        robot.stop()

        return {"success": True, "action": f"Rotated {degrees} degrees"}

    def stop():
        if robot and robot.connected:
            robot.stop()
        return {"success": True, "action": "Stopped"}

    def get_distance():
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        robot.request_distance()
        time.sleep(0.3)  # Wait for response
        distance = robot.get_distance()
        return {"distance_cm": distance}

    def get_objects():
        if not yolo:
            return {"error": "YOLO not initialized"}
        return yolo.get_detection_summary()

    def find_object(object_type: str):
        if not yolo:
            return {"error": "YOLO not initialized"}

        detection = yolo.find_object(object_type)
        if detection:
            return {
                "found": True,
                "object": detection.to_dict()
            }
        return {"found": False, "message": f"No {object_type} found in view"}

    def scan_surroundings(angles: list = None):
        if angles is None:
            angles = [45, 90, 135]

        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        results = {}
        for angle in angles:
            robot.set_camera_pan(angle)
            time.sleep(0.5)  # Wait for servo and new frame

            if yolo:
                detections = yolo.get_detection_summary()
                results[f"angle_{angle}"] = detections

        # Return to center
        robot.set_camera_pan(90)

        return {"scan_results": results}

    def look_at(pan: int, tilt: int):
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        robot.set_camera_pan(pan)
        robot.set_camera_tilt(tilt)
        return {"success": True, "pan": pan, "tilt": tilt}

    def center_camera():
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}
        robot.center_camera()
        return {"success": True}

    def get_light_sensors():
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        robot.request_light()
        time.sleep(0.3)
        left, right = robot.get_light_sensors()
        return {"left_voltage": left, "right_voltage": right}

    def get_line_sensors():
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        robot.request_line_sensors()
        time.sleep(0.3)
        value = robot.get_line_sensors()
        return {
            "raw_value": value,
            "sensors": {
                "left": bool(value & 0x01),
                "center": bool(value & 0x02),
                "right": bool(value & 0x04)
            }
        }

    def get_battery():
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        robot.request_power()
        time.sleep(0.3)
        voltage = robot.get_battery_voltage()
        percentage = min(100, max(0, int((voltage - 7) / 1.4 * 100)))
        return {"voltage": voltage, "percentage": percentage}

    def set_leds(r: int, g: int, b: int, mode: int = 0):
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        if mode > 0:
            robot.set_led_mode(mode)
        else:
            robot.set_all_leds(r, g, b)
        return {"success": True}

    def beep(duration_ms: int = 200):
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}
        robot.beep(duration_ms)
        return {"success": True}

    def capture_photo(filename: str = None):
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        if filename is None:
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

        save_path = Path(__file__).parent / "captures" / filename
        save_path.parent.mkdir(exist_ok=True)

        frame = robot.capture_image(str(save_path))
        if frame is not None:
            return {"success": True, "path": str(save_path)}
        return {"error": "Failed to capture image"}

    async def analyze_image(question: str):
        if not robot or not robot.connected:
            return {"error": "Robot not connected"}

        frame = robot.get_frame()
        if frame is None:
            return {"error": "No frame available"}

        # Encode frame to JPEG bytes
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        try:
            result = await agent.analyze_image_with_vision_llm(image_bytes, question)
            return {"analysis": result}
        except Exception as e:
            return {"error": str(e)}

    def wait(duration_ms: int):
        time.sleep(duration_ms / 1000)
        return {"success": True, "waited_ms": duration_ms}

    def report(message: str, type: str = "info"):
        if agent:
            agent.add_report(message, type)
        # Also broadcast to connected websockets
        broadcast_message({
            "type": "report",
            "message": message,
            "msg_type": type,
            "timestamp": datetime.now().isoformat()
        })
        return {"success": True}

    return {
        "move": move,
        "rotate": rotate,
        "stop": stop,
        "get_distance": get_distance,
        "get_objects": get_objects,
        "find_object": find_object,
        "scan_surroundings": scan_surroundings,
        "look_at": look_at,
        "center_camera": center_camera,
        "get_light_sensors": get_light_sensors,
        "get_line_sensors": get_line_sensors,
        "get_battery": get_battery,
        "set_leds": set_leds,
        "beep": beep,
        "capture_photo": capture_photo,
        "analyze_image": analyze_image,
        "wait": wait,
        "report": report
    }


def broadcast_message(message: dict):
    """Broadcast a message to all connected websockets (thread-safe)"""
    if not connected_websockets:
        return

    # Use the stored main event loop for thread-safe broadcasting
    if main_event_loop is None:
        return

    for ws in connected_websockets[:]:  # Copy list to avoid modification during iteration
        try:
            # Schedule the coroutine on the main event loop from any thread
            asyncio.run_coroutine_threadsafe(ws.send_json(message), main_event_loop)
        except Exception:
            pass


# Frame processing thread
def frame_processor():
    """Background thread that processes frames with YOLO"""
    frame_skip = config.get("vision", {}).get("frame_skip", 2)
    frame_count = 0
    last_log = time.time()

    while True:
        if robot and robot.connected and yolo:
            frame = robot.get_frame()
            if frame is not None:
                frame_count += 1
                if frame_count % frame_skip == 0:
                    yolo.process_frame(frame)
                # Log every 5 seconds
                if time.time() - last_log > 5:
                    print(f"Frame processor: processed {frame_count} frames")
                    last_log = time.time()
            elif time.time() - last_log > 5:
                print("Frame processor: no frames available from robot")
                last_log = time.time()
        time.sleep(0.033)  # ~30 FPS max


# Autonomous System update thread
def autonomous_processor():
    """Background thread that runs the FSD-style autonomous system"""
    while True:
        if autonomous and autonomous.running:
            try:
                autonomous.update()
            except Exception as e:
                print(f"Autonomous system error: {e}")
                import traceback
                traceback.print_exc()
        time.sleep(0.1)  # 10Hz control loop


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global robot, yolo, agent, autonomous, main_event_loop

    print("Starting AI Robot Control Server...")

    # Store the main event loop for thread-safe broadcasts
    main_event_loop = asyncio.get_running_loop()

    # Initialize YOLO
    yolo_config = config.get("vision", {})
    yolo = YOLOProcessor(
        model_path=yolo_config.get("yolo_model", "yolov8n.pt"),
        confidence_threshold=yolo_config.get("confidence_threshold", 0.5)
    )
    print("YOLO initialized")

    # Initialize agent
    llm_config = config.get("llm", {})
    agent_config = AgentConfig(
        provider=llm_config.get("provider", "ollama"),
        model=llm_config.get("model", "qwen2.5:7b-instruct"),
        base_url=llm_config.get("base_url", "http://localhost:11434"),
        api_key=llm_config.get("api_key"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 1024),
        vision_model=config.get("vision_llm", {}).get("model", "moondream")
    )
    agent = RobotAgent(agent_config)
    agent.register_all_handlers(create_tool_handlers())
    print("Agent initialized")

    # Start frame processor thread
    processor_thread = threading.Thread(target=frame_processor, daemon=True)
    processor_thread.start()

    # Start autonomous system processor thread
    autonomous_thread = threading.Thread(target=autonomous_processor, daemon=True)
    autonomous_thread.start()
    print("Autonomous system processor thread started")

    yield

    # Cleanup
    if robot and robot.connected:
        robot.disconnect()
    main_event_loop = None
    print("Server stopped")


app = FastAPI(title="AI Robot Control", lifespan=lifespan)

# Mount static files
static_path = Path(__file__).parent / "web" / "static"
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Serve main web interface"""
    return HTMLResponse(content=get_index_html())


@app.post("/connect")
async def connect_robot(ip: str = None):
    """Connect to the robot"""
    global robot, autonomous

    if robot and robot.connected:
        return {"status": "already_connected"}

    robot_config = config.get("robot", {})
    robot_ip = ip or robot_config.get("ip", "192.168.1.100")

    robot = RobotConnection(
        ip=robot_ip,
        command_port=robot_config.get("command_port", 5000),
        video_port=robot_config.get("video_port", 8000)
    )

    if robot.connect():
        # Initialize the FSD-style autonomous system
        autonomous = AutonomousSystem(robot, yolo, broadcast_message)
        print("Autonomous system initialized")
        return {"status": "connected", "ip": robot_ip}
    else:
        return {"status": "failed", "ip": robot_ip}


@app.post("/disconnect")
async def disconnect_robot():
    """Disconnect from the robot"""
    global robot, is_autonomous, autonomous

    is_autonomous = False
    if autonomous_task:
        autonomous_task.cancel()

    # Stop autonomous system
    if autonomous:
        autonomous.stop_all()
        autonomous = None

    if robot:
        robot.disconnect()
        robot = None
    return {"status": "disconnected"}


@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "connected": robot.connected if robot else False,
        "autonomous": is_autonomous,
        "agent_state": agent.get_state() if agent else None,
        "yolo_fps": yolo.fps if yolo else 0
    }


@app.post("/task")
async def run_task(task: str):
    """Run an autonomous task"""
    global is_autonomous, autonomous_task

    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    if is_autonomous:
        raise HTTPException(400, "Already running a task")

    is_autonomous = True

    async def task_runner():
        global is_autonomous
        try:
            result = await agent.run_task(task)
            broadcast_message({"type": "task_complete", "result": result})
        finally:
            is_autonomous = False

    autonomous_task = asyncio.create_task(task_runner())
    return {"status": "started", "task": task}


@app.post("/stop")
async def stop_task():
    """Stop the current task"""
    global is_autonomous, ai_follow_mode, ai_patrol_mode

    if robot and robot.connected:
        robot.stop()

    if agent:
        agent.stop()

    # Stop the FSD autonomous system
    if autonomous:
        autonomous.stop_all()

    # Disable all AI modes
    with ai_mode_lock:
        ai_follow_mode = False
        ai_patrol_mode = False

    is_autonomous = False
    return {"status": "stopped"}


@app.post("/manual/{action}")
async def manual_control(action: str, speed: int = 1200):
    """Manual control endpoint"""
    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    if is_autonomous:
        raise HTTPException(400, "Cannot use manual control during autonomous mode")

    actions = {
        "forward": lambda: robot.move_forward(speed),
        "backward": lambda: robot.move_backward(speed),
        "left": lambda: robot.turn_left(speed),
        "right": lambda: robot.turn_right(speed),
        "stop": lambda: robot.stop(),
        "strafe_left": lambda: robot.strafe_left(speed),
        "strafe_right": lambda: robot.strafe_right(speed)
    }

    if action in actions:
        actions[action]()
        return {"status": "ok", "action": action}

    raise HTTPException(400, f"Unknown action: {action}")


@app.post("/led")
async def led_control(r: int = 0, g: int = 0, b: int = 0, mode: int = -1):
    """LED control endpoint"""
    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    if mode >= 0:
        robot.set_led_mode(mode)
    else:
        # Must be in manual mode (1) to set LED colors directly
        robot.set_led_mode(1)
        time.sleep(0.05)  # Give time for mode to be set
        robot.set_all_leds(r, g, b)

    return {"status": "ok", "r": r, "g": g, "b": b, "mode": mode}


@app.post("/buzzer")
async def buzzer_control(state: int = 0, duration_ms: int = 0):
    """Buzzer control endpoint"""
    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    if duration_ms > 0:
        robot.beep(duration_ms)
    elif state:
        robot.buzzer_on()
    else:
        robot.buzzer_off()

    return {"status": "ok"}


# Track current camera position globally for calibration
current_camera_pan = 90
current_camera_tilt = 90


@app.post("/servo")
async def servo_control(pan: int = None, tilt: int = None):
    """Camera servo control endpoint"""
    global current_camera_pan, current_camera_tilt

    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    if pan is not None:
        pan = max(0, min(180, pan))
        robot.set_camera_pan(pan)
        current_camera_pan = pan
        # Also update motion controller if active
        if autonomous and autonomous.motion:
            autonomous.motion.current_pan = pan

    if tilt is not None:
        tilt = max(0, min(180, tilt))
        robot.set_camera_tilt(tilt)
        current_camera_tilt = tilt
        # Also update motion controller if active
        if autonomous and autonomous.motion:
            autonomous.motion.current_tilt = tilt

    return {"status": "ok", "pan": pan, "tilt": tilt}


@app.post("/servo/center")
async def servo_center():
    """Center camera servos at calibrated forward position"""
    global current_camera_pan, current_camera_tilt

    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    # Load calibration to get the actual center position
    cal = CameraCalibration.load_from_config(str(CONFIG_PATH))
    pan_center = cal.pan_center
    tilt_center = cal.tilt_center

    robot.set_camera_pan(pan_center)
    robot.set_camera_tilt(tilt_center)

    current_camera_pan = pan_center
    current_camera_tilt = tilt_center

    # Update motion controller if active
    if autonomous and autonomous.motion:
        autonomous.motion.current_pan = pan_center
        autonomous.motion.current_tilt = tilt_center

    return {"status": "ok", "pan": pan_center, "tilt": tilt_center}


# ==================== Camera Calibration ====================

@app.get("/calibration")
async def get_calibration():
    """Get current camera calibration values"""
    cal = CameraCalibration.load_from_config(str(CONFIG_PATH))
    return {
        "pan_center": cal.pan_center,
        "tilt_center": cal.tilt_center,
        "pan_range": list(cal.pan_range),
        "tilt_range": list(cal.tilt_range),
        "calibrated": cal.calibrated
    }


@app.post("/calibration")
async def set_calibration(pan_center: int = None, tilt_center: int = None):
    """
    Set camera calibration values.

    To calibrate:
    1. Use /servo endpoint to manually point camera straight ahead (same direction robot faces)
    2. Note the pan and tilt values
    3. Call this endpoint with those values to save as calibration

    Example: If camera faces forward at pan=95, tilt=85, call:
    POST /calibration?pan_center=95&tilt_center=85
    """
    cal = CameraCalibration.load_from_config(str(CONFIG_PATH))

    if pan_center is not None:
        cal.pan_center = max(0, min(180, pan_center))

    if tilt_center is not None:
        cal.tilt_center = max(0, min(180, tilt_center))

    cal.calibrated = True
    cal.save_to_config(str(CONFIG_PATH))

    # Update motion controller if autonomous system is active
    if autonomous and autonomous.motion:
        autonomous.motion.calibration = cal
        print(f"Updated motion controller calibration: pan_center={cal.pan_center}")

    return {
        "status": "ok",
        "pan_center": cal.pan_center,
        "tilt_center": cal.tilt_center,
        "calibrated": cal.calibrated
    }


@app.post("/calibration/set-current")
async def calibrate_current_position():
    """
    Set the current camera position as the calibrated center.
    Point the camera where the robot is facing, then call this endpoint.
    """
    global current_camera_pan, current_camera_tilt

    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    # Use the globally tracked camera position (updated by /servo endpoint)
    pan = current_camera_pan
    tilt = current_camera_tilt

    print(f"Calibrating camera: current position pan={pan}, tilt={tilt}")

    cal = CameraCalibration(
        pan_center=pan,
        tilt_center=tilt,
        calibrated=True
    )
    cal.save_to_config(str(CONFIG_PATH))

    # Update motion controller calibration
    if autonomous and autonomous.motion:
        autonomous.motion.calibration = cal
        autonomous.motion.current_pan = pan
        autonomous.motion.current_tilt = tilt

    return {
        "status": "ok",
        "message": f"Calibration saved: camera center at pan={pan}, tilt={tilt}",
        "pan_center": pan,
        "tilt_center": tilt
    }


@app.post("/ai/follow")
async def ai_follow_control(enabled: bool = None, target: str = None, speed: int = None):
    """Control AI follow mode using FSD-style autonomous system"""
    global ai_follow_mode, ai_follow_target, ai_follow_speed, ai_patrol_mode

    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    if not autonomous:
        raise HTTPException(400, "Autonomous system not initialized")

    with ai_mode_lock:
        if target is not None:
            ai_follow_target = target
            autonomous.set_follow_target(target)

        if speed is not None:
            ai_follow_speed = max(500, min(3000, speed))
            autonomous.set_speed(ai_follow_speed)

        if enabled is not None:
            ai_follow_mode = enabled
            if enabled:
                # Disable patrol when enabling follow
                ai_patrol_mode = False
                autonomous.start_follow(ai_follow_target, ai_follow_speed)
            else:
                autonomous.stop_all()

        return {
            "status": "ok",
            "enabled": ai_follow_mode,
            "target": ai_follow_target,
            "speed": ai_follow_speed
        }


@app.get("/ai/follow/status")
async def ai_follow_status():
    """Get AI follow mode status"""
    with ai_mode_lock:
        return {
            "enabled": ai_follow_mode,
            "target": ai_follow_target,
            "speed": ai_follow_speed
        }


@app.get("/ai/targets")
async def get_available_targets():
    """Get list of YOLO detectable object classes"""
    # Standard COCO dataset classes that YOLO can detect
    coco_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    return {"targets": coco_classes}


@app.post("/ai/patrol")
async def ai_patrol_control(enabled: bool = None, speed: int = None):
    """Control AI patrol mode using FSD-style autonomous system"""
    global ai_patrol_mode, ai_follow_speed, ai_follow_mode

    if not robot or not robot.connected:
        raise HTTPException(400, "Robot not connected")

    if not autonomous:
        raise HTTPException(400, "Autonomous system not initialized")

    with ai_mode_lock:
        if speed is not None:
            ai_follow_speed = max(500, min(3000, speed))
            autonomous.set_speed(ai_follow_speed)

        if enabled is not None:
            ai_patrol_mode = enabled
            if enabled:
                # Disable follow mode when enabling patrol
                ai_follow_mode = False
                autonomous.start_patrol(ai_follow_speed)
            else:
                autonomous.stop_all()

        return {
            "status": "ok",
            "enabled": ai_patrol_mode,
            "speed": ai_follow_speed
        }


@app.get("/ai/patrol/status")
async def ai_patrol_status():
    """Get AI patrol mode status"""
    with ai_mode_lock:
        return {
            "enabled": ai_patrol_mode,
            "speed": ai_follow_speed
        }


@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream"""
    def generate():
        while True:
            frame = None

            # Try to get YOLO annotated frame first
            if yolo:
                frame = yolo.get_annotated_frame()

            # Fallback to raw robot frame if YOLO has no frame
            if frame is None and robot and robot.connected:
                frame = robot.get_frame()

            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # No frame available, wait a bit
                time.sleep(0.1)
                continue

            time.sleep(0.033)

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    connected_websockets.append(websocket)

    try:
        while True:
            # Request fresh sensor data from robot
            if robot and robot.connected:
                robot.request_distance()

            await asyncio.sleep(0.3)  # Wait for response

            # Send status updates
            with ai_mode_lock:
                follow_info = {
                    "enabled": ai_follow_mode,
                    "target": ai_follow_target,
                    "speed": ai_follow_speed
                }
                patrol_info = {
                    "enabled": ai_patrol_mode,
                    "speed": ai_follow_speed
                }

            # Get autonomous system state for detailed info
            autonomous_state = None
            if autonomous:
                autonomous_state = autonomous.get_state()

            status = {
                "type": "status",
                "connected": robot.connected if robot else False,
                "autonomous": is_autonomous,
                "distance": robot.get_distance() if robot and robot.connected else 0,
                "detections": yolo.get_detection_summary() if yolo else {},
                "agent_state": agent.get_state() if agent else None,
                "follow_mode": follow_info,
                "patrol_mode": patrol_info,
                "autonomous_state": autonomous_state
            }
            await websocket.send_json(status)
            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


def get_index_html():
    """Generate the web interface HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Robot Control</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; margin-bottom: 20px; color: #00d4ff; }

        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }

        .panel {
            background: #16213e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        .panel h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.2rem; }

        .video-container {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            cursor: crosshair;
        }
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
            pointer-events: none;
        }
        .crosshair {
            position: absolute;
            width: 20px;
            height: 20px;
            border: 2px solid #00d4ff;
            border-radius: 50%;
            pointer-events: none;
            transform: translate(-50%, -50%);
            opacity: 0.8;
        }
        .crosshair::before, .crosshair::after {
            content: '';
            position: absolute;
            background: #00d4ff;
        }
        .crosshair::before {
            width: 2px;
            height: 30px;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }
        .crosshair::after {
            width: 30px;
            height: 2px;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
        }
        .servo-info {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
        }

        .controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 200px; margin: 0 auto; }
        .controls button {
            padding: 15px;
            font-size: 1.5rem;
            background: #0f3460;
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            transition: background 0.2s;
        }
        .controls button:hover { background: #1a5aa0; }
        .controls button:active { background: #00d4ff; }

        .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .status-item {
            background: #0f3460;
            padding: 10px;
            border-radius: 5px;
        }
        .status-item label { font-size: 0.8rem; color: #888; }
        .status-item .value { font-size: 1.2rem; font-weight: bold; }

        .task-input {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .task-input input {
            flex: 1;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #0f3460;
            background: #1a1a2e;
            color: white;
        }
        .task-input button, .connect-btn {
            padding: 10px 20px;
            background: #00d4ff;
            border: none;
            border-radius: 5px;
            color: #1a1a2e;
            font-weight: bold;
            cursor: pointer;
        }
        .task-input button:hover, .connect-btn:hover { background: #00b8e6; }

        .stop-btn {
            background: #ff4444 !important;
            color: white !important;
        }

        .reports {
            max-height: 200px;
            overflow-y: auto;
            font-size: 0.9rem;
        }
        .report {
            padding: 8px;
            margin-bottom: 5px;
            border-radius: 4px;
            background: #0f3460;
        }
        .report.warning { border-left: 3px solid #ffaa00; }
        .report.error { border-left: 3px solid #ff4444; }
        .report.success { border-left: 3px solid #00ff88; }
        .report .time { color: #666; font-size: 0.8rem; }

        .connection-panel {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
        }
        .connection-panel input {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #0f3460;
            background: #1a1a2e;
            color: white;
            width: 200px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.connected { background: #00ff88; }
        .status-dot.disconnected { background: #ff4444; }

        .detections {
            font-size: 0.9rem;
            max-height: 150px;
            overflow-y: auto;
        }

        /* Speed slider */
        .speed-control {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 15px;
            padding: 10px;
            background: #0f3460;
            border-radius: 8px;
        }
        .speed-control input[type="range"] {
            flex: 1;
            height: 8px;
            -webkit-appearance: none;
            background: #1a1a2e;
            border-radius: 4px;
            outline: none;
        }
        .speed-control input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }
        .speed-value { min-width: 60px; text-align: center; font-weight: bold; }

        /* LED controls */
        .led-controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }
        .color-picker-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .color-picker-wrapper input[type="color"] {
            width: 50px;
            height: 35px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .led-modes {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        .led-modes button {
            padding: 6px 12px;
            font-size: 0.8rem;
            background: #0f3460;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
        }
        .led-modes button:hover { background: #1a5aa0; }
        .led-modes button.active { background: #00d4ff; color: #1a1a2e; }

        /* Buzzer controls */
        .buzzer-controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .buzzer-controls button {
            padding: 8px 16px;
            background: #0f3460;
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
        }
        .buzzer-controls button:hover { background: #1a5aa0; }

        /* Sensor graphs */
        .sensor-graph {
            height: 80px;
            background: #0a0a1a;
            border-radius: 5px;
            margin-top: 10px;
            position: relative;
            overflow: hidden;
        }
        .sensor-graph canvas {
            width: 100%;
            height: 100%;
        }

        /* Help overlay */
        .help-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .help-overlay.show { display: flex; }
        .help-content {
            background: #16213e;
            padding: 30px;
            border-radius: 15px;
            max-width: 600px;
            border: 1px solid #0f3460;
        }
        .help-content h2 { margin-bottom: 20px; }
        .shortcut-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .shortcut {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .shortcut kbd {
            background: #0f3460;
            padding: 5px 10px;
            border-radius: 4px;
            font-family: monospace;
            min-width: 30px;
            text-align: center;
        }
        .help-close {
            margin-top: 20px;
            padding: 10px 20px;
            background: #00d4ff;
            border: none;
            border-radius: 5px;
            color: #1a1a2e;
            cursor: pointer;
            font-weight: bold;
        }

        /* Gamepad indicator */
        .gamepad-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: #0f3460;
            border-radius: 5px;
            font-size: 0.85rem;
        }
        .gamepad-status.connected { border: 1px solid #00ff88; }

        /* AI Mode controls */
        .ai-mode-panel {
            margin-top: 20px;
        }
        .ai-mode-toggle {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }
        .ai-mode-toggle button {
            padding: 12px 24px;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .ai-mode-toggle button.follow-btn {
            background: #0f3460;
            color: white;
        }
        .ai-mode-toggle button.follow-btn.active {
            background: #00ff88;
            color: #1a1a2e;
        }
        .ai-mode-toggle button.follow-btn:hover {
            background: #1a5aa0;
        }
        .ai-mode-toggle button.follow-btn.active:hover {
            background: #00cc6a;
        }
        .target-select {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        .target-select select {
            padding: 8px 12px;
            background: #0f3460;
            border: 1px solid #1a5aa0;
            border-radius: 5px;
            color: white;
            font-size: 0.9rem;
            cursor: pointer;
        }
        .target-select select:focus {
            outline: none;
            border-color: #00d4ff;
        }
        .follow-status {
            padding: 10px;
            background: #0a0a1a;
            border-radius: 5px;
            font-size: 0.85rem;
        }
        .follow-status .tracking {
            color: #00ff88;
        }
        .follow-status .searching {
            color: #ffaa00;
        }
        .follow-status .lost {
            color: #ff4444;
        }
        .follow-indicator {
            display: flex;
            gap: 5px;
            margin-top: 8px;
        }
        .follow-indicator .bar {
            height: 8px;
            background: #1a5aa0;
            border-radius: 4px;
            transition: width 0.1s;
        }
        .follow-indicator .bar.left { margin-left: auto; }
        .follow-indicator .bar.right { margin-right: auto; }
        .follow-indicator .bar.active-left { background: #00d4ff; }
        .follow-indicator .bar.active-right { background: #00d4ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Robot Control</h1>

        <div class="connection-panel">
            <input type="text" id="robotIp" placeholder="Robot IP (e.g., 192.168.1.100)" value="192.168.0.105">
            <button class="connect-btn" id="connectBtn" onclick="toggleConnection()">Connect</button>
            <span class="status-dot disconnected" id="statusDot"></span>
            <span id="connectionStatus">Disconnected</span>
            <button class="connect-btn" onclick="toggleHelp()" style="margin-left:auto;background:#0f3460;color:white;">? Help</button>
            <div class="gamepad-status" id="gamepadStatus">🎮 No gamepad</div>
        </div>

        <div class="grid">
            <div>
                <div class="panel">
                    <h2>Camera Feed <small style="font-weight:normal;color:#888">(click & drag to move camera)</small></h2>
                    <div class="video-container" id="videoContainer">
                        <img id="videoFeed" src="/video_feed" alt="Video Feed">
                        <div class="crosshair" id="crosshair" style="display:none;"></div>
                        <div class="servo-info">Pan: <span id="panValue">90</span> | Tilt: <span id="tiltValue">90</span></div>
                    </div>
                    <div style="margin-top:10px; text-align:center; display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
                        <button class="connect-btn" onclick="centerCamera()" style="padding:8px 16px;">Center Camera</button>
                        <button class="connect-btn" onclick="calibrateCamera()" style="padding:8px 16px; background:#0f3460;">Set as Forward</button>
                    </div>
                    <div id="calibrationStatus" style="margin-top:8px; font-size:0.85rem; color:#888; text-align:center;"></div>
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h2>AI Task</h2>
                    <div class="task-input">
                        <input type="text" id="taskInput" placeholder="Enter task (e.g., 'Explore the room and find a person')">
                        <button onclick="runTask()">Run</button>
                        <button class="stop-btn" onclick="stopTask()">Stop</button>
                    </div>
                    <div class="reports" id="reports"></div>
                </div>
            </div>

            <div>
                <div class="panel">
                    <h2>Manual Control</h2>
                    <div class="controls">
                        <button onclick="move('strafe_left')" title="Strafe Left (Q)">↖</button>
                        <button onclick="move('forward')" title="Forward (W)">↑</button>
                        <button onclick="move('strafe_right')" title="Strafe Right (E)">↗</button>
                        <button onclick="move('left')" title="Turn Left (A)">←</button>
                        <button onclick="move('stop')" title="Stop (Space)">⬤</button>
                        <button onclick="move('right')" title="Turn Right (D)">→</button>
                        <button></button>
                        <button onclick="move('backward')" title="Backward (S)">↓</button>
                        <button></button>
                    </div>
                    <div class="speed-control">
                        <span>🐢</span>
                        <input type="range" id="speedSlider" min="500" max="3000" value="1200" oninput="updateSpeed(this.value)">
                        <span>🐇</span>
                        <span class="speed-value" id="speedValue">1200</span>
                    </div>
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h2>LEDs</h2>
                    <div class="led-controls">
                        <div class="color-picker-wrapper">
                            <input type="color" id="ledColor" value="#00d4ff" onchange="setLedColor(this.value)">
                            <button onclick="setLedColor(document.getElementById('ledColor').value)" style="padding:8px 12px;background:#0f3460;border:none;border-radius:5px;color:white;cursor:pointer;">Apply</button>
                            <button onclick="setLedOff()" style="padding:8px 12px;background:#0f3460;border:none;border-radius:5px;color:white;cursor:pointer;">Off</button>
                        </div>
                    </div>
                    <div class="led-modes" style="margin-top:10px;">
                        <button onclick="setLedMode(0)" id="ledMode0">Off</button>
                        <button onclick="setLedMode(1)" id="ledMode1">Manual</button>
                        <button onclick="setLedMode(2)" id="ledMode2">Follow</button>
                        <button onclick="setLedMode(3)" id="ledMode3">Blink</button>
                        <button onclick="setLedMode(4)" id="ledMode4">Breathe</button>
                        <button onclick="setLedMode(5)" id="ledMode5">Rainbow</button>
                    </div>
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h2>Buzzer</h2>
                    <div class="buzzer-controls">
                        <button onclick="beep(100)">Beep</button>
                        <button onclick="beep(300)">Beep x2</button>
                        <button onclick="beep(500)">Long Beep</button>
                        <button onclick="playTune('alert')">🚨 Alert</button>
                        <button onclick="playTune('happy')">😊 Happy</button>
                        <button onclick="playTune('sad')">😢 Sad</button>
                    </div>
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h2>Status</h2>
                    <div class="status-grid">
                        <div class="status-item">
                            <label>Distance</label>
                            <div class="value" id="distance">-- cm</div>
                        </div>
                        <div class="status-item">
                            <label>YOLO FPS</label>
                            <div class="value" id="fps">--</div>
                        </div>
                        <div class="status-item">
                            <label>Mode</label>
                            <div class="value" id="mode">Manual</div>
                        </div>
                        <div class="status-item">
                            <label>Battery</label>
                            <div class="value" id="battery">--%</div>
                        </div>
                    </div>
                    <div class="sensor-graph">
                        <canvas id="distanceGraph"></canvas>
                    </div>
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h2>Detections</h2>
                    <div class="detections" id="detections">No objects detected</div>
                </div>

                <div class="panel ai-mode-panel">
                    <h2>AI Modes</h2>
                    <div class="ai-mode-toggle">
                        <button class="follow-btn" id="followToggleBtn" onclick="toggleFollowMode()">
                            Follow
                        </button>
                        <button class="follow-btn" id="patrolToggleBtn" onclick="togglePatrolMode()">
                            Patrol
                        </button>
                    </div>
                    <div class="target-select">
                        <label>Target:</label>
                        <select id="followTarget" onchange="updateFollowTarget(this.value)">
                            <option value="person">Person</option>
                            <option value="cat">Cat</option>
                            <option value="dog">Dog</option>
                            <option value="car">Car</option>
                            <option value="bottle">Bottle</option>
                            <option value="cup">Cup</option>
                            <option value="cell phone">Cell Phone</option>
                            <option value="book">Book</option>
                            <option value="backpack">Backpack</option>
                            <option value="chair">Chair</option>
                        </select>
                        <label style="margin-left:10px;">Speed:</label>
                        <input type="range" id="aiModeSpeed" min="500" max="2500" value="1200"
                               onchange="updateAIModeSpeed(this.value)" style="width:80px;">
                        <span id="aiModeSpeedValue">1200</span>
                    </div>
                    <div class="follow-status" id="aiModeStatus">
                        <span class="lost">AI modes inactive</span>
                    </div>
                    <div class="follow-indicator">
                        <div style="flex:1;text-align:right;font-size:0.7rem;color:#888;">Turn Left</div>
                        <div class="bar left" id="followBarLeft" style="width:0%;max-width:45%;"></div>
                        <div style="width:10px;"></div>
                        <div class="bar right" id="followBarRight" style="width:0%;max-width:45%;"></div>
                        <div style="flex:1;text-align:left;font-size:0.7rem;color:#888;">Turn Right</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Help Overlay -->
    <div class="help-overlay" id="helpOverlay" onclick="if(event.target===this)toggleHelp()">
        <div class="help-content">
            <h2>Keyboard Shortcuts</h2>
            <div class="shortcut-grid">
                <div class="shortcut"><kbd>W</kbd> / <kbd>↑</kbd> Forward</div>
                <div class="shortcut"><kbd>S</kbd> / <kbd>↓</kbd> Backward</div>
                <div class="shortcut"><kbd>A</kbd> / <kbd>←</kbd> Turn Left</div>
                <div class="shortcut"><kbd>D</kbd> / <kbd>→</kbd> Turn Right</div>
                <div class="shortcut"><kbd>Q</kbd> Strafe Left</div>
                <div class="shortcut"><kbd>E</kbd> Strafe Right</div>
                <div class="shortcut"><kbd>Space</kbd> Stop</div>
                <div class="shortcut"><kbd>C</kbd> Center Camera</div>
                <div class="shortcut"><kbd>B</kbd> Beep</div>
                <div class="shortcut"><kbd>H</kbd> / <kbd>?</kbd> This Help</div>
                <div class="shortcut"><kbd>Esc</kbd> Close Help / Stop</div>
                <div class="shortcut"><kbd>1-5</kbd> LED Modes</div>
                <div class="shortcut"><kbd>F</kbd> Toggle Follow Mode</div>
                <div class="shortcut"><kbd>P</kbd> Toggle Patrol Mode</div>
            </div>
            <h3 style="margin-top:20px;margin-bottom:10px;">Mouse Controls</h3>
            <p>Click and drag on the video feed to control the camera pan/tilt.</p>
            <h3 style="margin-top:20px;margin-bottom:10px;">Gamepad Support</h3>
            <p>Connect an Xbox/PlayStation controller for analog control. Left stick moves, right stick controls camera.</p>
            <button class="help-close" onclick="toggleHelp()">Got it!</button>
        </div>
    </div>


    <script>
        let ws;
        let isConnected = false;
        let isDragging = false;
        let currentPan = 90;
        let currentTilt = 90;
        let servoThrottle = null;
        let currentSpeed = 1200;
        let currentLedMode = 0;
        let distanceHistory = [];
        let gamepadIndex = null;
        let gamepadLoop = null;

        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateUI(data);
            };
            ws.onclose = function() {
                setTimeout(connectWebSocket, 1000);
            };
        }

        // Camera servo control via mouse
        function initCameraControl() {
            const container = document.getElementById('videoContainer');
            const crosshair = document.getElementById('crosshair');

            container.addEventListener('mousedown', (e) => {
                if (!isConnected) return;
                isDragging = true;
                crosshair.style.display = 'block';
                updateServoFromMouse(e);
            });

            container.addEventListener('mousemove', (e) => {
                if (!isDragging || !isConnected) return;
                updateServoFromMouse(e);
            });

            container.addEventListener('mouseup', () => {
                isDragging = false;
            });

            container.addEventListener('mouseleave', () => {
                isDragging = false;
                crosshair.style.display = 'none';
            });

            // Touch support for mobile
            container.addEventListener('touchstart', (e) => {
                if (!isConnected) return;
                isDragging = true;
                crosshair.style.display = 'block';
                updateServoFromTouch(e);
            });

            container.addEventListener('touchmove', (e) => {
                if (!isDragging || !isConnected) return;
                e.preventDefault();
                updateServoFromTouch(e);
            });

            container.addEventListener('touchend', () => {
                isDragging = false;
            });
        }

        function updateServoFromMouse(e) {
            const container = document.getElementById('videoContainer');
            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            updateServoPosition(x, y, rect.width, rect.height);
        }

        function updateServoFromTouch(e) {
            const container = document.getElementById('videoContainer');
            const rect = container.getBoundingClientRect();
            const touch = e.touches[0];
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            updateServoPosition(x, y, rect.width, rect.height);
        }

        function updateServoPosition(x, y, width, height) {
            const crosshair = document.getElementById('crosshair');
            crosshair.style.left = x + 'px';
            crosshair.style.top = y + 'px';

            // Map mouse position to servo angles
            // Pan: 0 (left) to 180 (right), center is 90
            // Tilt: 0 (down) to 180 (up), center is 90
            // Invert X so moving mouse right turns camera right (which looks left in view)
            currentPan = Math.round(180 - (x / width) * 180);
            currentTilt = Math.round(180 - (y / height) * 180);

            // Clamp values
            currentPan = Math.max(0, Math.min(180, currentPan));
            currentTilt = Math.max(0, Math.min(180, currentTilt));

            // Update display
            document.getElementById('panValue').textContent = currentPan;
            document.getElementById('tiltValue').textContent = currentTilt;

            // Throttle servo commands to avoid overwhelming the robot
            if (!servoThrottle) {
                servoThrottle = setTimeout(() => {
                    sendServoCommand(currentPan, currentTilt);
                    servoThrottle = null;
                }, 50); // Send at most every 50ms
            }
        }

        async function sendServoCommand(pan, tilt) {
            try {
                await fetch(`/servo?pan=${pan}&tilt=${tilt}`, {method: 'POST'});
            } catch (e) {
                console.error('Servo command failed:', e);
            }
        }

        async function centerCamera() {
            if (!isConnected) {
                addReport('Not connected to robot', 'warning');
                return;
            }
            // /servo/center now returns the calibrated center position
            const resp = await fetch('/servo/center', {method: 'POST'});
            const data = await resp.json();
            currentPan = data.pan || 90;
            currentTilt = data.tilt || 90;
            document.getElementById('panValue').textContent = currentPan;
            document.getElementById('tiltValue').textContent = currentTilt;
            addReport(`Camera centered at calibrated position (Pan: ${currentPan}, Tilt: ${currentTilt})`, 'info');
        }

        async function calibrateCamera() {
            if (!isConnected) {
                addReport('Not connected to robot', 'warning');
                return;
            }
            const resp = await fetch('/calibration/set-current', {method: 'POST'});
            const data = await resp.json();
            const statusEl = document.getElementById('calibrationStatus');
            statusEl.textContent = `Calibrated: Forward = Pan ${data.pan_center}, Tilt ${data.tilt_center}`;
            statusEl.style.color = '#00ff88';
            addReport(`Camera calibrated: forward direction set to Pan=${data.pan_center}, Tilt=${data.tilt_center}`, 'info');
        }

        async function loadCalibration() {
            try {
                const resp = await fetch('/calibration');
                const data = await resp.json();
                const statusEl = document.getElementById('calibrationStatus');
                if (data.calibrated) {
                    statusEl.textContent = `Calibrated: Forward = Pan ${data.pan_center}, Tilt ${data.tilt_center}`;
                    statusEl.style.color = '#888';
                } else {
                    statusEl.textContent = 'Not calibrated - point camera forward and click "Set as Forward"';
                    statusEl.style.color = '#ff8800';
                }
            } catch (e) {
                console.log('Could not load calibration');
            }
        }

        // Speed control
        function updateSpeed(value) {
            currentSpeed = parseInt(value);
            document.getElementById('speedValue').textContent = value;
        }

        // LED controls
        async function setLedColor(hexColor) {
            if (!isConnected) return;
            const r = parseInt(hexColor.slice(1,3), 16);
            const g = parseInt(hexColor.slice(3,5), 16);
            const b = parseInt(hexColor.slice(5,7), 16);
            await fetch(`/led?r=${r}&g=${g}&b=${b}`, {method: 'POST'});
            // Update UI to show manual mode is active
            for (let i = 0; i <= 5; i++) {
                const btn = document.getElementById('ledMode' + i);
                if (btn) btn.classList.toggle('active', i === 1);
            }
        }

        async function setLedOff() {
            if (!isConnected) return;
            await fetch('/led?r=0&g=0&b=0', {method: 'POST'});
        }

        async function setLedMode(mode) {
            if (!isConnected) return;
            currentLedMode = mode;
            await fetch(`/led?mode=${mode}`, {method: 'POST'});
            // Update button styles
            for (let i = 0; i <= 5; i++) {
                const btn = document.getElementById('ledMode' + i);
                if (btn) btn.classList.toggle('active', i === mode);
            }
        }

        // Buzzer controls
        async function beep(duration) {
            if (!isConnected) return;
            await fetch(`/buzzer?duration_ms=${duration}`, {method: 'POST'});
        }

        async function playTune(tune) {
            if (!isConnected) return;
            const tunes = {
                'alert': [100, 100, 100, 100, 300],
                'happy': [100, 50, 100, 50, 200],
                'sad': [300, 100, 400]
            };
            const pattern = tunes[tune] || [200];
            for (const duration of pattern) {
                await beep(duration);
                await new Promise(r => setTimeout(r, duration + 50));
            }
        }

        // Help overlay
        function toggleHelp() {
            document.getElementById('helpOverlay').classList.toggle('show');
        }

        // Distance graph
        function updateDistanceGraph(distance) {
            distanceHistory.push(distance);
            if (distanceHistory.length > 100) distanceHistory.shift();

            const canvas = document.getElementById('distanceGraph');
            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw grid
            ctx.strokeStyle = '#1a3a5a';
            ctx.lineWidth = 1;
            for (let y = 0; y < canvas.height; y += 20) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }

            // Draw line
            if (distanceHistory.length < 2) return;
            ctx.strokeStyle = '#00d4ff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const maxDist = 200; // cm
            for (let i = 0; i < distanceHistory.length; i++) {
                const x = (i / (distanceHistory.length - 1)) * canvas.width;
                const y = canvas.height - (distanceHistory[i] / maxDist) * canvas.height;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw current value
            ctx.fillStyle = '#00d4ff';
            ctx.font = '12px sans-serif';
            ctx.fillText(`${distance.toFixed(1)} cm`, 5, 15);
        }

        // Gamepad support
        function initGamepad() {
            window.addEventListener('gamepadconnected', (e) => {
                gamepadIndex = e.gamepad.index;
                document.getElementById('gamepadStatus').textContent = '🎮 ' + e.gamepad.id.split('(')[0];
                document.getElementById('gamepadStatus').classList.add('connected');
                addReport('Gamepad connected: ' + e.gamepad.id, 'success');
                startGamepadLoop();
            });

            window.addEventListener('gamepaddisconnected', (e) => {
                if (e.gamepad.index === gamepadIndex) {
                    gamepadIndex = null;
                    document.getElementById('gamepadStatus').textContent = '🎮 No gamepad';
                    document.getElementById('gamepadStatus').classList.remove('connected');
                    addReport('Gamepad disconnected', 'info');
                    if (gamepadLoop) cancelAnimationFrame(gamepadLoop);
                }
            });
        }

        function startGamepadLoop() {
            let lastMove = '';
            let lastServoUpdate = 0;

            function loop() {
                if (gamepadIndex === null || !isConnected) {
                    gamepadLoop = requestAnimationFrame(loop);
                    return;
                }

                const gp = navigator.getGamepads()[gamepadIndex];
                if (!gp) {
                    gamepadLoop = requestAnimationFrame(loop);
                    return;
                }

                // Left stick - movement (axes 0, 1)
                const lx = gp.axes[0];
                const ly = gp.axes[1];
                const deadzone = 0.2;

                let moveCmd = 'stop';
                if (Math.abs(ly) > deadzone || Math.abs(lx) > deadzone) {
                    if (ly < -deadzone && Math.abs(lx) < 0.5) moveCmd = 'forward';
                    else if (ly > deadzone && Math.abs(lx) < 0.5) moveCmd = 'backward';
                    else if (lx < -deadzone && Math.abs(ly) < 0.5) moveCmd = 'left';
                    else if (lx > deadzone && Math.abs(ly) < 0.5) moveCmd = 'right';
                    else if (lx < -deadzone && ly < -deadzone) moveCmd = 'strafe_left';
                    else if (lx > deadzone && ly < -deadzone) moveCmd = 'strafe_right';
                }

                if (moveCmd !== lastMove) {
                    move(moveCmd);
                    lastMove = moveCmd;
                }

                // Right stick - camera (axes 2, 3)
                const rx = gp.axes[2];
                const ry = gp.axes[3];
                const now = Date.now();

                if ((Math.abs(rx) > deadzone || Math.abs(ry) > deadzone) && now - lastServoUpdate > 100) {
                    currentPan = Math.max(0, Math.min(180, 90 - rx * 45));
                    currentTilt = Math.max(0, Math.min(180, 90 - ry * 45));
                    sendServoCommand(Math.round(currentPan), Math.round(currentTilt));
                    document.getElementById('panValue').textContent = Math.round(currentPan);
                    document.getElementById('tiltValue').textContent = Math.round(currentTilt);
                    lastServoUpdate = now;
                }

                // Buttons
                if (gp.buttons[0].pressed) beep(100); // A - beep
                if (gp.buttons[3].pressed) centerCamera(); // Y - center camera

                gamepadLoop = requestAnimationFrame(loop);
            }
            loop();
        }

        // AI Mode controls
        let followModeEnabled = false;
        let patrolModeEnabled = false;

        async function toggleFollowMode() {
            if (!isConnected) {
                addReport('Not connected to robot', 'warning');
                return;
            }

            followModeEnabled = !followModeEnabled;
            // Disable patrol mode if enabling follow mode
            if (followModeEnabled) patrolModeEnabled = false;

            try {
                const resp = await fetch(`/ai/follow?enabled=${followModeEnabled}`, {method: 'POST'});
                const data = await resp.json();

                if (data.status === 'ok') {
                    updateAIModeButtons();
                    addReport(followModeEnabled ? 'Follow mode enabled - tracking ' + data.target : 'Follow mode disabled', followModeEnabled ? 'success' : 'info');
                }
            } catch (e) {
                console.error('Failed to toggle follow mode:', e);
                followModeEnabled = !followModeEnabled; // Revert
            }
        }

        async function togglePatrolMode() {
            if (!isConnected) {
                addReport('Not connected to robot', 'warning');
                return;
            }

            patrolModeEnabled = !patrolModeEnabled;
            // Disable follow mode if enabling patrol mode
            if (patrolModeEnabled) followModeEnabled = false;

            try {
                const resp = await fetch(`/ai/patrol?enabled=${patrolModeEnabled}`, {method: 'POST'});
                const data = await resp.json();

                if (data.status === 'ok') {
                    updateAIModeButtons();
                    addReport(patrolModeEnabled ? 'Patrol mode enabled - exploring environment' : 'Patrol mode disabled', patrolModeEnabled ? 'success' : 'info');
                }
            } catch (e) {
                console.error('Failed to toggle patrol mode:', e);
                patrolModeEnabled = !patrolModeEnabled; // Revert
            }
        }

        function updateAIModeButtons() {
            const followBtn = document.getElementById('followToggleBtn');
            const patrolBtn = document.getElementById('patrolToggleBtn');

            followBtn.textContent = followModeEnabled ? 'Stop Follow' : 'Follow';
            followBtn.classList.toggle('active', followModeEnabled);

            patrolBtn.textContent = patrolModeEnabled ? 'Stop Patrol' : 'Patrol';
            patrolBtn.classList.toggle('active', patrolModeEnabled);
        }

        async function updateFollowTarget(target) {
            if (!isConnected) return;
            try {
                await fetch(`/ai/follow?target=${encodeURIComponent(target)}`, {method: 'POST'});
                addReport('Follow target set to: ' + target, 'info');
            } catch (e) {
                console.error('Failed to update follow target:', e);
            }
        }

        async function updateAIModeSpeed(speed) {
            document.getElementById('aiModeSpeedValue').textContent = speed;
            if (!isConnected) return;
            try {
                // Update speed for both modes
                await fetch(`/ai/follow?speed=${speed}`, {method: 'POST'});
            } catch (e) {
                console.error('Failed to update AI mode speed:', e);
            }
        }

        function updateFollowStatusUI(data) {
            const statusEl = document.getElementById('aiModeStatus');
            const barLeft = document.getElementById('followBarLeft');
            const barRight = document.getElementById('followBarRight');

            // FSD-style state display
            const stateDisplay = {
                'SEARCH_EXPLORING': { label: '🔍 Exploring', color: 'searching' },
                'SEARCH_TURNING': { label: '🔄 Turning', color: 'searching' },
                'TARGET_ACQUIRED': { label: '🎯 Target Locked', color: 'tracking' },
                'APPROACHING': { label: '🚀 Approaching', color: 'tracking' },
                'MAINTAINING': { label: '✅ Following', color: 'tracking' },
                'TARGET_LOST': { label: '⚠️ Target Lost', color: 'lost' },
                'OBSTACLE_AVOIDING': { label: '↩️ Avoiding Obstacle', color: 'searching' },
                'IDLE': { label: '⏸️ Idle', color: 'lost' }
            };

            const stateInfo = stateDisplay[data.state] || { label: data.state, color: 'searching' };

            if (data.tracking) {
                statusEl.innerHTML = `<span class="${stateInfo.color}">${stateInfo.label}</span> | ` +
                    `Target: ${data.target} (${(data.confidence * 100).toFixed(0)}%) | ` +
                    `${data.error_x > 0 ? '→' : '←'} ${Math.abs(data.error_x * 100).toFixed(0)}% | ` +
                    `Pan: ${data.camera_pan || 90}°`;

                // Update turn indicator bars
                const turnAmount = Math.abs(data.error_x) * 100;
                if (data.error_x < 0) {
                    barLeft.style.width = turnAmount + '%';
                    barLeft.classList.add('active-left');
                    barRight.style.width = '0%';
                    barRight.classList.remove('active-right');
                } else {
                    barRight.style.width = turnAmount + '%';
                    barRight.classList.add('active-right');
                    barLeft.style.width = '0%';
                    barLeft.classList.remove('active-left');
                }
            } else if (data.searching) {
                statusEl.innerHTML = `<span class="${stateInfo.color}">${stateInfo.label}</span> | ` +
                    `Looking for ${data.target}... | Pan: ${data.camera_pan || 90}°`;
                barLeft.style.width = '0%';
                barRight.style.width = '0%';
            } else {
                statusEl.innerHTML = `<span class="${stateInfo.color}">${stateInfo.label}</span> | ${data.message || ''}`;
                barLeft.style.width = '0%';
                barRight.style.width = '0%';
            }
        }

        function updatePatrolStatusUI(data) {
            const statusEl = document.getElementById('aiModeStatus');

            // Soldier-style patrol state display
            const stateDisplay = {
                'PATROLLING': { label: '🚶 Patrolling', color: 'tracking' },
                'OBSTACLE_TURN': { label: '↩️ Turning', color: 'searching' },
                'BACKING_UP': { label: '⬅️ Backing Up', color: 'lost' },
                'ALERT': { label: '👀 Alert!', color: 'searching' },
                'IDLE': { label: '⏸️ Idle', color: 'lost' }
            };

            const stateInfo = stateDisplay[data.state] || { label: data.state, color: 'searching' };

            let details = '';
            if (data.distance) details += ` | Distance: ${data.distance.toFixed(1)}cm`;
            if (data.speed) details += ` | Speed: ${data.speed}`;
            if (data.camera_pan) details += ` | Pan: ${data.camera_pan}°`;

            statusEl.innerHTML = `<span class="${stateInfo.color}">${stateInfo.label}</span>${details}`;

            if (data.message) {
                statusEl.innerHTML += `<br><small style="color:#888">${data.message}</small>`;
            }
        }

        function updateUI(data) {
            if (data.type === 'status') {
                const distance = data.distance || 0;
                document.getElementById('distance').textContent = distance.toFixed(1) + ' cm';
                document.getElementById('fps').textContent = (data.detections?.fps || 0).toFixed(1);

                // Update mode display with FSD state info
                let modeText = data.autonomous ? 'Autonomous' : 'Manual';
                if (data.autonomous_state?.active_mode === 'follow') {
                    modeText = 'FSD Follow: ' + (data.autonomous_state.follow_state || '');
                } else if (data.autonomous_state?.active_mode === 'patrol') {
                    modeText = 'FSD Patrol: ' + (data.autonomous_state.patrol_state || '');
                } else if (data.follow_mode?.enabled) {
                    modeText = 'Following: ' + data.follow_mode.target;
                } else if (data.patrol_mode?.enabled) {
                    modeText = 'Patrol Mode';
                }
                document.getElementById('mode').textContent = modeText;
                updateDistanceGraph(distance);

                if (data.detections?.summary) {
                    document.getElementById('detections').textContent = data.detections.summary;
                }

                // Sync AI mode UI state from autonomous system
                if (data.autonomous_state) {
                    followModeEnabled = data.autonomous_state.active_mode === 'follow';
                    patrolModeEnabled = data.autonomous_state.active_mode === 'patrol';
                } else {
                    if (data.follow_mode) {
                        followModeEnabled = data.follow_mode.enabled;
                    }
                    if (data.patrol_mode) {
                        patrolModeEnabled = data.patrol_mode.enabled;
                    }
                }
                updateAIModeButtons();

                // Update AI mode status when both are inactive
                if (!followModeEnabled && !patrolModeEnabled) {
                    document.getElementById('aiModeStatus').innerHTML = '<span class="lost">AI modes inactive</span>';
                    document.getElementById('followBarLeft').style.width = '0%';
                    document.getElementById('followBarRight').style.width = '0%';
                }
            } else if (data.type === 'follow_status') {
                updateFollowStatusUI(data);
            } else if (data.type === 'patrol_status') {
                updatePatrolStatusUI(data);
            } else if (data.type === 'patrol_alert') {
                addReport(`Alert: ${data.object} detected (${(data.confidence * 100).toFixed(0)}% confidence)`, 'warning');
            } else if (data.type === 'report') {
                addReport(data.message, data.msg_type);
            } else if (data.type === 'task_complete') {
                addReport('Task completed: ' + data.result, 'success');
            }
        }

        function addReport(message, type) {
            const reports = document.getElementById('reports');
            const div = document.createElement('div');
            div.className = 'report ' + (type || 'info');
            div.innerHTML = '<span class="time">' + new Date().toLocaleTimeString() + '</span> ' + message;
            reports.insertBefore(div, reports.firstChild);
            if (reports.children.length > 20) {
                reports.removeChild(reports.lastChild);
            }
        }

        async function toggleConnection() {
            const btn = document.getElementById('connectBtn');
            const ip = document.getElementById('robotIp').value;

            if (!isConnected) {
                const resp = await fetch('/connect?ip=' + ip, {method: 'POST'});
                const data = await resp.json();
                if (data.status === 'connected' || data.status === 'already_connected') {
                    isConnected = true;
                    btn.textContent = 'Disconnect';
                    document.getElementById('statusDot').className = 'status-dot connected';
                    document.getElementById('connectionStatus').textContent = 'Connected to ' + ip;
                    addReport('Connected to robot at ' + ip, 'success');
                    loadCalibration();  // Load and display calibration status
                } else {
                    addReport('Failed to connect to ' + ip, 'error');
                }
            } else {
                await fetch('/disconnect', {method: 'POST'});
                isConnected = false;
                btn.textContent = 'Connect';
                document.getElementById('statusDot').className = 'status-dot disconnected';
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                addReport('Disconnected from robot', 'info');
            }
        }

        async function move(action) {
            if (!isConnected) {
                addReport('Not connected to robot', 'warning');
                return;
            }
            await fetch(`/manual/${action}?speed=${currentSpeed}`, {method: 'POST'});
        }

        async function runTask() {
            const task = document.getElementById('taskInput').value;
            if (!task) return;

            const resp = await fetch('/task?task=' + encodeURIComponent(task), {method: 'POST'});
            const data = await resp.json();
            if (data.status === 'started') {
                addReport('Started task: ' + task, 'info');
            } else {
                addReport('Failed to start task', 'error');
            }
        }

        async function stopTask() {
            await fetch('/stop', {method: 'POST'});
            // Stop all AI modes
            followModeEnabled = false;
            patrolModeEnabled = false;
            updateAIModeButtons();
            addReport('All tasks stopped', 'warning');
        }

        // Keyboard controls (only when not typing in an input field)
        document.addEventListener('keydown', function(e) {
            // Ignore if user is typing in an input or textarea
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            const keyMap = {
                'ArrowUp': 'forward', 'w': 'forward', 'W': 'forward',
                'ArrowDown': 'backward', 's': 'backward', 'S': 'backward',
                'ArrowLeft': 'left', 'a': 'left', 'A': 'left',
                'ArrowRight': 'right', 'd': 'right', 'D': 'right',
                'q': 'strafe_left', 'Q': 'strafe_left',
                'e': 'strafe_right', 'E': 'strafe_right',
                ' ': 'stop'
            };
            if (keyMap[e.key]) {
                e.preventDefault();
                move(keyMap[e.key]);
            }

            // Other shortcuts
            if (e.key === 'c' || e.key === 'C') { centerCamera(); }
            if (e.key === 'b' || e.key === 'B') { beep(200); }
            if (e.key === 'f' || e.key === 'F') { toggleFollowMode(); }
            if (e.key === 'p' || e.key === 'P') { togglePatrolMode(); }
            if (e.key === 'h' || e.key === 'H' || e.key === '?') { toggleHelp(); }
            if (e.key === 'Escape') {
                if (document.getElementById('helpOverlay').classList.contains('show')) {
                    toggleHelp();
                } else {
                    move('stop');
                }
            }
            // LED modes with number keys
            if (e.key >= '1' && e.key <= '5') { setLedMode(parseInt(e.key)); }
            if (e.key === '0') { setLedMode(0); }
        });

        document.addEventListener('keyup', function(e) {
            // Ignore if user is typing in an input or textarea
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            const moveKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
                              'w', 'a', 's', 'd', 'W', 'A', 'S', 'D',
                              'q', 'e', 'Q', 'E'];
            if (moveKeys.includes(e.key)) {
                move('stop');
            }
        });

        connectWebSocket();
        initCameraControl();
        initGamepad();
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    server_config = config.get("server", {})
    uvicorn.run(
        app,
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8080)
    )
