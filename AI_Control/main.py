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

# State
is_autonomous = False
autonomous_task: Optional[asyncio.Task] = None
connected_websockets: list = []


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
    """Broadcast a message to all connected websockets"""
    for ws in connected_websockets:
        try:
            asyncio.create_task(ws.send_json(message))
        except:
            pass


# Frame processing thread
def frame_processor():
    """Background thread that processes frames with YOLO"""
    frame_skip = config.get("vision", {}).get("frame_skip", 2)
    frame_count = 0

    while True:
        if robot and robot.connected and yolo:
            frame = robot.get_frame()
            if frame is not None:
                frame_count += 1
                if frame_count % frame_skip == 0:
                    yolo.process_frame(frame)
        time.sleep(0.033)  # ~30 FPS max


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global robot, yolo, agent

    print("Starting AI Robot Control Server...")

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

    yield

    # Cleanup
    if robot and robot.connected:
        robot.disconnect()
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
    global robot

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
        return {"status": "connected", "ip": robot_ip}
    else:
        return {"status": "failed", "ip": robot_ip}


@app.post("/disconnect")
async def disconnect_robot():
    """Disconnect from the robot"""
    global robot, is_autonomous

    is_autonomous = False
    if autonomous_task:
        autonomous_task.cancel()

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
    global is_autonomous

    if robot and robot.connected:
        robot.stop()

    if agent:
        agent.stop()

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


@app.get("/video_feed")
async def video_feed():
    """MJPEG video stream"""
    def generate():
        while True:
            if yolo:
                frame = yolo.get_annotated_frame()
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
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
            status = {
                "type": "status",
                "connected": robot.connected if robot else False,
                "autonomous": is_autonomous,
                "distance": robot.get_distance() if robot and robot.connected else 0,
                "detections": yolo.get_detection_summary() if yolo else {},
                "agent_state": agent.get_state() if agent else None
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
        }
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Robot Control</h1>

        <div class="connection-panel">
            <input type="text" id="robotIp" placeholder="Robot IP (e.g., 192.168.1.100)" value="192.168.1.100">
            <button class="connect-btn" id="connectBtn" onclick="toggleConnection()">Connect</button>
            <span class="status-dot disconnected" id="statusDot"></span>
            <span id="connectionStatus">Disconnected</span>
        </div>

        <div class="grid">
            <div>
                <div class="panel">
                    <h2>Camera Feed</h2>
                    <div class="video-container">
                        <img id="videoFeed" src="/video_feed" alt="Video Feed">
                    </div>
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
                        <button onclick="move('strafe_left')">↖</button>
                        <button onclick="move('forward')">↑</button>
                        <button onclick="move('strafe_right')">↗</button>
                        <button onclick="move('left')">←</button>
                        <button onclick="move('stop')">⬤</button>
                        <button onclick="move('right')">→</button>
                        <button></button>
                        <button onclick="move('backward')">↓</button>
                        <button></button>
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
                </div>

                <div class="panel" style="margin-top: 20px;">
                    <h2>Detections</h2>
                    <div class="detections" id="detections">No objects detected</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let isConnected = false;

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

        function updateUI(data) {
            if (data.type === 'status') {
                document.getElementById('distance').textContent = (data.distance || 0).toFixed(1) + ' cm';
                document.getElementById('fps').textContent = (data.detections?.fps || 0).toFixed(1);
                document.getElementById('mode').textContent = data.autonomous ? 'Autonomous' : 'Manual';

                if (data.detections?.summary) {
                    document.getElementById('detections').textContent = data.detections.summary;
                }
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
            await fetch('/manual/' + action, {method: 'POST'});
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
            addReport('Task stopped', 'warning');
        }

        // Keyboard controls
        document.addEventListener('keydown', function(e) {
            const keyMap = {
                'ArrowUp': 'forward', 'w': 'forward',
                'ArrowDown': 'backward', 's': 'backward',
                'ArrowLeft': 'left', 'a': 'left',
                'ArrowRight': 'right', 'd': 'right',
                ' ': 'stop'
            };
            if (keyMap[e.key]) {
                e.preventDefault();
                move(keyMap[e.key]);
            }
        });

        document.addEventListener('keyup', function(e) {
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'w', 'a', 's', 'd'].includes(e.key)) {
                move('stop');
            }
        });

        connectWebSocket();
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
