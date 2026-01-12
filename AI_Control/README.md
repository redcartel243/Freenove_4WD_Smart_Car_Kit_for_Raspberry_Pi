# AI Robot Control

An AI-powered control system for the Freenove 4WD Smart Car Kit. Uses YOLO for real-time object detection and an LLM with tool calling for autonomous navigation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      YOUR PC                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ YOLO v8     │  │ LLM Agent   │  │ Web Interface       │  │
│  │ (Detection) │  │ (Qwen2.5)   │  │ (FastAPI)           │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         └────────────────┼────────────────────┘              │
│                          ▼                                   │
│                  TCP Connection                              │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   RASPBERRY PI 3B (Robot)                     │
│  Motors │ Camera │ Ultrasonic │ IR Sensors │ LEDs │ Buzzer   │
└──────────────────────────────────────────────────────────────┘
```

## Features

- **Real-time Object Detection**: YOLOv8 running on your PC GPU
- **Agentic AI Control**: LLM with tool calling for autonomous navigation
- **Web Interface**: Monitor camera feed, control manually, give AI tasks
- **Sensor Integration**: Ultrasonic, IR line sensors, light sensors
- **Manual Override**: Keyboard/button controls when needed

## Requirements

### Hardware
- Freenove 4WD Smart Car Kit (assembled and working)
- Raspberry Pi 3B running the car's server software
- PC with:
  - NVIDIA GPU (RTX 4060 or similar, 8GB VRAM)
  - Python 3.10+

### Software
- Ollama with `qwen2.5:7b-instruct` model installed
- (Optional) Moondream model for vision LLM queries

## Installation

1. **Install Ollama** (for local LLM):
   ```bash
   # Windows: Download from https://ollama.ai
   # Then pull the model:
   ollama pull qwen2.5:7b-instruct
   ollama pull moondream  # Optional for vision
   ```

2. **Install Python dependencies**:
   ```bash
   cd AI_Control
   pip install -r requirements.txt
   ```

3. **Configure**:
   Edit `config.yaml` and set your robot's IP address.

## Usage

1. **Start the robot** (on Raspberry Pi):
   ```bash
   cd Code/Server
   python main.py
   ```

2. **Start the AI control server** (on your PC):
   ```bash
   cd AI_Control
   python main.py
   ```

3. **Open the web interface**:
   Go to `http://localhost:8080` in your browser.

4. **Connect to the robot**:
   Enter the Pi's IP address and click Connect.

5. **Give the AI a task**:
   Type something like:
   - "Explore the room and tell me what you see"
   - "Find a person and go towards them"
   - "Navigate to the door while avoiding obstacles"

## AI Tools

The LLM agent can use these tools:

| Tool | Description |
|------|-------------|
| `move(direction, speed, duration)` | Move forward/backward/left/right |
| `rotate(degrees, speed)` | Rotate in place |
| `stop()` | Emergency stop |
| `get_distance()` | Get ultrasonic reading |
| `get_objects()` | Get YOLO detections |
| `find_object(type)` | Look for specific object |
| `scan_surroundings()` | Pan camera and detect |
| `look_at(pan, tilt)` | Point camera |
| `set_leds(r, g, b)` | Control LEDs |
| `beep(duration)` | Make sound |
| `analyze_image(question)` | Ask vision LLM |
| `report(message)` | Report to user |

## Configuration

Edit `config.yaml`:

```yaml
robot:
  ip: "192.168.1.100"  # Your Pi's IP
  command_port: 5000
  video_port: 8000

llm:
  provider: "ollama"
  model: "qwen2.5:7b-instruct"

vision:
  yolo_model: "yolov8n.pt"  # or yolov8s.pt for better accuracy
```

## Keyboard Controls

When in manual mode:
- `W/↑` - Forward
- `S/↓` - Backward
- `A/←` - Turn left
- `D/→` - Turn right
- `Space` - Stop

## Troubleshooting

**Can't connect to robot:**
- Check Pi's IP address
- Ensure the robot server is running
- Check firewall settings

**YOLO is slow:**
- Make sure you have CUDA installed
- Try using `yolov8n.pt` (nano model)
- Increase `frame_skip` in config

**LLM not responding:**
- Check Ollama is running: `ollama list`
- Pull the model: `ollama pull qwen2.5:7b-instruct`
