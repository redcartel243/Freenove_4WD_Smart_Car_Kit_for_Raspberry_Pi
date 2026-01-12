"""
AI Tools Definition
Defines all the tools the LLM agent can use to control the robot
"""

from typing import Literal, Optional
from dataclasses import dataclass
import json


# Tool schemas for the LLM (OpenAI function calling format)
ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the robot in a direction. Use this for basic movement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "left", "right"],
                        "description": "Direction to move. 'left' and 'right' will turn the robot in place."
                    },
                    "speed": {
                        "type": "integer",
                        "description": "Speed from 500 (slow) to 2000 (fast). Default is 1200.",
                        "default": 1200
                    },
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration in milliseconds. Default is 500ms.",
                        "default": 500
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": "Rotate the robot by a specific angle. Positive = clockwise, negative = counter-clockwise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {
                        "type": "integer",
                        "description": "Degrees to rotate. Positive = right/clockwise, negative = left/counter-clockwise. Typical: 45, 90, -90, 180"
                    },
                    "speed": {
                        "type": "integer",
                        "description": "Rotation speed from 500 to 2000. Default is 1000.",
                        "default": 1000
                    }
                },
                "required": ["degrees"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Immediately stop all motors. Use in emergencies or when you need to halt.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_distance",
            "description": "Get the ultrasonic distance sensor reading. Returns distance to nearest obstacle in centimeters.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_objects",
            "description": "Get a list of objects currently visible to the camera (detected by YOLO). Returns object labels, positions, and sizes.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_object",
            "description": "Look for a specific type of object in the camera view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_type": {
                        "type": "string",
                        "description": "Type of object to find (e.g., 'person', 'chair', 'bottle', 'dog', 'cat')"
                    }
                },
                "required": ["object_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scan_surroundings",
            "description": "Pan the camera left and right to scan the environment. Returns objects seen at each angle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angles": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Camera pan angles to check (0-180, 90=center). Default: [45, 90, 135]",
                        "default": [45, 90, 135]
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_at",
            "description": "Point the camera in a specific direction using pan/tilt servos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pan": {
                        "type": "integer",
                        "description": "Pan angle (0=full right, 90=center, 180=full left)"
                    },
                    "tilt": {
                        "type": "integer",
                        "description": "Tilt angle (80=down, 90=level, 150=up)"
                    }
                },
                "required": ["pan", "tilt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "center_camera",
            "description": "Center the camera (pan=90, tilt=90).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_light_sensors",
            "description": "Get light sensor readings from left and right photoresistors. Returns voltage values.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_line_sensors",
            "description": "Get infrared line sensor readings. Returns a 3-bit value representing which sensors detect a line.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_battery",
            "description": "Get current battery voltage level.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_leds",
            "description": "Set the RGB LEDs on the robot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "r": {
                        "type": "integer",
                        "description": "Red value (0-255)"
                    },
                    "g": {
                        "type": "integer",
                        "description": "Green value (0-255)"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Blue value (0-255)"
                    },
                    "mode": {
                        "type": "integer",
                        "description": "LED mode: 0=solid, 1=follow, 2=blink, 3=rainbow breathe, 4=rainbow cycle",
                        "default": 0
                    }
                },
                "required": ["r", "g", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "beep",
            "description": "Make a beep sound with the buzzer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration of beep in milliseconds. Default is 200ms.",
                        "default": 200
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_photo",
            "description": "Capture and save a photo from the camera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename to save the photo. Default is auto-generated with timestamp.",
                        "default": None
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Send the current camera view to a vision LLM for detailed analysis. Use when you need to understand something the object detector can't identify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What do you want to know about the image? (e.g., 'What color is the object?', 'Is this a kitchen or living room?')"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a specified duration. Use between actions when needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "description": "Duration to wait in milliseconds"
                    }
                },
                "required": ["duration_ms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report",
            "description": "Send a status report or message to the user interface. Use this to communicate what you're doing or what you've found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to display to the user"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["info", "warning", "success", "error"],
                        "description": "Type of message for styling",
                        "default": "info"
                    }
                },
                "required": ["message"]
            }
        }
    }
]


def get_tools_for_ollama() -> list:
    """Convert tools to Ollama format"""
    return ROBOT_TOOLS


def get_tools_description() -> str:
    """Get a text description of all tools for the system prompt"""
    descriptions = []
    for tool in ROBOT_TOOLS:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        params = func.get("parameters", {}).get("properties", {})

        param_strs = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            param_strs.append(f"  - {pname} ({ptype}): {pdesc}")

        if param_strs:
            descriptions.append(f"- {name}: {desc}\n" + "\n".join(param_strs))
        else:
            descriptions.append(f"- {name}: {desc}")

    return "\n\n".join(descriptions)
