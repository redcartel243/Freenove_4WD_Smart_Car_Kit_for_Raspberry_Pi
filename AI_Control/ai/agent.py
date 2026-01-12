"""
AI Agent
The brain that controls the robot using an LLM with tool calling
"""

import json
import time
import base64
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import threading
import httpx

from .tools import ROBOT_TOOLS, get_tools_description


@dataclass
class AgentConfig:
    """Configuration for the AI agent"""
    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    vision_model: str = "moondream"


@dataclass
class AgentState:
    """Current state of the agent"""
    is_running: bool = False
    current_task: Optional[str] = None
    last_action: Optional[str] = None
    last_observation: Optional[str] = None
    message_history: List[Dict] = field(default_factory=list)
    reports: List[Dict] = field(default_factory=list)


class RobotAgent:
    """
    AI Agent that controls the robot using tool calling.
    Supports Ollama (local) and OpenAI-compatible APIs.
    """

    SYSTEM_PROMPT = """You are an AI controlling a 4-wheel drive robot. You can see through its camera (processed by YOLO object detection) and control its movements.

Your goal is to navigate safely, explore the environment, and complete tasks given by the user.

IMPORTANT RULES:
1. ALWAYS check distance before moving forward - stop if obstacle is closer than 20cm
2. Move slowly and carefully - use speed 800-1200 for normal movement
3. When you see an obstacle ahead, stop and turn to find a clear path
4. Report what you're doing using the 'report' tool so the user knows your status
5. Be methodical - scan surroundings before making decisions
6. If you're stuck or unsure, stop and report the situation

NAVIGATION STRATEGY:
- Check distance sensor frequently
- If distance < 30cm, slow down or stop
- If distance < 15cm, stop immediately and turn
- Scan left/right to find the clearest path
- Use object detection to understand your environment

You have access to these tools:
{tools}

When given a task, break it down into steps and execute them carefully.
Always explain what you're doing and why through the 'report' tool."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState()
        self.tool_handlers: Dict[str, Callable] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # HTTP client for LLM API calls
        self.http_client = httpx.Client(timeout=60.0)

        # Initialize system prompt with tool descriptions
        self.system_prompt = self.SYSTEM_PROMPT.format(tools=get_tools_description())

    def register_tool_handler(self, name: str, handler: Callable):
        """Register a handler function for a tool"""
        self.tool_handlers[name] = handler

    def register_all_handlers(self, handlers: Dict[str, Callable]):
        """Register multiple handlers at once"""
        self.tool_handlers.update(handlers)

    async def _call_llm(self, messages: List[Dict]) -> Dict:
        """Call the LLM API and get a response"""
        if self.config.provider == "ollama":
            return await self._call_ollama(messages)
        else:
            return await self._call_openai(messages)

    async def _call_ollama(self, messages: List[Dict]) -> Dict:
        """Call Ollama API"""
        url = f"{self.config.base_url}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": ROBOT_TOOLS,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }

        response = self.http_client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def _call_openai(self, messages: List[Dict]) -> Dict:
        """Call OpenAI-compatible API"""
        url = f"{self.config.base_url}/chat/completions"

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": ROBOT_TOOLS,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        response = self.http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool and return the result"""
        if tool_name not in self.tool_handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            handler = self.tool_handlers[tool_name]
            # Check if handler is async
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)

            # Convert result to string if needed
            if isinstance(result, dict):
                return json.dumps(result)
            return str(result)

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def run_task(self, task: str, max_iterations: int = 20) -> str:
        """
        Run a task with the agent.

        Args:
            task: Natural language task description
            max_iterations: Maximum number of LLM calls to prevent infinite loops

        Returns:
            Final response from the agent
        """
        with self._lock:
            self.state.is_running = True
            self.state.current_task = task
            self.state.message_history = []

        # Initialize conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]

        try:
            for iteration in range(max_iterations):
                if self._stop_event.is_set():
                    return "Task stopped by user"

                # Call LLM
                response = await self._call_llm(messages)

                # Parse response based on provider
                if self.config.provider == "ollama":
                    message = response.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])
                else:
                    # OpenAI format
                    choice = response.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls", [])

                # Add assistant message to history
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls if tool_calls else None})

                # If no tool calls, we're done
                if not tool_calls:
                    with self._lock:
                        self.state.is_running = False
                    return content

                # Execute each tool call
                for tool_call in tool_calls:
                    if self._stop_event.is_set():
                        return "Task stopped by user"

                    # Parse tool call (handle both Ollama and OpenAI formats)
                    if isinstance(tool_call, dict):
                        if "function" in tool_call:
                            # Ollama format
                            func = tool_call["function"]
                            tool_name = func.get("name", "")
                            arguments = func.get("arguments", {})
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                        else:
                            tool_name = tool_call.get("name", "")
                            arguments = tool_call.get("arguments", {})
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                    else:
                        continue

                    # Execute tool
                    with self._lock:
                        self.state.last_action = f"{tool_name}({arguments})"

                    result = await self._execute_tool(tool_name, arguments)

                    with self._lock:
                        self.state.last_observation = result

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": tool_name
                    })

            return "Max iterations reached"

        except Exception as e:
            with self._lock:
                self.state.is_running = False
            return f"Error: {str(e)}"

        finally:
            with self._lock:
                self.state.is_running = False
                self.state.message_history = messages

    def stop(self):
        """Stop the current task"""
        self._stop_event.set()

    def reset(self):
        """Reset the agent state"""
        self._stop_event.clear()
        with self._lock:
            self.state = AgentState()

    async def analyze_image_with_vision_llm(self, image_bytes: bytes, question: str) -> str:
        """
        Send an image to the vision LLM for analysis.
        Used by the analyze_image tool.
        """
        if self.config.provider == "ollama":
            # Ollama multimodal format
            url = f"{self.config.base_url}/api/chat"

            # Base64 encode the image
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            payload = {
                "model": self.config.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                        "images": [image_b64]
                    }
                ],
                "stream": False
            }

            response = self.http_client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "Unable to analyze image")

        else:
            # OpenAI vision format
            url = f"{self.config.base_url}/chat/completions"

            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            payload = {
                "model": self.config.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ]
            }

            response = self.http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "Unable to analyze image")

    def get_state(self) -> Dict:
        """Get current agent state as dict"""
        with self._lock:
            return {
                "is_running": self.state.is_running,
                "current_task": self.state.current_task,
                "last_action": self.state.last_action,
                "last_observation": self.state.last_observation,
                "reports": self.state.reports.copy()
            }

    def add_report(self, message: str, msg_type: str = "info"):
        """Add a report message"""
        with self._lock:
            self.state.reports.append({
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "type": msg_type
            })
            # Keep only last 50 reports
            if len(self.state.reports) > 50:
                self.state.reports = self.state.reports[-50:]
