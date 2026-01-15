"""
FSD-Inspired Autonomous Navigation System

This module implements a layered autonomous navigation system:
1. Perception Layer - Fuses sensors and builds world model
2. Planning Layer - Decides what to do based on perception
3. Control Layer - Executes smooth, coordinated movements
"""

import time
import math
import threading
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto
from collections import deque


# ============== Camera Calibration ==============

@dataclass
class CameraCalibration:
    """Camera calibration data for body-camera alignment"""
    pan_center: int = 90    # Pan angle when camera faces forward (aligned with body)
    tilt_center: int = 90   # Tilt angle when camera is level
    pan_range: Tuple[int, int] = (30, 150)
    tilt_range: Tuple[int, int] = (30, 150)
    calibrated: bool = False

    def get_body_relative_pan(self, current_pan: int) -> float:
        """
        Get camera pan relative to body direction.
        Returns degrees: positive = looking right, negative = looking left
        """
        return current_pan - self.pan_center

    def get_forward_pan(self) -> int:
        """Get the pan angle that points straight ahead"""
        return self.pan_center

    @classmethod
    def load_from_config(cls, config_path: str = None) -> 'CameraCalibration':
        """Load calibration from config file"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            cal_config = config.get('camera_calibration', {})
            return cls(
                pan_center=cal_config.get('pan_center', 90),
                tilt_center=cal_config.get('tilt_center', 90),
                pan_range=tuple(cal_config.get('pan_range', [30, 150])),
                tilt_range=tuple(cal_config.get('tilt_range', [30, 150])),
                calibrated=cal_config.get('calibrated', False)
            )
        except Exception as e:
            print(f"Could not load camera calibration: {e}")
            return cls()

    def save_to_config(self, config_path: str = None):
        """Save calibration to config file"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            config['camera_calibration'] = {
                'pan_center': self.pan_center,
                'tilt_center': self.tilt_center,
                'pan_range': list(self.pan_range),
                'tilt_range': list(self.tilt_range),
                'calibrated': self.calibrated
            }

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            print(f"Camera calibration saved: pan_center={self.pan_center}, tilt_center={self.tilt_center}")
        except Exception as e:
            print(f"Could not save camera calibration: {e}")


# ============== Data Classes ==============

@dataclass
class TrackedObject:
    """An object being tracked over time"""
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]
    center: Tuple[float, float]  # (x, y) normalized -1 to 1
    size_ratio: float  # portion of frame
    estimated_distance: float  # cm, estimated from size
    velocity: Tuple[float, float] = (0, 0)  # estimated velocity
    last_seen: float = 0  # timestamp
    frames_tracked: int = 0

    def age(self) -> float:
        """How long since last seen"""
        return time.time() - self.last_seen


@dataclass
class ScanResult:
    """Result of scanning in a direction"""
    angle: int  # camera pan angle
    clearance: float  # estimated clearance in cm
    obstacles: List[str]  # detected obstacle types
    timestamp: float = 0


@dataclass
class WorldModel:
    """Simple world model built from perception"""
    # Tracked objects by class
    tracked_objects: Dict[str, TrackedObject] = field(default_factory=dict)

    # Clearance map from camera scans (angle -> clearance)
    clearance_map: Dict[int, ScanResult] = field(default_factory=dict)

    # Current sensor readings
    ultrasonic_distance: float = 100.0
    camera_pan: int = 90
    camera_tilt: int = 90

    # Movement state
    is_moving: bool = False
    current_speed: float = 0

    # Timestamps
    last_update: float = 0


# ============== Enums for States ==============

class FollowState(Enum):
    IDLE = auto()
    SEARCH_EXPLORING = auto()  # Moving forward slowly while sweeping camera
    SEARCH_TURNING = auto()    # Turning to explore new direction (obstacle ahead)
    TARGET_ACQUIRED = auto()   # Found target, centering camera
    APPROACHING = auto()       # Moving toward target with obstacle avoidance
    MAINTAINING = auto()       # At good distance, small adjustments only
    TARGET_LOST = auto()       # Brief pause before returning to search
    OBSTACLE_AVOIDING = auto() # Navigating around obstacle while following


class PatrolState(Enum):
    IDLE = auto()
    PATROLLING = auto()        # Smooth forward movement with camera sweep (main state)
    OBSTACLE_TURN = auto()     # Turning away from obstacle
    BACKING_UP = auto()        # Backing away from very close obstacle
    ALERT = auto()             # Detected something interesting (person, etc)


# ============== Perception Layer ==============

class PerceptionLayer:
    """
    Fuses multiple sensor inputs into a coherent world model.
    Tracks objects over time and estimates distances.
    """

    # Size-to-distance calibration (approximate)
    # Maps object class to (reference_size_ratio, reference_distance_cm)
    SIZE_DISTANCE_MAP = {
        "person": (0.25, 100),  # Person takes 25% of frame at 100cm
        "cat": (0.10, 80),
        "dog": (0.12, 80),
        "bottle": (0.05, 50),
        "cup": (0.04, 40),
        "chair": (0.20, 120),
        "cell phone": (0.03, 30),
    }

    def __init__(self):
        self.world = WorldModel()
        self.object_history: Dict[str, deque] = {}  # For velocity estimation
        self.lock = threading.Lock()

    def update_from_yolo(self, detections: List[dict], frame_width: int = 640, frame_height: int = 480):
        """Update world model from YOLO detections"""
        with self.lock:
            current_time = time.time()

            for det in detections:
                class_name = det.get("class_name", "unknown")
                confidence = det.get("confidence", 0)
                box = det.get("box", [0, 0, 0, 0])

                # Calculate normalized center (-1 to 1)
                center_x = ((box[0] + box[2]) / 2 - frame_width / 2) / (frame_width / 2)
                center_y = ((box[1] + box[3]) / 2 - frame_height / 2) / (frame_height / 2)

                # Calculate size ratio
                obj_width = box[2] - box[0]
                obj_height = box[3] - box[1]
                size_ratio = (obj_width * obj_height) / (frame_width * frame_height)

                # Estimate distance from size
                estimated_dist = self._estimate_distance(class_name, size_ratio)

                # Calculate velocity if we have history
                velocity = (0, 0)
                if class_name in self.object_history and len(self.object_history[class_name]) > 0:
                    prev = self.object_history[class_name][-1]
                    dt = current_time - prev[2]
                    if dt > 0 and dt < 0.5:  # Only if recent
                        velocity = (
                            (center_x - prev[0]) / dt,
                            (center_y - prev[1]) / dt
                        )

                # Store in history for velocity calculation
                if class_name not in self.object_history:
                    self.object_history[class_name] = deque(maxlen=10)
                self.object_history[class_name].append((center_x, center_y, current_time))

                # Update tracked object
                tracked = TrackedObject(
                    class_name=class_name,
                    confidence=confidence,
                    box=box,
                    center=(center_x, center_y),
                    size_ratio=size_ratio,
                    estimated_distance=estimated_dist,
                    velocity=velocity,
                    last_seen=current_time,
                    frames_tracked=self.world.tracked_objects.get(class_name, TrackedObject(
                        class_name="", confidence=0, box=[], center=(0,0),
                        size_ratio=0, estimated_distance=0
                    )).frames_tracked + 1
                )
                self.world.tracked_objects[class_name] = tracked

            self.world.last_update = current_time

    def update_ultrasonic(self, distance: float):
        """Update ultrasonic reading"""
        with self.lock:
            self.world.ultrasonic_distance = distance

    def update_camera_position(self, pan: int, tilt: int):
        """Update current camera position"""
        with self.lock:
            self.world.camera_pan = pan
            self.world.camera_tilt = tilt

    def add_scan_result(self, angle: int, distance: float, obstacles: List[str]):
        """Add result from a camera scan position"""
        with self.lock:
            self.world.clearance_map[angle] = ScanResult(
                angle=angle,
                clearance=distance,
                obstacles=obstacles,
                timestamp=time.time()
            )

    def get_target(self, class_name: str) -> Optional[TrackedObject]:
        """Get tracked object by class, None if not found or too old"""
        with self.lock:
            obj = self.world.tracked_objects.get(class_name)
            if obj and obj.age() < 1.0:  # Less than 1 second old
                return obj
            return None

    def get_best_clearance_direction(self) -> Tuple[int, float]:
        """Get the direction with best clearance"""
        with self.lock:
            if not self.world.clearance_map:
                return (90, 100)  # Default center, assume clear

            best_angle = 90
            best_clearance = 0

            for angle, result in self.world.clearance_map.items():
                # Only consider recent scans
                if time.time() - result.timestamp < 5.0:
                    if result.clearance > best_clearance:
                        best_clearance = result.clearance
                        best_angle = angle

            return (best_angle, best_clearance)

    def get_obstacles_in_view(self) -> List[TrackedObject]:
        """Get all obstacles currently visible"""
        with self.lock:
            obstacles = []
            obstacle_classes = {"chair", "couch", "bed", "dining table", "toilet",
                               "tv", "laptop", "refrigerator", "potted plant"}

            for class_name, obj in self.world.tracked_objects.items():
                if obj.age() < 0.5 and class_name in obstacle_classes:
                    obstacles.append(obj)

            return obstacles

    def _estimate_distance(self, class_name: str, size_ratio: float) -> float:
        """Estimate distance from object size"""
        if class_name in self.SIZE_DISTANCE_MAP and size_ratio > 0:
            ref_size, ref_dist = self.SIZE_DISTANCE_MAP[class_name]
            # Distance is inversely proportional to apparent size
            # d1/d2 = sqrt(s2/s1) for area
            return ref_dist * math.sqrt(ref_size / size_ratio)
        return 100  # Default


# ============== Motion Controller ==============

class MotionController:
    """
    Handles smooth, coordinated robot movement.
    Implements acceleration curves and coordinated camera/body movement.
    """

    def __init__(self, robot_connection, calibration: CameraCalibration = None):
        self.robot = robot_connection
        self.lock = threading.Lock()

        # Load calibration
        self.calibration = calibration or CameraCalibration.load_from_config()

        # Current state - start at calibrated center
        self.current_speed = 0
        self.target_speed = 0
        self.current_pan = self.calibration.pan_center
        self.current_tilt = self.calibration.tilt_center

        # Acceleration parameters
        self.acceleration = 800  # units per second
        self.deceleration = 1200  # faster stopping

        # Last update time
        self.last_update = time.time()

    def get_body_relative_pan(self) -> float:
        """Get how far camera is turned from body forward direction"""
        return self.calibration.get_body_relative_pan(self.current_pan)

    def get_forward_pan(self) -> int:
        """Get pan angle that points straight ahead"""
        return self.calibration.pan_center

    def update(self):
        """Called regularly to smooth motion"""
        with self.lock:
            now = time.time()
            dt = now - self.last_update
            self.last_update = now

            # Smooth speed changes
            if self.current_speed < self.target_speed:
                self.current_speed = min(
                    self.target_speed,
                    self.current_speed + self.acceleration * dt
                )
            elif self.current_speed > self.target_speed:
                self.current_speed = max(
                    self.target_speed,
                    self.current_speed - self.deceleration * dt
                )

    def set_target_speed(self, speed: float):
        """Set target speed (will accelerate/decelerate to reach it)"""
        with self.lock:
            self.target_speed = speed

    def move_toward_target(self, error_x: float, error_distance: float, base_speed: int):
        """
        Move toward a target based on position and distance errors.
        error_x: -1 to 1 (negative = target on left)
        error_distance: positive = too far, negative = too close
        """
        if not self.robot or not self.robot.connected:
            return

        # Calculate forward speed based on distance error
        forward_speed = 0
        if abs(error_distance) > 0.03:  # 3% deadzone
            forward_speed = int(error_distance * base_speed * 2.5)
            forward_speed = max(-base_speed, min(base_speed, forward_speed))

        # Calculate turn speed based on position error
        turn_speed = 0
        if abs(error_x) > 0.1:  # 10% deadzone
            turn_speed = int(error_x * base_speed * 1.2)

        # Combine into differential drive
        left_speed = forward_speed + turn_speed
        right_speed = forward_speed - turn_speed

        # Clamp
        left_speed = max(-base_speed, min(base_speed, left_speed))
        right_speed = max(-base_speed, min(base_speed, right_speed))

        # Apply
        if abs(left_speed) > 100 or abs(right_speed) > 100:
            self.robot.set_motors(left_speed, left_speed, right_speed, right_speed)
        else:
            self.robot.stop()

    def rotate(self, speed: int, direction: int = 1):
        """Rotate in place. direction: 1 = right, -1 = left"""
        if not self.robot or not self.robot.connected:
            return
        if direction > 0:
            self.robot.turn_right(abs(speed))
        else:
            self.robot.turn_left(abs(speed))

    def move_forward(self, speed: int):
        """Move forward"""
        if self.robot and self.robot.connected:
            self.robot.move_forward(speed)

    def move_backward(self, speed: int):
        """Move backward"""
        if self.robot and self.robot.connected:
            self.robot.move_backward(speed)

    def strafe_left(self, speed: int):
        """Strafe left using mecanum wheels"""
        if self.robot and self.robot.connected:
            # Mecanum strafe: front-left back, front-right forward, back-left forward, back-right back
            self.robot.set_motors(-speed, speed, speed, -speed)

    def strafe_right(self, speed: int):
        """Strafe right using mecanum wheels"""
        if self.robot and self.robot.connected:
            # Mecanum strafe: front-left forward, front-right back, back-left back, back-right forward
            self.robot.set_motors(speed, -speed, -speed, speed)

    def move_diagonal(self, forward_speed: int, strafe_speed: int):
        """
        Move diagonally using mecanum wheels.
        Combines forward/backward with left/right strafe.
        """
        if not self.robot or not self.robot.connected:
            return

        # Mecanum wheel math:
        # FL = forward + strafe
        # FR = forward - strafe
        # BL = forward - strafe
        # BR = forward + strafe
        fl = forward_speed + strafe_speed
        fr = forward_speed - strafe_speed
        bl = forward_speed - strafe_speed
        br = forward_speed + strafe_speed

        self.robot.set_motors(fl, bl, fr, br)

    def stop(self):
        """Stop all movement"""
        self.target_speed = 0
        self.current_speed = 0
        if self.robot and self.robot.connected:
            self.robot.stop()

    def set_camera(self, pan: int, tilt: int = None, smooth: bool = True):
        """Set camera position"""
        if not self.robot or not self.robot.connected:
            return

        pan = max(0, min(180, pan))

        if smooth:
            # Move gradually
            step = 5 if abs(pan - self.current_pan) > 5 else 1
            if pan > self.current_pan:
                self.current_pan = min(pan, self.current_pan + step)
            else:
                self.current_pan = max(pan, self.current_pan - step)
        else:
            self.current_pan = pan

        self.robot.set_camera_pan(self.current_pan)

        if tilt is not None:
            tilt = max(0, min(180, tilt))
            self.current_tilt = tilt
            self.robot.set_camera_tilt(self.current_tilt)

    def center_camera(self):
        """Center the camera at calibrated forward position"""
        self.set_camera(
            self.calibration.pan_center,
            self.calibration.tilt_center,
            smooth=False
        )


# ============== Follow Mode Controller ==============

class FollowModeController:
    """
    Improved follow mode that:
    - Explores while searching (moves forward with camera sweep)
    - Avoids obstacles while approaching target
    - Smoothly tracks and follows
    """

    # Safety thresholds
    STOP_DISTANCE = 25   # cm - emergency stop
    SLOW_DISTANCE = 50   # cm - slow down

    def __init__(self, perception: PerceptionLayer, motion: MotionController, broadcast_fn):
        self.perception = perception
        self.motion = motion
        self.broadcast = broadcast_fn

        self.state = FollowState.IDLE
        self.target_class = "person"
        self.base_speed = 1200

        # Search/explore state
        self.sweep_pan = 90
        self.sweep_direction = 1
        self.turn_start_time = 0
        self.turn_direction = 1

        # Tracking state
        self.target_lost_time = 0
        self.frames_on_target = 0

        # Obstacle avoidance state
        self.avoid_direction = 1
        self.avoid_start_time = 0

        # Target distance goal (size ratio - 15% of frame = good distance)
        self.target_size_ratio = 0.15

        self.lock = threading.Lock()

    def set_target(self, class_name: str):
        """Set the target class to follow"""
        with self.lock:
            self.target_class = class_name

    def set_speed(self, speed: int):
        """Set base speed"""
        with self.lock:
            self.base_speed = speed

    def start(self):
        """Start follow mode"""
        with self.lock:
            self.state = FollowState.SEARCH_EXPLORING
            self.sweep_pan = 90
            self.sweep_direction = 1
            self.motion.center_camera()
            self._broadcast_state("Exploring to find target...")
            print(f"Follow mode: starting SEARCH_EXPLORING for {self.target_class}")

    def stop(self):
        """Stop follow mode"""
        with self.lock:
            self.state = FollowState.IDLE
            self.motion.stop()
            self.motion.center_camera()

    def update(self) -> bool:
        """Main update loop - called at ~10Hz"""
        with self.lock:
            if self.state == FollowState.IDLE:
                return False

            target = self.perception.get_target(self.target_class)
            distance = self.perception.world.ultrasonic_distance

            if self.state == FollowState.SEARCH_EXPLORING:
                self._do_search_exploring(target, distance)

            elif self.state == FollowState.SEARCH_TURNING:
                self._do_search_turning(target)

            elif self.state == FollowState.TARGET_ACQUIRED:
                self._do_target_acquired(target, distance)

            elif self.state == FollowState.APPROACHING:
                self._do_approaching(target, distance)

            elif self.state == FollowState.MAINTAINING:
                self._do_maintaining(target, distance)

            elif self.state == FollowState.TARGET_LOST:
                self._do_target_lost(target)

            elif self.state == FollowState.OBSTACLE_AVOIDING:
                self._do_obstacle_avoiding(target, distance)

            return True

    def _do_search_exploring(self, target: Optional[TrackedObject], distance: float):
        """Move forward slowly while sweeping camera to find target"""
        # Check if we found the target - INSTANT LOCK-ON, no waiting
        if target and target.confidence > 0.4:
            self.motion.stop()
            # Immediately go to APPROACHING - skip the "acquired" phase
            self.state = FollowState.APPROACHING
            self._broadcast_state(f"Target locked! Following...")
            return

        # Check for obstacles
        if distance < self.STOP_DISTANCE:
            # Too close - turn to explore new direction
            self.motion.stop()
            self.state = FollowState.SEARCH_TURNING
            self.turn_start_time = time.time()
            self.turn_direction = 1 if self.sweep_pan < 90 else -1  # Turn toward clearer side
            self._broadcast_state("Obstacle ahead, turning...")
            return

        # Sweep camera while moving forward slowly
        self.sweep_pan += self.sweep_direction * 5
        if self.sweep_pan >= 140:
            self.sweep_pan = 140
            self.sweep_direction = -1
        elif self.sweep_pan <= 40:
            self.sweep_pan = 40
            self.sweep_direction = 1

        self.motion.set_camera(self.sweep_pan, smooth=False)

        # Move forward at reduced speed while searching
        search_speed = int(self.base_speed * 0.4)  # 40% speed while searching
        if distance < self.SLOW_DISTANCE:
            search_speed = int(search_speed * 0.5)  # Even slower when obstacle nearby

        self.motion.move_forward(search_speed)

        self._broadcast_state(f"Exploring... Distance: {distance:.0f}cm",
                             searching=True, distance=distance)

    def _do_search_turning(self, target: Optional[TrackedObject]):
        """Turn to explore new direction when path is blocked"""
        # Check if we found target while turning
        if target and target.confidence > 0.5:
            self.motion.stop()
            self.state = FollowState.TARGET_ACQUIRED
            self.frames_on_target = 0
            self._broadcast_state(f"Target found! ({target.confidence:.0%})")
            return

        # Turn for about 1 second (~60 degrees)
        if time.time() - self.turn_start_time < 1.0:
            self.motion.rotate(int(self.base_speed * 0.5), self.turn_direction)
        else:
            self.motion.stop()
            self.state = FollowState.SEARCH_EXPLORING
            self.sweep_pan = 90
            self.motion.center_camera()
            self._broadcast_state("Continuing exploration...")

    def _do_target_acquired(self, target: Optional[TrackedObject], distance: float):
        """Target found - center camera on it"""
        if not target:
            self.target_lost_time = time.time()
            self.state = FollowState.TARGET_LOST
            self._broadcast_state("Target lost...")
            return

        self.frames_on_target += 1

        # Center camera on target
        error_x = target.center[0]
        cam_adjust = -int(error_x * 25)
        new_pan = max(30, min(150, self.motion.current_pan + cam_adjust))
        self.motion.set_camera(new_pan, smooth=True)

        # Once centered for a few frames, start approaching
        if self.frames_on_target > 3 and abs(error_x) < 0.25:
            self.state = FollowState.APPROACHING
            self._broadcast_state("Approaching target...")

        self._broadcast_state(f"Locking on target...",
                             tracking=True, error_x=error_x, confidence=target.confidence)

    def _do_approaching(self, target: Optional[TrackedObject], distance: float):
        """
        Move toward target while avoiding obstacles.

        SNAKE/DOG BEHAVIOR:
        - Camera (head) locks onto target and tracks it smoothly
        - Body continuously aligns with camera direction
        - Forward movement based on distance, turning based on camera offset
        - Like a snake following prey: head leads, body follows
        """
        if not target:
            self.motion.stop()
            self.target_lost_time = time.time()
            self.state = FollowState.TARGET_LOST
            return

        # Check for obstacle between us and target
        if distance < self.STOP_DISTANCE:
            # Obstacle in the way - need to go around
            self.motion.stop()
            self.state = FollowState.OBSTACLE_AVOIDING
            self.avoid_start_time = time.time()
            # Go around on the side where target is
            self.avoid_direction = 1 if target.center[0] > 0 else -1
            self._broadcast_state("Obstacle in path, navigating around...")
            return

        error_x = target.center[0]  # Target position in camera frame (-1 to 1)
        error_size = self.target_size_ratio - target.size_ratio  # Positive = too far

        # === CAMERA TRACKING (HEAD) ===
        # Camera aggressively tracks target - fast response
        cam_adjust = -int(error_x * 30)  # Increased from 20 for faster tracking
        new_pan = max(30, min(150, self.motion.current_pan + cam_adjust))
        self.motion.set_camera(new_pan, smooth=False)  # Immediate camera movement

        # === BODY ALIGNMENT (SNAKE/DOG BEHAVIOR) ===
        # Body continuously tries to align with camera direction
        # Use calibration to know where "forward" is
        camera_offset = self.motion.get_body_relative_pan()  # Positive = camera looking right

        # Calculate body turn rate - proportional to camera offset
        # The more the camera is turned, the more the body needs to catch up
        body_turn_rate = -camera_offset / 60.0  # Normalized turn rate (-1 to 1)
        body_turn_rate = max(-1.0, min(1.0, body_turn_rate))  # Clamp

        # === FORWARD MOVEMENT ===
        # Speed based on distance to target
        speed = self.base_speed
        if distance < self.SLOW_DISTANCE:
            speed = int(self.base_speed * 0.6)

        # Calculate forward speed based on distance error
        forward_speed = 0
        if error_size > 0.02:  # Target too far
            forward_speed = int(min(error_size * 3, 1.0) * speed)  # Approach
        elif error_size < -0.02:  # Target too close
            forward_speed = int(max(error_size * 2, -0.5) * speed)  # Back up slowly

        # === COMBINED MOVEMENT ===
        # Mix forward movement with body alignment turn
        # Turn speed proportional to camera offset
        turn_speed = int(body_turn_rate * speed * 0.7)

        # Apply differential drive for 4WD
        left_speed = forward_speed + turn_speed
        right_speed = forward_speed - turn_speed

        # Clamp speeds to valid range
        max_speed = min(speed, 3000)  # Robot max is 4095
        left_speed = max(-max_speed, min(max_speed, left_speed))
        right_speed = max(-max_speed, min(max_speed, right_speed))

        # Debug output
        print(f"[FOLLOW] fwd={forward_speed}, turn={turn_speed}, L={left_speed}, R={right_speed}, cam_offset={camera_offset:.1f}")

        # Apply movement - use robot directly with proper 4-motor format
        if abs(left_speed) > 200 or abs(right_speed) > 200:
            # Robot expects: left_upper, left_lower, right_upper, right_lower
            self.motion.robot.set_motors(
                int(left_speed), int(left_speed),
                int(right_speed), int(right_speed)
            )
        else:
            # Speeds too low to move effectively
            self.motion.stop()

        # Check if at good distance AND body aligned (camera near center)
        if abs(error_size) < 0.03 and abs(camera_offset) < 15:
            self.state = FollowState.MAINTAINING
            self._broadcast_state("At target distance")

        self._broadcast_state(f"Following: {target.estimated_distance:.0f}cm",
                             tracking=True, error_x=error_x, error_size=error_size,
                             confidence=target.confidence, distance=distance,
                             camera_offset=camera_offset)

    def _do_maintaining(self, target: Optional[TrackedObject], distance: float):
        """
        Maintain distance from target with small adjustments.
        Uses same snake/dog behavior but with gentler movements.
        """
        if not target:
            self.motion.stop()
            self.target_lost_time = time.time()
            self.state = FollowState.TARGET_LOST
            return

        # Obstacle check
        if distance < self.STOP_DISTANCE:
            self.motion.stop()
            self._broadcast_state("Obstacle detected, holding position")
            return

        error_x = target.center[0]
        error_size = self.target_size_ratio - target.size_ratio

        # If target moves significantly, go back to approaching
        if abs(error_size) > 0.05 or abs(error_x) > 0.4:
            self.state = FollowState.APPROACHING
            return

        # === CAMERA TRACKING (gentler in maintain mode) ===
        cam_adjust = -int(error_x * 20)
        new_pan = max(30, min(150, self.motion.current_pan + cam_adjust))
        self.motion.set_camera(new_pan, smooth=True)

        # === BODY ALIGNMENT (snake/dog - gentle corrections) ===
        camera_offset = self.motion.get_body_relative_pan()  # Use calibration

        # Gentle body turn to keep aligned
        if abs(camera_offset) > 10:  # Only correct if camera offset is significant
            body_turn_rate = -camera_offset / 90.0  # Gentler than approaching
            body_turn_rate = max(-0.5, min(0.5, body_turn_rate))

            # Small forward/back adjustment for distance
            speed = self.base_speed // 2  # 50% speed for gentle corrections
            forward_speed = int(error_size * speed * 2) if abs(error_size) > 0.02 else 0
            turn_speed = int(body_turn_rate * speed)

            left_speed = forward_speed + turn_speed
            right_speed = forward_speed - turn_speed

            print(f"[MAINTAIN] fwd={forward_speed}, turn={turn_speed}, L={left_speed}, R={right_speed}")

            if abs(left_speed) > 200 or abs(right_speed) > 200:
                self.motion.robot.set_motors(
                    int(left_speed), int(left_speed),
                    int(right_speed), int(right_speed)
                )
            else:
                self.motion.stop()
        else:
            # Camera centered - just small distance adjustments if needed
            if abs(error_size) > 0.02:
                speed = int(error_size * self.base_speed * 0.5)
                speed = max(-self.base_speed // 2, min(self.base_speed // 2, speed))
                print(f"[MAINTAIN] distance adjust: speed={speed}")
                if speed > 200:
                    self.motion.move_forward(speed)
                elif speed < -200:
                    self.motion.move_backward(abs(speed))
                else:
                    self.motion.stop()
            else:
                self.motion.stop()

        self._broadcast_state(f"Maintaining: {target.estimated_distance:.0f}cm",
                             tracking=True, error_x=error_x, error_size=error_size,
                             confidence=target.confidence, camera_offset=camera_offset)

    def _do_target_lost(self, target: Optional[TrackedObject]):
        """Target lost - wait briefly then return to exploring"""
        if target and target.confidence > 0.5:
            self.state = FollowState.TARGET_ACQUIRED
            self.frames_on_target = 0
            self._broadcast_state("Target reacquired!")
            return

        lost_duration = time.time() - self.target_lost_time

        if lost_duration < 1.5:
            # Brief loss - hold position and wait
            self._broadcast_state(f"Target lost ({lost_duration:.1f}s)...", searching=True)
        else:
            # Go back to exploring
            self.state = FollowState.SEARCH_EXPLORING
            self.sweep_pan = 90
            self.motion.center_camera()
            self._broadcast_state("Resuming exploration...")

    def _do_obstacle_avoiding(self, target: Optional[TrackedObject], distance: float):
        """Navigate around obstacle while trying to keep target in view"""
        # If path is now clear, resume approaching
        if distance > self.SLOW_DISTANCE:
            self.motion.stop()
            if target:
                self.state = FollowState.APPROACHING
                self._broadcast_state("Path clear, resuming approach")
            else:
                self.state = FollowState.SEARCH_EXPLORING
                self._broadcast_state("Path clear, searching for target")
            return

        # Turn and move to go around obstacle
        elapsed = time.time() - self.avoid_start_time

        if elapsed < 0.8:
            # Turn away from obstacle
            self.motion.rotate(int(self.base_speed * 0.5), self.avoid_direction)
        elif elapsed < 1.5:
            # Move forward briefly
            self.motion.move_forward(int(self.base_speed * 0.4))
        else:
            # Reset and check again
            self.motion.stop()
            self.avoid_start_time = time.time()

        # Keep camera looking for target
        if target:
            error_x = target.center[0]
            cam_adjust = -int(error_x * 20)
            new_pan = max(30, min(150, self.motion.current_pan + cam_adjust))
            self.motion.set_camera(new_pan, smooth=True)

        self._broadcast_state(f"Avoiding obstacle... Distance: {distance:.0f}cm",
                             searching=True, distance=distance)

    def _broadcast_state(self, message: str, **kwargs):
        """Broadcast current state to UI"""
        state_info = {
            "type": "follow_status",
            "target": self.target_class,
            "state": self.state.name,
            "message": message,
            "tracking": kwargs.get("tracking", False),
            "searching": kwargs.get("searching", self.state in [
                FollowState.SEARCH_EXPLORING, FollowState.SEARCH_TURNING, FollowState.TARGET_LOST
            ]),
            "error_x": kwargs.get("error_x", 0),
            "error_size": kwargs.get("error_size", 0),
            "confidence": kwargs.get("confidence", 0),
            "distance": kwargs.get("distance", 0),
            "camera_pan": self.motion.current_pan
        }
        self.broadcast(state_info)


# ============== Patrol Mode Controller ==============

class PatrolModeController:
    """
    Soldier-style patrol mode:
    - Smooth continuous forward movement
    - Camera slowly sweeping side to side (like head turning)
    - Only stops/turns when obstacle detected
    - Alerts on interesting objects (person, cat, dog)
    """

    # Safety thresholds
    STOP_DISTANCE = 25   # cm - emergency stop
    SLOW_DISTANCE = 50   # cm - slow down
    TURN_DISTANCE = 40   # cm - start turning

    def __init__(self, perception: PerceptionLayer, motion: MotionController, broadcast_fn):
        self.perception = perception
        self.motion = motion
        self.broadcast = broadcast_fn

        self.state = PatrolState.IDLE
        self.base_speed = 1000

        # Camera sweep (slow, soldier-like head movement)
        self.sweep_pan = 90
        self.sweep_direction = 1
        self.sweep_speed = 2  # Slow sweep

        # Obstacle avoidance state
        self.turn_start_time = 0
        self.turn_direction = 1
        self.backup_start_time = 0

        # Alert tracking
        self.last_alert_time = 0
        self.alert_pause_until = 0

        self.lock = threading.Lock()

    def set_speed(self, speed: int):
        """Set base speed"""
        with self.lock:
            self.base_speed = speed

    def start(self):
        """Start patrol mode - immediately start patrolling"""
        with self.lock:
            self.state = PatrolState.PATROLLING
            self.sweep_pan = 90
            self.sweep_direction = 1
            self.motion.center_camera()
            self._broadcast_state("Starting patrol...")
            print(f"Patrol mode: starting PATROLLING at speed {self.base_speed}")

    def stop(self):
        """Stop patrol mode"""
        with self.lock:
            self.state = PatrolState.IDLE
            self.motion.stop()
            self.motion.center_camera()

    def update(self) -> bool:
        """Main update loop - called at ~10Hz"""
        with self.lock:
            if self.state == PatrolState.IDLE:
                return False

            distance = self.perception.world.ultrasonic_distance

            # Always check for interesting objects
            self._check_for_alerts()

            if self.state == PatrolState.PATROLLING:
                self._do_patrolling(distance)

            elif self.state == PatrolState.OBSTACLE_TURN:
                self._do_obstacle_turn(distance)

            elif self.state == PatrolState.BACKING_UP:
                self._do_backing_up(distance)

            elif self.state == PatrolState.ALERT:
                self._do_alert()

            return True

    def _do_patrolling(self, distance: float):
        """Main patrol state - move forward with camera sweep"""

        # Emergency stop - too close
        if distance < self.STOP_DISTANCE:
            self.motion.stop()
            self.state = PatrolState.BACKING_UP
            self.backup_start_time = time.time()
            self._broadcast_state(f"Obstacle! Backing up... ({distance:.0f}cm)")
            return

        # Need to turn - obstacle approaching
        if distance < self.TURN_DISTANCE:
            self.motion.stop()
            self.state = PatrolState.OBSTACLE_TURN
            self.turn_start_time = time.time()
            # Turn toward the side with more space (based on camera position)
            self.turn_direction = -1 if self.sweep_pan > 90 else 1
            self._broadcast_state(f"Obstacle ahead, turning...")
            return

        # Calculate speed based on distance
        if distance < self.SLOW_DISTANCE:
            speed = int(self.base_speed * 0.5)
        else:
            speed = int(self.base_speed * 0.7)  # Patrol at 70% speed for safety

        # Move forward
        self.motion.move_forward(speed)

        # Slow camera sweep (soldier-like head movement)
        self.sweep_pan += self.sweep_direction * self.sweep_speed
        if self.sweep_pan >= 120:
            self.sweep_pan = 120
            self.sweep_direction = -1
        elif self.sweep_pan <= 60:
            self.sweep_pan = 60
            self.sweep_direction = 1

        self.motion.set_camera(self.sweep_pan, smooth=True)

        self._broadcast_state(f"Patrolling... {distance:.0f}cm clear",
                             distance=distance, speed=speed)

    def _do_obstacle_turn(self, distance: float):
        """Turn away from obstacle"""
        # If path is now clear, resume patrolling
        if distance > self.SLOW_DISTANCE:
            self.motion.stop()
            self.state = PatrolState.PATROLLING
            self.sweep_pan = 90
            self._broadcast_state("Path clear, resuming patrol")
            return

        # Turn for about 0.6 seconds (~45 degrees)
        elapsed = time.time() - self.turn_start_time
        if elapsed < 0.6:
            self.motion.rotate(int(self.base_speed * 0.6), self.turn_direction)
            self._broadcast_state(f"Turning {'right' if self.turn_direction > 0 else 'left'}...",
                                 distance=distance)
        else:
            self.motion.stop()
            # Check if we need to turn more or back up
            if distance < self.STOP_DISTANCE:
                self.state = PatrolState.BACKING_UP
                self.backup_start_time = time.time()
            else:
                self.state = PatrolState.PATROLLING
                self.sweep_pan = 90
                self._broadcast_state("Turn complete, patrolling")

    def _do_backing_up(self, distance: float):
        """Back up from obstacle"""
        elapsed = time.time() - self.backup_start_time

        if elapsed < 0.5:
            self.motion.move_backward(int(self.base_speed * 0.5))
            self._broadcast_state(f"Backing up... {distance:.0f}cm", distance=distance)
        else:
            self.motion.stop()
            # After backing up, turn to find new path
            self.state = PatrolState.OBSTACLE_TURN
            self.turn_start_time = time.time()
            self.turn_direction = 1  # Default turn right
            self._broadcast_state("Looking for new path...")

    def _do_alert(self):
        """Pause during alert (detected something interesting)"""
        if time.time() > self.alert_pause_until:
            self.state = PatrolState.PATROLLING
            self._broadcast_state("Resuming patrol...")

    def _check_for_alerts(self):
        """Check for interesting objects during patrol"""
        # Don't spam alerts
        if time.time() - self.last_alert_time < 5.0:
            return

        interesting = ["person", "cat", "dog"]

        for class_name in interesting:
            target = self.perception.get_target(class_name)
            if target and target.confidence > 0.6:
                self.last_alert_time = time.time()

                # Brief pause on alert
                if self.state == PatrolState.PATROLLING:
                    self.motion.stop()
                    self.state = PatrolState.ALERT
                    self.alert_pause_until = time.time() + 1.5  # Pause 1.5 seconds

                    # Point camera at the object
                    error_x = target.center[0]
                    new_pan = max(30, min(150, 90 - int(error_x * 40)))
                    self.motion.set_camera(new_pan, smooth=False)

                self.broadcast({
                    "type": "patrol_alert",
                    "object": class_name,
                    "confidence": round(target.confidence, 2),
                    "distance": round(target.estimated_distance, 0),
                    "position": "left" if target.center[0] < -0.2 else "right" if target.center[0] > 0.2 else "ahead"
                })
                self._broadcast_state(f"Alert! {class_name} detected!")
                break

    def _broadcast_state(self, message: str, **kwargs):
        """Broadcast current state to UI"""
        state_info = {
            "type": "patrol_status",
            "state": self.state.name,
            "message": message,
            "distance": kwargs.get("distance", self.perception.world.ultrasonic_distance),
            "speed": kwargs.get("speed", 0),
            "camera_pan": self.motion.current_pan
        }
        self.broadcast(state_info)


# ============== Main Autonomous System ==============

class AutonomousSystem:
    """
    Main autonomous system that coordinates all components.
    """

    def __init__(self, robot_connection, yolo_processor, broadcast_fn):
        self.robot = robot_connection
        self.yolo = yolo_processor
        self.broadcast = broadcast_fn

        # Initialize layers
        self.perception = PerceptionLayer()
        self.motion = MotionController(robot_connection)

        # Initialize mode controllers
        self.follow_controller = FollowModeController(
            self.perception, self.motion, broadcast_fn
        )
        self.patrol_controller = PatrolModeController(
            self.perception, self.motion, broadcast_fn
        )

        # State
        self.active_mode = None  # "follow" or "patrol"
        self.running = False
        self.lock = threading.Lock()

    def _stop_all_unlocked(self):
        """Internal: Stop all modes without acquiring lock (caller must hold lock)"""
        self.follow_controller.stop()
        self.patrol_controller.stop()
        self.motion.stop()
        self.active_mode = None
        self.running = False

    def start_follow(self, target_class: str = "person", speed: int = 1200):
        """Start follow mode"""
        with self.lock:
            self._stop_all_unlocked()
            self.follow_controller.set_target(target_class)
            self.follow_controller.set_speed(speed)
            self.follow_controller.start()
            self.active_mode = "follow"
            self.running = True
            print(f"Follow mode started: target={target_class}, speed={speed}")

    def start_patrol(self, speed: int = 1000):
        """Start patrol mode"""
        with self.lock:
            self._stop_all_unlocked()
            self.patrol_controller.set_speed(speed)
            self.patrol_controller.start()
            self.active_mode = "patrol"
            self.running = True
            print(f"Patrol mode started: speed={speed}")

    def stop_all(self):
        """Stop all autonomous modes"""
        with self.lock:
            self._stop_all_unlocked()
            print("All autonomous modes stopped")

    def set_speed(self, speed: int):
        """Set speed for active mode"""
        self.follow_controller.set_speed(speed)
        self.patrol_controller.set_speed(speed)

    def set_follow_target(self, target_class: str):
        """Set follow target class"""
        self.follow_controller.set_target(target_class)

    def update(self):
        """
        Main update loop - call this at ~10Hz.
        Updates perception and active controller.
        """
        if not self.running:
            return

        # Update perception from YOLO
        if self.yolo:
            detections = self.yolo.get_detections_raw()
            if detections:
                self.perception.update_from_yolo(detections)

        # Update perception from ultrasonic
        if self.robot and self.robot.connected:
            self.robot.request_distance()
            # Give sensor time to respond
            time.sleep(0.05)
            self.perception.update_ultrasonic(self.robot.get_distance())
            self.perception.update_camera_position(
                self.motion.current_pan,
                self.motion.current_tilt
            )

        # Update motion controller
        self.motion.update()

        # Update active mode
        with self.lock:
            if self.active_mode == "follow":
                self.follow_controller.update()
            elif self.active_mode == "patrol":
                self.patrol_controller.update()

    def get_state(self) -> dict:
        """Get current system state"""
        with self.lock:
            state = {
                "active_mode": self.active_mode,
                "running": self.running,
            }

            if self.active_mode == "follow":
                state["follow_state"] = self.follow_controller.state.name
                state["target"] = self.follow_controller.target_class
            elif self.active_mode == "patrol":
                state["patrol_state"] = self.patrol_controller.state.name

            return state
