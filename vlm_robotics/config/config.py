"""Configuration parameters for VLM Robotics simulation and control."""

import os
import warnings
from pathlib import Path

# Suppress harmless GLFW shutdown warnings on Windows
try:
    import glfw
    warnings.filterwarnings("ignore", category=glfw.GLFWError)
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

# Simulation Scene
SCENE_XML_PATH = os.path.join(BASE_DIR, "simulation", "worlds", "tabletop_scene.xml")

# Ollama / Local VLM Configuration
_raw_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not _raw_host.startswith("http://") and not _raw_host.startswith("https://"):
    _raw_host = f"http://{_raw_host}"
OLLAMA_HOST = _raw_host.replace("0.0.0.0", "127.0.0.1")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen2.5vl:3b")

# Camera Configuration
DEFAULT_CAMERA = "workspace_cam"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Robot Workspace Boundaries (Safety enforcement)
WORKSPACE_LIMITS = {
    "x_min": 0.10,
    "x_max": 0.32,
    "y_min": -0.28,
    "y_max": 0.28,
    "z_min": 0.005,
    "z_max": 0.35,
}

# OpenMANIPULATOR-X Joint Limits (radians)
JOINT_LIMITS = {
    "joint1": (-3.14159, 3.14159),
    "joint2": (-1.5, 1.5),
    "joint3": (-1.5, 1.4),
    "joint4": (-1.7, 1.97),
}

# Default Robot Poses
HOME_POSE = [0.0, -0.4, 0.6, 0.2]       # [joint1, joint2, joint3, joint4]
READY_POSE = [0.0, -0.2, 0.4, 0.1]

# Gripper Configuration
GRIPPER_OPEN = 0.012     # Slide joint pos (open, 4.8cm clearance)
GRIPPER_CLOSE = -0.009   # Slide joint pos (pinch / closed)

# Ground Truth Object Metadata (used to calibrate and anchor perception)
KNOWN_OBJECTS = {
    "red_cube": {
        "body_name": "red_cube",
        "type": "cube",
        "color": "red",
        "dimensions": [0.03, 0.03, 0.03],
        "default_pos": [0.16, -0.10, 0.015],
    },
    "blue_cube": {
        "body_name": "blue_cube",
        "type": "cube",
        "color": "blue",
        "dimensions": [0.03, 0.03, 0.03],
        "default_pos": [0.17, 0.06, 0.015],
    },
    "green_cube": {
        "body_name": "green_cube",
        "type": "cube",
        "color": "green",
        "dimensions": [0.03, 0.03, 0.03],
        "default_pos": [0.18, -0.02, 0.015],
    },
    "orange_cube": {
        "body_name": "orange_cube",
        "type": "cube",
        "color": "orange",
        "dimensions": [0.03, 0.03, 0.03],
        "default_pos": [0.26, 0.03, 0.015],
    },
    "bottle": {
        "body_name": "bottle",
        "type": "bottle",
        "color": "yellow",
        "dimensions": [0.026, 0.026, 0.07],
        "default_pos": [0.18, -0.21, 0.035],
    },
    "glass1": {
        "body_name": "glass1",
        "type": "glass",
        "color": "blue",
        "dimensions": [0.028, 0.028, 0.06],
        "default_pos": [0.21, 0.18, 0.035],
    },
    "glass2": {
        "body_name": "glass2",
        "type": "glass",
        "color": "purple",
        "dimensions": [0.028, 0.028, 0.06],
        "default_pos": [0.24, -0.13, 0.035],
    },
    "glass3": {
        "body_name": "glass3",
        "type": "glass",
        "color": "amber",
        "dimensions": [0.028, 0.028, 0.06],
        "default_pos": [0.26, -0.06, 0.035],
    },
    "glass4": {
        "body_name": "glass4",
        "type": "glass",
        "color": "emerald green",
        "dimensions": [0.028, 0.028, 0.06],
        "default_pos": [0.11, -0.16, 0.035],
    },
    "glass5": {
        "body_name": "glass5",
        "type": "glass",
        "color": "rose pink",
        "dimensions": [0.028, 0.028, 0.06],
        "default_pos": [0.25, 0.11, 0.035],
    },
    "glass6": {
        "body_name": "glass6",
        "type": "glass",
        "color": "teal cyan",
        "dimensions": [0.028, 0.028, 0.06],
        "default_pos": [0.13, 0.13, 0.035],
    },
    "box": {
        "body_name": "box",
        "type": "container",
        "color": "cyan",
        "dimensions": [0.09, 0.09, 0.04],
        "default_pos": [0.11, 0.22, 0.025],
    },
}
