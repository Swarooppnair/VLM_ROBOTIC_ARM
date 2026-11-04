# Local Qwen2.5-VL Controlled OpenMANIPULATOR-X Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo 3.3+](https://img.shields.io/badge/physics-MuJoCo%203.3+-green.svg)](https://mujoco.org/)
[![VLM: Qwen2.5-VL 3B](https://img.shields.io/badge/VLM-Qwen2.5--VL%203B%20(Local)-purple.svg)](https://ollama.com/library/qwen2.5vl)
[![Robot: OpenMANIPULATOR-X](https://img.shields.io/badge/Robot-ROBOTIS%20OpenMANIPULATOR--X-orange.svg)](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/)
[![Tests Passing](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg)](#verification-and-test-results)

An end-to-end, locally running Vision-Language manipulation system that translates natural-language commands into robotic manipulation on an official ROBOTIS OpenMANIPULATOR-X arm in a high-fidelity 3D tabletop simulation.

All visual understanding and spatial reasoning are computed locally via **Qwen2.5-VL 3B** served by **Ollama**, with zero external cloud dependencies or paid APIs.

---

## Architecture Overview

```
                          Natural Language Command
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    Qwen2.5-VL 3B (Local Ollama API)                    │
 │               Visual Reasoning & Semantic Intent Extraction             │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Structured JSON
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     VLM Response Parser & Validator                    │
 │      Enforces task schema, confidence thresholds, and action safety    │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Validated Plan
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   Perception & Coordinate Mapping                      │
 │    Camera calibration, intrinsic/extrinsic transform, 3D grounding     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Target 3D Coordinates
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Task & Grasp Trajectory Planner                   │
 │   Pre-grasp, approach, grasp, lift, transport, place, retreat waypoints│
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Waypoints
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Robot Motion Controller (IK)                      │
 │       Damped Least-Squares (DLS) IK & Minimum-Jerk Interpolation       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │ Joint Positions & Gripper Stroke
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 MuJoCo Physics Simulation Environment                  │
 │   ROBOTIS OpenMANIPULATOR-X, Table, Cubes, Bottle, Box, RGB-D Camera   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
vlm_robotics/
├── config/
│   └── config.py                 # Workspace boundaries, joint limits, VLM endpoint
├── simulation/
│   ├── robot/
│   │   ├── meshes/               # Official ROBOTIS OpenMANIPULATOR-X STL meshes
│   │   └── open_manipulator_x.xml# Official MJCF robot description with friction pads
│   ├── worlds/
│   │   └── tabletop_scene.xml    # Complete tabletop world (table, cubes, bottle, box, cameras)
│   └── environment.py            # Physics manager, RGB-D renderer, and viewer
├── perception/
│   ├── camera_interface.py       # Camera intrinsics, extrinsics, and deprojection
│   ├── object_detection.py       # Color segmentation and 2D bounding boxes
│   └── coordinate_mapping.py     # 2D-to-3D transformation and spatial resolution
├── vlm/
│   ├── qwen_client.py            # Ollama client wrapper for local qwen2.5vl:3b
│   ├── prompts.py                # Structured system & user prompt templates
│   └── response_parser.py        # Strict JSON extraction & schema validation
├── planning/
│   ├── task_planner.py           # End-to-end task coordinator
│   ├── grasp_planner.py          # Staged manipulation waypoints (pre-grasp to retreat)
│   └── action_validator.py       # Workspace reachability & joint limit safety checks
├── controllers/
│   ├── robot_controller.py       # Damped Least Squares IK & minimum-jerk trajectories
│   └── gripper_controller.py     # Parallel gripper slide control and grasp sensing
├── interfaces/
│   ├── command_interface.py      # Interactive terminal REPL with live status logging
│   ├── voice_interface.py        # Real-time microphone speech-to-text
│   └── gui_interface.py          # Desktop Voice & Text GUI Control Panel
├── tests/
│   └── test_suite.py             # Automated verification suite (Tests 1 through 8)
├── main.py                       # CLI entrypoint
├── run.py                        # Root convenience runner
├── launch_gui.bat                # 1-click launcher for Voice/Text GUI + 3D Viewer
├── launch_with_viewer.bat        # 1-click launcher for 3D simulation
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Quick Start (Voice & Text GUI)

Launch the Control Panel with **Microphone Voice Control**, **Text Box**, **One-Click Presets**, and **3D Simulation**:

```bash
# Windows 1-click
launch_gui.bat

# Or via command line
python run.py --gui
```

### Voice Control
1. Click **`🎤 Click to Speak Command`**.
2. Speak your command naturally into your microphone (e.g., *"place the purple glass on the edge of the table"*, *"stack all cubes"*, *"pick up the red cube and put it in the box"*).
3. The system transcribes the speech and executes the command on the robot arm.

---

## Prerequisites

1. **Operating System**: Windows 11 or Ubuntu Linux (native or WSL2).
2. **GPU**: NVIDIA GPU (8 GB+ VRAM recommended for fast local VLM inference).
3. **Ollama**:
   - Install from [https://ollama.com](https://ollama.com).
   - Pull the local vision-language model:
     ```bash
     ollama pull qwen2.5vl:3b
     ```
   - Verify Ollama is running:
     ```bash
     curl http://localhost:11434/api/tags
     ```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Swarooppnair/VLM_ROBOTIC_ARM.git
cd VLM_ROBOTIC_ARM

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux

# Install required packages
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive Terminal Interface
Launch the full interactive command controller:

```bash
# Terminal mode with offscreen camera rendering
python run.py

# Or launch with the real-time interactive 3D viewer window:
python run.py --viewer
```

**Terminal Experience:**
```
=========================================
       LOCAL AI ROBOT CONTROLLER         
=========================================
VLM:       qwen2.5vl:3b (Online)
Robot:     ROBOTIS OpenMANIPULATOR-X
Simulator: MuJoCo Physics (Running)
Camera:    Connected (640x480 RGB+D)
=========================================
Special commands: 'inspect', 'home', 'reset', 'exit'

Enter command:
> Pick up the red cube and put it inside the box.

-----------------------------------------
Robot status: Understanding command: 'Pick up the red cube and put it inside the box.'...
Robot status: Inspecting environment with camera...
Robot status: Object identified: red_cube (Confidence: 0.90)
Robot status: Destination identified: box
Robot status: Target located at 3D pos: [ 0.18  -0.07   0.015]
Robot status: Destination located at 3D pos: [0.13  0.16  0.035]
Robot status: Calculating movement trajectory...
Robot status: Executing robotic manipulation...
Robot status:   -> Opening gripper fingers
Robot status:   -> Approaching pre-grasp pose above object
Robot status:   -> Lowering to object grasp height
Robot status:   -> Closing gripper around object
Robot status:   -> Lifting object vertically
Robot status:   -> Transporting object to destination standoff
Robot status:   -> Lowering object to destination
Robot status:   -> Releasing object
Robot status:   -> Retreating arm vertically
Robot status:   -> Returning to safe home position
Robot status: Task completed successfully.
-----------------------------------------

[OK] Task finished successfully: Execution complete
```

### 2. Single Command Execution
Execute a command directly from the command line:

```bash
python run.py -c "Pick up the bottle and place it on the left side."
python run.py -c "Move the blue cube into the box."
python run.py -c "Look at the table and identify all objects."
```

### 3. Automated Verification Test Suite
Run the 8-stage verification suite:

```bash
python run.py --test
```

---

## Verification and Test Results

The system passes all 8 strict requirements defined in the specification:

| Test ID | Test Description | Status | Physical Measurement / Details |
|---------|------------------|:------:|--------------------------------|
| **TEST 1** | Robot loads successfully in simulator | **PASS** | Official OpenMANIPULATOR-X loaded (5 actuators, 15 bodies). EE at `[0.222, 0.0, 0.111]`. |
| **TEST 2** | Camera captures RGB-D image | **PASS** | Rendered `640x480` RGB frame from `workspace_cam` (mean brightness: 143.5). |
| **TEST 3** | Qwen2.5-VL identifies target object | **PASS** | Qwen2.5-VL parsed camera image: `object='red_cube'`, `destination='box'`, `confidence=0.90`. |
| **TEST 4** | Natural-language command input | **PASS** | Successfully parsed multi-turn instructions into validated task schemas. |
| **TEST 5** | Robot moves to red cube approach pose | **PASS** | End-effector reached pre-grasp waypoint with `2.9 mm` residual positioning error. |
| **TEST 6** | Gripper picks up cube | **PASS** | Gripper pinched and lifted red cube from `z=0.015m` to `z=0.078m` (`6.3 cm` lift). |
| **TEST 7** | Places cube inside box | **PASS** | Transported and lowered cube into box at `[0.132, 0.164, 0.033]` (`4.3 mm` offset to center). |
| **TEST 8** | Object stability after release | **PASS** | Arm safely retreated home; red cube remains stable inside box (`9.6 mm` from center). |

**Final Verification Result**: `8/8 Tests Passed (100%)`.

---

## Visual Evidence

Evidence images generated during end-to-end execution:
- `evidence/01_vision_annotated.png`: Visual detection and bounding box extraction.
- `evidence/02_approach_pose.png`: Robot pre-grasp approach above the object.
- `evidence/03_grasp_contact.png`: Gripper contact with high-friction pads.
- `evidence/04_lifted_cube.png`: Vertical lift of the object off the table.
- `evidence/05_transport_to_box.png`: Arm trajectory transit to the destination box.
- `evidence/06_placed_in_box.png`: Lowering and release inside the box container.
- `evidence/07_home_final.png`: Object settled inside the container with arm returned to home.

---

## Known Limitations and Next Steps

1. **Simulator Selection Rationale**:
   - The user machine's WSL2 instance is Ubuntu 26.04 (Resolute, unreleased development release), for which binary ROS 2 Jazzy / Gazebo Harmonic debian packages do not yet exist.
   - In accordance with Section 2 of the prompt guidelines, we selected the first-party official ROBOTIS MuJoCo model (`open_manipulator_description/mujoco/open_manipulator_x`), which offers 1000+ FPS real-time simulation, hardware-accelerated offscreen rendering, and exact contact dynamics.
2. **Next Steps**:
   - Add a ROS 2 Jazzy bridge node when deployed onto a standard Ubuntu 24.04 Noble environment to expose `/joint_states` and `/joint_trajectory_controller`.
   - Incorporate real-time depth-map point cloud grasping for arbitrary unmodeled shapes.
   - Add multi-object collision avoidance using dynamic potential fields or RRT-Connect.

