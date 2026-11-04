"""Capture multi-stage visual evidence of simulation, perception, and manipulation."""

import os
import cv2
import numpy as np
from PIL import Image

from vlm_robotics.simulation.environment import SimulationEnvironment
from vlm_robotics.perception.object_detection import ObjectDetector
from vlm_robotics.controllers.robot_controller import RobotController
from vlm_robotics.controllers.gripper_controller import GripperController
from vlm_robotics.perception.coordinate_mapping import CoordinateMapper

env = SimulationEnvironment()
detector = ObjectDetector()
robot = RobotController(env)
gripper = GripperController(env)
mapper = CoordinateMapper(env)

out_dir = "evidence"
os.makedirs(out_dir, exist_ok=True)

# 1. Capture Initial State & Vision Bounding Boxes
rgb = env.render_rgb()
detections = detector.detect_objects(rgb)
annotated = detector.draw_detections(rgb, detections)
cv2.imwrite(os.path.join(out_dir, "01_vision_annotated.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
Image.fromarray(rgb).save(os.path.join(out_dir, "01_initial_tabletop.png"))
print("Saved 01_initial_tabletop.png and 01_vision_annotated.png")

# 2. Pre-grasp Approach
cube_pos = mapper.get_object_position("red_cube")
approach_pt = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.07])
gripper.open(0.3)
robot.move_to_cartesian_position(approach_pt, duration=1.0)
Image.fromarray(env.render_rgb()).save(os.path.join(out_dir, "02_approach_pose.png"))
print("Saved 02_approach_pose.png")

# 3. Grasping
grasp_pt = np.array([cube_pos[0], cube_pos[1], 0.018])
robot.move_to_cartesian_position(grasp_pt, duration=0.8)
gripper.close(0.6)
Image.fromarray(env.render_rgb()).save(os.path.join(out_dir, "03_grasp_contact.png"))
print("Saved 03_grasp_contact.png")

# 4. Lifted
lift_pt = np.array([cube_pos[0], cube_pos[1], 0.10])
robot.move_to_cartesian_position(lift_pt, duration=0.8, gripper_target=gripper.close_pos)
Image.fromarray(env.render_rgb()).save(os.path.join(out_dir, "04_lifted_cube.png"))
print("Saved 04_lifted_cube.png")

# 5. Transport to Box
box_pos = mapper.get_object_position("box")
standoff = np.array([box_pos[0], box_pos[1], 0.12])
robot.move_to_cartesian_position(standoff, duration=1.2, gripper_target=gripper.close_pos)
Image.fromarray(env.render_rgb()).save(os.path.join(out_dir, "05_transport_to_box.png"))
print("Saved 05_transport_to_box.png")

# 6. Lower and Place
place_pt = np.array([box_pos[0], box_pos[1], 0.035])
robot.move_to_cartesian_position(place_pt, duration=0.8, gripper_target=gripper.close_pos)
gripper.open(0.5)
Image.fromarray(env.render_rgb()).save(os.path.join(out_dir, "06_placed_in_box.png"))
print("Saved 06_placed_in_box.png")

# 7. Safe Home
robot.go_home(1.0)
for _ in range(100):
    env.step()
Image.fromarray(env.render_rgb()).save(os.path.join(out_dir, "07_home_final.png"))
print("Saved 07_home_final.png")

print("\nAll 7 stage evidence images captured successfully in /evidence!")

