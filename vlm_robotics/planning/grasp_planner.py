"""Grasp and waypoint trajectory planner for pick-and-place manipulation."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from vlm_robotics.planning.action_validator import ActionValidator


class GraspPlanner:
    """Generates structured sequence of 3D waypoints for grasping and placing."""

    def __init__(self, hover_height: float = 0.07, lift_height: float = 0.10):
        self.hover_height = hover_height
        self.lift_height = lift_height
        self.validator = ActionValidator()

    def plan_pick_and_place(
        self,
        object_pos: np.ndarray,
        dest_pos: np.ndarray,
        object_type: str = "cube",
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Generate sequence of waypoints for complete pick-and-place operation.
        Returns: (success, message, waypoints)
        """
        # Grasp height depends on object type
        if "bottle" in object_type or "glass" in object_type:
            grasp_z = 0.028
            hover = 0.08
            place_z = max(0.035, dest_pos[2])
        else:
            # Cube grasp: align gripper with center of 3cm cube
            grasp_z = max(0.015, object_pos[2])
            hover = self.hover_height
            place_z = max(0.015, dest_pos[2])

        pre_grasp = np.array([object_pos[0], object_pos[1], grasp_z + hover])
        grasp = np.array([object_pos[0], object_pos[1], grasp_z])
        lift = np.array([object_pos[0], object_pos[1], grasp_z + self.lift_height])

        # Destination waypoints
        pre_place = np.array([dest_pos[0], dest_pos[1], max(0.18, place_z + 0.08)])
        place = np.array([dest_pos[0], dest_pos[1], place_z])
        retreat = np.array([dest_pos[0], dest_pos[1], max(0.20, place_z + 0.08)])

        # Validate waypoints
        for name, pt in [("pre_grasp", pre_grasp), ("grasp", grasp), ("lift", lift),
                         ("pre_place", pre_place), ("place", place)]:
            valid, msg = self.validator.validate_cartesian_target(pt)
            if not valid:
                return False, f"Waypoint '{name}' failed safety validation: {msg}", []

        # Slower, calm, and deliberate robotic arm movements (comfortable human speed)
        waypoints = [
            {"action": "open_gripper", "target_pos": None, "desc": "Opening gripper fingers"},
            {"action": "move", "target_pos": pre_grasp, "duration": 3.0, "desc": "Approaching pre-grasp pose above object"},
            {"action": "move", "target_pos": grasp, "duration": 2.2, "desc": "Lowering to object grasp height"},
            {"action": "close_gripper", "target_pos": None, "desc": "Closing gripper around object"},
            {"action": "move", "target_pos": lift, "duration": 2.2, "desc": "Lifting object vertically"},
            {"action": "move", "target_pos": pre_place, "duration": 3.5, "desc": "Transporting object to destination standoff"},
            {"action": "move", "target_pos": place, "duration": 2.2, "desc": "Lowering object to destination"},
            {"action": "open_gripper", "target_pos": None, "desc": "Releasing object"},
            {"action": "move", "target_pos": retreat, "duration": 2.0, "desc": "Retreating arm vertically"},
            {"action": "go_home", "target_pos": None, "duration": 3.0, "desc": "Returning to safe home position"},
        ]

        return True, "Trajectory planned successfully", waypoints

    def plan_pick_only(
        self,
        object_pos: np.ndarray,
        object_type: str = "cube",
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Generate sequence of waypoints for pick-and-lift only."""
        grasp_z = max(0.018, object_pos[2])
        pre_grasp = np.array([object_pos[0], object_pos[1], grasp_z + self.hover_height])
        grasp = np.array([object_pos[0], object_pos[1], grasp_z])
        lift = np.array([object_pos[0], object_pos[1], grasp_z + self.lift_height])

        waypoints = [
            {"action": "open_gripper", "target_pos": None, "desc": "Opening gripper fingers"},
            {"action": "move", "target_pos": pre_grasp, "duration": 3.0, "desc": "Approaching pre-grasp pose"},
            {"action": "move", "target_pos": grasp, "duration": 2.2, "desc": "Lowering to grasp height"},
            {"action": "close_gripper", "target_pos": None, "desc": "Closing gripper"},
            {"action": "move", "target_pos": lift, "duration": 2.2, "desc": "Lifting object"},
        ]
        return True, "Pick trajectory planned", waypoints
