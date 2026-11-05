"""Action and safety validator for robotic arm trajectories and workspaces."""

import numpy as np
from typing import Tuple, Dict, Any, Optional

from vlm_robotics.config.config import WORKSPACE_LIMITS, JOINT_LIMITS


class ActionValidator:
    """Validates commands, spatial waypoints, and joint constraints for physical safety."""

    @classmethod
    def validate_cartesian_target(cls, target_pos: np.ndarray) -> Tuple[bool, str]:
        """Verify target (x, y, z) is within safe reach envelope."""
        x, y, z = target_pos

        # Check radius from base
        r = np.sqrt(x**2 + y**2)
        if r < 0.12:
            return False, f"Target too close to robot base (r={r:.3f}m, min=0.12m)."
        if r > 0.32:
            return False, f"Target exceeds maximum reach (r={r:.3f}m, max=0.32m)."

        # Check height limits
        if z < WORKSPACE_LIMITS["z_min"]:
            return False, f"Target height below table clearance (z={z:.3f}m)."
        if z > WORKSPACE_LIMITS["z_max"]:
            return False, f"Target height exceeds maximum ceiling (z={z:.3f}m)."

        return True, "Target is within safe workspace"

    @classmethod
    def validate_joint_angles(cls, q: np.ndarray) -> Tuple[bool, str]:
        """Verify joint angles are within official mechanical limits."""
        joint_names = ["joint1", "joint2", "joint3", "joint4"]
        for i, name in enumerate(joint_names):
            q_min, q_max = JOINT_LIMITS[name]
            if q[i] < q_min - 0.01 or q[i] > q_max + 0.01:
                return False, f"Joint {name} ({q[i]:.3f} rad) exceeds limits [{q_min}, {q_max}]."

        return True, "Joint configuration valid"

    @classmethod
    def validate_vlm_plan(cls, plan: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify high-level plan structure before execution."""
        task = plan.get("task")
        if task not in ("pick_and_place", "pyramid", "stack", "line", "sort", "multi_step", "pick", "inspect", "home", "status"):
            return False, f"Invalid task type '{task}'."

        if task in ("pick_and_place", "pick"):
            if not plan.get("object"):
                return False, "Plan is missing target object."

        if task == "pick_and_place":
            if not plan.get("destination"):
                return False, "Pick and place requires a valid destination."

        return True, "Plan validated successfully"

