"""Gripper controller for OpenMANIPULATOR-X parallel gripper."""

import time
import numpy as np
from typing import Optional

from vlm_robotics.config.config import GRIPPER_OPEN, GRIPPER_CLOSE


class GripperController:
    """Controls opening, closing, and grasping force of the parallel gripper."""

    def __init__(self, env):
        self.env = env
        self.open_pos = GRIPPER_OPEN
        self.close_pos = GRIPPER_CLOSE
        self.current_target = self.open_pos

    def open(self, duration: float = 1.5):
        """Open gripper fingers to maximum stroke with realistic pacing."""
        self.current_target = self.open_pos
        arm_ctrl = self.env.data.ctrl[self.env.arm_actuator_ids].copy()
        dt = 0.02
        steps = max(5, int(duration / dt))
        for _ in range(steps):
            t0 = time.perf_counter()
            self.env.set_control(arm_ctrl, gripper_target=self.open_pos)
            self.env.step(n_steps=10)
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def close(self, duration: float = 1.5):
        """Pinch gripper fingers to grasp an object with realistic pacing."""
        self.current_target = self.close_pos
        arm_ctrl = self.env.data.ctrl[self.env.arm_actuator_ids].copy()
        dt = 0.02
        steps = max(5, int(duration / dt))
        for _ in range(steps):
            t0 = time.perf_counter()
            self.env.set_control(arm_ctrl, gripper_target=self.close_pos)
            self.env.step(n_steps=10)
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def get_state(self) -> float:
        """Get current slide position of the gripper."""
        return self.env.get_gripper_position()

    def is_grasping(self, threshold: float = 0.003) -> bool:
        """Check if gripper stopped before fully closing, indicating object contact."""
        current = self.get_state()
        # If gripper target is closed (-0.009) but current position is higher (e.g. > -0.005),
        # an object is physically held between the fingers.
        return (current - self.close_pos) > threshold

