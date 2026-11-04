"""Robot motion controller for OpenMANIPULATOR-X using numerical IK and smooth trajectories."""

import time
import numpy as np
import mujoco
from typing import Tuple, Optional, List

from vlm_robotics.config.config import (
    HOME_POSE,
    READY_POSE,
    JOINT_LIMITS,
    WORKSPACE_LIMITS,
)


class RobotController:
    """Controls the 4-DOF OpenMANIPULATOR-X arm movements and Inverse Kinematics."""

    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.ee_id = env.ee_body_id
        self.site_id = getattr(env, "grasp_site_id", -1)
        self.joint_ids = env.arm_joint_ids
        self.qpos_adrs = env.arm_qpos_adrs
        self.dof_adrs = env.arm_dof_adrs
        self.actuator_ids = env.arm_actuator_ids

        # Joint limit arrays
        self.q_min = np.array([JOINT_LIMITS[f"joint{i}"][0] for i in range(1, 5)])
        self.q_max = np.array([JOINT_LIMITS[f"joint{i}"][1] for i in range(1, 5)])

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_steps: int = 150,
        tol: float = 0.003,
        damping: float = 1e-3,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve inverse kinematics for the target 3D position using Damped Least Squares (DLS).
        Returns: (q_solution, success, residual_error)
        """
        # Save current state to restore after IK calculation
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        # Target base yaw angle directly from (x, y)
        target_yaw = np.arctan2(target_pos[1], target_pos[0])

        if q_init is not None:
            q_current = q_init.copy()
        else:
            q_current = self.get_current_joint_positions().copy()
        q_current[0] = target_yaw

        for i in range(4):
            self.data.qpos[self.qpos_adrs[i]] = q_current[i]
        mujoco.mj_forward(self.model, self.data)

        success = False
        error_norm = float("inf")

        for _ in range(max_steps):
            if self.site_id != -1:
                current_pos = self.data.site_xpos[self.site_id]
            else:
                current_pos = self.data.xpos[self.ee_id]
            err = target_pos - current_pos
            error_norm = np.linalg.norm(err)

            if error_norm < tol:
                success = True
                break

            # Compute translation Jacobian for grasp site / end-effector
            jacp = np.zeros((3, self.model.nv))
            if self.site_id != -1:
                mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
            else:
                mujoco.mj_jac(self.model, self.data, jacp, None, current_pos, self.ee_id)

            # Extract arm joint DOF columns
            J = jacp[:, self.dof_adrs]

            # Damped Least Squares: J_pinv = J.T @ (J @ J.T + lambda^2 * I)^-1
            JJT = J @ J.T
            damped_inv = np.linalg.inv(JJT + damping * np.eye(3))
            J_pinv = J.T @ damped_inv
            dq = J_pinv @ err

            # Step and clamp to limits
            q_current += 0.5 * dq
            q_current = np.clip(q_current, self.q_min, self.q_max)

            for i in range(4):
                self.data.qpos[self.qpos_adrs[i]] = q_current[i]
            mujoco.mj_forward(self.model, self.data)

        # Restore simulation state
        self.data.qpos[:] = qpos_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)

        return q_current, success, error_norm

    def get_current_joint_positions(self) -> np.ndarray:
        return self.env.get_arm_joint_positions()

    def move_to_joint_positions(
        self,
        q_target: np.ndarray,
        duration: float = 1.0,
        gripper_target: Optional[float] = None,
        dt_substep: float = 0.01,
    ):
        """
        Execute smooth minimum-jerk trajectory to target joint angles.
        """
        q_start = self.get_current_joint_positions()
        steps = max(10, int(duration / dt_substep))

        for step in range(steps + 1):
            t_start = time.perf_counter()
            t = step / steps
            # Minimum jerk polynomial: s(t) = 10*t^3 - 15*t^4 + 6*t^5
            s = 10.0 * (t ** 3) - 15.0 * (t ** 4) + 6.0 * (t ** 5)
            q_interp = q_start + s * (q_target - q_start)

            self.env.set_control(q_interp, gripper_target=gripper_target)
            # Step physics for substep interval
            sim_steps = int(dt_substep / self.model.opt.timestep)
            self.env.step(n_steps=max(1, sim_steps))

            elapsed = time.perf_counter() - t_start
            if elapsed < dt_substep:
                time.sleep(dt_substep - elapsed)

    def move_to_cartesian_position(
        self,
        target_pos: np.ndarray,
        duration: float = 1.2,
        gripper_target: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Move end-effector to 3D Cartesian coordinates.
        """
        q_sol, success, err = self.solve_ik(target_pos)
        if not success and err > 0.01:
            return False, f"Target position {np.round(target_pos, 3)} unreachable (error {err:.4f}m)."

        self.move_to_joint_positions(q_sol, duration=duration, gripper_target=gripper_target)
        return True, "Target reached"

    def go_home(self, duration: float = 1.2):
        """Move arm to safe home configuration."""
        self.move_to_joint_positions(np.array(HOME_POSE), duration=duration)
