"""MuJoCo Simulation Environment for OpenMANIPULATOR-X tabletop manipulation."""

import os
import time
import threading
import numpy as np
import mujoco
from typing import Tuple, Optional, Dict, Any
from PIL import Image

from vlm_robotics.config.config import (
    SCENE_XML_PATH,
    DEFAULT_CAMERA,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    HOME_POSE,
    GRIPPER_OPEN,
)


class SimulationEnvironment:
    """Manages the MuJoCo physics simulation, robot states, cameras, and objects."""

    def __init__(self, scene_path: str = SCENE_XML_PATH, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT):
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Simulation scene file not found: {scene_path}")
        
        self.scene_path = scene_path
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.gl_lock = threading.RLock()
        self._viewer_active = False
        self._is_idle = True
        self._viewer_thread = None

        # Load MuJoCo Model and Data
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Offscreen Renderer
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        # Body, Joint, and Site IDs
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_link")
        self.grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 5)
        ]
        self.gripper_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left_joint")

        # Position and DOF addresses in qpos and qvel
        self.arm_qpos_adrs = [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids]
        self.arm_dof_adrs = [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids]
        self.gripper_qpos_adr = self.model.jnt_qposadr[self.gripper_joint_id]
        self.gripper_dof_adr = self.model.jnt_dofadr[self.gripper_joint_id]

        # Actuator IDs (joint1, joint2, joint3, joint4, gripper_left_joint)
        self.arm_actuator_ids = [0, 1, 2, 3]
        self.gripper_actuator_id = 4

        # Viewer reference
        self.viewer = None

        # Reset to home
        self.reset()

    def reset(self):
        """Reset simulation physics and set robot to home pose."""
        mujoco.mj_resetData(self.model, self.data)

        # Set default joint positions
        for i, q in enumerate(HOME_POSE):
            self.data.qpos[self.arm_qpos_adrs[i]] = q
            self.data.ctrl[self.arm_actuator_ids[i]] = q

        # Gripper open
        self.data.qpos[self.gripper_qpos_adr] = GRIPPER_OPEN
        self.data.ctrl[self.gripper_actuator_id] = GRIPPER_OPEN

        # Settle simulation physics
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

    def step(self, n_steps: int = 1):
        """Advance physics simulation by n steps."""
        with self.lock:
            for _ in range(n_steps):
                mujoco.mj_step(self.model, self.data)

    def set_control(self, arm_targets: np.ndarray, gripper_target: Optional[float] = None):
        """Set position actuator control inputs."""
        for i in range(4):
            self.data.ctrl[self.arm_actuator_ids[i]] = float(arm_targets[i])
        if gripper_target is not None:
            self.data.ctrl[self.gripper_actuator_id] = float(gripper_target)

    def get_arm_joint_positions(self) -> np.ndarray:
        """Get current 4-DOF arm joint angles."""
        return np.array([self.data.qpos[adr] for adr in self.arm_qpos_adrs], dtype=np.float64)

    def get_gripper_position(self) -> float:
        """Get current gripper opening displacement."""
        return float(self.data.qpos[self.gripper_qpos_adr])

    def get_ee_position(self) -> np.ndarray:
        """Get 3D world position of the end-effector link."""
        return self.data.xpos[self.ee_body_id].copy()

    def get_grasp_position(self) -> np.ndarray:
        """Get 3D world position of the grasp center between the gripper pads."""
        if self.grasp_site_id != -1:
            return self.data.site_xpos[self.grasp_site_id].copy()
        return self.data.xpos[self.ee_body_id].copy()

    def get_ee_orientation(self) -> np.ndarray:
        """Get 3x3 rotation matrix of the end-effector."""
        return self.data.xmat[self.ee_body_id].reshape(3, 3).copy()

    def get_object_position(self, object_name: str) -> Optional[np.ndarray]:
        """Query real-time 3D position of an object body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id == -1:
            return None
        return self.data.xpos[body_id].copy()

    def get_all_object_positions(self) -> Dict[str, np.ndarray]:
        """Return positions of all tracked workspace objects."""
        names = [
            "red_cube", "blue_cube", "green_cube", "orange_cube",
            "bottle", "glass1", "glass2", "glass3", "glass4", "glass5", "glass6", "box"
        ]
        positions = {}
        for name in names:
            pos = self.get_object_position(name)
            if pos is not None:
                positions[name] = pos
        return positions

    def render_rgb(self, camera_name: str = DEFAULT_CAMERA) -> np.ndarray:
        """Render RGB image from specified camera under thread-safe OpenGL lock."""
        with self.gl_lock:
            with self.lock:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if cam_id == -1:
                    raise ValueError(f"Camera '{camera_name}' not found in model.")
                self.renderer.update_scene(self.data, camera=cam_id)
                return self.renderer.render()

    def render_rgbd(self, camera_name: str = DEFAULT_CAMERA) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB and Depth image from specified camera under thread-safe OpenGL lock."""
        with self.gl_lock:
            with self.lock:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                if cam_id == -1:
                    raise ValueError(f"Camera '{camera_name}' not found in model.")
                self.renderer.disable_depth_rendering()
                self.renderer.update_scene(self.data, camera=cam_id)
                rgb = self.renderer.render()
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render()
                self.renderer.disable_depth_rendering()
                return rgb, depth

    def save_image(self, filepath: str, camera_name: str = DEFAULT_CAMERA) -> str:
        """Capture and save camera image to disk."""
        rgb = self.render_rgb(camera_name)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        img = Image.fromarray(rgb)
        img.save(filepath)
        return filepath

    def _viewer_loop(self):
        """Background thread keeping the 3D viewer interactive and synced at 60 FPS."""
        while self._viewer_active:
            if self.viewer is not None and self.viewer.is_running():
                with self.gl_lock:
                    with self.lock:
                        if self._is_idle:
                            mujoco.mj_step(self.model, self.data)
                        try:
                            self.viewer.sync()
                        except Exception:
                            pass
            elif self.viewer is not None and not self.viewer.is_running():
                break
            time.sleep(0.016)

    def open_viewer(self):
        """Open interactive 3D viewer window safely."""
        with self.gl_lock:
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer_active = True
                self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
                self._viewer_thread.start()
                print("Interactive 3D Viewer launched.")
            except Exception as e:
                print(f"Interactive viewer could not be launched: {e}")

    def close_viewer(self):
        """Safely close 3D viewer."""
        self._viewer_active = False
        with self.gl_lock:
            if self.viewer is not None:
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None

    def close(self):
        """Clean shutdown of both renderer and viewer."""
        self.close_viewer()
        with self.gl_lock:
            if self.renderer is not None:
                try:
                    self.renderer.close()
                except Exception:
                    pass
                self.renderer = None
