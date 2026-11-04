"""Camera interface for RGB/Depth capture and intrinsic/extrinsic transformations."""

import numpy as np
import mujoco
from typing import Tuple, Dict, Any


class CameraInterface:
    """Handles camera image acquisition, calibration, and coordinate projections."""

    def __init__(self, env, camera_name: str = "workspace_cam"):
        self.env = env
        self.camera_name = camera_name
        self.cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found.")

        self.width = env.width
        self.height = env.height

        # Calculate camera intrinsics from FOV
        fovy = env.model.cam_fovy[self.cam_id]
        f = (self.height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        self.fx = f
        self.fy = f
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_rgb(self) -> np.ndarray:
        """Capture RGB frame."""
        return self.env.render_rgb(self.camera_name)

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture RGB and Depth frames."""
        return self.env.render_rgbd(self.camera_name)

    def get_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera extrinsic position and 3x3 rotation matrix in world coordinates.
        Returns: (pos, rot_mat)
        """
        pos = self.env.data.cam_xpos[self.cam_id].copy()
        rot_mat = self.env.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return pos, rot_mat

    def project_world_to_pixel(self, world_pt: np.ndarray) -> Tuple[int, int, float]:
        """
        Project 3D world coordinate into 2D camera pixel coordinates.
        Returns: (u, v, depth)
        """
        cam_pos, cam_mat = self.get_extrinsics()
        # Transform world point into camera coordinates
        # MuJoCo camera looks along -Z with +Y up and +X right
        rel_pos = world_pt - cam_pos
        x_cam = np.dot(cam_mat[0], rel_pos)
        y_cam = np.dot(cam_mat[1], rel_pos)
        z_cam = np.dot(cam_mat[2], rel_pos)

        # MuJoCo OpenGL convention: camera looks along -Z
        depth = -z_cam
        if depth <= 0:
            return -1, -1, depth

        u = int(self.fx * (x_cam / depth) + self.cx)
        v = int(-self.fy * (y_cam / depth) + self.cy)
        return u, v, depth

    def deproject_pixel_to_world(self, u: int, v: int, depth: float) -> np.ndarray:
        """
        Deproject 2D pixel coordinate + depth into 3D world coordinate.
        """
        cam_pos, cam_mat = self.get_extrinsics()
        # Normalized camera ray
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = -(v - self.cy) * depth / self.fy
        z_cam = -depth

        # Transform to world
        cam_ray = np.array([x_cam, y_cam, z_cam])
        world_pt = cam_pos + cam_mat.T @ cam_ray
        return world_pt

