"""Camera interface for RGB/Depth capture, intrinsic/extrinsic transformations, and Multi-Camera systems."""

import cv2
import numpy as np
import mujoco
from typing import Tuple, Dict, Any, Optional, List


class CameraInterface:
    """Handles camera image acquisition, calibration, and coordinate projections for a specific camera."""

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

    def get_colorized_depth(self, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
        """
        Capture depth buffer and return an 8-bit RGB colorized depth heatmap.
        """
        _, depth = self.get_rgbd()
        # Filter non-finite and clip to reasonable workspace depth (e.g. 0.05m to 1.5m)
        depth_clean = np.nan_to_num(depth, nan=1.5, posinf=1.5, neginf=0.05)
        d_min = np.percentile(depth_clean, 2)
        d_max = np.percentile(depth_clean, 98)
        if d_max - d_min < 1e-4:
            d_max = d_min + 1.0
        depth_norm = np.clip((depth_clean - d_min) / (d_max - d_min), 0.0, 1.0)
        depth_8u = (depth_norm * 255.0).astype(np.uint8)
        color_bgr = cv2.applyColorMap(depth_8u, colormap)
        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

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


class MultiCameraSystem:
    """Coordinates multi-camera acquisition (Overhead, Wrist, Third-Person) and PIP composition."""

    AVAILABLE_CAMERAS = ["workspace_cam", "overhead_cam", "wrist_cam", "cinematic_cam"]

    def __init__(self, env):
        self.env = env
        self.cameras: Dict[str, CameraInterface] = {}
        for cam_name in self.AVAILABLE_CAMERAS:
            try:
                self.cameras[cam_name] = CameraInterface(env, cam_name)
            except Exception:
                pass

    def get_camera(self, name: str) -> Optional[CameraInterface]:
        return self.cameras.get(name)

    def capture_all_rgb(self) -> Dict[str, np.ndarray]:
        """Capture RGB frames from all available cameras."""
        frames = {}
        for name, cam in self.cameras.items():
            frames[name] = cam.get_rgb()
        return frames

    def create_multiview_composite(
        self,
        main_camera: str = "workspace_cam",
        hud_overlay: Optional[np.ndarray] = None,
        pip_camera: str = "wrist_cam",
        overhead_camera: str = "overhead_cam",
        show_depth: bool = True,
    ) -> np.ndarray:
        """
        Render a research laboratory multi-camera dashboard:
        - Primary canvas: main_camera RGB frame (or custom HUD frame)
        - Top-Right Picture-in-Picture: wrist camera (eye-in-hand)
        - Bottom-Right thumbnail: overhead camera
        - Bottom-Left thumbnail: colorized depth map
        """
        if hud_overlay is not None:
            base_img = hud_overlay.copy()
        elif main_camera in self.cameras:
            base_img = self.cameras[main_camera].get_rgb().copy()
        else:
            base_img = self.env.render_rgb()

        H, W = base_img.shape[:2]

        # 1. Overlay Wrist Camera (PiP Top-Right)
        if pip_camera in self.cameras:
            wrist_rgb = self.cameras[pip_camera].get_rgb()
            pip_w = int(W * 0.30)
            pip_h = int(H * 0.30)
            wrist_resized = cv2.resize(wrist_rgb, (pip_w, pip_h))

            # Add decorative border and header
            cv2.rectangle(wrist_resized, (0, 0), (pip_w - 1, pip_h - 1), (0, 229, 255), 2)
            cv2.rectangle(wrist_resized, (0, 0), (pip_w, 20), (20, 20, 30), -1)
            cv2.putText(
                wrist_resized, "WRIST EYE-IN-HAND", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 229, 255), 1, cv2.LINE_AA
            )

            x_off = W - pip_w - 12
            y_off = 12
            base_img[y_off:y_off + pip_h, x_off:x_off + pip_w] = wrist_resized

        # 2. Overlay Overhead Camera (Thumbnail Mid/Lower-Right)
        if overhead_camera in self.cameras:
            oh_rgb = self.cameras[overhead_camera].get_rgb()
            pip_w = int(W * 0.26)
            pip_h = int(H * 0.26)
            oh_resized = cv2.resize(oh_rgb, (pip_w, pip_h))

            cv2.rectangle(oh_resized, (0, 0), (pip_w - 1, pip_h - 1), (137, 180, 250), 2)
            cv2.rectangle(oh_resized, (0, 0), (pip_w, 20), (20, 20, 30), -1)
            cv2.putText(
                oh_resized, "OVERHEAD VIEW", (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (137, 180, 250), 1, cv2.LINE_AA
            )

            x_off = W - pip_w - 12
            y_off = H - pip_h - 12
            base_img[y_off:y_off + pip_h, x_off:x_off + pip_w] = oh_resized

        # 3. Overlay Colorized Depth (Thumbnail Bottom-Left)
        if show_depth and main_camera in self.cameras:
            try:
                depth_rgb = self.cameras[main_camera].get_colorized_depth()
                pip_w = int(W * 0.24)
                pip_h = int(H * 0.24)
                depth_resized = cv2.resize(depth_rgb, (pip_w, pip_h))

                cv2.rectangle(depth_resized, (0, 0), (pip_w - 1, pip_h - 1), (203, 166, 247), 2)
                cv2.rectangle(depth_resized, (0, 0), (pip_w, 20), (20, 20, 30), -1)
                cv2.putText(
                    depth_resized, "DEPTH (TURBO)", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (203, 166, 247), 1, cv2.LINE_AA
                )

                x_off = 12
                y_off = H - pip_h - 12
                base_img[y_off:y_off + pip_h, x_off:x_off + pip_w] = depth_resized
            except Exception:
                pass

        return base_img


