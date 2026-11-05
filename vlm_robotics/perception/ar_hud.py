"""Augmented Reality (AR) HUD and Visual Telemetry Renderer for Robotic Manipulation."""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class ARHudRenderer:
    """Renders research-grade AR overlays, bounding boxes, trajectory ribbons, and telemetry onto camera frames."""

    # Futuristic UI color palette (BGR for OpenCV)
    COLOR_CYAN = (255, 229, 0)      # #00e5ff
    COLOR_GREEN = (118, 230, 0)     # #00e676
    COLOR_AMBER = (0, 179, 255)     # #ffb300
    COLOR_RED = (80, 80, 245)       # #f55050
    COLOR_PURPLE = (247, 166, 203)  # #cba6f7
    COLOR_WHITE = (240, 240, 245)
    COLOR_DARK = (24, 24, 32)
    COLOR_DIM_BG = (16, 18, 22)

    def __init__(self, camera_interface=None):
        self.camera_interface = camera_interface
        self.ee_trail: List[np.ndarray] = []
        self.max_trail_points = 50

    def add_trail_point(self, pt_3d: np.ndarray):
        """Append end-effector 3D position to the active trajectory trail."""
        self.ee_trail.append(pt_3d.copy())
        if len(self.ee_trail) > self.max_trail_points:
            self.ee_trail.pop(0)

    def clear_trail(self):
        """Clear motion trail history."""
        self.ee_trail.clear()

    def draw_corner_box(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 10,
    ):
        """Draw sleek futuristic sci-fi corner brackets instead of plain rectangles."""
        x1, y1 = pt1
        x2, y2 = pt2
        cl = min(corner_len, abs(x2 - x1) // 3, abs(y2 - y1) // 3)

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + cl), color, thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness)

    def render_hud(
        self,
        rgb_image: np.ndarray,
        camera_interface,
        target_obj: Optional[str] = None,
        destination: Optional[str] = None,
        stage: str = "IDLE",
        all_objects: Optional[Dict[str, np.ndarray]] = None,
        planned_waypoints: Optional[List[Dict[str, Any]]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        fps: float = 60.0,
    ) -> np.ndarray:
        """
        Render comprehensive AR HUD over camera frame:
        - 3D Projected Target Markers & Bounding Brackets
        - Affordance tags (e.g. [TOP-GRASP], [CONTAINER])
        - Dynamic 3D Trajectory Ribbons
        - Live Mission-Control Status Header & Telemetry Bars
        """
        vis = rgb_image.copy()
        H, W = vis.shape[:2]

        # 1. Draw 3D Trajectory Waypoints & Motion Trail
        if camera_interface is not None:
            # Draw planned path ribbons
            if planned_waypoints:
                prev_px = None
                for wp in planned_waypoints:
                    pt3d = wp.get("target_pos")
                    if pt3d is not None:
                        u, v, d = camera_interface.project_world_to_pixel(pt3d)
                        if 0 <= u < W and 0 <= v < H and d > 0:
                            # Waypoint circle
                            action = wp.get("action", "")
                            wp_color = self.COLOR_CYAN if "pre" in action else (
                                self.COLOR_GREEN if "grasp" in action else self.COLOR_AMBER
                            )
                            cv2.circle(vis, (u, v), 5, wp_color, -1)
                            cv2.circle(vis, (u, v), 8, wp_color, 1)

                            if prev_px is not None:
                                cv2.line(vis, prev_px, (u, v), wp_color, 2, cv2.LINE_AA)
                            prev_px = (u, v)

            # Draw historical EE trail
            if len(self.ee_trail) >= 2:
                trail_pixels = []
                for pt3d in self.ee_trail:
                    u, v, d = camera_interface.project_world_to_pixel(pt3d)
                    if 0 <= u < W and 0 <= v < H and d > 0:
                        trail_pixels.append((u, v))

                for i in range(1, len(trail_pixels)):
                    alpha = i / len(trail_pixels)
                    # Fade from dim cyan to bright cyan
                    thick = max(1, int(3 * alpha))
                    cv2.line(vis, trail_pixels[i - 1], trail_pixels[i], self.COLOR_CYAN, thick, cv2.LINE_AA)

            # 2. Draw 3D Object Projections & Affordances
            if all_objects:
                for obj_name, pos_3d in all_objects.items():
                    u, v, d = camera_interface.project_world_to_pixel(pos_3d)
                    if not (0 <= u < W and 0 <= v < H and d > 0):
                        continue

                    is_target = (target_obj is not None and obj_name == target_obj)
                    is_dest = (destination is not None and obj_name in destination)

                    # Dynamic sizing based on distance
                    box_rad = max(12, int(22.0 / d))
                    p1 = (u - box_rad, v - box_rad)
                    p2 = (u + box_rad, v + box_rad)

                    if is_target:
                        box_color = self.COLOR_GREEN
                        label = f"TARGET: {obj_name.upper()} | AFFORDANCE: TOP-GRASP"
                        thick = 2
                    elif is_dest:
                        box_color = self.COLOR_AMBER
                        label = f"DESTINATION: {obj_name.upper()}"
                        thick = 2
                    else:
                        box_color = (120, 140, 160)
                        label = obj_name
                        thick = 1

                    self.draw_corner_box(vis, p1, p2, box_color, thickness=thick, corner_len=6)
                    cv2.circle(vis, (u, v), 2, box_color, -1)

                    # Label badge
                    label_font_scale = 0.35 if not is_target else 0.40
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 1)
                    badge_y = max(18, p1[1] - 4)
                    badge_x = max(2, min(W - lw - 8, p1[0] - 6))

                    cv2.rectangle(
                        vis,
                        (badge_x, badge_y - lh - 4),
                        (badge_x + lw + 6, badge_y + 2),
                        self.COLOR_DARK,
                        -1,
                    )
                    cv2.putText(
                        vis,
                        label,
                        (badge_x + 3, badge_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        box_color,
                        1,
                        cv2.LINE_AA,
                    )

        # 3. Top Mission-Control HUD Bar
        header_h = 32
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (W, header_h), self.COLOR_DIM_BG, -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
        cv2.line(vis, (0, header_h), (W, header_h), self.COLOR_CYAN, 1)

        # Stage color badge
        stage_color = self.COLOR_GREEN if stage in ("GRASP", "SUCCESS", "VERIFY") else (
            self.COLOR_AMBER if stage in ("APPROACH", "TRANSPORT") else (
                self.COLOR_RED if "FAIL" in stage or "RECOVERY" in stage else self.COLOR_CYAN
            )
        )
        status_text = f"[VLM-ROBOTICS OS]  MODE: AUTONOMOUS  |  STAGE: {stage}"
        cv2.putText(vis, status_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, stage_color, 1, cv2.LINE_AA)

        fps_text = f"SIM: {fps:.1f} FPS  |  VLM: QWEN2.5-VL (OLLAMA)"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(vis, fps_text, (W - tw - 12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # 4. Telemetry Readout Strip at Bottom
        if telemetry:
            telemetry_h = 28
            bot_y = H - telemetry_h
            overlay_bot = vis.copy()
            cv2.rectangle(overlay_bot, (0, bot_y), (W, H), self.COLOR_DIM_BG, -1)
            cv2.addWeighted(overlay_bot, 0.85, vis, 0.15, 0, vis)
            cv2.line(vis, (0, bot_y), (W, bot_y), (60, 70, 85), 1)

            ee_pos = telemetry.get("ee_pos", [0, 0, 0])
            grip = telemetry.get("gripper_pos", 0.0)
            holding = telemetry.get("is_grasping", False)
            hold_str = "SECURED" if holding else "OPEN"
            hold_color = self.COLOR_GREEN if holding else self.COLOR_WHITE

            telem_str = (
                f"EEF: X={ee_pos[0]:+.3f} Y={ee_pos[1]:+.3f} Z={ee_pos[2]:+.3f}m | "
                f"GRIP: {grip:+.4f}m [{hold_str}]"
            )
            cv2.putText(vis, telem_str, (12, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_CYAN, 1, cv2.LINE_AA)

        return vis

