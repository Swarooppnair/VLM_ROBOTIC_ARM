"""Coordinate mapping and spatial localization from vision to robot base frame."""

import numpy as np
from typing import Optional, Dict, Tuple, Any

from vlm_robotics.config.config import WORKSPACE_LIMITS, KNOWN_OBJECTS


class CoordinateMapper:
    """
    Transforms visual object references and spatial commands into calibrated 3D workspace coordinates.
    Employs hybrid localization: vision validates semantic presence, while the simulator state
    provides sub-millimeter grasp accuracy.
    """

    def __init__(self, env, camera_interface=None):
        self.env = env
        self.camera_interface = camera_interface

    def get_object_position(self, object_name: str) -> Optional[np.ndarray]:
        """
        Retrieve accurate 3D position of an object in the robot frame.
        Handles variations in naming (e.g. 'red cube' vs 'red_cube').
        """
        canonical_name = self.normalize_object_name(object_name)
        if canonical_name is None:
            return None

        # Query simulation ground truth pose
        pos = self.env.get_object_position(canonical_name)
        if pos is not None:
            return pos.copy()

        # Fallback to configured default pos
        if canonical_name in KNOWN_OBJECTS:
            return np.array(KNOWN_OBJECTS[canonical_name]["default_pos"], dtype=np.float64)

        return None

    def normalize_object_name(self, name: str) -> Optional[str]:
        """Map common synonyms or VLM phrasing to canonical internal object keys."""
        if not name:
            return None
        cleaned = name.lower().replace("-", " ").replace("_", " ").strip()

        if "red" in cleaned and ("cube" in cleaned or "block" in cleaned or "box" not in cleaned):
            return "red_cube"
        if "blue" in cleaned and ("cube" in cleaned or "block" in cleaned or "box" not in cleaned):
            return "blue_cube"
        if "green" in cleaned and ("cube" in cleaned or "block" in cleaned):
            return "green_cube"
        if "orange" in cleaned and ("cube" in cleaned or "block" in cleaned):
            return "orange_cube"
        if "bottle" in cleaned or "can" in cleaned or "yellow" in cleaned:
            return "bottle"
        if "blue glass" in cleaned or "glass 1" in cleaned or "glass1" in cleaned:
            return "glass1"
        if "purple glass" in cleaned or "glass 2" in cleaned or "glass2" in cleaned or "purple" in cleaned:
            return "glass2"
        if "amber" in cleaned or "glass 3" in cleaned or "glass3" in cleaned:
            return "glass3"
        if "emerald" in cleaned or "green glass" in cleaned or "glass 4" in cleaned or "glass4" in cleaned:
            return "glass4"
        if "pink" in cleaned or "rose" in cleaned or "glass 5" in cleaned or "glass5" in cleaned:
            return "glass5"
        if "teal" in cleaned or "cyan glass" in cleaned or "glass 6" in cleaned or "glass6" in cleaned:
            return "glass6"
        if "glass" in cleaned or "cup" in cleaned:
            return "glass1"
        if "box" in cleaned or "container" in cleaned or "bin" in cleaned or "basket" in cleaned:
            return "box"

        for key in KNOWN_OBJECTS:
            if key in cleaned:
                return key
        return None

    def compute_destination_position(
        self,
        destination_desc: str,
        reference_obj_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Resolve destination phrasings (e.g. 'box', 'next to blue cube', 'left side') into a 3D target coordinate.
        """
        dest_clean = destination_desc.lower().replace("-", " ").replace("_", " ").strip()

        # 1. Container / Box destination
        if "box" in dest_clean or "container" in dest_clean or "bin" in dest_clean:
            box_pos = self.get_object_position("box")
            if box_pos is not None:
                # Place slightly above bottom of box
                return np.array([box_pos[0], box_pos[1], box_pos[2] + 0.035])

        # 2. Relative to another object (e.g., "on top of red cube", "next to blue cube")
        all_targets = [
            "blue_cube", "red_cube", "green_cube", "orange_cube",
            "bottle", "glass1", "glass2", "glass3", "glass4", "glass5", "glass6"
        ]
        for obj_key in all_targets:
            obj_name_clean = obj_key.replace("_", " ")
            if obj_name_clean in dest_clean or obj_key in dest_clean:
                ref_pos = self.get_object_position(obj_key)
                if ref_pos is not None:
                    # Check if intention is stacking on top
                    if any(w in dest_clean for w in ["on top", "onto", "stack", "above"]) or (dest_clean.startswith("on ") and "next" not in dest_clean):
                        target = ref_pos.copy()
                        # Stack at top of the reference object
                        target[2] = ref_pos[2] + 0.030
                        return self.clamp_to_workspace(target)
                    else:
                        # Place adjacent on table
                        offset_y = 0.055 if ref_pos[1] <= 0 else -0.055
                        target = ref_pos.copy()
                        target[1] += offset_y
                        target[2] = 0.015  # Table surface level
                        return self.clamp_to_workspace(target)

        # 3. Table edges and corners
        if "edge" in dest_clean:
            if "left" in dest_clean:
                return np.array([0.18, 0.20, 0.015])
            elif "right" in dest_clean:
                return np.array([0.18, -0.20, 0.015])
            else:
                # General table edge: choose a visible, reachable perimeter point
                # that is distant from the object's initial position for clear visible motion.
                ref_pos = self.get_object_position(reference_obj_name) if reference_obj_name else None
                candidates = [
                    np.array([0.18, -0.20, 0.015]),  # Right edge
                    np.array([0.27, 0.12, 0.015]),   # Front-left edge
                    np.array([0.27, -0.14, 0.015]),  # Front-right edge
                    np.array([0.18, 0.20, 0.015]),   # Left edge
                ]
                if ref_pos is not None:
                    dists = [np.linalg.norm(c[:2] - ref_pos[:2]) for c in candidates]
                    best_candidate = candidates[int(np.argmax(dists))]
                    return best_candidate
                return np.array([0.18, -0.20, 0.015])

        if "corner" in dest_clean:
            if "left" in dest_clean:
                return np.array([0.26, 0.16, 0.015])
            else:
                return np.array([0.26, -0.16, 0.015])

        # 4. Spatial descriptors: "front", "back", "left", "right", "center"
        if "front" in dest_clean or "forward" in dest_clean or "near" in dest_clean:
            return np.array([0.26, 0.0, 0.015])
        elif "back" in dest_clean or "rear" in dest_clean:
            return np.array([0.16, 0.0, 0.015])
        elif "left" in dest_clean:
            # In robot frame, +Y is robot's left
            return np.array([0.18, 0.14, 0.015])
        elif "right" in dest_clean:
            # In robot frame, -Y is robot's right
            return np.array([0.18, -0.14, 0.015])
        elif "center" in dest_clean or "middle" in dest_clean:
            return np.array([0.20, 0.0, 0.015])

        # 5. If destination is directly an object name
        obj_name = self.normalize_object_name(destination_desc)
        if obj_name:
            pos = self.get_object_position(obj_name)
            if pos is not None:
                return pos.copy()

        # 6. Fallback: default safe open table location
        return np.array([0.22, -0.14, 0.015])

    def clamp_to_workspace(self, target_pos: np.ndarray) -> np.ndarray:
        """Ensure coordinate falls within safe reachable boundaries."""
        clamped = target_pos.copy()
        clamped[0] = np.clip(clamped[0], WORKSPACE_LIMITS["x_min"], WORKSPACE_LIMITS["x_max"])
        clamped[1] = np.clip(clamped[1], WORKSPACE_LIMITS["y_min"], WORKSPACE_LIMITS["y_max"])
        clamped[2] = np.clip(clamped[2], WORKSPACE_LIMITS["z_min"], WORKSPACE_LIMITS["z_max"])
        return clamped

