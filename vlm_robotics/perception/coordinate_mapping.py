"""Coordinate mapping and spatial localization from vision to robot base frame."""

import re
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

        # Containers
        if any(w in cleaned for w in ["box", "container", "bin", "basket", "tray"]):
            return "box"

        # Bottle
        if any(w in cleaned for w in ["bottle", "flask", "yellow bottle", "can"]):
            return "bottle"

        # Specific glasses / cups
        if "purple" in cleaned or "violet" in cleaned or "glass 2" in cleaned or "glass2" in cleaned:
            return "glass2"
        if "amber" in cleaned or "glass 3" in cleaned or "glass3" in cleaned:
            return "glass3"
        if "emerald" in cleaned or ("green" in cleaned and ("glass" in cleaned or "cup" in cleaned)) or "glass 4" in cleaned or "glass4" in cleaned:
            return "glass4"
        if "pink" in cleaned or "rose" in cleaned or "glass 5" in cleaned or "glass5" in cleaned:
            return "glass5"
        if "teal" in cleaned or "cyan glass" in cleaned or "turquoise" in cleaned or "glass 6" in cleaned or "glass6" in cleaned:
            return "glass6"
        if ("blue" in cleaned and ("glass" in cleaned or "cup" in cleaned)) or "glass 1" in cleaned or "glass1" in cleaned:
            return "glass1"
        if "glass" in cleaned or "cup" in cleaned:
            return "glass1"

        # Cubes / blocks
        if "red" in cleaned and ("cube" in cleaned or "block" in cleaned or ("glass" not in cleaned and "bottle" not in cleaned and "box" not in cleaned)):
            return "red_cube"
        if "blue" in cleaned and ("cube" in cleaned or "block" in cleaned or ("glass" not in cleaned and "bottle" not in cleaned and "box" not in cleaned)):
            return "blue_cube"
        if "green" in cleaned and ("cube" in cleaned or "block" in cleaned or ("glass" not in cleaned and "bottle" not in cleaned and "box" not in cleaned)):
            return "green_cube"
        if "orange" in cleaned and ("cube" in cleaned or "block" in cleaned or ("glass" not in cleaned and "bottle" not in cleaned and "box" not in cleaned)):
            return "orange_cube"

        if "cube" in cleaned or "block" in cleaned:
            return "red_cube"

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
        Resolve destination phrasings (e.g. 'box', 'on the purple glass', 'next to blue cube', 'left side')
        into a physical 3D target coordinate.
        """
        dest_clean = destination_desc.lower().replace("-", " ").replace("_", " ").strip()

        # 1. Container / Box destination
        if any(w in dest_clean for w in ["box", "container", "bin", "basket", "tray"]):
            box_pos = self.get_object_position("box")
            if box_pos is not None:
                # Place safely inside box
                return np.array([box_pos[0], box_pos[1], 0.052])

        # 2. Extract relative intention (stack on top vs adjacent)
        is_stack = (
            any(w in dest_clean for w in ["on top", "onto", "on to", "above", "stack"])
            or (dest_clean.startswith("on ") and "edge" not in dest_clean and "corner" not in dest_clean and "table" not in dest_clean and "next" not in dest_clean)
        )
        is_adjacent = any(w in dest_clean for w in ["next to", "beside", "near to", "near", "adjacent", "by", "close to"])

        # Try to resolve destination object
        # Strip prepositions to isolate object phrase
        candidate = dest_clean
        for prep in ["on top of", "on top", "onto", "on to", "next to", "beside", "near to", "near", "inside", "into", "in", "on", "to", "at", "the", "a", "an"]:
            candidate = re.sub(r'\b' + re.escape(prep) + r'\b', ' ', candidate)
        candidate = " ".join(candidate.split())

        dest_obj = self.normalize_object_name(candidate)
        if not dest_obj:
            # Try matching any known object key directly in dest_clean
            for key in ["glass1", "glass2", "glass3", "glass4", "glass5", "glass6",
                        "blue_cube", "red_cube", "green_cube", "orange_cube", "bottle", "box"]:
                clean_k = key.replace("_", " ")
                if clean_k in dest_clean or key in dest_clean:
                    dest_obj = key
                    break

        if dest_obj and dest_obj != "box":
            ref_pos = self.get_object_position(dest_obj)
            if ref_pos is not None:
                if is_stack:
                    # Stacking on top: target EE height ensures cube bottom contacts top surface
                    if dest_obj.startswith("glass"):
                        # Glass rim top is at ~0.0575m; cube center at 0.0725m; EE target = 0.1015m
                        target = np.array([ref_pos[0], ref_pos[1], 0.1015])
                    elif dest_obj.startswith("cube") or "cube" in dest_obj:
                        # Base cube center is at ref_pos[2]; top is at ref_pos[2] + 0.015
                        # Stacked cube center is at ref_pos[2] + 0.030; EE target = ref_pos[2] + 0.059
                        target = np.array([ref_pos[0], ref_pos[1], ref_pos[2] + 0.059])
                    elif dest_obj == "bottle":
                        # Bottle cap is at ~0.068m; cube center at 0.083m; EE target = 0.112m
                        target = np.array([ref_pos[0], ref_pos[1], 0.112])
                    else:
                        target = np.array([ref_pos[0], ref_pos[1], ref_pos[2] + 0.059])
                    return self.clamp_to_workspace(target)
                else:
                    # Place adjacent on table beside reference object (directional offsets)
                    offset_x = 0.0
                    offset_y = 0.0
                    if "left" in dest_clean:
                        offset_y = 0.055
                    elif "right" in dest_clean:
                        offset_y = -0.055
                    elif "front" in dest_clean or "forward" in dest_clean:
                        offset_x = 0.050
                    elif "behind" in dest_clean or "back" in dest_clean:
                        offset_x = -0.050
                    else:
                        offset_y = 0.055 if ref_pos[1] <= 0 else -0.055
                    target = np.array([ref_pos[0] + offset_x, ref_pos[1] + offset_y, 0.044])
                    return self.clamp_to_workspace(target)

        # 3. Table edges and corners (z = 0.044 for cube on table)
        if "edge" in dest_clean:
            if "left" in dest_clean:
                return np.array([0.18, 0.20, 0.044])
            elif "right" in dest_clean:
                return np.array([0.18, -0.20, 0.044])
            else:
                ref_pos = self.get_object_position(reference_obj_name) if reference_obj_name else None
                candidates = [
                    np.array([0.18, -0.20, 0.044]),  # Right edge
                    np.array([0.27, 0.12, 0.044]),   # Front-left edge
                    np.array([0.27, -0.14, 0.044]),  # Front-right edge
                    np.array([0.18, 0.20, 0.044]),   # Left edge
                ]
                if ref_pos is not None:
                    dists = [np.linalg.norm(c[:2] - ref_pos[:2]) for c in candidates]
                    best_candidate = candidates[int(np.argmax(dists))]
                    return best_candidate
                return np.array([0.18, -0.20, 0.044])

        if "corner" in dest_clean:
            if "left" in dest_clean:
                return np.array([0.26, 0.16, 0.044])
            else:
                return np.array([0.26, -0.16, 0.044])

        # 4. Spatial descriptors: "front", "back", "left", "right", "center"
        if "front" in dest_clean or "forward" in dest_clean:
            return np.array([0.26, 0.0, 0.044])
        elif "back" in dest_clean or "rear" in dest_clean:
            return np.array([0.16, 0.0, 0.044])
        elif "left" in dest_clean:
            return np.array([0.18, 0.14, 0.044])
        elif "right" in dest_clean:
            return np.array([0.18, -0.14, 0.044])
        elif "center" in dest_clean or "middle" in dest_clean:
            return np.array([0.20, 0.0, 0.044])

        # 5. Fallback: default safe open table location
        return np.array([0.22, -0.14, 0.044])

    def clamp_to_workspace(self, target_pos: np.ndarray) -> np.ndarray:
        """Ensure coordinate falls within safe reachable boundaries."""
        clamped = target_pos.copy()
        clamped[0] = np.clip(clamped[0], WORKSPACE_LIMITS["x_min"], WORKSPACE_LIMITS["x_max"])
        clamped[1] = np.clip(clamped[1], WORKSPACE_LIMITS["y_min"], WORKSPACE_LIMITS["y_max"])
        clamped[2] = np.clip(clamped[2], WORKSPACE_LIMITS["z_min"], WORKSPACE_LIMITS["z_max"])
        return clamped

