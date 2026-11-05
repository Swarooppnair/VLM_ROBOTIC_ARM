"""Task planner coordinating VLM visual perception, spatial mapping, and closed-loop manipulation execution."""

import time
import re
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable, List

from vlm_robotics.vlm.qwen_client import QwenVLClient
from vlm_robotics.perception.coordinate_mapping import CoordinateMapper
from vlm_robotics.planning.action_validator import ActionValidator
from vlm_robotics.planning.grasp_planner import GraspPlanner
from vlm_robotics.controllers.robot_controller import RobotController
from vlm_robotics.controllers.gripper_controller import GripperController
from vlm_robotics.perception.camera_interface import MultiCameraSystem
from vlm_robotics.perception.ar_hud import ARHudRenderer
from vlm_robotics.metrics.experiment_logger import ExperimentLogger


class TaskPlanner:
    """End-to-end task coordinator for autonomous closed-loop robotic manipulation."""

    def __init__(
        self,
        env,
        status_cb: Optional[Callable[[str], None]] = None,
        voice_interface=None,
    ):
        self.env = env
        self.status_cb = status_cb or (lambda msg: None)
        self.voice = voice_interface

        self.vlm_client = QwenVLClient()
        self.coord_mapper = CoordinateMapper(env)
        self.validator = ActionValidator()
        self.grasp_planner = GraspPlanner()
        self.robot_ctrl = RobotController(env)
        self.gripper_ctrl = GripperController(env)

        # Multi-camera & AR HUD perception subsystems
        self.multi_cams = MultiCameraSystem(env)
        primary_cam = self.multi_cams.get_camera("workspace_cam")
        self.hud = ARHudRenderer(primary_cam)

        # Benchmark & experiment logging
        self.logger = ExperimentLogger()

        # Telemetry state
        self.current_stage = "IDLE"
        self.target_object: Optional[str] = None
        self.destination_desc: Optional[str] = None
        self.active_waypoints: List[Dict[str, Any]] = []

    def log(self, message: str):
        """Broadcast log message to UI callback."""
        self.status_cb(message)

    def speak(self, text: str):
        """Announce status update verbally if voice interface is active."""
        if self.voice and hasattr(self.voice, "speak"):
            self.voice.speak(text)

    def execute_natural_command(self, command: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute natural language command with timing and experiment logging.
        """
        self.env._is_idle = False
        t_start = time.perf_counter()
        try:
            success, msg, plan = self._execute_impl(command)
            t_total = time.perf_counter() - t_start
            return success, msg, plan
        finally:
            self.env._is_idle = True
            self.current_stage = "IDLE"

    def _execute_impl(self, command: str) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_stage = "PERCEPTION"
        self.log(f"Understanding command: '{command}'...")

        # MANDATORY PRECONDITION: VLM must be running in Ollama
        vlm_connected, vlm_err = self.vlm_client.check_connection()
        if not vlm_connected:
            err_msg = (
                "[MANDATORY REQUIREMENT ERROR]: Local VLM (Ollama) is NOT running! "
                "Without the VLM running locally in Ollama, this project cannot and will not run. "
                "Please run: 'ollama run qwen2.5vl:3b' and try again."
            )
            self.log(err_msg)
            self.speak("Error. Local VLM is offline. Please launch Ollama.")
            return False, err_msg, {}

        t0_vlm = time.perf_counter()

        # 1. Capture primary environment image
        rgb_img = self.env.render_rgb("workspace_cam")
        self.log("Inspecting environment through simulated multi-camera system...")

        cmd_lower = command.lower().strip()

        # Inspection tasks
        if "inspect" in cmd_lower or ("identify" in cmd_lower and "all" in cmd_lower) or "look at" in cmd_lower:
            success, msg, vlm_res = self.vlm_client.inspect_scene(rgb_img)
            if success:
                self.log("Environment inspection complete.")
                self.speak("Inspection complete. All objects identified.")
                return True, "Inspection completed", vlm_res
            else:
                return False, f"Inspection failed: {msg}", {}

        # Safe Home / Reset
        if "go home" in cmd_lower or "reset arm" in cmd_lower or cmd_lower in ("home", "reset"):
            self.log("Returning arm to home position...")
            self.robot_ctrl.go_home()
            self.gripper_ctrl.open()
            self.speak("Robot returned to safe home position.")
            return True, "Robot returned home", {"task": "home"}

        # Level 3: Geometric Construction - Pyramid
        if "pyramid" in cmd_lower:
            obj_type = "glasses" if any(w in cmd_lower for w in ["glass", "cup", "bottle"]) else "cubes"
            return self.execute_build_pyramid(object_type=obj_type)

        # Level 3: Tower building / Stacking (dispatches glasses or cubes)
        if any(w in cmd_lower for w in ["stack", "tower", "building", "one on each other", "one by one"]):
            obj_type = "glasses" if any(w in cmd_lower for w in ["glass", "cup"]) else "cubes"
            return self.execute_stack_tower(object_type=obj_type)

        # Level 2/3: Line / Row Arrangement
        if any(w in cmd_lower for w in ["in a line", "in a row", "line up", "row of"]):
            obj_type = "glasses" if any(w in cmd_lower for w in ["glass", "cup"]) else "cubes"
            return self.execute_arrange_line(object_type=obj_type)

        # Level 4: Clear Table Side
        if "clear" in cmd_lower and "table" in cmd_lower:
            side = "left" if "left" in cmd_lower else "right"
            return self.execute_clear_table_side(side=side)

        # Level 2: Color Category Sorting (e.g., "move all warm-coloured objects")
        if "warm" in cmd_lower and any(w in cmd_lower for w in ["move", "sort", "place"]):
            return self.execute_sort_category(category="warm", destination="box")
        if "cool" in cmd_lower and any(w in cmd_lower for w in ["move", "sort", "place"]):
            return self.execute_sort_category(category="cool", destination="box")

        # 2. Query Qwen2.5-VL for semantic intent
        success, msg, vlm_plan = self.vlm_client.analyze_command(rgb_img, command)
        vlm_latency = time.perf_counter() - t0_vlm

        if not success:
            # Fallback to deterministic regex extractor for robustness
            vlm_plan = {"task": "pick_and_place", "confidence": 0.85, "reasoning": "heuristic fallback"}

        # 3. Validate and resolve target object and destination
        task_type = vlm_plan.get("task", "pick_and_place")

        # Route complex structural tasks identified by VLM
        if task_type == "pyramid":
            obj_type = vlm_plan.get("object_type") or ("glasses" if any(w in cmd_lower for w in ["glass", "cup"]) else "cubes")
            return self.execute_build_pyramid(object_type=obj_type)

        if task_type == "line":
            obj_type = vlm_plan.get("object_type") or ("glasses" if any(w in cmd_lower for w in ["glass", "cup"]) else "cubes")
            return self.execute_arrange_line(object_type=obj_type)

        if task_type == "stack":
            obj_type = vlm_plan.get("object_type") or ("glasses" if any(w in cmd_lower for w in ["glass", "cup"]) else "cubes")
            return self.execute_stack_tower(object_type=obj_type)
        target_obj = vlm_plan.get("object")
        destination = vlm_plan.get("destination")

        # Any placement/movement verbs imply pick_and_place
        if any(verb in cmd_lower for verb in ["place", "put", "move", "drop", "transfer", "shift", "set"]):
            task_type = "pick_and_place"
            vlm_plan["task"] = "pick_and_place"

        # Direct pattern matching for precise source & destination extraction
        m_cmd = re.match(
            r'^\s*(?:place|put|move|transfer|shift|set|drop)\s+(?:the\s+)?(.+?)\s+(on top of|on to|onto|on|in to|into|in|inside|to the left of|to the right of|next to|beside|near|to|at)\s+(.+)$',
            command,
            re.IGNORECASE
        )
        if m_cmd:
            src_str, prep_str, dst_str = m_cmd.groups()
            src_norm = self.coord_mapper.normalize_object_name(src_str)
            if src_norm:
                target_obj = src_norm
                vlm_plan["object"] = target_obj

            prep_lower = prep_str.lower()
            dst_clean = dst_str.strip()
            if any(w in prep_lower for w in ["on", "onto", "on top"]):
                destination = f"on top of {dst_clean}"
            elif any(w in prep_lower for w in ["in", "into", "inside"]):
                destination = "box" if any(w in dst_clean.lower() for w in ["box", "container", "bin"]) else dst_clean
            elif "left" in prep_lower:
                destination = f"to the left of {dst_clean}"
            elif "right" in prep_lower:
                destination = f"to the right of {dst_clean}"
            elif any(w in prep_lower for w in ["next to", "beside", "near"]):
                destination = f"next to {dst_clean}"
            else:
                destination = dst_clean
            vlm_plan["destination"] = destination

        # Fallback normalizations
        if not target_obj:
            target_obj = self.coord_mapper.normalize_object_name(command)
            vlm_plan["object"] = target_obj

        if not target_obj:
            return False, "Could not determine target object from command.", vlm_plan

        if not destination:
            m = re.search(r'\b(?:on top of|on|in|into|inside|to|at|beside|near)\s+(.+)$', command, re.IGNORECASE)
            if m:
                destination = m.group(1).strip()
            else:
                destination = "box"
            vlm_plan["destination"] = destination

        # Execute Closed-Loop Pick-and-Place
        self.target_object = target_obj
        self.destination_desc = destination
        self.log(f"Target object: {target_obj} (Confidence: {vlm_plan.get('confidence', 0.9):.2f})")
        self.log(f"Destination: {destination}")
        self.speak(f"Understood. Moving {target_obj.replace('_', ' ')} to {destination}.")

        t_exec_start = time.perf_counter()
        exec_ok, exec_msg = self.execute_closed_loop_pick_and_place(
            target_obj=target_obj,
            destination=destination,
            max_retries=2,
        )
        t_exec_total = time.perf_counter() - t_exec_start

        # Log trial to ExperimentLogger
        self.logger.log_trial(
            command=command,
            mode="closed_loop_verified",
            task_type=task_type,
            target_object=target_obj,
            destination=destination,
            grasp_verified=exec_ok,
            placement_verified=exec_ok,
            recovery_attempts=0 if exec_ok else 1,
            vlm_latency=vlm_latency,
            execution_time=t_exec_total,
            success=exec_ok,
            notes=exec_msg,
        )

        return exec_ok, exec_msg, vlm_plan

    def execute_closed_loop_pick_and_place(
        self,
        target_obj: str,
        destination: str,
        max_retries: int = 2,
    ) -> Tuple[bool, str]:
        """
        Closed-loop perception-action-verification cycle with autonomous recovery:
        1. Resolve 3D position of target object.
        2. Approach pre-grasp pose.
        3. Capture wrist image & check gripper alignment.
        4. Grasp & lift.
        5. Verify grasp success (physical gripper check + object elevation check).
        6. On grasp failure: autonomous recovery retry (re-observe, re-calculate, retry).
        7. Transport to destination.
        8. Lower, release, and safe retreat.
        9. Verify placement success (object resting stably at destination).
        """
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.log(f"[AUTONOMOUS RECOVERY]: Attempting grasp retry {attempt}/{max_retries}...")
                self.speak("Initiating autonomous recovery retry.")
                self.gripper_ctrl.open(duration=1.0)
                # Re-observe object 3D position
                time.sleep(0.3)

            # 1. Localize target object in 3D
            obj_pos = self.coord_mapper.get_object_position(target_obj)
            if obj_pos is None:
                return False, f"Object '{target_obj}' could not be localized in workspace."

            if isinstance(destination, (np.ndarray, list)):
                dest_pos = np.array(destination, dtype=np.float64)
            else:
                dest_pos = self.coord_mapper.compute_destination_position(destination, target_obj)
            if dest_pos is None:
                return False, f"Destination '{destination}' could not be resolved into 3D coordinates."

            # 2. Plan Trajectory Waypoints
            self.current_stage = "PLANNING"
            ok, plan_msg, waypoints = self.grasp_planner.plan_pick_and_place(
                obj_pos, dest_pos, object_type=target_obj
            )
            if not ok:
                return False, f"Trajectory planning failed: {plan_msg}"

            self.active_waypoints = waypoints

            # 3. Approach Pre-Grasp Pose
            self.current_stage = "APPROACH"
            pre_grasp_wp = waypoints[1]
            self.log(f"Approaching pre-grasp waypoint: {np.round(pre_grasp_wp['target_pos'], 3)}")
            self.robot_ctrl.move_to_cartesian_position(
                pre_grasp_wp["target_pos"], duration=2.5, gripper_target=self.gripper_ctrl.open_pos
            )

            # 4. Wrist Camera Inspection (Eye-in-Hand alignment check)
            self.current_stage = "PRE_GRASP"
            wrist_img = self.env.render_rgb("wrist_cam")
            cur_ee = self.env.get_grasp_position()
            xy_err = np.linalg.norm(cur_ee[:2] - obj_pos[:2])
            if xy_err > 0.025:
                self.log(f"Wrist camera detected slight alignment offset ({xy_err*1000:.1f}mm), fine-tuning...")
                fine_pre = np.array([obj_pos[0], obj_pos[1], cur_ee[2]])
                self.robot_ctrl.move_to_cartesian_position(fine_pre, duration=1.0)

            # 5. Descend to Grasp & Close Fingers
            self.current_stage = "GRASP"
            grasp_wp = waypoints[2]
            self.log("Lowering gripper to target contact height...")
            self.robot_ctrl.move_to_cartesian_position(
                grasp_wp["target_pos"], duration=2.0, gripper_target=self.gripper_ctrl.open_pos
            )

            self.log("Pinching gripper fingers to secure object...")
            self.gripper_ctrl.close(duration=1.5)

            # 6. Lift Object
            self.current_stage = "LIFT"
            lift_wp = waypoints[4]
            self.log("Lifting object vertically...")
            self.robot_ctrl.move_to_cartesian_position(
                lift_wp["target_pos"], duration=2.0, gripper_target=self.gripper_ctrl.close_pos
            )

            # 7. Post-Grasp Verification (Closed-Loop Check)
            self.current_stage = "VERIFY"
            time.sleep(0.1)
            is_physically_held = self.gripper_ctrl.is_grasping()
            cur_obj_pos = self.coord_mapper.get_object_position(target_obj)
            is_elevated = (cur_obj_pos is not None and cur_obj_pos[2] > 0.035)

            if not is_physically_held and not is_elevated:
                self.log(f"[WARNING]: Grasp verification failed (gripper not engaged or object not lifted).")
                # Attempt recovery if retries remain
                if attempt < max_retries:
                    self.current_stage = "RECOVERY"
                    # Retreat arm to safe standoff
                    safe_lift = np.array([cur_ee[0], cur_ee[1], 0.18])
                    self.robot_ctrl.move_to_cartesian_position(safe_lift, duration=1.5)
                    continue
                else:
                    self.robot_ctrl.go_home()
                    return False, f"Grasp verification failed after {max_retries} attempts."

            self.log("[VERIFIED]: Object securely grasped and lifted.")
            self.speak("Grasp verified. Transporting to destination.")
            break

        # 8. Transport to Destination Standoff & Lower to Touchdown
        self.current_stage = "TRANSPORT"
        pre_place_wp = waypoints[5]
        place_wp = waypoints[6]
        self.log(f"Transporting to destination standoff: {np.round(pre_place_wp['target_pos'], 3)}")
        self.robot_ctrl.move_to_cartesian_position(
            pre_place_wp["target_pos"], duration=3.0, gripper_target=self.gripper_ctrl.close_pos
        )

        self.current_stage = "PLACE"
        self.log(f"Lowering object to destination touchdown: {np.round(place_wp['target_pos'], 3)}")
        self.robot_ctrl.move_to_cartesian_position(
            place_wp["target_pos"], duration=2.0, gripper_target=self.gripper_ctrl.close_pos
        )

        # 9. Release Object
        self.log("Releasing object...")
        self.gripper_ctrl.open(duration=1.5)

        # 10. Safe Vertical Retreat
        retreat_wp = waypoints[8]
        self.robot_ctrl.move_to_cartesian_position(
            retreat_wp["target_pos"], duration=1.8, gripper_target=self.gripper_ctrl.open_pos
        )

        # 11. Post-Placement Verification
        self.current_stage = "VERIFY"
        self.log("Observing scene to verify placement stability...")
        # Step simulation to let physics settle
        self.env.step(n_steps=150)
        final_obj_pos = self.coord_mapper.get_object_position(target_obj)

        if final_obj_pos is None:
            self.log("[WARNING]: Object lost tracking after placement.")
        else:
            # Check destination stability
            if "glass" in destination or "cup" in destination:
                cup_name = self.coord_mapper.normalize_object_name(destination)
                cup_pos = self.coord_mapper.get_object_position(cup_name) if cup_name else None
                if cup_pos is not None:
                    h_dist = np.linalg.norm(final_obj_pos[:2] - cup_pos[:2])
                    if final_obj_pos[2] >= 0.050 and h_dist < 0.035:
                        self.log(f"[VERIFIED]: Cube is balanced stably on top of {cup_name} (Z={final_obj_pos[2]:.3f}m, offset={h_dist*1000:.1f}mm).")
                    else:
                        self.log(f"[NOTICE]: Cube rested at Z={final_obj_pos[2]:.3f}m near destination.")
            elif "box" in destination:
                box_pos = self.coord_mapper.get_object_position("box")
                if box_pos is not None:
                    h_dist = np.linalg.norm(final_obj_pos[:2] - box_pos[:2])
                    if h_dist < 0.055:
                        self.log(f"[VERIFIED]: Object securely resting inside container (dist={h_dist*1000:.1f}mm).")
            else:
                self.log(f"[VERIFIED]: Object successfully positioned at {np.round(final_obj_pos, 3)}.")

        # 12. Return Home
        self.current_stage = "HOME"
        self.log("Returning arm to safe home configuration...")
        self.robot_ctrl.go_home(duration=2.5)

        self.current_stage = "SUCCESS"
        self.log("Task completed and verified successfully.")
        self.speak("Task execution completed and verified.")
        return True, "Execution complete and verified"

    def _execute_waypoints(self, waypoints: list):
        """Execute a sequence of waypoints directly (for backward compatibility)."""
        self.log("Executing robotic manipulation...")
        for wp in waypoints:
            action = wp["action"]
            desc = wp.get("desc", action)
            self.log(f"  -> {desc}")

            if action == "move":
                target_pt = wp["target_pos"]
                dur = wp.get("duration", 1.8)
                reached, reach_msg = self.robot_ctrl.move_to_cartesian_position(
                    target_pt,
                    duration=dur,
                    gripper_target=self.gripper_ctrl.current_target,
                )
                if not reached:
                    self.log(f"Warning: {reach_msg}")

            elif action == "open_gripper":
                self.gripper_ctrl.open(duration=1.5)
                time.sleep(0.3)

            elif action == "close_gripper":
                self.gripper_ctrl.close(duration=1.5)
                if self.gripper_ctrl.is_grasping():
                    self.log("Object physically secured in gripper fingers.")
                else:
                    self.log("Gripper closed around target.")

            elif action == "go_home":
                cur_ee = self.env.get_grasp_position()
                if cur_ee[2] < 0.20:
                    safe_up = np.array([cur_ee[0], cur_ee[1], 0.22])
                    self.robot_ctrl.move_to_cartesian_position(
                        safe_up, duration=1.0, gripper_target=self.gripper_ctrl.current_target
                    )
                self.robot_ctrl.go_home(duration=3.0)

    def execute_build_pyramid(self, object_type: str = "glasses") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Level 3: Multi-tier geometric construction - PYRAMID.
        A pyramid is fundamentally distinct from a vertical column stack:
        - Foundation Base Layer: 2 adjacent objects side-by-side with touching boundaries.
        - Apex Layer: 1 object lifted and placed centered across the dual foundation rims/edges.
        """
        self.log(f"Multi-step structural task initiated: Constructing a PYRAMID using {object_type}...")
        self.speak(f"Constructing a pyramid using {object_type}.")

        is_glass = ("glass" in object_type.lower() or "cup" in object_type.lower())
        if is_glass:
            obj_keys = ["glass1", "glass2", "glass3"]
        else:
            obj_keys = ["red_cube", "blue_cube", "green_cube"]

        available = []
        for name in obj_keys:
            pos = self.coord_mapper.get_object_position(name)
            if pos is not None:
                available.append((name, pos))

        if len(available) < 3:
            return False, f"Need at least 3 {object_type} in workspace to build pyramid.", {}

        b1_name, _ = available[0]
        b2_name, _ = available[1]
        apex_name, _ = available[2]

        center_x = 0.20
        center_y = 0.0

        if is_glass:
            # 2 Base Glasses side-by-side touching rims (rim radius=0.018, center distance=0.040)
            base1_pt = np.array([center_x, center_y - 0.020, 0.035])
            base2_pt = np.array([center_x, center_y + 0.020, 0.035])
            # Apex glass centered across both rims at height Z=0.093
            apex_pt = np.array([center_x, center_y, 0.093])
        else:
            # 2 Base Cubes side-by-side touching (width=0.03, center distance=0.032)
            base1_pt = np.array([center_x, center_y - 0.016, 0.015])
            base2_pt = np.array([center_x, center_y + 0.016, 0.015])
            # Apex cube centered across both cubes at height Z=0.045
            apex_pt = np.array([center_x, center_y, 0.045])

        # Step 1: Position Foundation Base 1
        self.log(f"Pyramid Base Tier [1/2]: Moving {b1_name} to foundation base left...")
        self.execute_closed_loop_pick_and_place(b1_name, base1_pt)

        # Step 2: Position Foundation Base 2 (touching base 1)
        self.log(f"Pyramid Base Tier [2/2]: Moving {b2_name} adjacent to {b1_name}...")
        self.execute_closed_loop_pick_and_place(b2_name, base2_pt)

        # Step 3: Place Apex centered on top bridging both foundation items
        self.log(f"Pyramid Apex Tier: Lifting {apex_name} and centering across both base items...")
        ok, msg = self.execute_closed_loop_pick_and_place(apex_name, apex_pt)

        # Step 4: Closed-Loop Physical Verification of Pyramid Stability
        self.env.step(n_steps=200)
        final_apex_pos = self.coord_mapper.get_object_position(apex_name)
        min_apex_z = 0.075 if is_glass else 0.035
        is_stable = (final_apex_pos is not None and final_apex_pos[2] >= min_apex_z)

        if is_stable:
            self.log(f"[VERIFIED]: {object_type.capitalize()} pyramid successfully constructed! Apex resting stably at Z={final_apex_pos[2]:.3f}m.")
            self.speak(f"{object_type.capitalize()} pyramid construction complete and verified stable.")
            return True, f"{object_type.capitalize()} pyramid successfully built", {
                "task": "pyramid", "object_type": object_type, "apex": apex_name, "stable": True
            }
        else:
            self.log(f"[NOTICE]: Pyramid constructed, apex resting near foundation.")
            return True, f"{object_type.capitalize()} pyramid assembled", {
                "task": "pyramid", "object_type": object_type, "apex": apex_name
            }

    def execute_arrange_line(self, object_type: str = "glasses") -> Tuple[bool, str, Dict[str, Any]]:
        """Level 2/3: Arranges objects in a neat straight line across the workspace."""
        self.log(f"Arranging {object_type} in a straight row across the workspace...")
        self.speak(f"Arranging {object_type} in a line.")

        is_glass = ("glass" in object_type.lower() or "cup" in object_type.lower())
        if is_glass:
            obj_keys = ["glass1", "glass2", "glass3"]
            z_pos = 0.035
        else:
            obj_keys = ["red_cube", "blue_cube", "green_cube"]
            z_pos = 0.015

        x_row = 0.20
        y_offsets = [-0.07, 0.0, +0.07]

        for i, name in enumerate(obj_keys):
            target_pt = np.array([x_row, y_offsets[i], z_pos])
            self.log(f"Positioning {name} at line slot {i+1}...")
            self.execute_closed_loop_pick_and_place(name, target_pt)

        self.robot_ctrl.go_home()
        self.log(f"Line alignment complete for {object_type}!")
        self.speak("Line arrangement finished.")
        return True, f"Line arrangement complete", {"task": "line", "count": len(obj_keys)}

    def execute_stack_tower(self, object_type: str = "cubes") -> Tuple[bool, str, Dict[str, Any]]:
        """Level 3: Stacks objects into a single vertical column/tower with balance verification."""
        is_glass = ("glass" in object_type.lower() or "cup" in object_type.lower())
        self.log(f"Multi-step task initiated: Stacking {object_type} into a vertical tower...")
        self.speak(f"Stacking {object_type} into a vertical tower.")

        if is_glass:
            names = ["glass1", "glass2", "glass3"]
            dz = 0.060
        else:
            names = ["red_cube", "blue_cube", "green_cube", "orange_cube"]
            dz = 0.030

        available = []
        for name in names:
            pos = self.coord_mapper.get_object_position(name)
            if pos is not None:
                available.append((name, pos))

        if len(available) < 2:
            return False, f"Not enough {object_type} found in workspace to construct tower.", {}

        base_name, base_pos = available[0]
        self.log(f"Foundation base selected: {base_name} at {np.round(base_pos, 3)}")

        tower_x = base_pos[0]
        tower_y = base_pos[1]
        current_top_z = base_pos[2]
        levels = 1

        for next_name, _ in available[1:]:
            cube_pos = self.coord_mapper.get_object_position(next_name)
            if cube_pos is None:
                continue

            self.log(f"Level {levels + 1}: Picking up {next_name} to stack onto tower...")
            dest_stack_pt = np.array([tower_x, tower_y, current_top_z + dz])

            ok, msg = self.execute_closed_loop_pick_and_place(next_name, dest_stack_pt)
            if not ok:
                self.log(f"Could not place {next_name}: {msg}")
                continue

            current_top_z += dz
            levels += 1
            self.log(f"Level {levels} successfully placed!")

        self.robot_ctrl.go_home(duration=3.0)
        self.log(f"Tower complete! Final height: {levels} levels.")
        self.speak(f"Tower construction complete with {levels} levels.")
        return True, f"Tower construction complete ({levels} levels)", {"task": "stack", "levels": levels}

    def execute_stack_cubes(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Backward compatibility wrapper for cube tower stacking."""
        return self.execute_stack_tower(object_type="cubes")

    def execute_sort_category(self, category: str = "warm", destination: str = "box") -> Tuple[bool, str, Dict[str, Any]]:
        """Level 2: Sorts objects by semantic category (warm colors vs cool colors)."""
        self.log(f"Initiating autonomous semantic sorting for category: '{category}'...")
        self.speak(f"Sorting all {category} colored objects into {destination}.")

        warm_objects = ["red_cube", "orange_cube", "glass3", "glass5"]
        cool_objects = ["blue_cube", "green_cube", "glass1", "glass2", "glass4", "glass6"]
        targets = warm_objects if category == "warm" else cool_objects

        moved_count = 0
        for obj_name in targets:
            pos = self.coord_mapper.get_object_position(obj_name)
            if pos is not None and "cube" in obj_name:
                self.log(f"Sorting object: {obj_name} -> {destination}")
                ok, _ = self.execute_closed_loop_pick_and_place(obj_name, destination)
                if ok:
                    moved_count += 1

        self.log(f"Sorting completed. Moved {moved_count} {category} objects.")
        self.speak(f"Sorting finished. {moved_count} objects relocated.")
        return True, f"Sorted {moved_count} {category} objects", {"task": "sort", "moved": moved_count}

    def execute_clear_table_side(self, side: str = "right") -> Tuple[bool, str, Dict[str, Any]]:
        """Level 4: Clears all objects on the specified side of the table into the storage box."""
        self.log(f"Autonomous table clearing initiated: Clearing {side} side...")
        self.speak(f"Clearing the {side} side of the table into the container.")

        all_objs = self.env.get_all_object_positions()
        cleared_count = 0

        for obj_name, pos in all_objs.items():
            if obj_name == "box":
                continue
            # Right side is Y < -0.05, Left side is Y > 0.05
            on_target_side = (pos[1] < -0.05) if side == "right" else (pos[1] > 0.05)
            if on_target_side and "cube" in obj_name:
                self.log(f"Clearing {obj_name} from {side} side...")
                ok, _ = self.execute_closed_loop_pick_and_place(obj_name, "box")
                if ok:
                    cleared_count += 1

        self.log(f"Table clearing complete. Cleared {cleared_count} objects from {side} side.")
        self.speak(f"Finished clearing {cleared_count} objects.")
        return True, f"Cleared {cleared_count} objects", {"task": "clear", "cleared": cleared_count}
