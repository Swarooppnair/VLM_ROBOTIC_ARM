"""Task planner coordinating VLM visual perception, spatial mapping, and manipulation execution."""

import time
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable

from vlm_robotics.vlm.qwen_client import QwenVLClient
from vlm_robotics.perception.coordinate_mapping import CoordinateMapper
from vlm_robotics.planning.action_validator import ActionValidator
from vlm_robotics.planning.grasp_planner import GraspPlanner
from vlm_robotics.controllers.robot_controller import RobotController
from vlm_robotics.controllers.gripper_controller import GripperController


class TaskPlanner:
    """End-to-end task coordinator for natural-language robotic manipulation."""

    def __init__(
        self,
        env,
        status_cb: Optional[Callable[[str], None]] = None,
    ):
        self.env = env
        self.status_cb = status_cb or (lambda msg: None)

        self.vlm_client = QwenVLClient()
        self.coord_mapper = CoordinateMapper(env)
        self.validator = ActionValidator()
        self.grasp_planner = GraspPlanner()
        self.robot_ctrl = RobotController(env)
        self.gripper_ctrl = GripperController(env)

    def log(self, message: str):
        self.status_cb(message)

    def execute_natural_command(self, command: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Full perception-to-action pipeline:
        1. Capture workspace RGB image.
        2. Query local Qwen2.5-VL for structured semantic intent.
        3. Validate VLM plan.
        4. Map objects to physical 3D workspace coordinates.
        5. Plan trajectory waypoints.
        6. Execute pick-and-place with OpenMANIPULATOR-X.
        """
        self.env._is_idle = False
        try:
            return self._execute_impl(command)
        finally:
            self.env._is_idle = True

    def _execute_impl(self, command: str) -> Tuple[bool, str, Dict[str, Any]]:
        self.log(f"Understanding command: '{command}'...")

        # 1. Capture image
        rgb_img = self.env.render_rgb()
        self.log("Inspecting environment with camera...")

        # Quick check for pure inspection command
        cmd_lower = command.lower()
        if "inspect" in cmd_lower or ("identify" in cmd_lower and "all" in cmd_lower) or "look at" in cmd_lower:
            success, msg, vlm_res = self.vlm_client.inspect_scene(rgb_img)
            if success:
                self.log("Environment inspection complete.")
                return True, "Inspection completed", vlm_res
            else:
                return False, f"Inspection failed: {msg}", {}

        # Home / reset command
        if "go home" in cmd_lower or "reset arm" in cmd_lower or cmd_lower.strip() in ("home", "reset"):
            self.log("Returning arm to home position...")
            self.robot_ctrl.go_home()
            self.gripper_ctrl.open()
            return True, "Robot returned home", {"task": "home"}

        # 2. Query Qwen2.5-VL
        success, msg, vlm_plan = self.vlm_client.analyze_command(rgb_img, command)
        if not success:
            return False, f"VLM reasoning failed: {msg}", {}

        # 3. Validate plan schema
        valid, val_msg = self.validator.validate_vlm_plan(vlm_plan)
        if not valid:
            return False, f"Safety check failed on VLM plan: {val_msg}", vlm_plan

        task_type = vlm_plan["task"]

        # Handle building / stacking cubes
        if task_type == "stack" or any(w in cmd_lower for w in ["stack", "building", "tower", "one on each other", "one by one"]):
            return self.execute_stack_cubes()

        target_obj = vlm_plan.get("object")
        destination = vlm_plan.get("destination")

        # Any command containing placement/movement verbs implies pick_and_place
        if any(verb in cmd_lower for verb in ["place", "put", "move", "drop", "transfer", "shift", "set"]):
            task_type = "pick_and_place"
            vlm_plan["task"] = "pick_and_place"

        # Parse semantic source object and destination directly from natural command
        import re
        cmd_matched = False
        m_cmd = re.match(
            r'^\s*(?:place|put|move|transfer|shift|set|drop)\s+(?:the\s+)?(.+?)\s+(on top of|on to|onto|on|in to|into|in|inside|next to|beside|near|to|at)\s+(.+)$',
            command,
            re.IGNORECASE
        )
        if m_cmd:
            src_str, prep_str, dst_str = m_cmd.groups()
            src_norm = self.coord_mapper.normalize_object_name(src_str)
            if src_norm:
                target_obj = src_norm
                vlm_plan["object"] = target_obj
                cmd_matched = True

            prep_lower = prep_str.lower()
            dst_clean = dst_str.strip()
            if any(w in prep_lower for w in ["on", "onto", "on top"]):
                destination = f"on top of {dst_clean}"
            elif any(w in prep_lower for w in ["in", "into", "inside"]):
                destination = "box" if any(w in dst_clean.lower() for w in ["box", "container", "bin"]) else dst_clean
            elif any(w in prep_lower for w in ["next to", "beside", "near"]):
                destination = f"next to {dst_clean}"
            else:
                destination = dst_clean
            vlm_plan["destination"] = destination
            task_type = "pick_and_place"
            vlm_plan["task"] = "pick_and_place"

        if not cmd_matched:
            # Fallback to general object extraction
            cmd_obj = self.coord_mapper.normalize_object_name(command)
            if cmd_obj:
                target_obj = cmd_obj
                vlm_plan["object"] = target_obj
            elif not target_obj:
                target_obj = self.coord_mapper.normalize_object_name(command)
                vlm_plan["object"] = target_obj

        # If user explicitly said "on <glass/cube>", ensure destination doesn't get downgraded by VLM
        if destination and any(w in cmd_lower for w in ["on ", "on top", "onto"]):
            if "next to" in destination or "next_to" in destination:
                destination = destination.replace("next to", "on top of").replace("next_to", "on top of")
                vlm_plan["destination"] = destination

        # Auto-extract destination clause from user command if still missing
        if not destination:
            m = re.search(r'\b(?:on top of|on|in|into|inside|to|at|beside|near)\s+(.+)$', command, re.IGNORECASE)
            if m:
                destination = m.group(1).strip()
                vlm_plan["destination"] = destination
                task_type = "pick_and_place"
                vlm_plan["task"] = "pick_and_place"

        self.log(f"Object identified: {target_obj} (Confidence: {vlm_plan['confidence']:.2f})")
        if destination:
            self.log(f"Destination identified: {destination}")

        # 4. Resolve 3D Coordinates
        obj_pos = self.coord_mapper.get_object_position(target_obj)
        if obj_pos is None:
            return False, f"Could not localize object '{target_obj}' in workspace.", vlm_plan

        self.log(f"Target located at 3D pos: {np.round(obj_pos, 3)}")

        # 5. Plan Waypoints
        if task_type == "pick_and_place":
            if not destination:
                destination = "box"  # Default fallback if unspecified
            dest_pos = self.coord_mapper.compute_destination_position(destination, target_obj)
            if dest_pos is None:
                return False, f"Could not determine valid 3D coordinates for destination '{destination}'.", vlm_plan

            self.log(f"Destination located at 3D pos: {np.round(dest_pos, 3)}")
            self.log("Calculating movement trajectory...")
            ok, plan_msg, waypoints = self.grasp_planner.plan_pick_and_place(obj_pos, dest_pos, object_type=target_obj)
            if not ok:
                return False, f"Trajectory planning error: {plan_msg}", vlm_plan

        elif task_type == "pick":
            self.log("Calculating pick trajectory...")
            ok, plan_msg, waypoints = self.grasp_planner.plan_pick_only(obj_pos, object_type=target_obj)
            if not ok:
                return False, f"Trajectory planning error: {plan_msg}", vlm_plan

        else:
            return True, f"Handled task: {task_type}", vlm_plan

        # 6. Execute Motion Waypoints
        self._execute_waypoints(waypoints)
        self.log("Task completed successfully.")
        return True, "Execution complete", vlm_plan

    def _execute_waypoints(self, waypoints: list):
        """Execute a sequence of waypoints with calm, controlled velocity."""
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

            elif action == "go_home":
                # Ensure arm lifts above any workspace objects before swinging home
                cur_ee = self.env.get_grasp_position()
                if cur_ee[2] < 0.20:
                    safe_up = np.array([cur_ee[0], cur_ee[1], 0.22])
                    self.robot_ctrl.move_to_cartesian_position(
                        safe_up, duration=1.0, gripper_target=self.gripper_ctrl.current_target
                    )
                self.robot_ctrl.go_home(duration=3.0)

    def execute_stack_cubes(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Multi-step planner: stacks tabletop cubes on top of each other to build a tower."""
        self.log("Multi-step task initiated: Stacking cubes into a building...")

        cube_names = ["red_cube", "blue_cube", "green_cube", "orange_cube"]
        available = []
        for name in cube_names:
            pos = self.coord_mapper.get_object_position(name)
            if pos is not None:
                available.append((name, pos))

        if len(available) < 2:
            return False, "Not enough cubes found in workspace to construct building.", {}

        # First cube serves as the ground foundation
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

            self.log(f"Level {levels + 1}: Picking up {next_name} to stack on top of {base_name}...")
            # Target stack position on top of the current tower height
            dest_stack_pt = np.array([tower_x, tower_y, current_top_z + 0.030])

            ok, msg, waypoints = self.grasp_planner.plan_pick_and_place(
                cube_pos, dest_stack_pt, object_type="cube"
            )
            if not ok:
                self.log(f"Could not plan trajectory for {next_name}: {msg}")
                continue

            self._execute_waypoints(waypoints)
            current_top_z += 0.030
            levels += 1
            self.log(f"Level {levels} successfully added to building!")

        self.robot_ctrl.go_home(duration=3.0)
        self.log(f"Building construction complete! Final height: {levels} stories ({levels*3} cm).")
        return True, f"Tower construction complete ({levels} levels)", {"task": "stack", "levels": levels}

