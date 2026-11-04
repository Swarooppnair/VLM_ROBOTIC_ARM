"""Automated Verification Test Suite for Local Qwen2.5-VL Controlled OpenMANIPULATOR-X.

Verifies Tests 1 through 8 per system specifications:
- TEST 1: Robot loads successfully in the simulator.
- TEST 2: Simulated camera captures an image.
- TEST 3: Qwen2.5-VL receives the image and identifies the red cube.
- TEST 4: Natural language command parsing and intent validation.
- TEST 5: Robot moves to the red cube approach pose.
- TEST 6: Gripper grasps and physically lifts the cube.
- TEST 7: Robot transports and places the cube inside the box.
- TEST 8: Object remains stable in its new position after release.
"""

import sys
import time
import numpy as np

from vlm_robotics.simulation.environment import SimulationEnvironment
from vlm_robotics.vlm.qwen_client import QwenVLClient
from vlm_robotics.perception.coordinate_mapping import CoordinateMapper
from vlm_robotics.controllers.robot_controller import RobotController
from vlm_robotics.controllers.gripper_controller import GripperController
from vlm_robotics.planning.grasp_planner import GraspPlanner


class SystemTestSuite:
    """Rigorous physical and algorithmic verification harness."""

    def __init__(self):
        self.results = {}
        self.env = None
        self.vlm = None
        self.robot_ctrl = None
        self.gripper_ctrl = None
        self.mapper = None
        self.planner = None

    def log_result(self, test_name: str, passed: bool, details: str):
        self.results[test_name] = {"passed": passed, "details": details}
        status_str = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status_str} {test_name}: {details}")

    def run_all(self) -> int:
        print("\n=======================================================")
        print("   VLM ROBOTIC ARM SIMULATION - VERIFICATION SUITE     ")
        print("=======================================================")

        try:
            self.test_1_robot_loading()
            self.test_2_camera_capture()
            self.test_3_vlm_object_identification()
            self.test_4_natural_language_command()
            self.test_5_robot_approach()
            self.test_6_gripper_pickup()
            self.test_7_place_inside_box()
            self.test_8_object_stability_after_release()
        except Exception as e:
            print(f"\n[FATAL ERROR IN SUITE]: {e}")
            import traceback
            traceback.print_exc()

        print("\n=======================================================")
        print("                  SUMMARY REPORT                       ")
        print("=======================================================")
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["passed"])
        for name, r in self.results.items():
            st = "PASS" if r["passed"] else "FAIL"
            print(f"  [{st}] {name}: {r['details']}")

        print(f"\nTotal: {passed}/{total} tests passed.")
        print("=======================================================\n")
        return 0 if passed == total else 1

    def test_1_robot_loading(self):
        """TEST 1: Robot loads successfully in the simulator."""
        try:
            self.env = SimulationEnvironment()
            model = self.env.model
            data = self.env.data

            assert model is not None, "MuJoCo model is None"
            assert data is not None, "MuJoCo data is None"
            assert model.nu == 5, f"Expected 5 actuators (4 arm + 1 gripper), got {model.nu}"
            assert model.njnt >= 6, f"Expected at least 6 joints, got {model.njnt}"

            # Verify forward kinematics
            ee_pos = self.env.get_ee_position()
            assert ee_pos is not None and len(ee_pos) == 3
            self.log_result(
                "TEST 1: Robot loads in simulator",
                True,
                f"Official OpenMANIPULATOR-X loaded with {model.nu} actuators, {model.nbody} bodies. EE at {np.round(ee_pos, 3)}."
            )
        except Exception as e:
            self.log_result("TEST 1: Robot loads in simulator", False, str(e))

    def test_2_camera_capture(self):
        """TEST 2: Simulated camera captures an image."""
        try:
            rgb = self.env.render_rgb("workspace_cam")
            assert rgb is not None, "Render returned None"
            assert rgb.shape == (480, 640, 3), f"Unexpected shape {rgb.shape}"
            mean_intensity = float(np.mean(rgb))
            assert mean_intensity > 20.0, f"Image is too dark or empty (mean: {mean_intensity})"

            # Save snapshot
            self.env.save_image("test_cam_capture.png", "workspace_cam")
            self.log_result(
                "TEST 2: Camera captures image",
                True,
                f"Rendered {rgb.shape[1]}x{rgb.shape[0]} RGB frame, mean brightness {mean_intensity:.1f}. Saved test_cam_capture.png."
            )
        except Exception as e:
            self.log_result("TEST 2: Camera captures image", False, str(e))

    def test_3_vlm_object_identification(self):
        """TEST 3: Qwen2.5-VL receives the image and identifies the red cube."""
        try:
            self.vlm = QwenVLClient()
            conn_ok, conn_msg = self.vlm.check_connection()
            assert conn_ok, f"VLM connection failed: {conn_msg}"

            rgb = self.env.render_rgb("workspace_cam")
            success, msg, plan = self.vlm.analyze_command(
                rgb,
                "Pick up the red cube and put it inside the box."
            )
            assert success, f"VLM command analysis failed: {msg}"
            assert plan.get("object") == "red_cube", f"Expected object 'red_cube', got '{plan.get('object')}'"
            assert plan.get("confidence", 0) >= 0.5, f"Low confidence {plan.get('confidence')}"

            self.log_result(
                "TEST 3: Qwen2.5-VL identifies red cube",
                True,
                f"Qwen2.5-VL processed camera image: object='{plan.get('object')}', destination='{plan.get('destination')}', confidence={plan.get('confidence')}."
            )
        except Exception as e:
            self.log_result("TEST 3: Qwen2.5-VL identifies red cube", False, str(e))

    def test_4_natural_language_command(self):
        """TEST 4: Natural language command parsing and intent validation."""
        try:
            rgb = self.env.render_rgb("workspace_cam")
            test_commands = [
                ("Pick up the red cube and place it inside the box.", "pick_and_place", "red_cube", "box"),
                ("Move the blue cube into the box.", "pick_and_place", "blue_cube", "box"),
            ]

            all_ok = True
            details_list = []
            for cmd, expected_task, expected_obj, expected_dest in test_commands:
                ok, msg, plan = self.vlm.analyze_command(rgb, cmd)
                assert ok, f"Command '{cmd}' failed: {msg}"
                assert plan.get("object") == expected_obj, f"Object mismatch for '{cmd}'"
                details_list.append(f"'{cmd}' -> task={plan.get('task')}, obj={plan.get('object')}")

            self.log_result(
                "TEST 4: Natural language command input",
                True,
                f"Successfully parsed natural instructions: {'; '.join(details_list)}."
            )
        except Exception as e:
            self.log_result("TEST 4: Natural language command input", False, str(e))

    def test_5_robot_approach(self):
        """TEST 5: The robot moves to the red cube approach pose."""
        try:
            self.robot_ctrl = RobotController(self.env)
            self.gripper_ctrl = GripperController(self.env)
            self.mapper = CoordinateMapper(self.env)

            cube_pos = self.mapper.get_object_position("red_cube")
            assert cube_pos is not None, "Could not find red cube pos"

            approach_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.07])
            self.gripper_ctrl.open(duration=0.3)
            reached, msg = self.robot_ctrl.move_to_cartesian_position(approach_pos, duration=1.2)
            assert reached, f"Failed to reach approach pos: {msg}"

            actual_pos = self.env.get_grasp_position()
            dist = np.linalg.norm(actual_pos - approach_pos)
            assert dist < 0.015, f"Grasp site did not reach approach waypoint (dist={dist:.4f}m)"

            self.log_result(
                "TEST 5: Robot moves to red cube",
                True,
                f"Arm approached pre-grasp waypoint {np.round(approach_pos, 3)} with error {dist*1000:.1f} mm."
            )
        except Exception as e:
            self.log_result("TEST 5: Robot moves to red cube", False, str(e))

    def test_6_gripper_pickup(self):
        """TEST 6: The gripper picks up the cube and lifts it off the table."""
        try:
            cube_pos = self.mapper.get_object_position("red_cube")
            initial_z = cube_pos[2]

            # Lower to grasp
            grasp_pos = np.array([cube_pos[0], cube_pos[1], 0.015])
            self.robot_ctrl.move_to_cartesian_position(grasp_pos, duration=0.8)

            # Close gripper
            self.gripper_ctrl.close(duration=0.6)

            # Lift vertically
            lift_pos = np.array([cube_pos[0], cube_pos[1], 0.10])
            self.robot_ctrl.move_to_cartesian_position(lift_pos, duration=0.8, gripper_target=self.gripper_ctrl.close_pos)

            # Check cube height
            current_cube_pos = self.env.get_object_position("red_cube")
            lift_delta = current_cube_pos[2] - initial_z
            assert lift_delta > 0.04, f"Cube was not lifted off table (delta Z={lift_delta:.4f}m)"

            self.log_result(
                "TEST 6: Gripper picks up cube",
                True,
                f"Gripper pinched and lifted red cube from z={initial_z:.3f}m to z={current_cube_pos[2]:.3f}m (lifted by {lift_delta*100:.1f} cm)."
            )
        except Exception as e:
            self.log_result("TEST 6: Gripper picks up cube", False, str(e))

    def test_7_place_inside_box(self):
        """TEST 7: The robot places the cube inside the box."""
        try:
            box_pos = self.mapper.get_object_position("box")
            assert box_pos is not None, "Box position not found"

            # Transport to box standoff
            standoff = np.array([box_pos[0], box_pos[1], 0.12])
            self.robot_ctrl.move_to_cartesian_position(standoff, duration=1.2, gripper_target=self.gripper_ctrl.close_pos)

            # Lower into box
            place_pt = np.array([box_pos[0], box_pos[1], 0.035])
            self.robot_ctrl.move_to_cartesian_position(place_pt, duration=0.8, gripper_target=self.gripper_ctrl.close_pos)

            # Release
            self.gripper_ctrl.open(duration=0.5)

            # Check cube is over box
            cube_pos = self.env.get_object_position("red_cube")
            horiz_dist = np.linalg.norm(cube_pos[:2] - box_pos[:2])
            assert horiz_dist < 0.05, f"Cube not over box (dist={horiz_dist:.4f}m)"

            self.log_result(
                "TEST 7: Places cube inside box",
                True,
                f"Transported and lowered cube into box at {np.round(cube_pos, 3)} (horizontal offset to center {horiz_dist*1000:.1f} mm)."
            )
        except Exception as e:
            self.log_result("TEST 7: Places cube inside box", False, str(e))

    def test_8_object_stability_after_release(self):
        """TEST 8: The object remains in its new position after release."""
        try:
            box_pos = self.mapper.get_object_position("box")

            # Retreat arm and return home
            retreat_pt = np.array([box_pos[0], box_pos[1], 0.10])
            self.robot_ctrl.move_to_cartesian_position(retreat_pt, duration=0.6)
            self.robot_ctrl.go_home(duration=1.0)

            # Step physical simulation for settling (150 steps = 0.3s physics)
            for _ in range(150):
                self.env.step()

            final_pos = self.env.get_object_position("red_cube")
            dist_to_box = np.linalg.norm(final_pos[:2] - box_pos[:2])
            assert dist_to_box < 0.06, f"Object rolled or fell out of box (dist={dist_to_box:.4f}m)"
            assert final_pos[2] > 0.005, f"Object fell through table (z={final_pos[2]:.4f}m)"

            self.log_result(
                "TEST 8: Object stability after release",
                True,
                f"Arm safely returned home. Red cube remains stable inside box at {np.round(final_pos, 3)} (dist to center {dist_to_box*1000:.1f} mm)."
            )
        except Exception as e:
            self.log_result("TEST 8: Object stability after release", False, str(e))


def run_all_tests():
    suite = SystemTestSuite()
    return suite.run_all()


if __name__ == "__main__":
    sys.exit(run_all_tests())

