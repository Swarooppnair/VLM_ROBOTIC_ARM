"""Terminal-based Command Interface for VLM Robot Controller."""

import os
import sys
import time
from typing import Optional

from vlm_robotics.simulation.environment import SimulationEnvironment
from vlm_robotics.planning.task_planner import TaskPlanner
from vlm_robotics.vlm.qwen_client import QwenVLClient
from vlm_robotics.config.config import VLM_MODEL


class CommandInterface:
    """Terminal CLI offering natural language control of the simulated robot."""

    def __init__(self, with_viewer: bool = False):
        self.with_viewer = with_viewer
        self.env = SimulationEnvironment()
        self.vlm = QwenVLClient()
        self.planner = TaskPlanner(self.env, status_cb=self._on_status)

        if self.with_viewer:
            self.env.open_viewer()

    def _on_status(self, msg: str):
        """Format and print status update in terminal."""
        print(f"Robot status: {msg}")

    def display_banner(self):
        """Display system status header."""
        vlm_ok, vlm_msg = self.vlm.check_connection()
        vlm_status = f"{VLM_MODEL} (Online)" if vlm_ok else f"Offline ({vlm_msg})"

        print("=========================================")
        print("       LOCAL AI ROBOT CONTROLLER         ")
        print("=========================================")
        print(f"VLM:       {vlm_status}")
        print("Robot:     ROBOTIS OpenMANIPULATOR-X")
        print("Simulator: MuJoCo Physics (Running)")
        print("Camera:    Connected (640x480 RGB+D)")
        print("=========================================")
        print("Special commands: 'inspect', 'home', 'reset', 'exit'\n")

    def run(self):
        """Main interactive REPL loop."""
        self.display_banner()

        while True:
            try:
                user_cmd = input("\nEnter command:\n> ").strip()
                if not user_cmd:
                    continue

                if user_cmd.lower() in ("exit", "quit", "q"):
                    print("\nShutting down controller. Goodbye!")
                    break

                if user_cmd.lower() in ("help", "?"):
                    print("\nExample commands:")
                    print("  - Pick up the red cube and put it inside the box.")
                    print("  - Move the blue cube into the box.")
                    print("  - Pick up the bottle and place it on the left side.")
                    print("  - Move the red cube next to the blue cube.")
                    print("  - Look at the table and identify all objects.")
                    print("  - Go home / Reset")
                    continue

                print("\n-----------------------------------------")
                success, msg, details = self.planner.execute_natural_command(user_cmd)
                print("-----------------------------------------")

                if success:
                    print(f"\n[OK] Task finished successfully: {msg}")
                    if "identified_objects" in details:
                        print("Identified objects:")
                        for obj in details["identified_objects"]:
                            print(f"  * {obj.get('name', 'obj')}: {obj.get('color', '')} {obj.get('type', '')}")
                else:
                    print(f"\n[ERROR] Task halted: {msg}")

            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting.")
                break
            except Exception as e:
                print(f"\n[UNEXPECTED ERROR] {e}")

        if self.env.viewer is not None:
            self.env.close_viewer()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local Qwen2.5-VL OpenMANIPULATOR-X Controller")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D viewer window")
    args = parser.parse_args()

    cli = CommandInterface(with_viewer=args.viewer)
    cli.run()


if __name__ == "__main__":
    main()

