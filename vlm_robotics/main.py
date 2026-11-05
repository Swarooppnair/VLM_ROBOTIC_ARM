"""Main entrypoint for the Local Qwen2.5-VL OpenMANIPULATOR-X Controller."""

import sys
import argparse

from vlm_robotics.interfaces.command_interface import CommandInterface
from vlm_robotics.simulation.environment import SimulationEnvironment
from vlm_robotics.planning.task_planner import TaskPlanner
from vlm_robotics.vlm.qwen_client import QwenVLClient


def verify_vlm_prerequisite(abort_on_fail: bool = True) -> bool:
    """
    MANDATORY PRE-FLIGHT CHECK:
    Without Qwen2.5-VL actively running locally in Ollama, this project CANNOT and WILL NOT run.
    """
    client = QwenVLClient()
    connected, msg = client.check_connection()
    if not connected:
        print("\n" + "=" * 78)
        print(" [CRITICAL REQUIREMENT ERROR] LOCAL VLM IS NOT RUNNING")
        print("=" * 78)
        print(" This project is an Autonomous VLM Robotics System powered by local Qwen2.5-VL.")
        print(" WITHOUT THE VLM RUNNING IN OLLAMA LOCALLY, THIS PROJECT CANNOT AND WILL NOT RUN.")
        print(f"\n Detailed Status: {msg}")
        print("\n How to resolve this:")
        print("   1. Open a new terminal and start the Ollama service:")
        print("      ollama serve")
        print("   2. In another terminal, ensure the model is loaded and running:")
        print("      ollama run qwen2.5vl:3b")
        print("   3. Re-launch this application:")
        print("      python run.py --gui")
        print("=" * 78 + "\n")
        if abort_on_fail:
            sys.exit(1)
        return False
    return True


def run_single_command(command: str, with_viewer: bool = False):
    """Execute a single natural language command and exit."""
    verify_vlm_prerequisite(abort_on_fail=True)
    env = SimulationEnvironment()
    if with_viewer:
        env.open_viewer()

    def print_status(msg):
        print(f"Robot status: {msg}")

    planner = TaskPlanner(env, status_cb=print_status)
    print(f"\nExecuting command: '{command}'")
    success, msg, plan = planner.execute_natural_command(command)
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'} - {msg}")
    print(f"VLM Plan: {plan}")

    if env.viewer is not None:
        import time
        time.sleep(2.0)
        env.close_viewer()

    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(
        description="Local Qwen2.5-VL Controlled OpenMANIPULATOR-X Robotic Manipulation System"
    )
    parser.add_argument("--command", "-c", type=str, default=None, help="Direct natural language command to execute")
    parser.add_argument("--viewer", "-v", action="store_true", help="Launch interactive 3D visualizer window")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch Graphical Control Panel with Microphone and Text Box")
    parser.add_argument("--test", "-t", action="store_true", help="Run full automated verification test suite")
    args = parser.parse_args()

    if args.test:
        from vlm_robotics.tests.test_suite import run_all_tests
        return run_all_tests()
    elif args.gui:
        verify_vlm_prerequisite(abort_on_fail=True)
        from vlm_robotics.interfaces.gui_interface import launch_gui
        launch_gui(with_viewer=not args.viewer if False else True)
        return 0
    elif args.command:
        return run_single_command(args.command, with_viewer=args.viewer)
    else:
        verify_vlm_prerequisite(abort_on_fail=True)
        cli = CommandInterface(with_viewer=args.viewer)
        cli.run()
        return 0


if __name__ == "__main__":
    sys.exit(main())

