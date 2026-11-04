"""Main entrypoint for the Local Qwen2.5-VL OpenMANIPULATOR-X Controller."""

import sys
import argparse

from vlm_robotics.interfaces.command_interface import CommandInterface
from vlm_robotics.simulation.environment import SimulationEnvironment
from vlm_robotics.planning.task_planner import TaskPlanner


def run_single_command(command: str, with_viewer: bool = False):
    """Execute a single natural language command and exit."""
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
        from vlm_robotics.interfaces.gui_interface import launch_gui
        launch_gui(with_viewer=not args.viewer if False else True)
        return 0
    elif args.command:
        return run_single_command(args.command, with_viewer=args.viewer)
    else:
        cli = CommandInterface(with_viewer=args.viewer)
        cli.run()
        return 0


if __name__ == "__main__":
    sys.exit(main())

