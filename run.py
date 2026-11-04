"""Convenience launcher for VLM Robotics Controller."""

import sys
import os

# Add workspace directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vlm_robotics.main import main

if __name__ == "__main__":
    sys.exit(main())

