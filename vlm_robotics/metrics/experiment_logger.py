"""Research Experiment Logger and Performance Metrics for Autonomous Manipulation."""

import os
import csv
import time
from typing import Dict, Any, List, Optional
from datetime import datetime


class ExperimentLogger:
    """Logs autonomous manipulation trials, latencies, success rates, and exports CSV benchmarks."""

    def __init__(self, log_dir: str = "experiments"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "experiment_log.csv")
        self.trials: List[Dict[str, Any]] = []

        self._ensure_csv_headers()

    def _ensure_csv_headers(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "command",
                    "mode",
                    "task_type",
                    "target_object",
                    "destination",
                    "grasp_verified",
                    "placement_verified",
                    "recovery_attempts",
                    "vlm_latency_s",
                    "execution_time_s",
                    "task_success",
                    "notes",
                ])

    def log_trial(
        self,
        command: str,
        mode: str,
        task_type: str,
        target_object: str,
        destination: str,
        grasp_verified: bool,
        placement_verified: bool,
        recovery_attempts: int,
        vlm_latency: float,
        execution_time: float,
        success: bool,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record a completed or attempted manipulation trial."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "command": command,
            "mode": mode,
            "task_type": task_type,
            "target_object": target_object,
            "destination": destination,
            "grasp_verified": grasp_verified,
            "placement_verified": placement_verified,
            "recovery_attempts": recovery_attempts,
            "vlm_latency_s": round(vlm_latency, 3),
            "execution_time_s": round(execution_time, 2),
            "task_success": success,
            "notes": notes,
        }
        self.trials.append(entry)

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                entry["timestamp"],
                entry["command"],
                entry["mode"],
                entry["task_type"],
                entry["target_object"],
                entry["destination"],
                entry["grasp_verified"],
                entry["placement_verified"],
                entry["recovery_attempts"],
                entry["vlm_latency_s"],
                entry["execution_time_s"],
                entry["task_success"],
                entry["notes"],
            ])

        return entry

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute aggregated metrics across recorded trials."""
        total = len(self.trials)
        if total == 0:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "overall_success_rate": 0.0,
                "grasp_success_rate": 0.0,
                "placement_success_rate": 0.0,
                "avg_recovery_attempts": 0.0,
                "avg_execution_time_s": 0.0,
                "avg_vlm_latency_s": 0.0,
            }

        successful = sum(1 for t in self.trials if t["task_success"])
        grasps = sum(1 for t in self.trials if t["grasp_verified"])
        placements = sum(1 for t in self.trials if t["placement_verified"])
        recoveries = sum(t["recovery_attempts"] for t in self.trials)
        avg_exec = sum(t["execution_time_s"] for t in self.trials) / total
        avg_vlm = sum(t["vlm_latency_s"] for t in self.trials) / total

        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "overall_success_rate": round((successful / total) * 100.0, 1),
            "grasp_success_rate": round((grasps / total) * 100.0, 1),
            "placement_success_rate": round((placements / total) * 100.0, 1),
            "avg_recovery_attempts": round(recoveries / total, 2),
            "avg_execution_time_s": round(avg_exec, 2),
            "avg_vlm_latency_s": round(avg_vlm, 3),
        }

