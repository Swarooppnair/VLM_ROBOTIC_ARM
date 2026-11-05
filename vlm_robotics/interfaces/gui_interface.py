"""Research-Grade Desktop Mission-Control GUI for Autonomous VLM Robotics Manipulation.

Features:
- Live Multi-Camera vision stream (Third-Person, Overhead, Wrist PiP, and Colorized Depth)
- AR HUD with 3D object detection bounding brackets, affordance labels, and trajectory trails
- Real-time physical telemetry: 6-DoF End-Effector pose, velocities, joint angles, torques, grip force
- Voice (Microphone) and Natural-Language Text Command Interface with Local TTS audio feedback
- Multi-Level task presets (Level 1 pick-and-place, Level 2 relative & sorting, Level 3 stacking, Level 4 clearing)
- Research experiment logging, performance metrics, and CSV export
"""

import os
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
from PIL import Image, ImageTk

from vlm_robotics.simulation.environment import SimulationEnvironment
from vlm_robotics.planning.task_planner import TaskPlanner
from vlm_robotics.vlm.qwen_client import QwenVLClient
from vlm_robotics.interfaces.voice_interface import VoiceInterface
from vlm_robotics.perception.camera_interface import MultiCameraSystem, CameraInterface
from vlm_robotics.perception.ar_hud import ARHudRenderer
from vlm_robotics.metrics.experiment_logger import ExperimentLogger
from vlm_robotics.config.config import VLM_MODEL


class RobotControlGUI:
    """Advanced Mission Control Dashboard for Autonomous VLM Manipulation."""

    # Industrial Sci-Fi Laboratory Dark Theme
    BG_DARK = "#0d1117"
    CARD_BG = "#161b22"
    CARD_INNER = "#21262d"
    TEXT_MAIN = "#f0f6fc"
    TEXT_MUTED = "#8b949e"
    ACCENT_CYAN = "#00e5ff"
    ACCENT_BLUE = "#58a6ff"
    ACCENT_GREEN = "#3fb950"
    ACCENT_AMBER = "#d29922"
    ACCENT_RED = "#f85149"
    ACCENT_PURPLE = "#bc8cff"

    def __init__(self, with_viewer: bool = False):
        self.with_viewer = with_viewer

        # 1. Initialize Robotics & Simulation Subsystems
        self.env = SimulationEnvironment()
        self.vlm = QwenVLClient()
        self.voice = VoiceInterface()
        self.multi_cams = MultiCameraSystem(self.env)
        self.primary_cam = self.multi_cams.get_camera("workspace_cam") or CameraInterface(self.env, "workspace_cam")
        self.hud = ARHudRenderer(self.primary_cam)
        self.logger = ExperimentLogger()

        self.msg_queue: queue.Queue = queue.Queue()
        self.is_executing = False
        self.planner = TaskPlanner(self.env, status_cb=self._queue_robot_status, voice_interface=self.voice)

        if self.with_viewer:
            self.env.open_viewer()

        # Telemetry smoothing variables
        self.last_ee_pos = self.env.get_ee_position()
        self.last_time = time.perf_counter()
        self.current_fps = 60.0
        self.frame_count = 0
        self.fps_timer = time.perf_counter()

        # 2. Build GUI Window
        self.root = tk.Tk()
        self.root.title("Autonomous VLM Robotic Manipulation — Laboratory Mission Control")
        self.root.geometry("1260x860")
        self.root.minsize(1080, 740)
        self.root.configure(bg=self.BG_DARK)

        self._apply_styles()
        self._build_layout()

        # Start periodic video & queue update loops
        self.root.after(30, self._update_camera_and_telemetry)
        self.root.after(50, self._process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Initial Welcome Log
        self._log_message("SYSTEM", f"Autonomous Robotics Platform Initialized.")
        self._log_message("SYSTEM", "Multi-Camera Vision Online: Workspace, Overhead, Wrist PiP, and Depth TURBO.")

        # Strict VLM pre-flight check
        vlm_ok, vlm_msg = self.vlm.check_connection()
        if vlm_ok:
            self._log_message("SUCCESS", f"✔ Local VLM Online: {vlm_msg}")
            self.vlm_badge.configure(text=f"● VLM: {VLM_MODEL} (ONLINE)", fg=self.ACCENT_GREEN)
            self._log_message("SYSTEM", "Closed-loop perception-action-verification engine READY.")
        else:
            self._log_message("ERROR", "✖ [CRITICAL REQUIREMENT ERROR] Local VLM (Ollama) is NOT running!")
            self._log_message("ERROR", "Without Qwen2.5-VL running in Ollama locally, this project CANNOT run.")
            self._log_message("SYSTEM", "To start the VLM: open a terminal and run 'ollama run qwen2.5vl:3b'")
            self.vlm_badge.configure(text="✖ VLM: OFFLINE (OLLAMA REQUIRED)", fg=self.ACCENT_RED)
            self.send_btn.configure(state=tk.DISABLED, bg="#334155")
            self.mic_btn.configure(state=tk.DISABLED, bg="#334155")

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.BG_DARK, foreground=self.TEXT_MAIN)
        style.configure("TFrame", background=self.BG_DARK)
        style.configure("Card.TFrame", background=self.CARD_BG)
        style.configure("TLabel", background=self.CARD_BG, foreground=self.TEXT_MAIN)

    def _build_layout(self):
        # 1. Top Status Bar
        top_bar = tk.Frame(self.root, bg=self.CARD_BG, padx=16, pady=8, highlightthickness=1, highlightbackground=self.CARD_INNER)
        top_bar.pack(fill=tk.X, padx=12, pady=(10, 6))

        title_lbl = tk.Label(
            top_bar,
            text="⚡ VLM AUTONOMOUS ROBOTICS MISSION CONTROL",
            font=("Segoe UI", 13, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_CYAN,
        )
        title_lbl.pack(side=tk.LEFT)

        badges_frame = tk.Frame(top_bar, bg=self.CARD_BG)
        badges_frame.pack(side=tk.RIGHT)

        self.fps_badge = tk.Label(
            badges_frame,
            text="SIM: 60.0 FPS",
            font=("Consolas", 9, "bold"),
            bg=self.CARD_INNER,
            fg=self.ACCENT_GREEN,
            padx=8,
            pady=2,
        )
        self.fps_badge.pack(side=tk.LEFT, padx=4)

        self.vlm_badge = tk.Label(
            badges_frame,
            text=f"VLM: {VLM_MODEL}",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_INNER,
            fg=self.ACCENT_CYAN,
            padx=8,
            pady=2,
        )
        self.vlm_badge.pack(side=tk.LEFT, padx=4)

        self.robot_badge = tk.Label(
            badges_frame,
            text="ROBOT: OpenMANIPULATOR-X",
            font=("Segoe UI", 9),
            bg=self.CARD_INNER,
            fg=self.TEXT_MUTED,
            padx=8,
            pady=2,
        )
        self.robot_badge.pack(side=tk.LEFT, padx=4)

        self.activity_badge = tk.Label(
            badges_frame,
            text="● IDLE",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_INNER,
            fg=self.ACCENT_GREEN,
            padx=8,
            pady=2,
        )
        self.activity_badge.pack(side=tk.LEFT, padx=4)

        # 2. Main Workspace (2-Column Layout: Left=Vision & Telemetry, Right=Mission Controls & Log)
        main_content = tk.Frame(self.root, bg=self.BG_DARK)
        main_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Left Column (Width ~680)
        left_col = tk.Frame(main_content, bg=self.BG_DARK)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Vision Feed Canvas Card
        vision_card = tk.Frame(left_col, bg=self.CARD_BG, padx=8, pady=8, highlightthickness=1, highlightbackground=self.CARD_INNER)
        vision_card.pack(fill=tk.BOTH, expand=True)

        vis_header = tk.Frame(vision_card, bg=self.CARD_BG)
        vis_header.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            vis_header,
            text="📹 MULTI-CAMERA AR VISION STREAM (WORKSPACE + WRIST PiP + OVERHEAD + DEPTH)",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_BLUE,
        ).pack(side=tk.LEFT)

        self.video_label = tk.Label(vision_card, bg="#05070a")
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=2)

        # Live Telemetry Strip
        telemetry_card = tk.Frame(left_col, bg=self.CARD_BG, padx=12, pady=8, highlightthickness=1, highlightbackground=self.CARD_INNER)
        telemetry_card.pack(fill=tk.X, pady=(6, 0))

        tk.Label(
            telemetry_card,
            text="📊 PHYSICAL ROBOT TELEMETRY (SIMULATOR GROUND TRUTH)",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_AMBER,
        ).pack(anchor="w", pady=(0, 4))

        telem_grid = tk.Frame(telemetry_card, bg=self.CARD_BG)
        telem_grid.pack(fill=tk.X)

        self.lbl_eef_pos = tk.Label(telem_grid, text="EEF Pose: [0.000, 0.000, 0.000]m", font=("Consolas", 9), bg=self.CARD_BG, fg=self.TEXT_MAIN)
        self.lbl_eef_pos.grid(row=0, column=0, sticky="w", padx=4, pady=2)

        self.lbl_eef_vel = tk.Label(telem_grid, text="Velocity: 0.000 m/s", font=("Consolas", 9), bg=self.CARD_BG, fg=self.TEXT_MAIN)
        self.lbl_eef_vel.grid(row=0, column=1, sticky="w", padx=12, pady=2)

        self.lbl_joints = tk.Label(telem_grid, text="Joints: [0.00, 0.00, 0.00, 0.00] rad", font=("Consolas", 9), bg=self.CARD_BG, fg=self.TEXT_MAIN)
        self.lbl_joints.grid(row=1, column=0, sticky="w", padx=4, pady=2)

        self.lbl_gripper = tk.Label(telem_grid, text="Gripper: 0.0120m [OPEN]", font=("Consolas", 9), bg=self.CARD_BG, fg=self.ACCENT_GREEN)
        self.lbl_gripper.grid(row=1, column=1, sticky="w", padx=12, pady=2)

        # Right Column (Mission Controls & Console)
        right_col = tk.Frame(main_content, bg=self.BG_DARK, width=540)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(6, 0))

        # Voice Input Card
        voice_card = tk.Frame(right_col, bg=self.CARD_BG, padx=12, pady=8, highlightthickness=1, highlightbackground=self.CARD_INNER)
        voice_card.pack(fill=tk.X, pady=(0, 6))

        voice_top = tk.Frame(voice_card, bg=self.CARD_BG)
        voice_top.pack(fill=tk.X)
        tk.Label(
            voice_top,
            text="🎙️ SPEECH RECOGNITION & VOICE COMMAND",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_CYAN,
        ).pack(side=tk.LEFT)

        self.voice_status_lbl = tk.Label(voice_top, text="Idle", font=("Segoe UI", 9, "italic"), bg=self.CARD_BG, fg=self.TEXT_MUTED)
        self.voice_status_lbl.pack(side=tk.RIGHT)

        self.mic_btn = tk.Button(
            voice_card,
            text="🎤  Click to Speak Command",
            font=("Segoe UI", 10, "bold"),
            bg="#1f3d5c",
            fg=self.TEXT_MAIN,
            activebackground="#2b5988",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=6,
            command=self._on_mic_click,
        )
        self.mic_btn.pack(fill=tk.X, pady=(6, 2))

        # Text Command Input Card
        text_card = tk.Frame(right_col, bg=self.CARD_BG, padx=12, pady=8, highlightthickness=1, highlightbackground=self.CARD_INNER)
        text_card.pack(fill=tk.X, pady=4)

        tk.Label(
            text_card,
            text="⌨️ NATURAL-LANGUAGE PROMPT INPUT",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_BLUE,
        ).pack(anchor="w", pady=(0, 4))

        input_row = tk.Frame(text_card, bg=self.CARD_BG)
        input_row.pack(fill=tk.X)

        self.cmd_entry = tk.Entry(
            input_row,
            font=("Segoe UI", 10),
            bg=self.CARD_INNER,
            fg=self.TEXT_MAIN,
            insertbackground=self.ACCENT_CYAN,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#30363d",
            highlightcolor=self.ACCENT_CYAN,
        )
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 6))
        self.cmd_entry.bind("<Return>", lambda e: self._on_send_click())
        self.cmd_entry.insert(0, "place the blue cube on the purple glass")

        self.send_btn = tk.Button(
            input_row,
            text="▶ Execute",
            font=("Segoe UI", 10, "bold"),
            bg="#238636",
            fg="#ffffff",
            activebackground="#2ea043",
            relief=tk.FLAT,
            cursor="hand2",
            padx=12,
            pady=4,
            command=self._on_send_click,
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Advanced Research Manipulation Presets (Levels 1 - 4)
        preset_card = tk.Frame(right_col, bg=self.CARD_BG, padx=12, pady=8, highlightthickness=1, highlightbackground=self.CARD_INNER)
        preset_card.pack(fill=tk.X, pady=4)

        tk.Label(
            preset_card,
            text="🎯 BENCHMARK MANIPULATION TASKS (LEVELS 1 - 4)",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_PURPLE,
        ).pack(anchor="w", pady=(0, 4))

        p_row1 = tk.Frame(preset_card, bg=self.CARD_BG)
        p_row1.pack(fill=tk.X, pady=2)
        self._create_btn(p_row1, "L1: Red Cube to Box", lambda: self._fill_and_execute("place the red cube in the box"))
        self._create_btn(p_row1, "L2: Blue Cube on Purple Glass", lambda: self._fill_and_execute("place the blue cube on the purple glass"))

        p_row2 = tk.Frame(preset_card, bg=self.CARD_BG)
        p_row2.pack(fill=tk.X, pady=2)
        self._create_btn(p_row2, "L3: Glass Pyramid", lambda: self._fill_and_execute("build a pyramid using glasses"))
        self._create_btn(p_row2, "L3: Cube Pyramid", lambda: self._fill_and_execute("build a pyramid of cubes"))

        p_row3 = tk.Frame(preset_card, bg=self.CARD_BG)
        p_row3.pack(fill=tk.X, pady=2)
        self._create_btn(p_row3, "L2: Sort Warm Colors", lambda: self._fill_and_execute("move all warm-coloured objects into the box"))
        self._create_btn(p_row3, "L3: Stack Cubes Tower", lambda: self._fill_and_execute("stack the cubes into a tower"))

        p_row4 = tk.Frame(preset_card, bg=self.CARD_BG)
        p_row4.pack(fill=tk.X, pady=2)
        self._create_btn(p_row4, "L4: Clear Right Table", lambda: self._fill_and_execute("clear the right side of the table"))
        self._create_btn(p_row4, "🔍 Inspect Tabletop", lambda: self._fill_and_execute("identify all objects on the table"))

        # Research Evaluation & Metrics Panel
        metrics_card = tk.Frame(right_col, bg=self.CARD_BG, padx=12, pady=6, highlightthickness=1, highlightbackground=self.CARD_INNER)
        metrics_card.pack(fill=tk.X, pady=4)

        m_top = tk.Frame(metrics_card, bg=self.CARD_BG)
        m_top.pack(fill=tk.X)
        tk.Label(
            m_top,
            text="📈 EXPERIMENT EVALUATION & RESEARCH METRICS",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_GREEN,
        ).pack(side=tk.LEFT)

        export_btn = tk.Button(
            m_top,
            text="📊 Export CSV",
            font=("Segoe UI", 8, "bold"),
            bg=self.CARD_INNER,
            fg=self.ACCENT_CYAN,
            relief=tk.FLAT,
            cursor="hand2",
            padx=6,
            command=self._export_csv,
        )
        export_btn.pack(side=tk.RIGHT)

        self.metrics_lbl = tk.Label(
            metrics_card,
            text="Total: 0 | Success: 0.0% | Grasp: 0.0% | Avg Latency: 0.00s",
            font=("Consolas", 8),
            bg=self.CARD_BG,
            fg=self.TEXT_MUTED,
        )
        self.metrics_lbl.pack(anchor="w", pady=(2, 0))

        # Event Console Log
        log_card = tk.Frame(right_col, bg=self.CARD_BG, padx=12, pady=8, highlightthickness=1, highlightbackground=self.CARD_INNER)
        log_card.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        tk.Label(
            log_card,
            text="📜 AUTONOMOUS EXECUTION EVENT LOG",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 4))

        self.log_text = scrolledtext.ScrolledText(
            log_card,
            font=("Consolas", 8),
            bg="#090d13",
            fg=self.TEXT_MAIN,
            insertbackground=self.ACCENT_CYAN,
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_text.tag_config("TIME", foreground="#6e7681")
        self.log_text.tag_config("SYSTEM", foreground=self.ACCENT_CYAN)
        self.log_text.tag_config("COMMAND", foreground=self.ACCENT_BLUE, font=("Consolas", 8, "bold"))
        self.log_text.tag_config("ROBOT", foreground="#c9d1d9")
        self.log_text.tag_config("SUCCESS", foreground=self.ACCENT_GREEN, font=("Consolas", 8, "bold"))
        self.log_text.tag_config("ERROR", foreground=self.ACCENT_RED, font=("Consolas", 8, "bold"))
        self.log_text.tag_config("VOICE", foreground=self.ACCENT_AMBER)

    def _create_btn(self, parent, text: str, command):
        btn = tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 8),
            bg=self.CARD_INNER,
            fg=self.TEXT_MAIN,
            activebackground="#30363d",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            padx=6,
            pady=3,
            command=command,
        )
        btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        return btn

    def _update_camera_and_telemetry(self):
        """Periodic loop updating the AR HUD vision stream and physical telemetry."""
        try:
            # 1. Capture and render live camera frame with AR HUD
            t_now = time.perf_counter()
            self.frame_count += 1
            if t_now - self.fps_timer >= 1.0:
                self.current_fps = self.frame_count / (t_now - self.fps_timer)
                self.frame_count = 0
                self.fps_timer = t_now
                self.fps_badge.configure(text=f"SIM: {self.current_fps:.1f} FPS")

            # Telemetry queries
            cur_ee = self.env.get_grasp_position()
            cur_grip = self.env.get_gripper_position()
            is_holding = self.planner.gripper_ctrl.is_grasping()
            arm_q = self.env.get_arm_joint_positions()

            dt = max(1e-4, t_now - self.last_time)
            ee_vel = np.linalg.norm(cur_ee - self.last_ee_pos) / dt
            self.last_ee_pos = cur_ee.copy()
            self.last_time = t_now

            # Add to AR trajectory trail
            if self.is_executing:
                self.hud.add_trail_point(cur_ee)

            # Generate AR HUD on primary camera
            raw_rgb = self.primary_cam.get_rgb()
            hud_frame = self.hud.render_hud(
                rgb_image=raw_rgb,
                camera_interface=self.primary_cam,
                target_obj=self.planner.target_object,
                destination=self.planner.destination_desc,
                stage=self.planner.current_stage,
                all_objects=self.env.get_all_object_positions(),
                planned_waypoints=self.planner.active_waypoints,
                telemetry={"ee_pos": cur_ee, "gripper_pos": cur_grip, "is_grasping": is_holding},
                fps=self.current_fps,
            )

            # Generate multi-view composite (Overhead, Wrist PiP, and Colorized Depth)
            multiview_comp = self.multi_cams.create_multiview_composite(
                hud_overlay=hud_frame,
                pip_camera="wrist_cam",
                overhead_camera="overhead_cam",
                show_depth=True,
            )

            # Resize to fit UI video canvas smoothly (~620x465)
            comp_h, comp_w = multiview_comp.shape[:2]
            scale = 620.0 / comp_w
            new_w = 620
            new_h = int(comp_h * scale)
            img_pil = Image.fromarray(multiview_comp).resize((new_w, new_h), Image.BILINEAR)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk

            # 2. Update Physical Telemetry Labels
            self.lbl_eef_pos.configure(text=f"EEF Pose: [{cur_ee[0]:+.3f}, {cur_ee[1]:+.3f}, {cur_ee[2]:+.3f}]m")
            self.lbl_eef_vel.configure(text=f"Velocity: {ee_vel:.3f} m/s")
            self.lbl_joints.configure(text=f"Joints: [{arm_q[0]:+.2f}, {arm_q[1]:+.2f}, {arm_q[2]:+.2f}, {arm_q[3]:+.2f}] rad")

            grip_state_str = "SECURED (CONTACT)" if is_holding else "OPEN / SLACK"
            grip_color = self.ACCENT_GREEN if is_holding else self.TEXT_MAIN
            self.lbl_gripper.configure(text=f"Gripper: {cur_grip:+.4f}m [{grip_state_str}]", fg=grip_color)

        except Exception as e:
            pass

        # Schedule next video frame in 33 ms (~30 FPS)
        self.root.after(33, self._update_camera_and_telemetry)

    def _on_mic_click(self):
        """Handle microphone voice input."""
        if self.is_executing:
            self._log_message("SYSTEM", "Robot is currently executing a task. Please wait.")
            return

        if self.voice.is_listening():
            return

        self.mic_btn.configure(text="🔴  Listening... (Speak your command)", bg="#822025")
        self.voice_status_lbl.configure(text="Listening...", fg=self.ACCENT_RED)
        self._set_activity("LISTENING", self.ACCENT_RED)

        def _status_cb(msg: str):
            self.msg_queue.put(("VOICE_STATUS", msg))

        def _result_cb(success: bool, text_or_err: str):
            self.msg_queue.put(("VOICE_RESULT", (success, text_or_err)))

        self.voice.listen_and_transcribe(
            on_status=_status_cb,
            on_result=_result_cb,
        )

    def _on_send_click(self):
        cmd = self.cmd_entry.get().strip()
        if not cmd:
            return
        self._dispatch_command(cmd)

    def _fill_and_execute(self, cmd_str: str):
        self.cmd_entry.delete(0, tk.END)
        self.cmd_entry.insert(0, cmd_str)
        self._dispatch_command(cmd_str)

    def _dispatch_command(self, cmd_text: str):
        if self.is_executing:
            self._log_message("SYSTEM", "Robot is already busy executing. Please wait.")
            return

        self.is_executing = True
        self._set_activity("EXECUTING", self.ACCENT_AMBER)
        self.send_btn.configure(state=tk.DISABLED, bg="#334155")
        self.mic_btn.configure(state=tk.DISABLED, bg="#334155")

        self._log_message("COMMAND", f"> {cmd_text}")

        def _worker():
            try:
                success, msg, details = self.planner.execute_natural_command(cmd_text)
                self.msg_queue.put(("EXEC_DONE", (success, msg, details)))
            except Exception as e:
                self.msg_queue.put(("EXEC_DONE", (False, str(e), {})))

        threading.Thread(target=_worker, daemon=True).start()

    def _queue_robot_status(self, msg: str):
        self.msg_queue.put(("ROBOT_STATUS", msg))

    def _process_queue(self):
        try:
            while not self.msg_queue.empty():
                item = self.msg_queue.get_nowait()
                tag, payload = item

                if tag == "ROBOT_STATUS":
                    self._log_message("ROBOT", f"  {payload}")

                elif tag == "VOICE_STATUS":
                    self.voice_status_lbl.configure(text=payload)

                elif tag == "VOICE_RESULT":
                    success, text_or_err = payload
                    self.mic_btn.configure(text="🎤  Click to Speak Command", bg="#1f3d5c")
                    self.voice_status_lbl.configure(text="Idle", fg=self.TEXT_MUTED)

                    if success:
                        self._log_message("VOICE", f"🎙️ Transcribed: \"{text_or_err}\"")
                        self.cmd_entry.delete(0, tk.END)
                        self.cmd_entry.insert(0, text_or_err)
                        self._dispatch_command(text_or_err)
                    else:
                        self._log_message("ERROR", f"Voice error: {text_or_err}")
                        self._set_activity("IDLE", self.ACCENT_GREEN)

                elif tag == "EXEC_DONE":
                    success, msg, details = payload
                    self.is_executing = False
                    self.send_btn.configure(state=tk.NORMAL, bg="#238636")
                    self.mic_btn.configure(state=tk.NORMAL, bg="#1f3d5c")
                    self._set_activity("IDLE", self.ACCENT_GREEN)

                    if success:
                        self._log_message("SUCCESS", f"✔ Completed: {msg}")
                    else:
                        self._log_message("ERROR", f"✖ Failed: {msg}")

                    # Refresh metrics
                    self._update_metrics_display()

        except Exception as e:
            pass

        self.root.after(50, self._process_queue)

    def _set_activity(self, text: str, color: str):
        self.activity_badge.configure(text=f"● {text}", fg=color)

    def _update_metrics_display(self):
        metrics = self.logger.get_summary_metrics()
        self.metrics_lbl.configure(
            text=f"Total: {metrics['total_tasks']} | "
                 f"Success: {metrics['overall_success_rate']}% | "
                 f"Grasp: {metrics['grasp_success_rate']}% | "
                 f"Avg Latency: {metrics['avg_vlm_latency_s']}s"
        )

    def _export_csv(self):
        csv_path = self.logger.csv_path
        messagebox.showinfo(
            "Experiment Benchmark Exported",
            f"Experimental log successfully saved to:\n{os.path.abspath(csv_path)}"
        )

    def _log_message(self, tag: str, message: str):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.insert(tk.END, timestamp, "TIME")
        self.log_text.insert(tk.END, f"{message}\n", tag)
        self.log_text.see(tk.END)

    def _on_close(self):
        if self.env.viewer is not None:
            self.env.close_viewer()
        self.env.close()
        self.root.destroy()
        sys.exit(0)

    def start(self):
        self.root.mainloop()


def launch_gui(with_viewer: bool = False):
    app = RobotControlGUI(with_viewer=with_viewer)
    app.start()


if __name__ == "__main__":
    launch_gui()
