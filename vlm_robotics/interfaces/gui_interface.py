"""Modern Desktop GUI for VLM Robot Controller with Voice (Microphone) and Text input."""

import os
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional

from vlm_robotics.simulation.environment import SimulationEnvironment
from vlm_robotics.planning.task_planner import TaskPlanner
from vlm_robotics.vlm.qwen_client import QwenVLClient
from vlm_robotics.interfaces.voice_interface import VoiceInterface
from vlm_robotics.config.config import VLM_MODEL


class RobotControlGUI:
    """Tkinter Desktop Control Panel with Microphone and Text Box inputs."""

    # Color Palette (Catppuccin Mocha inspired dark theme)
    BG_DARK = "#181825"
    CARD_BG = "#1e1e2e"
    CARD_INNER = "#252538"
    TEXT_MAIN = "#cdd6f4"
    TEXT_MUTED = "#a6adc8"
    ACCENT_BLUE = "#89b4fa"
    ACCENT_GREEN = "#a6e3a1"
    ACCENT_RED = "#f38ba8"
    ACCENT_YELLOW = "#f9e2af"
    ACCENT_PURPLE = "#cba6f7"
    ACCENT_CYAN = "#89dceb"

    def __init__(self, with_viewer: bool = True):
        self.with_viewer = with_viewer

        # Core Robotics Subsystems
        self.env = SimulationEnvironment()
        self.vlm = QwenVLClient()
        self.voice = VoiceInterface()
        self.msg_queue: queue.Queue = queue.Queue()
        self.is_executing = False

        self.planner = TaskPlanner(self.env, status_cb=self._queue_robot_status)

        if self.with_viewer:
            self.env.open_viewer()

        # Build GUI
        self.root = tk.Tk()
        self.root.title("AI Robot Arm Controller — Voice & Text Interface")
        self.root.geometry("640x760")
        self.root.minsize(560, 680)
        self.root.configure(bg=self.BG_DARK)

        self._apply_styles()
        self._build_layout()

        # Periodic queue drain for thread-safe UI updates
        self.root.after(50, self._process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Welcome message
        self._log_message("SYSTEM", f"Control panel initialized. VLM: {VLM_MODEL}")
        self._log_message("SYSTEM", "3D Simulator running. Ready for voice or text commands.")

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.BG_DARK, foreground=self.TEXT_MAIN)
        style.configure("TFrame", background=self.BG_DARK)
        style.configure("Card.TFrame", background=self.CARD_BG)
        style.configure("TLabel", background=self.CARD_BG, foreground=self.TEXT_MAIN)

    def _build_layout(self):
        # 1. Header Card
        header_card = tk.Frame(self.root, bg=self.CARD_BG, padx=16, pady=12, highlightthickness=1, highlightbackground=self.CARD_INNER)
        header_card.pack(fill=tk.X, padx=12, pady=(12, 6))

        title_label = tk.Label(
            header_card,
            text="🤖 LOCAL AI ROBOT CONTROLLER",
            font=("Segoe UI", 14, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_CYAN,
        )
        title_label.pack(anchor="w")

        meta_frame = tk.Frame(header_card, bg=self.CARD_BG)
        meta_frame.pack(fill=tk.X, pady=(4, 0))

        vlm_badge = tk.Label(
            meta_frame,
            text=f"● VLM: {VLM_MODEL}",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_INNER,
            fg=self.ACCENT_GREEN,
            padx=8,
            pady=2,
        )
        vlm_badge.pack(side=tk.LEFT, padx=(0, 8))

        robot_badge = tk.Label(
            meta_frame,
            text="● Robot: OpenMANIPULATOR-X",
            font=("Segoe UI", 9),
            bg=self.CARD_INNER,
            fg=self.TEXT_MUTED,
            padx=8,
            pady=2,
        )
        robot_badge.pack(side=tk.LEFT, padx=(0, 8))

        sim_badge = tk.Label(
            meta_frame,
            text="● MuJoCo 3D",
            font=("Segoe UI", 9),
            bg=self.CARD_INNER,
            fg=self.ACCENT_PURPLE,
            padx=8,
            pady=2,
        )
        sim_badge.pack(side=tk.LEFT)

        # 2. Voice Input (Microphone) Section
        voice_card = tk.Frame(self.root, bg=self.CARD_BG, padx=16, pady=12, highlightthickness=1, highlightbackground=self.CARD_INNER)
        voice_card.pack(fill=tk.X, padx=12, pady=6)

        voice_top = tk.Frame(voice_card, bg=self.CARD_BG)
        voice_top.pack(fill=tk.X)

        voice_title = tk.Label(
            voice_top,
            text="🎙️ VOICE CONTROL (MICROPHONE)",
            font=("Segoe UI", 10, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_YELLOW,
        )
        voice_title.pack(side=tk.LEFT)

        self.voice_status_lbl = tk.Label(
            voice_top,
            text="Idle (Ready to listen)",
            font=("Segoe UI", 9, "italic"),
            bg=self.CARD_BG,
            fg=self.TEXT_MUTED,
        )
        self.voice_status_lbl.pack(side=tk.RIGHT)

        # Microphone Action Button
        self.mic_btn = tk.Button(
            voice_card,
            text="🎤  Click to Speak Command",
            font=("Segoe UI", 11, "bold"),
            bg="#2a4365",
            fg="#e2e8f0",
            activebackground="#2b6cb0",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=10,
            command=self._on_mic_click,
        )
        self.mic_btn.pack(fill=tk.X, pady=(8, 2))

        # 3. Text Command Section
        text_card = tk.Frame(self.root, bg=self.CARD_BG, padx=16, pady=12, highlightthickness=1, highlightbackground=self.CARD_INNER)
        text_card.pack(fill=tk.X, padx=12, pady=6)

        text_title = tk.Label(
            text_card,
            text="⌨️ TEXT COMMAND INPUT",
            font=("Segoe UI", 10, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_BLUE,
        )
        text_title.pack(anchor="w", pady=(0, 6))

        input_row = tk.Frame(text_card, bg=self.CARD_BG)
        input_row.pack(fill=tk.X)

        self.cmd_entry = tk.Entry(
            input_row,
            font=("Segoe UI", 11),
            bg=self.CARD_INNER,
            fg=self.TEXT_MAIN,
            insertbackground=self.ACCENT_CYAN,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#3b4261",
            highlightcolor=self.ACCENT_CYAN,
        )
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 8))
        self.cmd_entry.bind("<Return>", lambda e: self._on_send_click())
        self.cmd_entry.insert(0, "place the purple glass on the edge of the table")

        self.send_btn = tk.Button(
            input_row,
            text="▶ Send",
            font=("Segoe UI", 10, "bold"),
            bg="#1e5128",
            fg="#d8f3dc",
            activebackground="#2d6a4f",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            padx=16,
            pady=6,
            command=self._on_send_click,
        )
        self.send_btn.pack(side=tk.RIGHT)

        # 4. Quick Action Presets (One-click pills)
        preset_frame = tk.Frame(text_card, bg=self.CARD_BG)
        preset_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            preset_frame,
            text="Quick presets:",
            font=("Segoe UI", 8, "italic"),
            bg=self.CARD_BG,
            fg=self.TEXT_MUTED,
        ).pack(side=tk.LEFT, padx=(0, 6))

        presets = [
            ("🟣 Purple Glass -> Edge", "place the purple glass on the edge of the table"),
            ("🏗️ Build Tower", "place the cubes one on each other and make a building"),
            ("📦 Red Cube -> Box", "pick up the red cube and put it in the box"),
            ("🔍 Inspect Scene", "look at the table and identify all objects"),
            ("🏠 Home Arm", "go home"),
        ]

        for label, cmd_str in presets:
            btn = tk.Button(
                preset_frame,
                text=label,
                font=("Segoe UI", 8),
                bg=self.CARD_INNER,
                fg=self.TEXT_MAIN,
                activebackground="#3b4261",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                cursor="hand2",
                padx=6,
                pady=2,
                command=lambda c=cmd_str: self._fill_and_execute(c),
            )
            btn.pack(side=tk.LEFT, padx=3)

        # 5. Live Execution Status & Transcript Card
        log_card = tk.Frame(self.root, bg=self.CARD_BG, padx=16, pady=12, highlightthickness=1, highlightbackground=self.CARD_INNER)
        log_card.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 12))

        log_top = tk.Frame(log_card, bg=self.CARD_BG)
        log_top.pack(fill=tk.X, pady=(0, 6))

        log_title = tk.Label(
            log_top,
            text="📋 LIVE ROBOT STATUS & EXECUTION LOG",
            font=("Segoe UI", 10, "bold"),
            bg=self.CARD_BG,
            fg=self.TEXT_MAIN,
        )
        log_title.pack(side=tk.LEFT)

        self.activity_indicator = tk.Label(
            log_top,
            text="● IDLE",
            font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG,
            fg=self.ACCENT_GREEN,
        )
        self.activity_indicator.pack(side=tk.RIGHT)

        # Scrolled Text Box for logs
        self.log_text = scrolledtext.ScrolledText(
            log_card,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#11111b",
            fg="#cdd6f4",
            insertbackground=self.ACCENT_CYAN,
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure color tags
        self.log_text.tag_config("SYSTEM", foreground="#89dceb")
        self.log_text.tag_config("VOICE", foreground="#f9e2af", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("COMMAND", foreground="#89b4fa", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("ROBOT", foreground="#cdd6f4")
        self.log_text.tag_config("SUCCESS", foreground="#a6e3a1", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("ERROR", foreground="#f38ba8", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("TIME", foreground="#6c7086")

    def _on_mic_click(self):
        """Handle click on microphone button."""
        if self.is_executing:
            self._log_message("SYSTEM", "Robot is currently busy executing a task. Please wait.")
            return

        if self.voice.is_listening():
            return

        self.mic_btn.configure(
            text="🔴  Listening... (Speak your command)",
            bg="#9b2c2c",
            activebackground="#c53030",
        )
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
        """Handle send command from text entry box."""
        cmd = self.cmd_entry.get().strip()
        if not cmd:
            return
        self._dispatch_command(cmd)

    def _fill_and_execute(self, cmd_str: str):
        """Preset button helper: fills text box and runs command."""
        self.cmd_entry.delete(0, tk.END)
        self.cmd_entry.insert(0, cmd_str)
        self._dispatch_command(cmd_str)

    def _dispatch_command(self, cmd_text: str):
        """Dispatch natural command to background execution thread."""
        if self.is_executing:
            self._log_message("SYSTEM", "Robot is already busy executing. Please wait.")
            return

        self.is_executing = True
        self._set_activity("EXECUTING", self.ACCENT_YELLOW)
        self.send_btn.configure(state=tk.DISABLED, bg="#334155")
        self.mic_btn.configure(state=tk.DISABLED, bg="#334155")

        self._log_message("COMMAND", f"> {cmd_text}")

        def _worker():
            try:
                success, msg, details = self.planner.execute_natural_command(cmd_text)
                self.msg_queue.put(("EXEC_DONE", (success, msg, details)))
            except Exception as e:
                self.msg_queue.put(("EXEC_DONE", (False, str(e), {})))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _queue_robot_status(self, msg: str):
        """Thread-safe status callback from TaskPlanner."""
        self.msg_queue.put(("ROBOT_STATUS", msg))

    def _process_queue(self):
        """Drain background message queue onto the GUI thread."""
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
                    self.mic_btn.configure(
                        text="🎤  Click to Speak Command",
                        bg="#2a4365",
                        activebackground="#2b6cb0",
                    )
                    self.voice_status_lbl.configure(text="Idle", fg=self.TEXT_MUTED)

                    if success:
                        self._log_message("VOICE", f"🎙️ Transcribed: \"{text_or_err}\"")
                        self.cmd_entry.delete(0, tk.END)
                        self.cmd_entry.insert(0, text_or_err)
                        # Automatically execute recognized voice command
                        self._dispatch_command(text_or_err)
                    else:
                        self._log_message("ERROR", f"Voice input error: {text_or_err}")
                        self._set_activity("IDLE", self.ACCENT_GREEN)

                elif tag == "EXEC_DONE":
                    success, msg, details = payload
                    self.is_executing = False
                    self.send_btn.configure(state=tk.NORMAL, bg="#1e5128")
                    self.mic_btn.configure(state=tk.NORMAL, bg="#2a4365")
                    self._set_activity("IDLE", self.ACCENT_GREEN)

                    if success:
                        self._log_message("SUCCESS", f"✔ Completed: {msg}")
                        if "identified_objects" in details:
                            for obj in details["identified_objects"]:
                                self._log_message("SYSTEM", f"  * {obj.get('color', '')} {obj.get('name', 'obj')}")
                    else:
                        self._log_message("ERROR", f"✖ Failed: {msg}")

        except Exception as e:
            print(f"Queue processing error: {e}")

        # Re-schedule loop
        self.root.after(50, self._process_queue)

    def _set_activity(self, text: str, color: str):
        self.activity_indicator.configure(text=f"● {text}", fg=color)

    def _log_message(self, tag: str, message: str):
        """Append formatted timestamped message to scrolled text log."""
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.insert(tk.END, timestamp, "TIME")
        self.log_text.insert(tk.END, f"{message}\n", tag)
        self.log_text.see(tk.END)

    def _on_close(self):
        """Graceful shutdown of GUI and MuJoCo simulation."""
        if self.env.viewer is not None:
            self.env.close_viewer()
        self.root.destroy()
        sys.exit(0)

    def start(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()


def launch_gui(with_viewer: bool = True):
    app = RobotControlGUI(with_viewer=with_viewer)
    app.start()


if __name__ == "__main__":
    launch_gui()

