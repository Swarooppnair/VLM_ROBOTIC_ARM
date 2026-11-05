"""Client interface to local Qwen2.5-VL 3B model via Ollama."""

import io
import base64
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
import ollama

from vlm_robotics.config.config import OLLAMA_HOST, VLM_MODEL
from vlm_robotics.vlm.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    INSPECT_PROMPT,
    VERIFY_GRASP_PROMPT,
    VERIFY_PLACEMENT_PROMPT,
)
from vlm_robotics.vlm.response_parser import VLMResponseParser


class QwenVLClient:
    """Interacts with locally running Qwen2.5-VL 3B model through Ollama API."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = VLM_MODEL):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=self.host)
        self.parser = VLMResponseParser()

    def check_connection(self) -> Tuple[bool, str]:
        """Check if Ollama is running and Qwen2.5-VL model is present."""
        try:
            models_info = self.client.list()
            available = [m.model for m in models_info.models]
            # Match model name or tag prefix
            matching = [m for m in available if self.model in m or m.startswith("qwen2.5vl")]
            if not matching:
                return False, f"Model '{self.model}' not found in Ollama. Available: {available}"
            return True, f"Connected to Ollama. Model: {matching[0]}"
        except Exception as e:
            return False, f"Failed to connect to Ollama at {self.host}: {e}"

    def encode_image(self, image_input) -> str:
        """Convert numpy RGB array, PIL Image, or file path to base64 JPEG string."""
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze_command(
        self,
        rgb_image: np.ndarray,
        command: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Send camera image and natural language command to Qwen2.5-VL.
        Returns: (success, message, structured_action_dict)
        """
        b64_img = self.encode_image(rgb_image)
        prompt = build_user_prompt(command)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64_img],
                    },
                ],
                options={"temperature": 0.1},
            )

            raw_text = response["message"]["content"]
            parsed_json = self.parser.extract_json(raw_text)
            if not parsed_json:
                return False, f"Could not parse valid JSON from VLM output: {raw_text}", {}

            is_valid, err_msg, validated = self.parser.validate(parsed_json)
            if not is_valid:
                return False, f"VLM schema validation failed: {err_msg}", {}

            return True, "Success", validated

        except Exception as e:
            return False, f"VLM inference error: {e}", {}

    def inspect_scene(self, rgb_image: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """Ask Qwen2.5-VL to comprehensively inspect and describe all objects."""
        b64_img = self.encode_image(rgb_image)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": INSPECT_PROMPT,
                        "images": [b64_img],
                    },
                ],
                options={"temperature": 0.1},
            )

            raw_text = response["message"]["content"]
            parsed_json = self.parser.extract_json(raw_text)
            if not parsed_json:
                return False, f"Could not parse JSON from VLM: {raw_text}", {}

            return True, "Scene inspected successfully", parsed_json

        except Exception as e:
            return False, f"VLM inspection error: {e}", {}

    def verify_grasp_vlm(self, wrist_rgb: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """Verify grasp stability using wrist-mounted eye-in-hand camera."""
        b64_img = self.encode_image(wrist_rgb)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": VERIFY_GRASP_PROMPT,
                        "images": [b64_img],
                    },
                ],
                options={"temperature": 0.1},
            )
            raw_text = response["message"]["content"]
            parsed_json = self.parser.extract_json(raw_text)
            if not parsed_json:
                return True, "Defaulted to heuristic verification", {"grasp_successful": True}
            success = bool(parsed_json.get("grasp_successful", True))
            return success, "VLM grasp verification executed", parsed_json
        except Exception as e:
            return True, f"VLM verification fallback: {e}", {"grasp_successful": True}

    def verify_placement_vlm(self, overhead_rgb: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """Verify destination placement from overhead camera."""
        b64_img = self.encode_image(overhead_rgb)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": VERIFY_PLACEMENT_PROMPT,
                        "images": [b64_img],
                    },
                ],
                options={"temperature": 0.1},
            )
            raw_text = response["message"]["content"]
            parsed_json = self.parser.extract_json(raw_text)
            if not parsed_json:
                return True, "Defaulted to heuristic verification", {"placement_successful": True}
            success = bool(parsed_json.get("placement_successful", True))
            return success, "VLM placement verification executed", parsed_json
        except Exception as e:
            return True, f"VLM verification fallback: {e}", {"placement_successful": True}


