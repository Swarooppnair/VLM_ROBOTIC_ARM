"""Strict JSON parser and validator for Qwen2.5-VL responses."""

import re
import json
from typing import Dict, Any, Tuple, Optional


class VLMResponseParser:
    """Extracts, validates, and sanitizes JSON responses from local VLM."""

    ALLOWED_TASKS = {"pick_and_place", "stack", "pick", "inspect", "home", "status", "multi_step"}

    @classmethod
    def extract_json(cls, raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from raw LLM output string."""
        if not raw_text:
            return None

        # Clean markdown code blocks
        clean_text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Look for any outermost curly braces
            match2 = re.search(r"(\{.*\})", clean_text, re.DOTALL)
            if match2:
                json_str = match2.group(1)
            else:
                json_str = clean_text

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    @classmethod
    def validate(cls, parsed: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate parsed response against required robotics schema.
        Returns: (is_valid, error_message, validated_dict)
        """
        if not isinstance(parsed, dict):
            return False, "Response is not a valid JSON object.", {}

        task = parsed.get("task", "").lower()
        if not task:
            return False, "Missing 'task' field in VLM response.", {}

        # Allow slight synonyms
        if "stack" in task or "build" in task or "tower" in task:
            task = "stack"
        elif "pick" in task and ("place" in task or "move" in task):
            task = "pick_and_place"
        elif "pick" in task or "grab" in task or "grasp" in task:
            task = "pick"
        elif "inspect" in task or "identify" in task or "look" in task:
            task = "inspect"
        elif "home" in task or "reset" in task:
            task = "home"

        if task not in cls.ALLOWED_TASKS:
            return False, f"Unsupported task type '{task}'.", {}

        confidence = float(parsed.get("confidence", 0.9))
        if confidence < 0.2:
            return False, f"VLM confidence too low ({confidence:.2f}).", {}

        target_obj = parsed.get("object")
        if target_obj:
            target_obj = str(target_obj).lower().replace(" ", "_")

        destination = parsed.get("destination")
        if destination:
            destination = str(destination).lower()

        # If destination is present and task is pick, promote to pick_and_place
        if destination and task == "pick":
            task = "pick_and_place"

        # For pick / pick_and_place, an object is required
        if task in ("pick", "pick_and_place") and not target_obj:
            return False, "Task requires an 'object' specification.", {}

        validated = {
            "task": task,
            "object": target_obj,
            "destination": destination,
            "base_object": parsed.get("base_object"),
            "stacked_objects": parsed.get("stacked_objects"),
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
            "raw": parsed,
        }

        # If inspect, pass through identified_objects
        if "identified_objects" in parsed:
            validated["identified_objects"] = parsed["identified_objects"]

        return True, "Success", validated
