from typing import Dict, Optional

from .llm import LLMAgent


class LLMPDDLAgent(LLMAgent):
    """LLM agent variant that adds helpers for PDDL-style execution."""

    def __init__(self, args):
        super().__init__(args)

    def get_navigation_target(self, action: Dict, metadata: Optional[Dict]) -> Optional[Dict[str, float]]:
        """Return a position dict for navigation actions.

        The LLM plan may reference objects directly by objectId or by raw
        coordinates encoded in the identifier string. This helper extracts the
        most reliable 3D location we can use for navigation planning.
        """
        if action is None:
            return None

        object_id = action.get("object_id") or action.get("objectId")
        if not object_id:
            return None

        if metadata and "objects" in metadata:
            for obj in metadata["objects"]:
                if obj.get("objectId") == object_id:
                    position = obj.get("position")
                    if position:
                        return {"x": position.get("x", 0.0),
                                "y": position.get("y", 0.0),
                                "z": position.get("z", 0.0)}

        parts = object_id.split("|")
        if len(parts) >= 4:
            try:
                return {"x": float(parts[1]),
                        "y": float(parts[2]),
                        "z": float(parts[3])}
            except ValueError:
                return None

        return None
