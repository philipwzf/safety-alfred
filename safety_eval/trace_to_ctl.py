"""Utilities to convert recorded evaluation traces into CTL-compatible data structures.

The helper functions in this module translate the per-step JSON traces produced
by ``EpisodeLogger`` into the ``nodes``/``edges`` representation expected by the
legacy CTL tooling.  This allows us to reuse the existing safety checking stack
without re-generating VirtualHome-style trajectory trees.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json


def trace_to_ctl_sequence(trace_steps: Sequence[Dict[str, Any]]) -> List[Union[Dict[str, List[str]], str]]:
    """Convert a list of trace steps into the node/edge format expected by ``CTLParser``.

    Each step produced during evaluation contains the executed action and the
    resulting ``event_metadata`` snapshot.  The legacy CTL tooling expects an
    alternating list of state dictionaries and action-description strings.  We
    synthesise a best-effort translation by emitting the first state as the
    root, then interleaving the remaining states with formatted THOR actions.
    """

    if not trace_steps:
        raise ValueError("trace_to_ctl_sequence requires at least one step")

    ctl_sequence: List[Union[Dict[str, List[str]], str]] = []

    previous_state: Dict[str, List[str]] | None = None
    for index, step in enumerate(trace_steps):
        metadata = step.get("event_metadata") or {}
        state_dict = _state_from_metadata(metadata)

        if index == 0:
            # The first state becomes the root of the CTL trajectory.
            ctl_sequence.append(state_dict)
            previous_state = state_dict
            continue

        # Insert the action string representing the transition from the
        # previous step to the current state, then append the new state.
        action_string = _format_action(step.get("thor_action"), step.get("plan_action"))
        ctl_sequence.append(action_string)
        ctl_sequence.append(state_dict)
        previous_state = state_dict

    return ctl_sequence


def trace_file_to_ctl_sequence(trace_path: Union[str, Path]) -> List[Union[Dict[str, List[str]], str]]:
    """Load a saved trace JSON file and convert it to the CTL sequence format."""

    data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    if not isinstance(data, Sequence):
        raise TypeError(f"Expected sequence of steps in {trace_path}")
    return trace_to_ctl_sequence(data)  # type: ignore[arg-type]


def _state_from_metadata(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build the ``{"nodes": ..., "edges": ...}`` representation for a state."""

    nodes: List[str] = []
    edges: List[str] = []

    inventory_ids = {
        _normalise_object_id(inv.get("objectId") or inv.get("object_id") or inv.get("name"))
        for inv in (metadata.get("inventoryObjects") or [])
        if inv
    }

    object_entries: List[_ObjectEntry] = []

    # Build object-centric state strings.
    for obj in metadata.get("objects", []) or []:
        object_id = _normalise_object_id(obj.get("objectId") or obj.get("name"))
        if not object_id:
            continue

        state_tags = _object_state_tags(obj, inventory_ids)
        nodes.append(f"{object_id}, states:[{', '.join(sorted(state_tags))}]")

        bbox = _extract_bounding_box(obj.get("objectBounds"))
        parent_recs = {
            _normalise_object_id(rec)
            for rec in (obj.get("parentReceptacles") or [])
            if _normalise_object_id(rec)
        }
        receptacle_contents = {
            _normalise_object_id(child)
            for child in (obj.get("receptacleObjectIds") or [])
            if _normalise_object_id(child)
        }
        object_entries.append(_ObjectEntry(object_id, bbox, parent_recs, receptacle_contents))

    # Agent as an object for relational checks.
    agent_meta = metadata.get("agent", {}) or {}
    agent_states = []
    if agent_meta.get("isStanding"):
        agent_states.append("standing")
    if agent_meta.get("isCrouching"):
        agent_states.append("crouching")
    if agent_meta.get("isFalling"):
        agent_states.append("falling")
    nodes.append(f"agent, states:[{', '.join(agent_states) if agent_states else 'present'}]")
    object_entries.append(_ObjectEntry("agent", _agent_bounding_box(agent_meta)))
    edges = sorted(_compute_spatial_relationships(object_entries))

    # Ensure uniqueness while preserving deterministic order.
    nodes = sorted(dict.fromkeys(nodes))

    return {"nodes": nodes, "edges": edges}


def _object_state_tags(obj: Dict[str, Any], inventory_ids: Iterable[str]) -> List[str]:
    """Derive textual state tags for a THOR object."""

    tags: List[str] = []
    if obj.get("visible"):
        tags.append("visible")
    if obj.get("pickupable"):
        tags.append("pickupable")
    if obj.get("isPickedUp") or _normalise_object_id(obj.get("objectId")) in inventory_ids:
        tags.append("held")
    if obj.get("openable"):
        tags.append("open" if obj.get("isOpen") else "closed")
    if obj.get("toggleable"):
        tags.append("powered_on" if obj.get("isToggled") else "powered_off")
    if obj.get("dirtyable"):
        tags.append("dirty" if obj.get("isDirty") else "clean")
    if obj.get("cookable"):
        tags.append("cooked" if obj.get("isCooked") else "raw")
    if obj.get("sliceable"):
        tags.append("sliced" if obj.get("isSliced") else "unsliced")

    temperature = obj.get("temperature") or obj.get("ObjectTemperature")
    if isinstance(temperature, str) and temperature:
        tags.append(f"temp:{temperature.lower()}")

    return tags or ["default"]


def _format_action(thor_action: Dict[str, Any] | None, plan_action: Dict[str, Any] | None) -> str:
    """Format an action dictionary into the legacy string representation."""

    source = thor_action or plan_action or {}
    action_name = source.get("action") or "NoOp"
    parts = [f"'{action_name}'"]

    # Include up to two relevant object identifiers to retain context.
    for key in ("objectId", "object_id", "receptacleId", "receptacle_id", "targetObjectId", "object2Id"):
        value = source.get(key)
        if value:
            normalised = _normalise_object_id(value)
            parts.append(f"'{normalised}'")

    return "action: " + " ".join(parts)


def _normalise_object_id(object_id: Any) -> str:
    """Convert a THOR object identifier to a canonical string."""

    if not object_id:
        return ""
    if isinstance(object_id, str):
        return object_id.strip()
    return str(object_id)


class _BoundingBox:
    __slots__ = ("min", "max", "center")

    def __init__(self, minimum: Tuple[float, float, float], maximum: Tuple[float, float, float]):
        self.min = minimum
        self.max = maximum
        self.center = (
            (minimum[0] + maximum[0]) / 2.0,
            (minimum[1] + maximum[1]) / 2.0,
            (minimum[2] + maximum[2]) / 2.0,
        )


class _ObjectEntry:
    __slots__ = ("identifier", "bbox", "parent_receptacles", "receptacle_contents")

    def __init__(
        self,
        identifier: str,
        bbox: Optional[_BoundingBox],
        parent_receptacles: Optional[Iterable[str]] = None,
        receptacle_contents: Optional[Iterable[str]] = None,
    ):
        self.identifier = identifier
        self.bbox = bbox
        self.parent_receptacles = frozenset(filter(None, parent_receptacles or []))
        self.receptacle_contents = frozenset(filter(None, receptacle_contents or []))


def _extract_bounding_box(bounds: Optional[Dict[str, Any]]) -> Optional[_BoundingBox]:
    if not bounds:
        return None
    corners = bounds.get("objectBoundsCorners") or []
    if not corners:
        return None

    min_x = min(corner.get("x", 0.0) for corner in corners)
    max_x = max(corner.get("x", 0.0) for corner in corners)
    min_y = min(corner.get("y", 0.0) for corner in corners)
    max_y = max(corner.get("y", 0.0) for corner in corners)
    min_z = min(corner.get("z", 0.0) for corner in corners)
    max_z = max(corner.get("z", 0.0) for corner in corners)

    return _BoundingBox((min_x, min_y, min_z), (max_x, max_y, max_z))


def _agent_bounding_box(agent_meta: Dict[str, Any]) -> Optional[_BoundingBox]:
    position = agent_meta.get("position") if agent_meta else None
    if not position:
        return None

    x = float(position.get("x", 0.0))
    y = float(position.get("y", 0.0))
    z = float(position.get("z", 0.0))

    half_width = 0.25
    half_depth = 0.25
    half_height = 0.9

    minimum = (x - half_width, y - half_height, z - half_depth)
    maximum = (x + half_width, y + half_height, z + half_depth)
    return _BoundingBox(minimum, maximum)


def _compute_spatial_relationships(objects: Sequence[_ObjectEntry]) -> List[str]:
    relations: set[str] = set()

    for i, obj_a in enumerate(objects):
        bbox_a = obj_a.bbox
        if bbox_a is None:
            continue
        for j, obj_b in enumerate(objects):
            if i == j:
                continue

            if (
                obj_b.identifier in obj_a.parent_receptacles
                or obj_a.identifier in obj_b.receptacle_contents
            ):
                relations.add(f"INSIDE({obj_a.identifier}, {obj_b.identifier})")

            bbox_b = obj_b.bbox
            if bbox_b is None:
                continue

            if _is_inside(bbox_a, bbox_b):
                relations.add(f"INSIDE({obj_a.identifier}, {obj_b.identifier})")

            if _is_on_top(bbox_a, bbox_b):
                relations.add(f"ONTOP({obj_a.identifier}, {obj_b.identifier})")

        for obj_b in objects[i + 1 :]:
            bbox_b = obj_b.bbox
            if bbox_b is None:
                continue
            if _is_near(bbox_a, bbox_b):
                first, second = sorted([obj_a.identifier, obj_b.identifier])
                relations.add(f"NEAR({first}, {second})")

    return sorted(relations)


def _is_near(bbox_a: _BoundingBox, bbox_b: _BoundingBox, threshold: float = 0.25) -> bool:
    ax, ay, az = bbox_a.center
    bx, by, bz = bbox_b.center
    distance = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
    return distance <= threshold


def _horizontal_overlap(bbox_a: _BoundingBox, bbox_b: _BoundingBox) -> bool:
    overlap_x = not (bbox_a.max[0] < bbox_b.min[0] or bbox_b.max[0] < bbox_a.min[0])
    overlap_z = not (bbox_a.max[2] < bbox_b.min[2] or bbox_b.max[2] < bbox_a.min[2])
    return overlap_x and overlap_z


def _is_on_top(upper: _BoundingBox, lower: _BoundingBox, vertical_epsilon: float = 0.08) -> bool:
    if upper.min[1] < lower.max[1] - vertical_epsilon or upper.min[1] > lower.max[1] + vertical_epsilon:
        return False
    return _horizontal_overlap(upper, lower)


def _is_inside(inner: _BoundingBox, outer: _BoundingBox, margin: float = 0.02) -> bool:
    return (
        inner.min[0] >= outer.min[0] - margin
        and inner.max[0] <= outer.max[0] + margin
        and inner.min[1] >= outer.min[1] - margin
        and inner.max[1] <= outer.max[1] + margin
        and inner.min[2] >= outer.min[2] - margin
        and inner.max[2] <= outer.max[2] + margin
    )

trace = "logs/trajectories/trial_T20190909_135724_571540/r0_20250919_002926.json"
ctl_sequence = trace_file_to_ctl_sequence(trace)
with open("test.json", "w") as f:
    json.dump(ctl_sequence, f, indent=2)