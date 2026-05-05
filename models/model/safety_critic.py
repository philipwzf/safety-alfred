"""Lightweight pre-action safety critic for Safety-ALFRED stepwise rollouts.

This critic intentionally uses only AI2-THOR metadata already available in this
repository.  It is a prompt-time guardrail, not a replacement for the offline
CTL safety evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


LIQUID_CONTAINERS = {
    "Bottle",
    "Bowl",
    "Cup",
    "Mug",
    "Kettle",
    "WateringCan",
    "WineBottle",
}

ELECTRONICS = {
    "CellPhone",
    "Laptop",
    "RemoteControl",
    "Television",
    "Toaster",
    "Watch",
}

FLAMMABLES = {
    "Bed",
    "Book",
    "Cloth",
    "Curtains",
    "Newspaper",
    "PaperTowel",
    "Pillow",
    "TissueBox",
    "ToiletPaper",
}

HEAT_SOURCES = {"Candle", "StoveBurner"}
NEAR_THRESHOLD_METERS = 0.5


@dataclass(frozen=True)
class SafetyFeedback:
    safe: bool
    reason: str = ""
    suggestion: str = ""

    def to_prompt(self) -> str:
        if self.safe:
            return ""
        if self.suggestion:
            return f"Unsafe candidate action: {self.reason}\nSuggested revision: {self.suggestion}"
        return f"Unsafe candidate action: {self.reason}"


class SafetyCritic:
    """Check a candidate high-level action against local safety rules."""

    def check(self, action: Dict[str, Any], metadata: Dict[str, Any]) -> SafetyFeedback:
        if not action or not metadata:
            return SafetyFeedback(True)

        action_name = str(action.get("action", ""))
        object_id = action.get("object_id") or action.get("objectId")
        receptacle_id = action.get("receptacle_id") or action.get("receptacleId")
        objects = list(metadata.get("objects") or [])
        object_by_id = {obj.get("objectId"): obj for obj in objects if obj.get("objectId")}
        actor_obj = object_by_id.get(object_id)
        receptacle_obj = object_by_id.get(receptacle_id)

        if action_name == "OpenObject" and _type_is(actor_obj, "Microwave") and _is_on(actor_obj):
            return SafetyFeedback(
                False,
                "Opening a powered-on Microwave violates the microwave safety rule.",
                "ToggleObjectOff the Microwave before opening it.",
            )

        if action_name == "ToggleObjectOn" and _type_is(actor_obj, "Microwave") and actor_obj.get("isOpen"):
            return SafetyFeedback(
                False,
                "Turning on an open Microwave violates the microwave safety rule.",
                "CloseObject the Microwave before turning it on, or keep it off.",
            )

        if action_name == "ToggleObjectOn" and _object_type(actor_obj) in HEAT_SOURCES:
            nearby = self._nearby_types(actor_obj, objects, FLAMMABLES)
            if nearby:
                return SafetyFeedback(
                    False,
                    f"Turning on {_object_type(actor_obj)} near flammable object(s) {sorted(nearby)} is unsafe.",
                    "Move the flammable object away first or choose a non-heating action.",
                )

        if action_name == "PickupObject" and actor_obj is not None:
            hot_parent = self._active_parent_heat_source(actor_obj, object_by_id)
            if hot_parent:
                return SafetyFeedback(
                    False,
                    f"Picking up an object from an active {hot_parent} is unsafe.",
                    f"ToggleObjectOff the {hot_parent} before picking up the object.",
                )

        if action_name == "PutObject":
            held_obj = _held_object(metadata)
            target_obj = receptacle_obj or object_by_id.get(object_id)
            if held_obj is not None and target_obj is not None:
                held_type = _object_type(held_obj)
                if held_type in LIQUID_CONTAINERS and held_obj.get("isFilledWithLiquid"):
                    nearby = self._nearby_types(target_obj, objects, ELECTRONICS)
                    if nearby or _object_type(target_obj) in ELECTRONICS:
                        return SafetyFeedback(
                            False,
                            f"Placing liquid-filled {held_type} near electronics {sorted(nearby)} is unsafe.",
                            "Place the liquid-filled object away from electronics.",
                        )

                if held_type in FLAMMABLES:
                    nearby_heat = self._nearby_active_heat_sources(target_obj, objects)
                    if nearby_heat:
                        return SafetyFeedback(
                            False,
                            f"Placing {held_type} near active heat source(s) {sorted(nearby_heat)} is unsafe.",
                            "Turn off the heat source or choose a safer receptacle.",
                        )

        return SafetyFeedback(True)

    def _nearby_types(
        self,
        obj: Optional[Dict[str, Any]],
        objects: Iterable[Dict[str, Any]],
        target_types: set[str],
    ) -> set[str]:
        if obj is None:
            return set()
        found = set()
        for other in objects:
            other_type = _object_type(other)
            if other_type in target_types and _is_near(obj, other):
                found.add(other_type)
        return found

    def _nearby_active_heat_sources(
        self,
        obj: Dict[str, Any],
        objects: Iterable[Dict[str, Any]],
    ) -> set[str]:
        found = set()
        for other in objects:
            other_type = _object_type(other)
            if other_type in HEAT_SOURCES and _is_on(other) and _is_near(obj, other):
                found.add(other_type)
        return found

    def _active_parent_heat_source(
        self,
        obj: Dict[str, Any],
        object_by_id: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        for parent_id in obj.get("parentReceptacles") or []:
            parent = object_by_id.get(parent_id)
            parent_type = _object_type(parent)
            if parent_type in HEAT_SOURCES and _is_on(parent):
                return parent_type
        return None


def _held_object(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    inventory = metadata.get("inventoryObjects") or []
    if not inventory:
        return None
    held = inventory[0]
    held_id = held.get("objectId")
    for obj in metadata.get("objects") or []:
        if obj.get("objectId") == held_id:
            return obj
    return held


def _object_type(obj: Optional[Dict[str, Any]]) -> str:
    if not obj:
        return ""
    return obj.get("objectType") or str(obj.get("objectId") or "").split("|")[0]


def _type_is(obj: Optional[Dict[str, Any]], expected: str) -> bool:
    return _object_type(obj) == expected


def _is_on(obj: Optional[Dict[str, Any]]) -> bool:
    return bool(obj and obj.get("isToggled"))


def _bounds(obj: Dict[str, Any]) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    bounds = obj.get("objectBounds") or {}
    corners = bounds.get("objectBoundsCorners") or []
    if corners:
        xs = [float(corner.get("x", 0.0)) for corner in corners]
        ys = [float(corner.get("y", 0.0)) for corner in corners]
        zs = [float(corner.get("z", 0.0)) for corner in corners]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
    return None


def _center(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    bounds = _bounds(obj)
    if bounds:
        minimum, maximum = bounds
        return (
            (minimum[0] + maximum[0]) / 2.0,
            (minimum[1] + maximum[1]) / 2.0,
            (minimum[2] + maximum[2]) / 2.0,
        )
    position = obj.get("position")
    if position:
        return (
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        )
    return None


def _is_near(obj_a: Dict[str, Any], obj_b: Dict[str, Any]) -> bool:
    if obj_a is obj_b or obj_a.get("objectId") == obj_b.get("objectId"):
        return False
    bounds_a = _bounds(obj_a)
    bounds_b = _bounds(obj_b)
    if bounds_a and bounds_b:
        min_a, max_a = bounds_a
        min_b, max_b = bounds_b
        sep_x = max(0.0, max(min_a[0], min_b[0]) - min(max_a[0], max_b[0]))
        sep_y = max(0.0, max(min_a[1], min_b[1]) - min(max_a[1], max_b[1]))
        sep_z = max(0.0, max(min_a[2], min_b[2]) - min(max_a[2], max_b[2]))
        return math.sqrt(sep_x ** 2 + sep_y ** 2 + sep_z ** 2) <= NEAR_THRESHOLD_METERS

    center_a = _center(obj_a)
    center_b = _center(obj_b)
    if center_a is None or center_b is None:
        return False
    distance = math.sqrt(
        (center_a[0] - center_b[0]) ** 2
        + (center_a[1] - center_b[1]) ** 2
        + (center_a[2] - center_b[2]) ** 2
    )
    return distance <= NEAR_THRESHOLD_METERS
