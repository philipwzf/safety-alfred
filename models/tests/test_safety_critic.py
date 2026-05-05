import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.model.safety_critic import SafetyCritic


def _obj(object_id, object_type, x, z, **extra):
    return {
        "objectId": object_id,
        "objectType": object_type,
        "position": {"x": x, "y": 0.0, "z": z},
        **extra,
    }


def test_rejects_powered_microwave_open():
    metadata = {
        "objects": [
            _obj("Microwave|0|0|0", "Microwave", 0, 0, isToggled=True),
        ],
    }

    feedback = SafetyCritic().check(
        {"action": "OpenObject", "object_id": "Microwave|0|0|0"},
        metadata,
    )

    assert not feedback.safe
    assert "Microwave" in feedback.reason


def test_rejects_lit_candle_near_flammable():
    metadata = {
        "objects": [
            _obj("Candle|0|0|0", "Candle", 0, 0),
            _obj("Book|0.1|0|0", "Book", 0.1, 0),
        ],
    }

    feedback = SafetyCritic().check(
        {"action": "ToggleObjectOn", "object_id": "Candle|0|0|0"},
        metadata,
    )

    assert not feedback.safe
    assert "Book" in feedback.reason


def test_allows_unrelated_pickup():
    metadata = {
        "objects": [
            _obj("Apple|0|0|0", "Apple", 0, 0),
            _obj("Book|2|0|0", "Book", 2, 0),
        ],
    }

    feedback = SafetyCritic().check(
        {"action": "PickupObject", "object_id": "Apple|0|0|0"},
        metadata,
    )

    assert feedback.safe
