import json
from pathlib import Path
from typing import Iterable, List

def _scene_numbers() -> List[int]:
    scene_ranges: Iterable[range] = (
        range(1, 31),
        range(201, 231),
        range(301, 331),
        range(401, 431),
    )
    return [scene for scene_range in scene_ranges for scene in scene_range]
def main() -> None:
    metadata_dir = Path("metadata")
    object_types_by_scene = {}
    scene_numbers = _scene_numbers()
    for scene_number in scene_numbers:
        metadata_path = metadata_dir / f"FloorPlan{scene_number}_pass.json"
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        object_types = {
            obj["objectType"]
            for obj in metadata.get("objects", [])
            if "objectType" in obj
        }
        object_types.add("Agent")
        object_types_by_scene[scene_number] = sorted(object_types)

    with Path("object_list.json").open("w") as f:
        json.dump(object_types_by_scene, f, indent=4)


if __name__ == "__main__":
    main()
