import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Pattern, Set


def _load_json(path: Path):
    with path.open("r") as file:
        return json.load(file)


def _compile_object_patterns(object_names: Iterable[str]) -> Dict[str, Pattern[str]]:
    return {
        name: re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
        for name in object_names
    }


def _extract_rule_objects(
    rule: str,
    patterns: Dict[str, Pattern[str]],
    cache: Dict[str, Set[str]],
) -> Set[str]:
    if rule not in cache:
        cache[rule] = {
            name for name, pattern in patterns.items() if pattern.search(rule)
        }
    return cache[rule]


def _build_scene_rules(
    object_list: Dict[str, List[str]], safety_rules: Dict[str, List[str]]
) -> Dict[str, Dict[str, List[str]]]:
    all_object_names = {obj for objects in object_list.values() for obj in objects}
    patterns = _compile_object_patterns(all_object_names)
    rule_objects_cache: Dict[str, Set[str]] = {}

    scene_rules: Dict[str, Dict[str, List[str]]] = {}
    for scene, objects in object_list.items():
        scene_object_set = set(objects)
        scene_rules[scene] = set()
        applicable_rules = set()
        for obj in scene_object_set:
            for rule in safety_rules.get(obj, []):
                referenced_objects = _extract_rule_objects(
                    rule, patterns, rule_objects_cache
                )
                if len(referenced_objects) <= 2:
                    if referenced_objects.issubset(scene_object_set):
                        applicable_rules.add(rule)
                else:
                    if len(referenced_objects & scene_object_set) >= 2:
                        applicable_rules.add(rule)
            scene_rules[scene] = sorted(list(applicable_rules))
    return scene_rules


def main() -> None:
    base_path = Path(__file__).resolve().parent
    object_list_path = base_path / "object_list.json"
    safety_rules_path = base_path / ".." / "safety_rules_object.json"
    output_path = base_path / "scene_safety_rules.json"

    object_list = _load_json(object_list_path)
    safety_rules = _load_json(safety_rules_path)
    scene_rules = _build_scene_rules(object_list, safety_rules)

    with output_path.open("w") as file:
        json.dump(scene_rules, file, indent=4)


if __name__ == "__main__":
    main()
