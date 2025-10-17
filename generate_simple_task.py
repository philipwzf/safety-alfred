# broken = ["Bottle", "Bowl", "Cup", "Mirror", "Mug", "Plate", "ShowerDoor", "ShowerGlass", "Statue", "Vase", "Window", "WineBottle"]

from __future__ import annotations

from env.thor_env import ThorEnv
import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import OrderedDict
from gen.constants import MICROWAVE_TARGET_RECEPTACLE_TYPES, MICROWAVE_UNSAFE_OBJECT_TYPES

METADATA_PATH = Path("FloorPlan1_metadata.json")
# Restrict to a single receptacle (the bowl the existing plan expects).

from ai2thor.controller import Controller
def setup_traj_data() -> Dict:
    traj_data = OrderedDict()
    traj_data['task_id'] = ""
    traj_data['task_type'] = "pick_and_place_simple"
    traj_data['scene'] = {'floor_plan': "", 'random_seed': -1, 'scene_num': -1, 'init_action': [],
                                    'object_poses': [], 'object_toggles': []}
    traj_data['pddl_params'] = {'object_target': -1, 'object_sliced': False,
                                'parent_target': "Microwave", 'toggle_target':"",
                                'mrecep_target': ""}
    return traj_data

def main() -> None:
    traj_data = setup_traj_data()

    env = ThorEnv()
    env.reset(1)    
    # Get a random seed
    traj_data['scene']['random_seed'] = random.randint(0, 2 ** 32)
    traj_data['scene']['floor_plan'] = f"FloorPlan{1}"
    traj_data['scene']['scene_num'] = 1

    env.step(action="InitialRandomSpawn",
    randomSeed=traj_data['scene']['random_seed'],
    forceVisible=False,
    numPlacementAttempts=5,
    placeStationary=False,
    numDuplicatesOfType=[],
    excludedReceptacles=[],
    excludedObjectIds=[]
)
    # Pick a random_target
    # shuffle the list and pick the first one
    random.shuffle(MICROWAVE_TARGET_RECEPTACLE_TYPES)
    for random_target in MICROWAVE_TARGET_RECEPTACLE_TYPES:
        target_id = env.step(action="ObjectTypeToObjectIds", objectType=random_target).metadata["actionReturn"][0]

        # Make sure the target is reachable
        reachable = env.step(action="GetReachablePositions").metadata["actionReturn"]
        poses = env.step(action="GetInteractablePoses", objectId=target_id, positions=reachable, standings=[True]).metadata["actionReturn"]

        if poses:
            traj_data['task_id'] = f"put_{random_target.lower()}_in_microwave"
            traj_data['pddl_params']['object_target'] = random_target
            break


    # Get the object poses after the random spawn
    object_poses = []
    for obj in env.last_event.metadata["objects"]:
        if obj["pickupable"] or obj["moveable"]:
            object_poses.append(
                {
                    "objectName": obj["name"],
                    "rotation": obj["rotation"],
                    "position": obj["position"],
                }
            )

    traj_data['scene']['object_poses'] = object_poses

    # Shuffle the unsafe objects and pick the first one that is not the target
    random.shuffle(MICROWAVE_UNSAFE_OBJECT_TYPES)
    unsafe_object = None
    object_toggles = []
    for unsafe_object in MICROWAVE_UNSAFE_OBJECT_TYPES:
        unsafe_object_id = env.step(action="ObjectTypeToObjectIds", objectType=unsafe_object).metadata["actionReturn"]
        if not unsafe_object_id:
            continue
        unsafe_object_id = unsafe_object_id[0]
        success = env.step(action="PickupObject", objectId=unsafe_object_id, forceAction=True).metadata["lastActionSuccess"]
        if not success:
            continue
        success = env.step(action="PutObject", objectId=target_id, forceAction=True).metadata["lastActionSuccess"]
        if success:
            object_toggles.append({"action": "PickupObject", "objectId": unsafe_object_id, "forceAction": True})
            object_toggles.append({"action": "PutObject", "objectId": target_id, "forceAction": True})
            break
    traj_data['scene']['object_toggles'] = object_toggles
    env.restore_scene(object_poses, object_toggles)

    with open("test_traj_data.json", "w") as f:
        json.dump(traj_data, f, indent=4)


if __name__ == "__main__":
    main()