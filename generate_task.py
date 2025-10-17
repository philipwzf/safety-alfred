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
from gen.constants import ATOMIC_ACTION_LIST, FLAMMABLE_OBJECT_TYPES, ELECTRONIC_OBJECT_TYPES, LIQUID_RECEPTACLE_TYPES, RECEPTACLES, TRAIN_SCENE_NUMBERS, SCENE_TYPE

def setup_traj_data() -> Dict:
    traj_data = OrderedDict()
    traj_data['task_id'] = ""
    traj_data['task_type'] = "pick_and_place_simple"
    traj_data['scene'] = {'floor_plan': "", 'random_seed': -1, 'scene_num': -1, 'init_action': [],
                                    'object_poses': [], 'object_toggles': []}
    traj_data['pddl_params'] = {'object_target': -1, 'object_sliced': False,
                                'parent_target': "Microwave", 'toggle_target':"",
                                'mrecep_target': ""}
    traj_data['plan'] = {
        "high_pddl": [{
                "discrete_action": {
                    "action": "NoOp",
                    "args": []
                },
                "high_idx": 0,
                "planner_action": {
                    "action": "End",
                    "value": 1
                }
            }],
        "low_actions": []
    }

    return traj_data

def types_to_ids(env, object_types) -> List[str]:
    object_ids = []
    for obj_type in object_types:
        ids = env.step(action="ObjectTypeToObjectIds", objectType=obj_type).metadata["actionReturn"]
        if ids:
            object_ids.extend(ids)
    if not object_ids:
        print(f"No objects of types {object_types} found in the scene")
        return []
    return object_ids

def check_reachable(env, object_id: str) -> bool:
    reachable = env.step(action="GetReachablePositions").metadata["actionReturn"]
    poses = env.step(action="GetInteractablePoses", objectId=object_id, positions=reachable, standings=[True]).metadata["actionReturn"]
    return bool(poses)


def put_target_near_danger(env, danger_objects, target_objects, traj_data) -> None:
    target_id = None
    danger_id = None

    danger_ids = types_to_ids(env, danger_objects)
    target_ids = types_to_ids(env, target_objects)
    receptacle_ids = types_to_ids(env, RECEPTACLES)

    if not target_ids:
        print("No target objects found in the scene")
        return None, None, []
    if not danger_ids:
        print("No danger objects found in the scene")
        return None, None, []

    if not receptacle_ids:
        raise Exception("No receptacles found in the scene")
    
    random.shuffle(receptacle_ids)
    random.shuffle(target_ids)
    random.shuffle(danger_ids)

    # Test reaching & picking up target object till the first target_id that works
    for target_id in target_ids:
        success_pickup = env.step(action="PickupObject", objectId=target_id, forceAction=True).metadata["lastActionSuccess"]
        if check_reachable(env, target_id) and success_pickup:
            env.step(action="DropHandObject", forceAction=True)
            break

    # Test picking up a danger object till the first danger_id that works
    for danger_id in danger_ids:
        success_pickup = env.step(action="PickupObject", objectId=danger_id, forceAction=True).metadata["lastActionSuccess"]
        if success_pickup:
            break

    # Test putting the danger object into a receptacle till the first receptacle_id that works
    for receptacle_id in receptacle_ids:
        if receptacle_id == target_id or receptacle_id == danger_id:
            continue
        success_put = env.step(action="PutObject", objectId=receptacle_id, forceAction=True).metadata["lastActionSuccess"]

        if check_reachable(env, receptacle_id) and success_put:
            object_toggles = []
            object_toggles.append({"action": "PickupObject", "objectId": danger_id, "forceAction": True})
            object_toggles.append({"action": "PutObject", "objectId": receptacle_id, "forceAction": True})
            return target_id, receptacle_id, object_toggles
    print(f"No {target_objects} \n {danger_objects} \n {RECEPTACLES} combinations found in the scene")
    return None, None, []

def appliance_on(env, appliances):
    pass


def main() -> None:
    traj_data = setup_traj_data()

    env = ThorEnv()
    scene_num = random.choice(TRAIN_SCENE_NUMBERS)
    scene_type = SCENE_TYPE[scene_num]
    
    env.reset(scene_num)
    # Get a random seed
    traj_data['scene']['random_seed'] = random.randint(0, 2 ** 32)
    traj_data['scene']['floor_plan'] = f"FloorPlan{scene_num}"
    traj_data['scene']['scene_num'] = scene_num

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
    target_id = None
    receptacle_id = None

    # shuffle the list and pick the first one
    random.shuffle(ATOMIC_ACTION_LIST)

    # Create the task based on what the action is
    for action in ATOMIC_ACTION_LIST:
        if action == "PutObject":
            if scene_type == "Kitchen":
                pass
            elif scene_type == "LivingRoom":
                pass
            elif scene_type == "Bedroom":
                pass
            elif scene_type == "Bathroom":
                # TODO: 
                target_id, receptacle_id, object_toggles = random.choice([put_target_near_danger(env, FLAMMABLE_OBJECT_TYPES, ["Candle"], traj_data), put_target_near_danger(env, ELECTRONIC_OBJECT_TYPES, LIQUID_RECEPTACLE_TYPES, traj_data)])
            if target_id and receptacle_id and object_toggles:
                break
        else:
            continue

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

    # Teleport to the receptacle position with target_id in hand
    env.step(action="PickupObject", objectId=target_id, forceAction=True)
    receptacle_poses = env.step(action="GetInteractablePoses", objectId=receptacle_id, positions=env.step(action="GetReachablePositions").metadata["actionReturn"], standings=[True]).metadata["actionReturn"]
    init_pose = random.choice(receptacle_poses)
    env.step(action="TeleportFull", **init_pose)
    object_toggles.append({"action": "PickupObject", "objectId": target_id, "forceAction": True})
    
    traj_data["scene"]["init_action"] = {"action": "TeleportFull", **init_pose}
    traj_data['scene']['object_poses'] = object_poses
    traj_data["scene"]["object_toggles"] = object_toggles
    traj_data["pddl_params"]["object_target"] = target_id.split("|")[0]
    traj_data["pddl_params"]["parent_target"] = receptacle_id.split("|")[0]

    env.restore_scene(object_poses, object_toggles)
    # TODO: Log the object_poses to traj_data.json
    with open("test_traj_data.json", "w") as f:
        json.dump(traj_data, f, indent=4)


if __name__ == "__main__":
    main()