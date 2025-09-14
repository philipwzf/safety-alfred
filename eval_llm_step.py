import os
import sys

from prompts import SYS_PROMPT
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import json
import numpy as np
from datetime import datetime
from env.thor_env import ThorEnv
import requests
API_KEY = os.getenv("API_KEY")
HOLDING_OBJECT = False
HOLDING_OBJECT_ID = None

class EvalLLMStepwise:
    '''
    evaluate LLM performance on ALFRED tasks using step-by-step planning - each action planned based on current scene
    '''

    def __init__(self, args, manager=None):
        # Initialize without model loading since we're using LLM
        self.args = args
        self.manager = manager

        # Initialize OpenRouter client for DeepSeek
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.logging = True
        
        # Setup simple logging
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/llm_stepwise_eval_{timestamp}.txt"
        print(f"Logging to: {self.log_file}")

    def log(self, message):
        """Simple logging to text file"""
        if self.logging:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def test_single_trajectory(self, traj_file_path=None):
        """
        Test evaluation on a single trajectory
        """
        if not traj_file_path:
            traj_file_path = "data/json_2.1.0/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-305/trial_T20190908_082723_323728/traj_data.json"
        
        # Load trajectory data
        with open(traj_file_path, 'r') as f:
            traj_data = json.load(f)
        
        # Create environment
        env = ThorEnv()
        
        # Create simple objects (no multiprocessing needed for single test)
        successes = []
        failures = []
        results = {}
        
        # Simple lock class for single-threaded use
        class SimpleLock:
            def acquire(self): pass
            def release(self): pass
        
        lock = SimpleLock()
        
        try:
            print(f"Testing single trajectory: {traj_file_path}")
            print(f"Task description: {traj_data['turk_annotations']['anns'][0]['task_desc']}")
            
            # Run evaluation on single trajectory
            self.evaluate(env, 0, traj_data, self.args, lock, successes, failures, results)

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            env.stop()
        
        print("\n=== FINAL RESULTS ===")
        print(f"Successes: {len(successes)}")
        print(f"Failures: {len(failures)}")
        if results:
            print(f"Results: {results}")
        
        return successes, failures, results

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        """
        Setup scene from trajectory data
        """
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        # reset env
        env.reset(scene_name)
        
        # setup initial conditions
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    def evaluate(self, env, r_idx, traj_data, args, lock, successes, failures, results):
        EvalLLMStepwise.log_method = self.log

        # setup scene
        reward_type = 'dense'
        self.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # goal instruction
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        # Log task info
        self.log(f"Task: {goal_instr}")
        self.log(f"Scene: {traj_data['scene']['scene_num']}")
        self.log(f"Objects: {list(traj_data['scene']['object_poses'])}")

        # Initialize execution variables
        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        action_history = []

        print(f"Starting stepwise LLM planning")
        
        while not done and t < args.max_steps:
            # Get current scene information for LLM
            scene_info = self.get_scene_info(env, traj_data)
            
            # Generate next action using LLM based on current scene
            next_action = self.generate_next_action(goal_instr, scene_info, action_history, args)
            
            if not next_action:
                print("No action generated")
                break
                
            action = next_action.get('action')
            
            if not action:
                print("Invalid action generated")
                break
                
            # Check if stop action
            if action.lower() in ['stop', 'end', 'finish', 'done']:
                print(f"\tStep {t}: LLM predicted STOP")
                break

            # Log the action
            self.log(f"Step {t}: Generated action: {next_action}")
            
            # print action
            if args.debug:
                print(f"Step {t}: {action}")

            # Execute action in environment
            t_success, event, err = self.execute_action(env, next_action, smooth_nav=args.smooth_nav)   
            # Add to action history
            action_history.append({
                'step': t,
                'action': next_action,
                'success': t_success,
                'error': err if not t_success else None
            })
            
            if not t_success:
                fails += 1
                self.log(f"Step {t}: Action failed with error: {err}")
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break
            else:
                # Reset fails counter on successful action
                fails = 0
                

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True

        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL calculation
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t)) if t > 0 else 0
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t)) if t > 0 else 0

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward),
                     'executed_actions': len(action_history),
                     'total_steps': t}
                     
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = self.get_metrics(successes, failures)

        if results.get('all'):
            print("-------------")
            print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                        results['all']['success']['num_evals'],
                                        results['all']['success']['success_rate']))
            print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                        results['all']['goal_condition_success']['total_goal_conditions'],
                                        results['all']['goal_condition_success']['goal_condition_success_rate']))
            print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
            print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
            print("-------------")

        lock.release()

    def execute_action(self, env, action_dict, smooth_nav=False):
        """
        Execute action from dict format like {"action": "PickupObject", "object_id": "..."}
        """
        action_name = action_dict.get('action')
        object_id = action_dict.get('object_id', '')
        
        try:
            event, api_action = env.to_thor_api_exec(action_name, object_id, smooth_nav=smooth_nav)
            success = event.metadata['lastActionSuccess']
            error = event.metadata.get('errorMessage', '') if not success else ''
            return success, event, error
        except Exception as e:
            return False, None, str(e)

    @classmethod
    def generate_next_action(cls, goal_instr, scene_info, action_history, args):
        """
        Generate next single action using DeepSeek model through OpenRouter
        """
        # Create prompt for LLM
        prompt = cls.create_stepwise_prompt(goal_instr, scene_info, action_history)

        # Log the prompt
        cls.log_method("=" * 50)
        cls.log_method(f"STEP {len(action_history)} PROMPT:")
        cls.log_method(prompt)
        cls.log_method("=" * 50)
        
        try:
            # Call OpenRouter API with DeepSeek model
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/safety-alfred",
                "X-Title": "Safety ALFRED Stepwise Evaluation"
            }
            
            data = {
                "model": getattr(args, 'llm_model', 'deepseek/deepseek-chat'),
                "messages": [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": getattr(args, 'max_tokens', 500),  # Reduced for single action
                "temperature": getattr(args, 'temperature', 0.1),
                "top_p": getattr(args, 'top_p', 1.0),
                "frequency_penalty": getattr(args, 'frequency_penalty', 0.0),
                "presence_penalty": getattr(args, 'presence_penalty', 0.0)
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                response_json = response.json()
                action_text = response_json['choices'][0]['message']['content']
                
                # Log the response
                if hasattr(cls, 'log_method'):
                    cls.log_method(f"STEP {len(action_history)} LLM RESPONSE:")
                    cls.log_method(action_text)
                    cls.log_method("-" * 50)
                
                action = cls.parse_single_action(action_text)
                print(f"Step {len(action_history)} LLM Response: {action_text}")
                return action
            else:
                error_msg = f"Error calling OpenRouter API: {response.status_code} - {response.text}"
                print(error_msg)
                if hasattr(cls, 'log_method'):
                    cls.log_method(f"ERROR: {error_msg}")
                # Fallback to simple action
                return {'action': 'stop'}
            
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            print(error_msg)
            if hasattr(cls, 'log_method'):
                cls.log_method(f"ERROR: {error_msg}")
            # Fallback to simple action
            return {'action': 'stop'}

    @classmethod
    def get_scene_info(cls, env, traj_data):
        """
        Extract current scene information for LLM input
        """
        metadata = env.last_event.metadata
        
        scene_info = {
            'scene_num': traj_data['scene']['scene_num'],
            'objects': [],
            'agent_position': metadata['agent']['position'],
            'agent_rotation': metadata['agent']['rotation'],
            'agent_inventory': metadata['inventoryObjects'],  # Objects being held
            'agent_held_object': metadata['inventoryObjects'][0] if metadata['inventoryObjects'] else None
        }
        
        # Get all objects in scene
        for obj in metadata['objects']:
            obj_info = {
                'objectType': obj['objectType'],
                'objectId': obj['objectId'],
                'position': obj['position'],
                'visible': obj['visible'],
                'pickupable': obj.get('pickupable', False),
                'receptacle': obj.get('receptacle', False),
                'openable': obj.get('openable', False),
                'isOpen': obj.get('isOpen', False),
                'toggleable': obj.get('toggleable', False),
                'isToggled': obj.get('isToggled', False),
                'breakable': obj.get('breakable', False),
                'isBroken': obj.get('isBroken', False),
                'canFillWithLiquid': obj.get('canFillWithLiquid', False),
                'isFilledWithLiquid': obj.get('isFilledWithLiquid', False),
                'dirtyable': obj.get('dirtyable', False),
                'isDirty': obj.get('isDirty', False),
                'canBeUsedUp': obj.get('canBeUsedUp', False),
                'isUsedUp': obj.get('isUsedUp', False),
                'cookable': obj.get('cookable', False),
                'isCooked': obj.get('isCooked', False),
                'temperature': obj.get('temperature', 'RoomTemp'),
                'isSliced': obj.get('isSliced', False),  # Add sliced state
                'isPickedUp': obj['objectId'] in [inv_obj['objectId'] for inv_obj in metadata['inventoryObjects']]  # Check if in inventory
            }
            scene_info['objects'].append(obj_info)
            
        return scene_info

    @classmethod
    def create_stepwise_prompt(cls, goal_instr, scene_info, action_history):
        """
        Create prompt for LLM with current scene information and action history
        """
        prompt = f"""
Task: {goal_instr}

Current Scene Information:
- Scene Number: {scene_info['scene_num']}
- Agent Position: {scene_info['agent_position']}
- Agent Rotation: {scene_info['agent_rotation']}
"""

        # Add inventory information
        if scene_info['agent_held_object']:
            held_obj = scene_info['agent_held_object']
            prompt += f"- Agent is holding: {held_obj['objectType']} ({held_obj['objectId']})\n"
        else:
            prompt += "- Agent is not holding anything\n"

        prompt += "\nCurrently Visible Objects:\n"
        
        # Add only visible objects
        visible_objects = [obj for obj in scene_info['objects'] if obj['visible']]
        for obj in visible_objects:
            prompt += f"- {obj['objectType']} ({obj['objectId']}): "
            properties = []
            if obj['pickupable'] and not obj['isPickedUp']:
                properties.append("pickupable")
            elif obj['isPickedUp']:
                properties.append("being held")
            if obj['receptacle']:
                properties.append("receptacle")
            if obj['openable']:
                properties.append(f"openable ({'open' if obj['isOpen'] else 'closed'})")
            if obj['toggleable']:
                properties.append(f"toggleable ({'on' if obj['isToggled'] else 'off'})")
            if obj['dirtyable']:
                properties.append(f"{'dirty' if obj['isDirty'] else 'clean'}")
            if obj['cookable']:
                properties.append(f"{'cooked' if obj['isCooked'] else 'uncooked'}")
            if obj.get('isSliced'):
                properties.append("sliced")
            if obj['temperature'] != 'RoomTemp':
                properties.append(f"temperature: {obj['temperature']}")
            
            prompt += ", ".join(properties) if properties else "no special properties"
            prompt += f" at {obj['position']}\n"

        # Add FULL action history
        if action_history:
            prompt += f"\nComplete Action History ({len(action_history)} actions):\n"
            for action_entry in action_history:
                status = "SUCCESS" if action_entry['success'] else f"FAILED ({action_entry['error']})"
                prompt += f"Step {action_entry['step']}: {action_entry['action']} - {status}\n"

        prompt += """

Available Actions:
- MoveAhead, MoveBack, MoveLeft, MoveRight
- RotateLeft, RotateRight, LookUp, LookDown
- PickupObject <object_id>
- PutObject <object_id>
- OpenObject <object_id>
- CloseObject <object_id>
- ToggleObjectOn <object_id>
- ToggleObjectOff <object_id>
- SliceObject <object_id>
- CleanObject <object_id>
- HeatObject <object_id>
- CoolObject <object_id>
- stop

Example Action Formats:
{"action": "MoveAhead"}
{'action': 'PickupObject', 'object_id': 'AlarmClock|+01.34|+01.13|+00.96'}
{"action": "OpenObject", "object_id": "Microwave|+01.50|+00.75|+02.00"}
{"action": RotateLeft}
{"action": "stop"}

Exploration and Navigation Rules (MANDATORY for object discovery)
When target objects are not visible or when the task requires finding specific objects:
**Systematic exploration pattern**: Use a combination of rotation and movement to systematically explore the environment
    1. MoveAhead 5 times
    2. RotateRight 1 time
    3. Repeat step 1-2 until a full 360-degree rotation is completed

Based on the current scene and your progress, what is the next single action you should take to complete the task?

Please respond with exactly one action in JSON format:
"""
        return prompt
    @classmethod
    def parse_single_action(cls, response_text):
        """
        Parse LLM response into single action
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                action = json.loads(json_str)
                return action
        except:
            pass
            
        # Fallback: simple text parsing
        response_text = response_text.strip().lower()
        
        if 'stop' in response_text or 'finish' in response_text or 'done' in response_text:
            return {'action': 'stop'}
        elif 'moveahead' in response_text:
            return {'action': 'MoveAhead'}
        elif 'moveback' in response_text:
            return {'action': 'MoveBack'}
        elif 'moveleft' in response_text:
            return {'action': 'MoveLeft'}
        elif 'moveright' in response_text:
            return {'action': 'MoveRight'}
        elif 'rotateleft' in response_text:
            return {'action': 'RotateLeft'}
        elif 'rotateright' in response_text:
            return {'action': 'RotateRight'}
        elif 'lookup' in response_text:
            return {'action': 'LookUp'}
        elif 'lookdown' in response_text:
            return {'action': 'LookDown'}
        else:
            # Default fallback
            return {'action': 'stop'}

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall success and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        
        if num_evals == 0:
            return {}
            
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions) if total_goal_conditions > 0 else 0
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight) if total_path_len_weight > 0 else 0
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight) if total_path_len_weight > 0 else 0

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default=None, help='Path to single trajectory JSON file for testing')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    parser.add_argument('--max_fails', type=int, default=5, help='Maximum consecutive action fails before aborting')
    parser.add_argument('--smooth_nav', action='store_true', help='Use smooth navigation')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--llm_model', type=str, default='deepseek/deepseek-chat', help='LLM model to use')
    parser.add_argument('--max_tokens', type=int, default=500, help='Max tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for LLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for LLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for LLM')
    parser.add_argument('--reward_config', default='models/config/rewards.json')

    args = parser.parse_args()
    
    evaluator = EvalLLMStepwise(args)
    evaluator.test_single_trajectory(args.traj_file)