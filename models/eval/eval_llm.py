import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import json
from datetime import datetime
from env.thor_env import ThorEnv
from models.model.llm import LLMAgent


class EvalLLM:
    '''
    evaluate LLM performance on ALFRED tasks using zero-shot prompting - standalone version
    '''

    def __init__(self, args, manager=None):
        # Initialize without model loading since we're using LLM
        self.args = args
        self.manager = manager

        # Initialize OpenRouter client for DeepSeek
        # Initialize LLM agent
        self.llm_agent = LLMAgent(args)
        self.llm_agent.set_log_method(self.log)
        self.logging = args.debug if hasattr(args, 'debug') else False
        
        # Setup simple logging
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/llm_eval_{timestamp}.txt"
        print(f"Logging to: {self.log_file}")

    def log(self, message):
        """Simple logging to text file"""
        if self.logging:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def test_single_trajectory(self, traj_file_path=None, goto=False):
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
            self.evaluate(env, 0, traj_data, self.args, lock, successes, failures, results, goto=goto)

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
        # TODO: adjust visibility distance as needed, default is 1.5
        env.reset(scene_name)
        
        # setup initial conditions
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    def execute_action(self, env, action_dict, smooth_nav=False):
        """
        Execute action from dict format like {"action": "PickupObject", "object_id": "..."}
        Using direct scene interaction instead of va_interact
        """
        action_name = action_dict.get('action')
        object_id = action_dict.get('object_id', '')
        receptacle_id = action_dict.get('receptacle_id', '')
        if action_name == "PutObject":
            if not receptacle_id:
                raise ValueError("PutObject action requires a receptacle_id")
    
            # We only need the receptacle_id for PutObject (the object to put down is always the held object)
            object_id = receptacle_id

        try:
            # Use the same direct execution approach as eval_llm_step.py
            event, api_action = env.to_thor_api_exec(action_name, object_id, smooth_nav=smooth_nav)
            success = event.metadata['lastActionSuccess']
            error = event.metadata.get('errorMessage', '') if not success else ''
            self.log(f"Action: {action_name}, Object ID: {object_id}, Success: {success}, Error: {error}")
            return success, event, error
        except Exception as e:
            self.log(f"Error: {e} during action {action_name} with object {object_id}")
            return False, None, str(e)
        

    def evaluate(self, env, r_idx, traj_data, args, lock, successes, failures, results, goto=False):
        EvalLLM.log_method = self.log

        # setup scene
        reward_type = 'dense'
        self.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # goal instruction
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        # Log task info
        self.log(f"Task: {goal_instr}")
        self.log(f"Scene: {traj_data['scene']['scene_num']}")
        self.log(f"Objects: {list(traj_data['scene']['object_poses'])}")

        # Get scene information for LLM
        scene_info = self.get_scene_info(env, traj_data)

        # Test goal extraction
        subgoals = self.llm_agent.get_subgoals_from_scene(goal_instr, scene_info)
        
        # Generate LLM plan
        llm_plan = self.llm_agent.generate_plan(subgoals, scene_info, goto=goto)

        # Execute plan
        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        action_idx = 0

        print(f"Generated plan with {len(llm_plan)} actions")
        
        while not done and action_idx < len(llm_plan):
            # break if max_steps reached
            if t >= args.max_steps:
                print("Max steps reached")
                break
                
            action_data = llm_plan[action_idx]
            action = action_data.get('action')
            
            if not action:
                print("Invalid action in plan")
                break
                
            # Check if stop action
            if action.lower() in ['stop', 'end', 'finish', 'done']:
                print("\tLLM predicted STOP")
                break

            # print action
            if args.debug:
                print(f"Step {t}: {action_data}")

            # Execute action in environment
            t_success, event, err = self.execute_action(env, action_data, smooth_nav=args.smooth_nav)
            
            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1
            action_idx += 1

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
                     'llm_plan_length': len(llm_plan),
                     'executed_actions': action_idx}
                     
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
    

    def get_trajectory_files(self, data_dir, split):
        """Get all trajectory files for a given split"""
        import glob
        pattern = f"{data_dir}/{split}/*/trial_*/traj_data.json"
        return glob.glob(pattern)

    def test_batch(self, data_dir="data/json_2.1.0", split="valid_seen", num_runs=5):
        """
        Test evaluation on batch of trajectories with multiple runs
        """
        print(f"Starting batch evaluation on {split} split with {num_runs} runs per trajectory")
        
        # Get all trajectory files for the split
        traj_files = self.get_trajectory_files(data_dir, split)
        print(f"Found {len(traj_files)} trajectory files")
        
        if not traj_files:
            print(f"No trajectory files found in {data_dir}/{split}")
            return
        
        # Initialize batch tracking
        all_successes = []
        all_failures = []
        batch_results = {}
        
        # Create batch log file
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_log_file = f"logs/batch_eval_{split}_{batch_timestamp}.txt"
        
        # Simple lock class for single-threaded use
        class SimpleLock:
            def acquire(self): pass
            def release(self): pass
        
        lock = SimpleLock()
        
        # Track progress
        total_evaluations = len(traj_files) * num_runs
        current_eval = 0
        
        print(f"Running {total_evaluations} total evaluations...")
        
        for traj_idx, traj_file in enumerate(traj_files):
            print(f"\n{'='*60}")
            print(f"Processing trajectory {traj_idx + 1}/{len(traj_files)}: {traj_file}")
            
            try:
                # Load trajectory data
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                    
                # Extract task info
                task_id = traj_data['task_id']
                num_annotations = len(traj_data['turk_annotations']['anns'])
                
                # Run multiple times for this trajectory
                for run_idx in range(num_runs):
                    current_eval += 1
                    print(f"\nRun {run_idx + 1}/{num_runs} for {task_id} ({current_eval}/{total_evaluations})")
                    
                    # Use different annotations for variety if available
                    r_idx = run_idx % num_annotations
                    goal_desc = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
                    print(f"Goal: {goal_desc}")
                    
                    # Create fresh environment for each run
                    env = ThorEnv()
                    
                    try:
                        # Run evaluation
                        run_successes = []
                        run_failures = []
                        run_results = {}
                        
                        self.evaluate(env, r_idx, traj_data, self.args, lock, 
                                    run_successes, run_failures, run_results)
                        
                        # Add run info to results
                        for entry in run_successes + run_failures:
                            entry['run_idx'] = run_idx
                            entry['traj_file'] = traj_file
                            entry['total_runs'] = num_runs
                        
                        # Accumulate results
                        all_successes.extend(run_successes)
                        all_failures.extend(run_failures)
                        
                        # Log run result
                        run_success = len(run_successes) > 0
                        with open(batch_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"Traj {traj_idx+1}/{len(traj_files)}, "
                                f"Run {run_idx+1}/{num_runs}, "
                                f"Task: {task_id}, "
                                f"Success: {run_success}, "
                                f"Goal: {goal_desc}\n")
                        
                    except Exception as e:
                        print(f"Error in run {run_idx + 1}: {e}")
                        with open(batch_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] "
                                f"ERROR in run {run_idx+1}: {e}\n")
                    finally:
                        env.stop()
                    
                    # Print progress every 10 evaluations
                    if current_eval % 10 == 0:
                        current_sr = len(all_successes) / (len(all_successes) + len(all_failures)) if (len(all_successes) + len(all_failures)) > 0 else 0
                        print(f"Progress: {current_eval}/{total_evaluations} ({current_eval/total_evaluations*100:.1f}%) - Current SR: {current_sr:.3f}")
            
            except Exception as e:
                print(f"Error loading trajectory {traj_file}: {e}")
                continue
        
        # Calculate final batch results
        print(f"\n{'='*60}")
        print("BATCH EVALUATION COMPLETE")
        print(f"{'='*60}")
        
        if all_successes or all_failures:
            batch_results = self.get_metrics(all_successes, all_failures)
            
            # Print comprehensive results
            print(f"\nFINAL RESULTS ({split} split, {num_runs} runs per trajectory):")
            print(f"Total Evaluations: {len(all_successes) + len(all_failures)}")
            print(f"Total Trajectories: {len(traj_files)}")
            print(f"Successes: {len(all_successes)}")
            print(f"Failures: {len(all_failures)}")
            
            if batch_results:
                print(f"\nMETRICS:")
                print(f"Success Rate (SR): {batch_results['success']['success_rate']:.3f}")
                print(f"Goal Condition Success Rate (GC): {batch_results['goal_condition_success']['goal_condition_success_rate']:.3f}")
                print(f"Path Length Weighted SR: {batch_results['path_length_weighted_success_rate']:.3f}")
                print(f"Path Length Weighted GC: {batch_results['path_length_weighted_goal_condition_success_rate']:.3f}")
            
            # Save detailed results to JSON
            results_file = f"logs/batch_results_{split}_{batch_timestamp}.json"
            detailed_results = {
                'split': split,
                'num_runs': num_runs,
                'total_trajectories': len(traj_files),
                'total_evaluations': len(all_successes) + len(all_failures),
                'timestamp': batch_timestamp,
                'metrics': batch_results,
                'successes': all_successes,
                'failures': all_failures,
                'args': vars(self.args)
            }
            
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            print(f"\nDetailed results saved to: {results_file}")
            print(f"Log file saved to: {batch_log_file}")
            
            # Task type breakdown
            self.print_task_type_breakdown(all_successes, all_failures)
            
        else:
            print("No evaluations completed successfully")
        
        return all_successes, all_failures, batch_results

    def print_task_type_breakdown(self, successes, failures):
        """Print success rates broken down by task type"""
        from collections import defaultdict
        
        task_stats = defaultdict(lambda: {'successes': 0, 'failures': 0})
        
        for entry in successes:
            task_type = entry.get('type', 'unknown')
            task_stats[task_type]['successes'] += 1
        
        for entry in failures:
            task_type = entry.get('type', 'unknown')
            task_stats[task_type]['failures'] += 1
        
        if task_stats:
            print(f"\nTASK TYPE BREAKDOWN:")
            print(f"{'Task Type':<35} {'Success':<10} {'Total':<10} {'Rate':<10}")
            print("-" * 65)
            
            for task_type, stats in sorted(task_stats.items()):
                total = stats['successes'] + stats['failures']
                rate = stats['successes'] / total if total > 0 else 0
                print(f"{task_type:<35} {stats['successes']:<10} {total:<10} {rate:<10.3f}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_file', type=str, default=None, help='Path to single trajectory JSON file for testing')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--max_fails', type=int, default=5, help='Maximum consecutive action fails before aborting')
    parser.add_argument('--smooth_nav', action='store_true', help='Use smooth navigation')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--llm_model', type=str, default='deepseek/deepseek-chat', help='LLM model to use')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Max tokens for LLM response')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for LLM sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for LLM sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty for LLM')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty for LLM')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--batch', action='store_true', help='Run batch evaluation')
    parser.add_argument('--split', type=str, default='valid_seen', help='Data split to evaluate')
    parser.add_argument('--data_dir', type=str, default='data/json_2.1.0', help='Data directory')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs per trajectory')
    

    args = parser.parse_args()
    
    evaluator = EvalLLM(args)
    if args.batch:
        evaluator.test_batch(args.data_dir, args.split, args.num_runs)
    else:
        evaluator.test_single_trajectory(args.traj_file)
