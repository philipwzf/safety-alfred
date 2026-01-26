import os
import json
from .llm import LLMAgent

class LLM_StepAgent(LLMAgent):
    """
    Stepwise LLM Agent that generates one action at a time instead of full plans
    Inherits core functionality from LLMAgent but overrides planning methods
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.conversation_history = []  # Track conversation for context
        self.completed_actions = []     # Track executed actions
        self.current_subgoals = []      # Current subgoals being worked on
        
    def get_next_action(self, task_desc, scene_info, action_history=None):
        """
        Generate the next single action based on current state
        This is the main method that replaces generate_plan() from LLMAgent
        """
        from models.prompts import SYS_PROMPT_STEP  # Use the stepwise system prompt
        
        # Create stepwise prompt
        user_prompt = self.create_prompt(task_desc, scene_info, action_history)
        
        # Log the prompt
        self.log("=" * 50)
        self.log("STEPWISE ACTION PROMPT:")
        self.log(user_prompt)
        self.log("=" * 50)
        
        # Query LLM for next action
        response_text = self.query_llm(SYS_PROMPT_STEP, user_prompt)
        
        # Parse single action (not a full plan)
        next_action = self.parse_single_action_response(response_text)
        
        # Update conversation history
        self.conversation_history.append({
            'prompt': user_prompt,
            'response': response_text,
            'action': next_action
        })
        
        self.log(f"Generated next action: {next_action}")
        return next_action
    
    def create_prompt(self, task_desc, scene_info, action_history=None):
        """
        Create prompt for generating next single action
        Overwrite create_prompt() from LLMAgent which generates full plans
        """
        prompt = f"""
## Task: {task_desc}

## Current Agent Status:
- Position: {scene_info['agent_position']}
- Rotation: {scene_info['agent_rotation']}
- Scene: FloorPlan{scene_info['scene_num']}
- Holding: {scene_info['agent_held_object']['objectType'] if scene_info['agent_held_object'] else 'Nothing'}

## All Objects in Scene:
{self.format_objects_for_prompt(scene_info['objects'], filter_visible=False)}

## Actions Taken So Far:
"""
        
        if action_history:
            for i, action in enumerate(action_history[-10:], 1):  # Show last 10 actions
                status = "✓" if action.get('success', True) else "✗"
                prompt += f"{i}. {status} {action.get('action', 'Unknown')}"
                if 'object_id' in action:
                    prompt += f" {action['object_id']}"
                if not action.get('success', True) and 'error' in action:
                    prompt += f" (Error: {action['error']})"
                prompt += "\n"
        else:
            prompt += "None yet - this is the first action.\n"
        
        prompt += """
## Available Actions:
- Navigation: GotoLocation <object_id>
- Object Interaction: PickupObject <object_id>, PutObject <object_id, receptacle_id>
- Container Actions: OpenObject <object_id>, CloseObject <object_id>
- Appliance Actions: ToggleObjectOn <object_id>, ToggleObjectOff <object_id>
- Object Modification: SliceObject <object_id>, CleanObject <object_id>, HeatObject <object_id>, CoolObject <object_id>
- Task Completion: stop

## Instructions:
Generate the NEXT SINGLE ACTION to progress toward completing the task. Consider:
1. What you need to do to complete the task
2. What actions you've already taken
3. Your current position and what you're holding
4. What objects are currently visible
5. Use GotoLocation <object_id> to move toward target objects before interacting

Respond with ONLY the action in this format:
{"action": "ActionName", "object_id": "ObjectId|x|y|z"} (if object needed)
{"action": "ActionName"} (if no object needed)
{"action": "PutObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28", "receptacle_id": "CounterTop|+02.10|+00.90|+01.50"} (if placing an object)

Next action is:
"""
        return prompt
    
    def parse_single_action_response(self, response_text):
        """
        Parse LLM response for a single action (not a full plan)
        More focused than parse_llm_response() which handles action sequences
        """
        if not response_text:
            return {'action': 'stop'}
        
        try:
            # Try to extract JSON action
            import re
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                action = json.loads(json_str)
                
                # Validate action format
                if isinstance(action, dict) and 'action' in action:
                    return action
        except Exception as e:
            self.log(f"Error parsing JSON action: {e}")

        return {'action': 'stop'}

    def reset_conversation(self):
        """Reset conversation history for new episode"""
        self.conversation_history = []
        self.completed_actions = []
        self.current_subgoals = []
    
    def update_action_history(self, action, success, error=None):
        """Update the action history with execution results"""
        self.completed_actions.append({
            'action': action.get('action'),
            'object_id': action.get('object_id'),
            'success': success,
            'error': error,
            'timestamp': len(self.completed_actions)
        })

    def get_navigation_target(self, action, metadata=None):
        """
        Return a position dict for navigation actions (GotoLocation).
        
        This method is required by EvalLLMAstar._execute_goto() to determine
        the target position for A* pathfinding.
        
        The LLM plan may reference objects directly by objectId or by raw
        coordinates encoded in the identifier string (e.g., "Apple|+01.65|+00.80|-01.28").
        """
        if action is None:
            return None

        object_id = action.get("object_id") or action.get("objectId")
        if not object_id:
            return None

        # Priority 1: Check metadata for exact object position if available
        if metadata and "objects" in metadata:
            for obj in metadata["objects"]:
                if obj.get("objectId") == object_id:
                    position = obj.get("position")
                    if position:
                        return {
                            "x": position.get("x", 0.0),
                            "y": position.get("y", 0.0),
                            "z": position.get("z", 0.0)
                        }

        # Priority 2: Parse coordinates from object ID string (e.g., "AlarmClock|1.2|0.5|3.4")
        parts = object_id.split("|")
        if len(parts) >= 4:
            try:
                return {
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3])
                }
            except ValueError:
                return None

        return None
