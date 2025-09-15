import os
import json
import requests
from datetime import datetime

API_KEY = os.getenv("API_KEY")

class LLMAgent:
    """
    LLM Agent for ALFRED task execution using OpenRouter API
    """
    
    def __init__(self, args):
        self.args = args
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.log_method = None  # Will be set by caller
        
    def set_log_method(self, log_method):
        """Set logging method from caller"""
        self.log_method = log_method
    
    def log(self, message):
        """Log message if log method is available"""
        if self.log_method:
            self.log_method(message)
        else:
            print(message)
    
    def get_subgoals_from_scene(self, task_desc, scene_info):
        """
        Generate subgoals from task description and scene metadata
        """
        from models.prompts import SUBGOAL_PROMPT

        user_prompt = f"""
Task: {task_desc}

Scene Information:
- Scene Number: {scene_info['scene_num']}
- Agent Position: {scene_info['agent_position']}
- Agent Rotation: {scene_info['agent_rotation']}

Available Objects:
{self.format_objects_for_prompt(scene_info['objects'], filter_visible=False)}

Generate subgoals for the given task using only the provided objects. Format your output in JSON as specified above. DO NOT include any explanations or additional text. DO NOT wrap the JSON in markdown.
"""
       
        self.log("Generating subgoals from scene...")
        subgoals = self.query_llm(SUBGOAL_PROMPT, user_prompt)
        return subgoals

    def generate_plan(self, subgoals, scene_info):
        """
        Generate action plan from subgoals and scene information
        """
        from models.prompts import ACTION_SEQ_PROMPT
        
        # Create prompt for LLM
        user_prompt = self.create_prompt(subgoals, scene_info)
        
        # Log the prompt
        self.log("=" * 50)
        self.log("PLAN GENERATION PROMPT:")
        self.log(user_prompt)
        self.log("=" * 50)

        system_prompt = ACTION_SEQ_PROMPT
        plan_text = self.query_llm(system_prompt, user_prompt)
        plan = self.parse_llm_response(plan_text)
        
        self.log(f"Generated plan with {len(plan)} actions")
        for i, action in enumerate(plan):
            self.log(f"  {i+1}. {action}")
        
        return plan
    
    def create_prompt(self, subgoals_json, scene_info):
        """
        Create prompt for LLM with subgoals and scene information
        """
        # Parse subgoals with improved handling for markdown format
        subgoals = self.extract_subgoals(subgoals_json)

        prompt = f"""
## Current Agent Status:
- Position: {scene_info['agent_position']}
- Rotation: {scene_info['agent_rotation']}
- Scene: FloorPlan{scene_info['scene_num']}
- Holding: {scene_info['agent_held_object']['objectType'] if scene_info['agent_held_object'] else 'Nothing'}

## Subgoals to Achieve (in order):
"""
        
        # Add numbered subgoals for clarity
        if subgoals:
            for i, subgoal in enumerate(subgoals, 1):
                prompt += f"{i}. {subgoal}\n"
        else:
            prompt += "No valid subgoals found - will generate basic plan\n"
        
        prompt += "\n## Relevant Objects in Scene:\n"
        prompt += self.format_objects_for_prompt(scene_info['objects'], filter_visible=False)
            
        prompt += """
## Available Actions:
- Navigation: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight, LookUp, LookDown
- Object Interaction: PickupObject <object_id>, PutObject <object_id>
- Container Actions: OpenObject <object_id>, CloseObject <object_id>
- Appliance Actions: ToggleObjectOn <object_id>, ToggleObjectOff <object_id>
- Object Modification: SliceObject <object_id>, CleanObject <object_id>, HeatObject <object_id>, CoolObject <object_id>
- Task Completion: stop

## Planning Rules:
1. **Achieve subgoals in the given order** - each subgoal builds on the previous ones
2. **Navigation first** - Navigate to objects before interacting with them
3. **Object interactions** - Pick up objects with PickupObject before moving them
4. **State changes** - Use appropriate actions to change object states
5. **Exact object IDs** - Use the exact object IDs from the subgoals
6. **Find objects first** - Navigate around to find objects if needed
7. **Complete all subgoals** - Don't stop until all subgoals are achieved

## Output Format:
Generate a JSON array of actions that will achieve the subgoals in order:

[
    {"action": "MoveAhead"},
    {"action": "RotateRight"},
    {"action": "PickupObject", "object_id": "AlarmClock|+01.65|+00.80|-01.28"},
    {"action": "MoveLeft"},
    {"action": "ToggleObjectOn", "object_id": "DeskLamp|-01.31|+01.23|-00.99"},
    {"action": "stop"}
]

Generate the action sequence to achieve all subgoals:
"""
        return prompt

    def extract_subgoals(self, subgoals_json):
        """Extract subgoals handling markdown format"""
        if isinstance(subgoals_json, str):
            try:
                data = json.loads(subgoals_json)
                return data.get('subgoals', [])
            except:
                return []
        elif isinstance(subgoals_json, dict):
            return subgoals_json.get('subgoals', [])
        return []

    def query_llm(self, system_prompt, user_prompt):
        """
        Query LLM via OpenRouter API
        """
        try:
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            
            data = {
                "model": getattr(self.args, 'llm_model', 'deepseek/deepseek-chat'),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": getattr(self.args, 'max_tokens', 1000),
                "temperature": getattr(self.args, 'temperature', 0.6),
                "top_p": getattr(self.args, 'top_p', 1.0),
                "frequency_penalty": getattr(self.args, 'frequency_penalty', 0.0),
                "presence_penalty": getattr(self.args, 'presence_penalty', 0.0)
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            
            # Log the response
            self.log("LLM RESPONSE:")
            self.log(content)
            self.log("-" * 50)
            
            return content

        except Exception as e:
            error_msg = f"Unexpected error calling LLM: {e}"
            self.log(f"ERROR: {error_msg}")

        return None

    def parse_llm_response(self, response_text):
        """
        Parse LLM response into action list with robust error handling
        """
        if not response_text:
            return [{'action': 'stop'}]
            
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                plan = json.loads(json_str)
                
                # Validate that actions are properly formatted
                validated_plan = []
                for action in plan:
                    if isinstance(action, dict) and 'action' in action:
                        validated_plan.append(action)
                
                return validated_plan if validated_plan else [{'action': 'stop'}]
                
        except Exception as e:
            self.log(f"Error parsing JSON from LLM response: {e}")

        return [{'action': 'stop'}]

    def format_objects_for_prompt(self, objects, filter_visible=False):
        """
        Format objects list into a string for prompt inclusion
        
        Args:
            objects: List of object dictionaries from scene_info['objects']
            filter_visible: If True, only include visible objects
            include_position: If True, include position coordinates
        
        Returns:
            Formatted string ready for prompt inclusion
        """
        if not objects:
            return "No objects available.\n"
        
        prompt_section = ""
        
        for obj in objects:
            # Filter by visibility if requested
            if filter_visible and not obj.get('visible'):
                continue
                
            # Start with object type and ID
            line = f"- {obj['objectType']} ({obj['objectId']}): "
            
            # Collect object properties
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
            
            line += ", ".join(properties) if properties else "no special properties"
            line += f" at {obj['position']}\n"
            
            # Add visibility indicator if not filtering by visible
            if not filter_visible:
                visibility = "visible" if obj.get('visible') else "not visible"
                line += f" [{visibility}]"
                
            prompt_section += line + "\n"
        
        return prompt_section