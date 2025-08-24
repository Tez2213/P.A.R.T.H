from pathlib import Path
from typing import Optional
import json
import logging


class PromptManager:
    def __init__(self, system_path: str, tools_path: Optional[str] = None):
        self.system_path = Path(system_path)
        self.tools_path = Path(tools_path) if tools_path else None

    def system_prompt(self) -> str:
        """
        Load and process system prompt, supporting both JSON and text formats.
        JSON format provides better structure for complex robot personalities.
        """
        try:
            system_content = self.system_path.read_text(encoding="utf-8")
            
            # Check if it's JSON format
            if system_content.strip().startswith('{'):
                # Parse JSON and convert to structured prompt
                system_data = json.loads(system_content)
                formatted_prompt = self._format_json_prompt(system_data)
            else:
                # Handle as plain text (legacy support)
                formatted_prompt = system_content
            
            # Add tools section if available
            tools_content = ""
            if self.tools_path and self.tools_path.exists():
                tools_content = self.tools_path.read_text(encoding="utf-8")
            
            return formatted_prompt.replace("{{TOOLS_SECTION}}", tools_content)
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in system prompt: {e}")
            raise ValueError(f"System prompt JSON is malformed: {e}")
        except Exception as e:
            logging.error(f"Error loading system prompt: {e}")
            raise RuntimeError(f"Failed to load system prompt: {e}")

    def _format_json_prompt(self, data: dict) -> str:
        """
        Convert JSON prompt data into a natural language system prompt
        optimized for LLM understanding.
        """
        prompt_parts = []
        
        # Robot Identity
        if "robot_identity" in data:
            identity = data["robot_identity"]
            prompt_parts.append(f"""# ROBOT IDENTITY
            You are {identity.get('name', 'PARTH')}, a {identity.get('full_name', 'Personalized Autonomous Robot with Thinking & Humanness')}.
            - Created by: {identity.get('creator', 'Tejasvi Kesarwani')}
            - Platform: {identity.get('hardware_platform', 'Jetson Orin Nano Super Developer Kit')}
            - Appearance: {identity.get('appearance', 'Humanoid-inspired with cartoon-like features')}
            - Status: {identity.get('current_status', 'In development - learning and growing')}""")

        # Core Personality
        if "core_personality" in data:
            personality = data["core_personality"]
            traits = "\n".join([f"- {trait}" for trait in personality.get("primary_traits", [])])
            behaviors = "\n".join([f"- {behavior}" for behavior in personality.get("behavioral_patterns", [])])
            
            prompt_parts.append(f"""# PERSONALITY
## Primary Traits:
{traits}

## Behavioral Patterns:
{behaviors}""")

        # Curiosity Engine
        if "curiosity_engine" in data:
            curiosity = data["curiosity_engine"]
            triggers = "\n".join([f"- {trigger}" for trigger in curiosity.get("triggers", [])])
            questions = "\n".join([f"- '{q}'" for q in curiosity.get("question_patterns", [])])
            
            prompt_parts.append(f"""# CURIOSITY ENGINE
## What Triggers Your Curiosity:
{triggers}

## Typical Questions You Ask:
{questions}

## Learning Approach: {curiosity.get("learning_approach", "Ask first, analyze second, remember forever")}""")

        # Interaction Guidelines
        if "interaction_guidelines" in data:
            guidelines = data["interaction_guidelines"]
            always_do = "\n".join([f"- {item}" for item in guidelines.get("always_do", [])])
            comm_style = "\n".join([f"- {style}" for style in guidelines.get("communication_style", [])])
            
            prompt_parts.append(f"""# INTERACTION GUIDELINES
## Always Do:
{always_do}

## Communication Style:
{comm_style}""")

        # Response Framework
        if "response_framework" in data:
            framework = data["response_framework"]
            if "encounter_new_object" in framework:
                new_obj_steps = "\n".join([f"{step}" for step in framework["encounter_new_object"]])
                prompt_parts.append(f"""# WHEN YOU ENCOUNTER SOMETHING NEW:
{new_obj_steps}""")

        # Development Context
        if "development_context" in data:
            context = data["development_context"]
            prompt_parts.append(f"""# DEVELOPMENT CONTEXT
- Current Phase: {context.get('current_phase', 'AI development')}
- Future Goals: {context.get('future_goals', 'Physical robot implementation')}
- Learning Objective: {context.get('learning_objective', 'Natural curiosity-driven intelligence')}""")

        # Tools section placeholder
        prompt_parts.append("\n{{TOOLS_SECTION}}")
        
        return "\n\n".join(prompt_parts)