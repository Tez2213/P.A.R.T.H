#!/usr/bin/env python3
"""
P.A.R.T.H - Personalized Autonomous Robot with Thinking & Humanness
Main entry point with integrated vision and chat system
"""

import sys
import os
import logging
import warnings
import yaml
from chat.runner import ChatRunner

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

def configure_vision_logging_suppression():
    """Configure vision logging suppression based on config."""
    cfg = load_config()
    vision_cfg = cfg.get("vision", {})
    vision_logging_cfg = vision_cfg.get("logging", {})
    
    # Apply vision logging suppression if configured
    if vision_logging_cfg.get("suppress_ultralytics", True):
        os.environ['YOLO_VERBOSE'] = 'False'
        os.environ['ULTRALYTICS_VERBOSE'] = 'False'
        warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')
        logging.getLogger('ultralytics').setLevel(getattr(logging, vision_logging_cfg.get("log_level", "ERROR").upper()))

# Configure vision logging suppression before imports
configure_vision_logging_suppression()

def main():
    """Main function to run PARTH with integrated vision and chat."""
    
    # Load config to determine logging level
    cfg = load_config()
    log_config = cfg.get("logging", {})
    main_log_level = getattr(logging, log_config.get("level", "WARNING").upper())
    
    # Configure logging based on config
    logging.basicConfig(
        level=main_log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress additional loggers based on config
    vision_logging_cfg = cfg.get("vision", {}).get("logging", {})
    if vision_logging_cfg.get("suppress_ultralytics", True):
        logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('transformers').setLevel(logging.ERROR)
    
    try:
        # Initialize PARTH with vision integration
        print("🤖 Initializing P.A.R.T.H...")
        print("🔧 Setting up AI brain, memory, and vision systems...")
        
        runner = ChatRunner("config.yaml")
        
        print("\n" + "="*60)
        print("🎉 P.A.R.T.H is now online!")
        print("="*60)
        print("Hello! I'm PARTH - your curious AI companion!")
        print("I can see through my camera and chat with you about what I observe.")
        print("\n💡 Try asking me:")
        print("  • 'What do you see?' - I'll describe my current view")
        print("  • 'What is that?' - when I see something interesting")
        print("  • 'Can you look around?' - to get my visual observations")
        print("  • 'stats' - to see my system status")
        print("  • 'vision' - to check my vision system")
        print("  • 'exit' - to say goodbye")
        print("\n" + "="*60 + "\n")

        # Main interaction loop
        try:
            while True:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("\n🤖 PARTH: Goodbye! It was wonderful chatting and seeing the world with you! 👋")
                    runner.stop_vision()
                    break
                elif user_input.lower() == "stats":
                    stats = runner.get_stats()
                    print(f"\n📊 System Status:")
                    print(f"   🧠 AI Model: {stats['config']['model']}")
                    print(f"   💾 Memory: {stats['memory']['message_count']} conversations stored")
                    if 'vision' in stats:
                        print(f"   👁️  Vision: {stats['vision']['model_type']} ({'🟢 active' if stats['vision']['enabled'] else '🔴 inactive'})")
                        if stats['vision']['enabled'] and stats['vision']['description']:
                            print(f"   🔍 Current view: {stats['vision']['description']}")
                    print()
                    continue
                elif user_input.lower() in ["vision", "see", "look"]:
                    vision_desc = runner.get_vision_description()
                    print(f"\n👁️  PARTH: {vision_desc}\n")
                    continue
                elif not user_input:
                    continue
                    
                print("🤖 PARTH: ", end="")
                runner.ask(user_input, stream=True)
                print()
                
        except KeyboardInterrupt:
            print("\n\n🤖 PARTH: Goodbye! It was wonderful chatting with you! 👋")
            runner.stop_vision()
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            runner.stop_vision()
            
    except Exception as e:
        print(f"❌ Failed to initialize PARTH: {e}")
        print("💡 Please check your config.yaml file and ensure all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
