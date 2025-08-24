import cv2
import numpy as np
import time
import threading
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Callable
import queue
import base64
import os
import warnings
import sys
from contextlib import contextmanager


class FastVision:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize fast vision model with complete config-driven approach.
        No hardcoding - everything configurable.
        """
        # Load configuration first
        self.cfg = self._load_config(config_path)
        self.vision_cfg = self.cfg.get("vision", {})
        self.vision_logging_cfg = self.vision_cfg.get("logging", {})
        
        # Configure logging suppression based on config
        self._configure_logging_suppression()
        
        # Import YOLO after configuration
        self._import_yolo()
        
        # Initialize vision system only if enabled
        if not self.vision_cfg.get("enabled", False):
            logging.info("Vision system disabled in config")
            return
            
        # Vision state
        self.model = None
        self.current_frame = None
        self.current_detections = []
        self.vision_description = ""
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Callbacks for chat integration
        self.vision_callback = None
        
        # Load model
        self._load_model()
        
        # Setup logging
        self._setup_logging()
        
        log_level = getattr(logging, self.vision_logging_cfg.get("log_level", "INFO").upper())
        if logging.getLogger().isEnabledFor(log_level):
            logging.log(log_level, f"FastVision initialized with {self.vision_cfg.get('model_type')} model")

    def _configure_logging_suppression(self):
        """Configure logging suppression based on config settings."""
        if self.vision_logging_cfg.get("suppress_ultralytics", True):
            os.environ['YOLO_VERBOSE'] = 'False'
            os.environ['ULTRALYTICS_VERBOSE'] = 'False'
            
        # Set ultralytics logging level
        ultralytics_log_level = self.vision_logging_cfg.get("log_level", "ERROR").upper()
        logging.getLogger('ultralytics').setLevel(getattr(logging, ultralytics_log_level))
        
        if self.vision_logging_cfg.get("suppress_ultralytics", True):
            warnings.filterwarnings('ignore', category=UserWarning, module='ultralytics')

    def _import_yolo(self):
        """Import YOLO with optional output suppression."""
        if self.vision_logging_cfg.get("suppress_model_loading", True):
            with self._suppress_output():
                from ultralytics import YOLO
                self.YOLO = YOLO
        else:
            from ultralytics import YOLO
            self.YOLO = YOLO

    @contextmanager
    def _suppress_output(self):
        """Suppress stdout and stderr if configured."""
        if self.vision_logging_cfg.get("suppress_model_loading", True) or self.vision_logging_cfg.get("suppress_inference", True):
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        else:
            yield

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")

    def _setup_logging(self):
        """Setup logging from main config."""
        log_config = self.cfg.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())
        logging.basicConfig(level=level)

    def _load_model(self):
        """Load the vision model based on config."""
        if not self.vision_cfg.get("enabled", False):
            return
            
        model_type = self.vision_cfg.get("model_type", "yolov8n")
        model_path = self.vision_cfg.get("model_path", "./yolov8n.pt")
        
        try:
            # Load model with optional output suppression
            if self.vision_logging_cfg.get("suppress_model_loading", True):
                with self._suppress_output():
                    if model_type in ["yolov8n", "yolov8s"]:
                        # Check if custom model path exists, otherwise use default
                        if Path(model_path).exists():
                            self.model = self.YOLO(model_path, verbose=False)
                        else:
                            # Download default model silently
                            self.model = self.YOLO(f'{model_type}.pt', verbose=False)
                            
                        # Configure for performance
                        if self.vision_cfg.get("performance", {}).get("use_gpu", False):
                            self.model.to('cuda')
                            
                    else:
                        raise ValueError(f"Unsupported model type: {model_type}")
            else:
                if model_type in ["yolov8n", "yolov8s"]:
                    # Check if custom model path exists, otherwise use default
                    if Path(model_path).exists():
                        self.model = self.YOLO(model_path, verbose=False)
                    else:
                        # Download default model
                        self.model = self.YOLO(f'{model_type}.pt', verbose=False)
                        
                    # Configure for performance
                    if self.vision_cfg.get("performance", {}).get("use_gpu", False):
                        self.model.to('cuda')
                        
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
            log_level = getattr(logging, self.vision_logging_cfg.get("log_level", "INFO").upper())
            if logging.getLogger().isEnabledFor(log_level):
                logging.log(log_level, f"Loaded {model_type} model successfully")
                
        except Exception as e:
            logging.error(f"Failed to load vision model: {e}")
            raise

    def set_vision_callback(self, callback: Callable[[str, List], None]):
        """Set callback function for vision updates to chat system."""
        self.vision_callback = callback

    def process_frame(self, frame):
        """Process single frame for real-time detection with config optimizations."""
        if self.model is None:
            return frame, []
            
        detection_cfg = self.vision_cfg.get("detection", {})
        performance_cfg = self.vision_cfg.get("performance", {})
        
        # Resize for better performance if configured
        if performance_cfg.get("resize_input", True):
            camera_cfg = self.vision_cfg.get("camera", {})
            target_width = camera_cfg.get("width", 640)
            target_height = camera_cfg.get("height", 480)
            frame = cv2.resize(frame, (target_width, target_height))
        
        try:
            # YOLO inference with optional output suppression
            if self.vision_logging_cfg.get("suppress_inference", True):
                with self._suppress_output():
                    results = self.model(
                        frame, 
                        verbose=False,
                        conf=detection_cfg.get("confidence_threshold", 0.5),
                        max_det=detection_cfg.get("max_detections", 10)
                    )
            else:
                results = self.model(
                    frame, 
                    verbose=False,
                    conf=detection_cfg.get("confidence_threshold", 0.5),
                    max_det=detection_cfg.get("max_detections", 10)
                )
            
            # Extract detections
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        # Get class name
                        class_name = self.model.names[cls] if cls < len(self.model.names) else f"class_{cls}"
                        detections.append({
                            "class": class_name,
                            "confidence": conf,
                            "box": box.xyxy[0].tolist()
                        })
                
                # Draw annotations
                annotated_frame = results[0].plot()
                return annotated_frame, detections
                
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            
        return frame, []

    def get_vision_description(self, detections: List[Dict]) -> str:
        """Generate natural language description of what's being seen."""
        if not detections:
            return "I don't see any specific objects right now."
        
        # Group detections by class
        object_counts = {}
        for det in detections:
            cls = det["class"]
            if cls in object_counts:
                object_counts[cls] += 1
            else:
                object_counts[cls] = 1
        
        # Create description
        descriptions = []
        for obj, count in object_counts.items():
            if count == 1:
                descriptions.append(f"a {obj}")
            else:
                descriptions.append(f"{count} {obj}s")
        
        if len(descriptions) == 1:
            return f"I can see {descriptions[0]}."
        elif len(descriptions) == 2:
            return f"I can see {descriptions[0]} and {descriptions[1]}."
        else:
            return f"I can see {', '.join(descriptions[:-1])}, and {descriptions[-1]}."

    def real_time_detection(self, source: Optional[int] = None):
        """Real-time detection from webcam with config-driven parameters."""
        if not self.vision_cfg.get("enabled", False):
            logging.warning("Vision system is disabled in config")
            return
            
        # Use config source if not provided
        if source is None:
            source = self.vision_cfg.get("camera", {}).get("source", 0)
        
        cap = cv2.VideoCapture(source)
        
        # Configure camera settings from config
        camera_cfg = self.vision_cfg.get("camera", {})
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.get("width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.get("height", 480))
        cap.set(cv2.CAP_PROP_FPS, camera_cfg.get("fps", 30))
        
        # Performance settings
        performance_cfg = self.vision_cfg.get("performance", {})
        frame_skip = performance_cfg.get("frame_skip", 1)
        
        fps_counter = 0
        start_time = time.time()
        frame_counter = 0
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_counter += 1
                
                # Skip frames for performance if configured
                if frame_counter % frame_skip != 0:
                    continue
                
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Update current state
                self.current_frame = processed_frame
                self.current_detections = detections
                self.vision_description = self.get_vision_description(detections)
                
                # Notify chat system if callback is set
                if self.vision_callback and detections:
                    try:
                        self.vision_callback(self.vision_description, detections)
                    except Exception as e:
                        logging.error(f"Error in vision callback: {e}")
                
                # Calculate and display FPS if configured
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    
                    if self.vision_logging_cfg.get("show_fps", False):
                        print(f"Vision FPS: {fps:.1f}")
                    else:
                        log_level = getattr(logging, self.vision_logging_cfg.get("log_level", "INFO").upper())
                        if logging.getLogger().isEnabledFor(log_level):
                            logging.log(log_level, f"Vision FPS: {fps:.1f}")
                    
                    start_time = time.time()
                
                # Display if real_time enabled
                if self.vision_cfg.get("real_time", True):
                    cv2.imshow('PARTH Vision', processed_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            logging.info("Vision system stopped by user")
        except Exception as e:
            logging.error(f"Error in real-time detection: {e}")
        finally:
            self.stop()
            cap.release()
            cv2.destroyAllWindows()

    def start_background_vision(self):
        """Start vision processing in background thread."""
        if not self.vision_cfg.get("enabled", False):
            logging.warning("Vision system is disabled in config")
            return
            
        if not hasattr(self, '_vision_thread') or not self._vision_thread.is_alive():
            self._vision_thread = threading.Thread(target=self.real_time_detection, daemon=True)
            self._vision_thread.start()
            
            log_level = getattr(logging, self.vision_logging_cfg.get("log_level", "INFO").upper())
            if logging.getLogger().isEnabledFor(log_level):
                logging.log(log_level, "Background vision processing started")

    def stop(self):
        """Stop vision processing."""
        self.is_running = False
        if hasattr(self, '_vision_thread') and self._vision_thread.is_alive():
            self._vision_thread.join(timeout=2)
            
        log_level = getattr(logging, self.vision_logging_cfg.get("log_level", "INFO").upper())
        if logging.getLogger().isEnabledFor(log_level):
            logging.log(log_level, "Vision processing stopped")

    def get_current_vision(self) -> Dict:
        """Get current vision state for chat system."""
        return {
            "description": self.vision_description,
            "detections": self.current_detections,
            "enabled": self.vision_cfg.get("enabled", False),
            "model_type": self.vision_cfg.get("model_type", "none")
        }

    def capture_frame_description(self) -> str:
        """Capture and describe current frame for chat queries."""
        if not self.vision_cfg.get("enabled", False):
            return "Vision system is currently disabled."
            
        if self.current_detections:
            return f"Right now, {self.vision_description}"
        else:
            return "I'm looking around but don't see any specific objects I can identify right now."


# Factory function for easy integration
def create_vision_system(config_path: str = "config.yaml") -> FastVision:
    """Create and return configured vision system."""
    return FastVision(config_path)

# Usage example
if __name__ == "__main__":
    # Initialize with fastest model
    vision = FastVision("yolov8n")
    
    # Start real-time detection
    vision.real_time_detection()