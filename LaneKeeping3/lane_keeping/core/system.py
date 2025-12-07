"""
Lane Keeping System - Main integrated system combining detection, tracking, and steering.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from lane_keeping.core.detector import LaneDetector
from lane_keeping.core.tracker import LaneTracker
from lane_keeping.core.steering import SteeringGuidance, SteeringOutput
from lane_keeping.core.lane import Lane, LaneDetectionResult


class LaneKeepingSystem:
    """
    Integrated lane keeping system.
    
    Combines:
    - Lane detection (YOLOv11-based)
    - Temporal tracking and smoothing
    - Steering guidance computation
    
    Provides a single interface for lane centering applications.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize lane keeping system.
        
        Args:
            model_path: Path to trained model weights
            config: System configuration
            device: Inference device
        """
        self.config = config or self._default_config()
        
        # Initialize components
        self.detector = LaneDetector(
            model_path=model_path,
            config=self.config.get('detector', {}),
            device=device,
        )
        
        self.tracker = LaneTracker(
            config=self.config.get('tracker', {}),
        )
        
        self.steering = SteeringGuidance(
            config=self.config.get('steering', {}),
        )
        
        # System state
        self.is_initialized = False
        self.frame_count = 0
        self.last_timestamp = 0.0
        
        # Performance tracking
        self.timing_history: List[Dict] = []
        self.max_timing_history = 100
        
        # Callbacks
        self.on_detection: Optional[callable] = None
        self.on_steering: Optional[callable] = None
        self.on_fallback: Optional[callable] = None
    
    def _default_config(self) -> Dict:
        """Default system configuration."""
        return {
            'detector': {
                'input_size': (640, 640),
                'conf_threshold': 0.5,
                'use_polynomial': True,
            },
            'tracker': {
                'max_age': 10,
                'min_hits': 3,
                'use_kalman': True,
            },
            'steering': {
                'lookahead_distance': 10.0,
                'speed_gain': 0.5,
                'output_smoothing': 0.3,
            },
            'system': {
                'target_fps': 30,
                'warmup_frames': 10,
                'enable_visualization': True,
            },
        }
    
    def set_camera_params(
        self,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_size: Tuple[int, int],
    ) -> None:
        """
        Set camera parameters for ground projection.
        
        Args:
            intrinsics: 3x3 camera intrinsic matrix
            extrinsics: 4x4 camera to vehicle transform
            image_size: Image dimensions (width, height)
        """
        self.detector.set_camera_params(intrinsics, extrinsics, image_size)
        self.is_initialized = True
    
    def update_vehicle_state(
        self,
        speed: Optional[float] = None,
        yaw_rate: Optional[float] = None,
    ) -> None:
        """
        Update vehicle state for motion compensation.
        
        Args:
            speed: Vehicle speed in m/s
            yaw_rate: Yaw rate in rad/s
        """
        if speed is not None:
            self.steering.update_speed(speed)
            
        if speed is not None or yaw_rate is not None:
            dt = time.time() - self.last_timestamp if self.last_timestamp > 0 else 0.033
            self.tracker.update_ego_motion(speed=speed, yaw_rate=yaw_rate, dt=dt)
    
    def process_frame(
        self,
        image: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[LaneDetectionResult, SteeringOutput]:
        """
        Process a single frame through the full pipeline.
        
        Args:
            image: Input BGR image
            timestamp: Frame timestamp (optional)
            
        Returns:
            Tuple of (detection_result, steering_output)
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.perf_counter()
        
        # Detection
        detection_result = self.detector.detect(image)
        detection_time = time.perf_counter()
        
        # Tracking
        tracked_result = self.tracker.update(detection_result)
        tracking_time = time.perf_counter()
        
        # Steering guidance
        steering_output = self.steering.compute(
            tracked_result.ego_lane,
            timestamp=timestamp,
        )
        steering_time = time.perf_counter()
        
        # Update timing
        total_time = steering_time - start_time
        timing = {
            'frame_id': self.frame_count,
            'timestamp': timestamp,
            'detection_ms': (detection_time - start_time) * 1000,
            'tracking_ms': (tracking_time - detection_time) * 1000,
            'steering_ms': (steering_time - tracking_time) * 1000,
            'total_ms': total_time * 1000,
        }
        self._update_timing(timing)
        
        # Update state
        self.frame_count += 1
        self.last_timestamp = timestamp
        
        # Callbacks
        if self.on_detection is not None:
            self.on_detection(tracked_result)
        
        if self.on_steering is not None:
            self.on_steering(steering_output)
        
        if steering_output.fallback_active and self.on_fallback is not None:
            self.on_fallback(steering_output.fallback_reason)
        
        return tracked_result, steering_output
    
    def _update_timing(self, timing: Dict) -> None:
        """Update timing history."""
        self.timing_history.append(timing)
        if len(self.timing_history) > self.max_timing_history:
            self.timing_history.pop(0)
    
    def get_timing_stats(self) -> Dict:
        """Get timing statistics."""
        if not self.timing_history:
            return {}
        
        total_times = [t['total_ms'] for t in self.timing_history]
        detection_times = [t['detection_ms'] for t in self.timing_history]
        
        return {
            'mean_total_ms': np.mean(total_times),
            'std_total_ms': np.std(total_times),
            'p95_total_ms': np.percentile(total_times, 95),
            'max_total_ms': max(total_times),
            'mean_detection_ms': np.mean(detection_times),
            'mean_fps': 1000 / np.mean(total_times) if np.mean(total_times) > 0 else 0,
        }
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_callback: Optional[callable] = None,
        max_frames: Optional[int] = None,
    ) -> List[Tuple[LaneDetectionResult, SteeringOutput]]:
        """
        Process entire video file.
        
        Args:
            video_path: Path to input video
            output_callback: Callback for each frame result
            max_frames: Maximum frames to process
            
        Returns:
            List of (detection_result, steering_output) for each frame
        """
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames is not None and frame_idx >= max_frames:
                break
            
            timestamp = frame_idx / fps
            
            detection, steering = self.process_frame(frame, timestamp)
            results.append((detection, steering))
            
            if output_callback is not None:
                output_callback(frame_idx, frame, detection, steering)
            
            frame_idx += 1
        
        cap.release()
        return results
    
    def warmup(self, num_frames: int = 10) -> None:
        """Warmup system with dummy inputs."""
        self.detector.warmup(num_frames)
    
    def reset(self) -> None:
        """Reset system state."""
        self.tracker.reset()
        self.steering.reset()
        self.frame_count = 0
        self.last_timestamp = 0.0
        self.timing_history.clear()
    
    def get_status(self) -> Dict:
        """Get system status."""
        return {
            'is_initialized': self.is_initialized,
            'frame_count': self.frame_count,
            'active_tracks': len(self.tracker.tracks),
            'timing_stats': self.get_timing_stats(),
        }
