"""
Lane Keeping Module - YOLOv11-based Lane Detection and Centering
================================================================

Production-grade lane detection and lane-centering system for autonomous vehicles.

This module provides:
- Real-time lane boundary detection using YOLOv11-based architecture
- Polynomial lane representation in vehicle ground coordinates
- Temporal tracking with Kalman filtering for stable lane IDs
- Steering guidance computation (lateral offset, heading error)
- TensorRT/ONNX deployment support for edge devices

Modules:
    - core: Core components (detector, tracker, steering, system)
    - models: YOLOv11 lane detection models
    - data: Dataset handling and augmentation
    - processing: IPM, tracking, and steering guidance
    - evaluation: Metrics and benchmarking
    - visualization: Debug and overlay tools
    - deployment: Export and runtime utilities

Example Usage:
    >>> from lane_keeping import LaneKeepingSystem
    >>> 
    >>> config = {'model': {'backbone_variant': 'm', ...}, ...}
    >>> system = LaneKeepingSystem(config)
    >>> system.load_weights('model.pt')
    >>> 
    >>> result = system.process_frame(image)
    >>> print(f"Detected {len(result.lanes)} lanes")
    >>> print(f"Lateral offset: {result.guidance.lateral_offset_m:.2f}m")

Acceptance Criteria:
    - Lane detection precision/recall ≥ 0.95
    - Lateral localization error ≤ 0.15 m
    - End-to-end latency ≤ 40 ms (TensorRT FP16 on Jetson Orin NX)
"""

__version__ = "1.0.0"
__author__ = "Lane Keeping Team"

# Core components
from lane_keeping.core import (
    LaneDetector,
    LaneTracker,
    SteeringGuidance,
    LaneKeepingSystem,
)
from lane_keeping.core.lane import (
    Lane,
    LaneBoundary,
    LaneCenterline,
    LaneDetectionResult,
    LaneType,
)
from lane_keeping.core.steering import SteeringCommand

# Models
from lane_keeping.models.lane_yolo import LaneYOLO

__all__ = [
    # Version
    "__version__",
    # Core classes
    "LaneDetector",
    "LaneTracker", 
    "SteeringGuidance",
    "SteeringCommand",
    "LaneKeepingSystem",
    # Data classes
    "Lane",
    "LaneBoundary",
    "LaneCenterline",
    "LaneDetectionResult",
    "LaneType",
    # Models
    "LaneYOLO",
]
