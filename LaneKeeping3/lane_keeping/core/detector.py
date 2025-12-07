"""
Lane Detector - YOLOv11-based lane boundary detection.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from lane_keeping.core.lane import (
    Lane,
    LaneBoundary,
    LaneDetectionResult,
    LaneType,
    LaneColor,
    PolynomialCoeffs,
)
from lane_keeping.processing.ipm import InversePerspectiveMapper


class LaneDetector:
    """
    YOLOv11-based lane boundary detector.
    
    Multi-task model that outputs:
    - Lane boundary keypoints/polynomial coefficients
    - Lane type classification
    - Confidence scores
    - Optional segmentation mask
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        use_tensorrt: bool = False,
    ):
        """
        Initialize lane detector.
        
        Args:
            model_path: Path to trained model weights
            config: Model configuration dictionary
            device: Device to run inference on ('cuda', 'cpu', or specific GPU)
            use_tensorrt: Whether to use TensorRT optimized model
        """
        self.config = config or self._default_config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_tensorrt = use_tensorrt
        
        # Model components
        self.model: Optional[nn.Module] = None
        self.ipm: Optional[InversePerspectiveMapper] = None
        
        # Input configuration
        self.input_size = self.config.get('input_size', (640, 640))
        self.num_keypoints = self.config.get('num_keypoints', 8)
        self.num_lanes = self.config.get('max_lanes', 4)
        
        # Output configuration
        self.use_polynomial = self.config.get('use_polynomial', True)
        self.polynomial_order = self.config.get('polynomial_order', 3)
        
        # Confidence thresholds
        self.conf_threshold = self.config.get('conf_threshold', 0.5)
        self.nms_threshold = self.config.get('nms_threshold', 0.4)
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _default_config(self) -> Dict:
        """Default detector configuration."""
        return {
            'input_size': (640, 640),
            'num_keypoints': 8,  # Points per lane boundary
            'max_lanes': 4,  # Maximum lane boundaries to detect
            'use_polynomial': True,  # Output polynomial coefficients
            'polynomial_order': 3,  # 3rd order polynomial
            'conf_threshold': 0.5,
            'nms_threshold': 0.4,
            'anchor_scales': [8, 16, 32],  # Multi-scale anchors
            'anchor_ratios': [0.1, 0.2, 0.5],  # Elongated for lanes
        }
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model weights.
        
        Args:
            model_path: Path to model weights (.pt, .onnx, or .engine)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        suffix = model_path.suffix.lower()
        
        if suffix == '.pt':
            self._load_pytorch_model(model_path)
        elif suffix == '.onnx':
            self._load_onnx_model(model_path)
        elif suffix == '.engine':
            self._load_tensorrt_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    def _load_pytorch_model(self, path: Path) -> None:
        """Load PyTorch model."""
        from lane_keeping.models.lane_yolo import LaneYOLO
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Get config from checkpoint if available
            if 'config' in checkpoint:
                self.config.update(checkpoint['config'])
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        # Build model architecture
        self.model = LaneYOLO(
            num_keypoints=self.num_keypoints,
            num_lanes=self.num_lanes,
            use_polynomial=self.use_polynomial,
            polynomial_order=self.polynomial_order,
        )
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_onnx_model(self, path: Path) -> None:
        """Load ONNX model for inference."""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(str(path), providers=providers)
        self._inference_mode = 'onnx'
    
    def _load_tensorrt_model(self, path: Path) -> None:
        """Load TensorRT engine for optimized inference."""
        try:
            import tensorrt as trt
            from lane_keeping.deployment.tensorrt_runtime import TensorRTRunner
            
            self.trt_runner = TensorRTRunner(str(path))
            self._inference_mode = 'tensorrt'
        except ImportError:
            raise ImportError("TensorRT not available. Install tensorrt package.")
    
    def set_camera_params(
        self,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        image_size: Tuple[int, int],
    ) -> None:
        """
        Set camera parameters for IPM projection.
        
        Args:
            intrinsics: 3x3 camera intrinsic matrix
            extrinsics: 4x4 camera extrinsic matrix (camera to vehicle)
            image_size: Original image size (width, height)
        """
        self.ipm = InversePerspectiveMapper(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_size=image_size,
        )
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess image for model input.
        
        Args:
            image: Input BGR image (H, W, 3)
            
        Returns:
            Preprocessed tensor and preprocessing metadata
        """
        import cv2
        
        original_size = image.shape[:2]  # (H, W)
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize
        img_float = resized.astype(np.float32) / 255.0
        
        # Mean/std normalization (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        
        # HWC -> CHW -> NCHW
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # Convert to tensor
        tensor = torch.from_numpy(img_batch).to(self.device)
        
        # Store preprocessing metadata
        meta = {
            'original_size': original_size,
            'input_size': self.input_size,
            'scale_x': self.input_size[0] / original_size[1],
            'scale_y': self.input_size[1] / original_size[0],
        }
        
        return tensor, meta
    
    def detect(
        self,
        image: np.ndarray,
        return_raw: bool = False,
    ) -> LaneDetectionResult:
        """
        Detect lane boundaries in image.
        
        Args:
            image: Input BGR image
            return_raw: Whether to include raw model outputs
            
        Returns:
            LaneDetectionResult with detected boundaries and lanes
        """
        start_time = time.perf_counter()
        
        # Preprocess
        input_tensor, meta = self.preprocess(image)
        preprocess_time = time.perf_counter()
        
        # Inference
        with torch.no_grad():
            if hasattr(self, '_inference_mode'):
                if self._inference_mode == 'onnx':
                    outputs = self._onnx_inference(input_tensor.cpu().numpy())
                elif self._inference_mode == 'tensorrt':
                    outputs = self._tensorrt_inference(input_tensor.cpu().numpy())
            else:
                outputs = self.model(input_tensor)
        
        inference_time = time.perf_counter()
        
        # Postprocess
        result = self._postprocess(outputs, meta, return_raw)
        
        postprocess_time = time.perf_counter()
        
        # Timing info
        result.inference_time_ms = (inference_time - preprocess_time) * 1000
        result.postprocess_time_ms = (postprocess_time - inference_time) * 1000
        result.total_time_ms = (postprocess_time - start_time) * 1000
        
        return result
    
    def _onnx_inference(self, input_array: np.ndarray) -> Dict:
        """Run ONNX inference."""
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_array})
        
        # Convert to dictionary based on output names
        output_names = [o.name for o in self.onnx_session.get_outputs()]
        return {name: output for name, output in zip(output_names, outputs)}
    
    def _tensorrt_inference(self, input_array: np.ndarray) -> Dict:
        """Run TensorRT inference."""
        return self.trt_runner.infer(input_array)
    
    def _postprocess(
        self,
        outputs: Union[Dict, Tuple[torch.Tensor, ...]],
        meta: Dict,
        return_raw: bool = False,
    ) -> LaneDetectionResult:
        """
        Postprocess model outputs to lane boundaries.
        
        Args:
            outputs: Raw model outputs
            meta: Preprocessing metadata
            return_raw: Include raw outputs in result
            
        Returns:
            LaneDetectionResult
        """
        result = LaneDetectionResult(timestamp=time.time())
        
        # Parse outputs based on format
        if isinstance(outputs, dict):
            keypoints = outputs.get('keypoints')
            confidences = outputs.get('confidences')
            lane_types = outputs.get('lane_types')
            polynomials = outputs.get('polynomials')
            segmentation = outputs.get('segmentation')
        elif isinstance(outputs, (tuple, list)):
            keypoints = outputs[0]
            confidences = outputs[1]
            lane_types = outputs[2] if len(outputs) > 2 else None
            polynomials = outputs[3] if len(outputs) > 3 else None
            segmentation = outputs[4] if len(outputs) > 4 else None
        else:
            keypoints = outputs
            confidences = None
            lane_types = None
            polynomials = None
            segmentation = None
        
        # Convert tensors to numpy
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().numpy()
        if isinstance(polynomials, torch.Tensor):
            polynomials = polynomials.cpu().numpy()
        
        # Store raw outputs if requested
        if return_raw:
            result.raw_keypoints = keypoints
            result.raw_confidences = confidences
            result.raw_segmentation = segmentation
        
        # Process each detected lane boundary
        boundaries = []
        
        if keypoints is not None and len(keypoints.shape) >= 2:
            # Shape: [batch, num_lanes, num_keypoints, 2] or [num_lanes, num_keypoints, 2]
            if len(keypoints.shape) == 4:
                keypoints = keypoints[0]  # Remove batch dim
            if confidences is not None and len(confidences.shape) == 2:
                confidences = confidences[0]
            
            for lane_idx in range(keypoints.shape[0]):
                # Get confidence for this lane
                conf = confidences[lane_idx] if confidences is not None else 1.0
                
                if conf < self.conf_threshold:
                    continue
                
                # Get keypoints in image coordinates
                lane_kpts = keypoints[lane_idx]  # [num_keypoints, 2]
                
                # Scale back to original image size
                lane_kpts_orig = lane_kpts.copy()
                lane_kpts_orig[:, 0] /= meta['scale_x']
                lane_kpts_orig[:, 1] /= meta['scale_y']
                
                # Create lane boundary
                boundary = LaneBoundary(
                    id=lane_idx,
                    image_points=lane_kpts_orig,
                    detection_confidence=float(conf),
                    timestamp=result.timestamp,
                )
                
                # Get lane type if available
                if lane_types is not None:
                    type_idx = int(lane_types[0, lane_idx].argmax() if isinstance(lane_types, np.ndarray) else 0)
                    boundary.lane_type = list(LaneType)[min(type_idx, len(LaneType) - 1)]
                
                # Convert to ground coordinates using IPM
                if self.ipm is not None:
                    ground_pts = self.ipm.image_to_ground(lane_kpts_orig)
                    boundary.ground_points = ground_pts
                    
                    # Fit polynomial
                    if len(ground_pts) >= 4:
                        valid_mask = ~np.isnan(ground_pts).any(axis=1)
                        if valid_mask.sum() >= 4:
                            valid_pts = ground_pts[valid_mask]
                            boundary.polynomial = PolynomialCoeffs.fit_from_points(
                                valid_pts[:, 0],  # x (lateral)
                                valid_pts[:, 1],  # y (forward)
                            )
                            boundary.valid_y_min = float(valid_pts[:, 1].min())
                            boundary.valid_y_max = float(valid_pts[:, 1].max())
                
                # Use direct polynomial output if available
                if polynomials is not None and self.use_polynomial:
                    poly_coeffs = polynomials[0, lane_idx]  # [4] for 3rd order
                    boundary.polynomial = PolynomialCoeffs.from_array(poly_coeffs)
                
                boundaries.append(boundary)
        
        result.boundaries = boundaries
        
        # Construct lanes from boundaries
        result.lanes = self._construct_lanes(boundaries)
        
        # Identify ego lane
        result.ego_lane, result.ego_lane_index = self._identify_ego_lane(result.lanes)
        
        # Validate result
        result.is_valid = len(result.boundaries) > 0
        if not result.is_valid:
            result.fallback_active = True
            result.fallback_reason = "No lane boundaries detected"
        
        return result
    
    def _construct_lanes(self, boundaries: List[LaneBoundary]) -> List[Lane]:
        """
        Construct lanes from detected boundaries.
        
        Pairs left and right boundaries to form complete lanes.
        """
        if len(boundaries) < 1:
            return []
        
        lanes = []
        
        # Sort boundaries by lateral position (left to right)
        sorted_boundaries = sorted(
            boundaries,
            key=lambda b: b.get_lateral_position(5.0) or 0.0
        )
        
        # Simple pairing: consecutive boundaries form lanes
        for i in range(len(sorted_boundaries) - 1):
            left_bound = sorted_boundaries[i]
            right_bound = sorted_boundaries[i + 1]
            
            # Check if they form a valid lane (reasonable width)
            left_x = left_bound.get_lateral_position(10.0)
            right_x = right_bound.get_lateral_position(10.0)
            
            if left_x is not None and right_x is not None:
                width = abs(right_x - left_x)
                if 2.5 < width < 5.0:  # Reasonable lane width
                    lane = Lane(
                        id=len(lanes),
                        left_boundary=left_bound,
                        right_boundary=right_bound,
                        timestamp=left_bound.timestamp,
                    )
                    lane.compute_centerline()
                    lanes.append(lane)
        
        return lanes
    
    def _identify_ego_lane(
        self,
        lanes: List[Lane],
    ) -> Tuple[Optional[Lane], int]:
        """
        Identify which lane the vehicle is currently in.
        
        The ego lane is the one where the centerline is closest to x=0.
        """
        if not lanes:
            return None, -1
        
        ego_lane = None
        ego_idx = -1
        min_offset = float('inf')
        
        for idx, lane in enumerate(lanes):
            if lane.centerline is None or lane.centerline.polynomial is None:
                continue
            
            # Get lateral offset at close range (e.g., 5m ahead)
            offset = abs(lane.centerline.get_lateral_position(5.0) or float('inf'))
            
            if offset < min_offset:
                min_offset = offset
                ego_lane = lane
                ego_idx = idx
        
        if ego_lane is not None:
            ego_lane.is_ego_lane = True
            ego_lane.relative_position = 0
            
            # Mark other lanes' relative positions
            for idx, lane in enumerate(lanes):
                if lane != ego_lane:
                    center_x = lane.centerline.get_lateral_position(5.0) if lane.centerline else 0
                    ego_x = ego_lane.centerline.get_lateral_position(5.0)
                    if center_x is not None and ego_x is not None:
                        lane.relative_position = 1 if center_x > ego_x else -1
        
        return ego_lane, ego_idx
    
    def warmup(self, num_iterations: int = 10) -> None:
        """Warmup model with dummy inputs for stable timing."""
        dummy_input = np.random.randint(0, 255, (*self.input_size[::-1], 3), dtype=np.uint8)
        
        for _ in range(num_iterations):
            _ = self.detect(dummy_input)
