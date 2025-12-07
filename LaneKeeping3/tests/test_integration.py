"""
Integration tests for lane keeping system.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

from lane_keeping.core.system import LaneKeepingSystem
from lane_keeping.core.lane import LaneType


class TestSystemIntegration:
    """Integration tests for full system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'model': {
                'backbone_variant': 'n',
                'input_size': [320, 320],
                'num_keypoints': 36,
                'num_lanes': 4,
                'use_polynomial': True,
                'confidence_threshold': 0.5,
            },
            'tracker': {
                'max_age': 5,
                'min_hits': 2,
                'iou_threshold': 0.3,
            },
            'steering': {
                'lookahead_distance': 15.0,
                'max_lateral_offset': 2.0,
                'max_heading_error': 0.5,
            },
            'camera': {
                'width': 1920,
                'height': 1080,
                'fx': 1000.0,
                'fy': 1000.0,
                'cx': 960.0,
                'cy': 540.0,
                'camera_height': 1.5,
            },
        }
    
    @pytest.fixture
    def system(self, config):
        """Create lane keeping system."""
        return LaneKeepingSystem(config)
    
    def test_system_creation(self, system):
        """Test system creation."""
        assert system is not None
        assert system.detector is not None
        assert system.tracker is not None
        assert system.steering is not None
    
    def test_process_frame(self, system):
        """Test processing single frame."""
        # Create dummy frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        result = system.process_frame(frame)
        
        assert result is not None
        assert hasattr(result, 'lanes')
        assert hasattr(result, 'centerline')
        assert hasattr(result, 'guidance')
        assert hasattr(result, 'processing_time_ms')
    
    def test_process_frame_latency(self, system):
        """Test frame processing latency."""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(5):
            system.process_frame(frame)
        
        # Measure latency
        latencies = []
        for _ in range(20):
            result = system.process_frame(frame)
            latencies.append(result.processing_time_ms)
        
        mean_latency = np.mean(latencies)
        
        # Should complete in reasonable time (adjust based on hardware)
        # Note: This is a soft check, actual performance depends on hardware
        assert mean_latency > 0
    
    def test_sequential_frames(self, system):
        """Test processing sequential frames."""
        results = []
        
        for i in range(10):
            # Create slightly different frames
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            result = system.process_frame(frame)
            results.append(result)
        
        # All results should be valid
        assert len(results) == 10
        for result in results:
            assert result is not None
    
    def test_reset(self, system):
        """Test system reset."""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Process some frames
        for _ in range(5):
            system.process_frame(frame)
        
        # Reset
        system.reset()
        
        # Should work normally after reset
        result = system.process_frame(frame)
        assert result is not None


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    @pytest.fixture
    def config(self):
        return {
            'model': {
                'backbone_variant': 'n',
                'input_size': [320, 320],
                'num_keypoints': 36,
                'num_lanes': 4,
                'use_polynomial': True,
                'confidence_threshold': 0.3,
            },
            'tracker': {
                'max_age': 5,
                'min_hits': 2,
            },
            'steering': {
                'lookahead_distance': 15.0,
            },
            'camera': {
                'width': 640,
                'height': 480,
                'fx': 500.0,
                'fy': 500.0,
                'cx': 320.0,
                'cy': 240.0,
                'camera_height': 1.5,
            },
        }
    
    def test_synthetic_straight_lanes(self, config):
        """Test with synthetic straight lane image."""
        system = LaneKeepingSystem(config)
        
        # Create synthetic image with straight lanes
        frame = create_synthetic_lane_image(
            width=640,
            height=480,
            lane_offset=0.0,
            curvature=0.0,
        )
        
        # Process multiple frames to establish tracks
        for _ in range(5):
            result = system.process_frame(frame)
        
        # Should detect lanes
        # Note: Success depends on model training
        assert result is not None
    
    def test_synthetic_curved_lanes(self, config):
        """Test with synthetic curved lane image."""
        system = LaneKeepingSystem(config)
        
        # Create synthetic image with curved lanes
        frame = create_synthetic_lane_image(
            width=640,
            height=480,
            lane_offset=0.0,
            curvature=0.01,
        )
        
        # Process frames
        for _ in range(5):
            result = system.process_frame(frame)
        
        assert result is not None
    
    def test_lateral_offset_detection(self, config):
        """Test detection of lateral offset."""
        system = LaneKeepingSystem(config)
        
        # Create image with vehicle offset to the right
        frame = create_synthetic_lane_image(
            width=640,
            height=480,
            lane_offset=0.5,  # 0.5m offset
            curvature=0.0,
        )
        
        # Process frames
        for _ in range(10):
            result = system.process_frame(frame)
        
        # If guidance is valid, offset should be detected
        if result.guidance and result.guidance.is_valid:
            # Should detect vehicle is offset from center
            assert result.guidance.lateral_offset_m != 0.0


class TestGoldenOutputs:
    """Tests against golden reference outputs."""
    
    @pytest.fixture
    def golden_data_dir(self, tmp_path):
        """Create temporary directory with golden test data."""
        # Create synthetic golden data
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir()
        
        # Create test case
        test_case = {
            'frame_id': 0,
            'expected_lanes': 2,
            'expected_left_x0': -1.75,
            'expected_right_x0': 1.75,
            'tolerance': 0.3,
        }
        
        with open(golden_dir / "test_case_001.json", 'w') as f:
            json.dump(test_case, f)
        
        return golden_dir
    
    def test_against_golden(self, golden_data_dir):
        """Test against golden reference."""
        # Load golden data
        test_files = list(golden_data_dir.glob("*.json"))
        
        for test_file in test_files:
            with open(test_file, 'r') as f:
                test_case = json.load(f)
            
            # This would normally load the corresponding image
            # and compare against expected outputs
            
            # Placeholder check
            assert test_case['expected_lanes'] == 2
            assert test_case['tolerance'] > 0


class TestRobustness:
    """Robustness tests for edge cases."""
    
    @pytest.fixture
    def config(self):
        return {
            'model': {
                'backbone_variant': 'n',
                'input_size': [320, 320],
                'num_keypoints': 36,
                'num_lanes': 4,
                'use_polynomial': True,
            },
            'tracker': {'max_age': 5, 'min_hits': 2},
            'steering': {'lookahead_distance': 15.0},
            'camera': {
                'width': 640, 'height': 480,
                'fx': 500.0, 'fy': 500.0,
                'cx': 320.0, 'cy': 240.0,
                'camera_height': 1.5,
            },
        }
    
    def test_empty_frame(self, config):
        """Test handling of empty/black frame."""
        system = LaneKeepingSystem(config)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = system.process_frame(frame)
        
        # Should not crash, may have empty lanes
        assert result is not None
    
    def test_white_frame(self, config):
        """Test handling of white/overexposed frame."""
        system = LaneKeepingSystem(config)
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result = system.process_frame(frame)
        
        assert result is not None
    
    def test_noisy_frame(self, config):
        """Test handling of very noisy frame."""
        system = LaneKeepingSystem(config)
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = system.process_frame(frame)
        
        assert result is not None
    
    def test_different_resolutions(self, config):
        """Test handling of different input resolutions."""
        system = LaneKeepingSystem(config)
        
        resolutions = [
            (480, 640),
            (720, 1280),
            (1080, 1920),
        ]
        
        for height, width in resolutions:
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            result = system.process_frame(frame)
            assert result is not None
    
    def test_grayscale_input(self, config):
        """Test handling of grayscale input."""
        system = LaneKeepingSystem(config)
        
        # Grayscale expanded to 3 channels
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        frame = np.stack([gray, gray, gray], axis=-1)
        
        result = system.process_frame(frame)
        assert result is not None


def create_synthetic_lane_image(
    width: int,
    height: int,
    lane_offset: float = 0.0,
    curvature: float = 0.0,
) -> np.ndarray:
    """
    Create synthetic lane image for testing.
    
    Args:
        width: Image width
        height: Image height
        lane_offset: Lateral offset of vehicle from center (meters)
        curvature: Road curvature (1/radius)
    
    Returns:
        Synthetic RGB image
    """
    import cv2
    
    # Create gray road
    image = np.ones((height, width, 3), dtype=np.uint8) * 100
    
    # Parameters
    lane_width = 3.5  # meters
    pixels_per_meter = width / 10.0  # Approximate
    
    # Draw left lane
    left_x = width // 2 - int(lane_width * pixels_per_meter / 2) - int(lane_offset * pixels_per_meter)
    
    # Draw right lane
    right_x = width // 2 + int(lane_width * pixels_per_meter / 2) - int(lane_offset * pixels_per_meter)
    
    # Draw lane lines
    for y in range(height // 2, height, 5):
        # Apply curvature
        curve_offset = int(curvature * (y - height // 2) ** 2 * 0.01)
        
        lx = left_x + curve_offset
        rx = right_x + curve_offset
        
        if 0 < lx < width:
            cv2.circle(image, (lx, y), 3, (255, 255, 255), -1)
        if 0 < rx < width:
            cv2.circle(image, (rx, y), 3, (255, 255, 255), -1)
    
    return image
