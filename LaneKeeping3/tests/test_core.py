"""
Unit tests for lane detection core components.
"""

import pytest
import numpy as np
from dataclasses import dataclass

from lane_keeping.core.lane import (
    LaneBoundary,
    Lane,
    LaneType,
    LaneCenterline,
    LaneDetectionResult,
)


class TestLaneBoundary:
    """Tests for LaneBoundary class."""
    
    def test_creation(self):
        """Test basic LaneBoundary creation."""
        coeffs = np.array([0.0, 0.0, 0.1, 1.0])
        boundary = LaneBoundary(
            coefficients=coeffs,
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        assert np.allclose(boundary.coefficients, coeffs)
        assert boundary.confidence == 0.9
        assert boundary.lane_type == LaneType.SOLID
    
    def test_evaluate_polynomial(self):
        """Test polynomial evaluation at points."""
        # y = 1.0 (constant lane)
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        boundary = LaneBoundary(
            coefficients=coeffs,
            confidence=0.9,
            lane_type=LaneType.DASHED,
        )
        
        # Evaluate at multiple y values
        y_values = np.array([0.0, 5.0, 10.0])
        x_values = boundary.evaluate(y_values)
        
        assert np.allclose(x_values, np.array([1.0, 1.0, 1.0]))
    
    def test_evaluate_linear(self):
        """Test linear polynomial evaluation."""
        # y = 0.5 + 0.1*x -> x = (y - 0.5) / 0.1
        # Or: x = c0 + c1*y -> x = 0.0 + 1.0*y
        coeffs = np.array([0.0, 1.0, 0.0, 0.0])
        boundary = LaneBoundary(
            coefficients=coeffs,
            confidence=0.95,
            lane_type=LaneType.SOLID,
        )
        
        y_values = np.array([0.0, 1.0, 2.0])
        x_values = boundary.evaluate(y_values)
        
        expected = np.array([0.0, 1.0, 2.0])
        assert np.allclose(x_values, expected)
    
    def test_evaluate_quadratic(self):
        """Test quadratic polynomial evaluation."""
        # x = 1.0 + 0.0*y + 0.1*y^2
        coeffs = np.array([1.0, 0.0, 0.1, 0.0])
        boundary = LaneBoundary(
            coefficients=coeffs,
            confidence=0.85,
            lane_type=LaneType.SOLID,
        )
        
        y_values = np.array([0.0, 1.0, 2.0])
        x_values = boundary.evaluate(y_values)
        
        expected = np.array([1.0, 1.1, 1.4])
        assert np.allclose(x_values, expected)


class TestLane:
    """Tests for Lane class."""
    
    def test_creation(self):
        """Test Lane creation with left and right boundaries."""
        left = LaneBoundary(
            coefficients=np.array([-1.5, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        right = LaneBoundary(
            coefficients=np.array([1.5, 0.0, 0.0, 0.0]),
            confidence=0.85,
            lane_type=LaneType.DASHED,
        )
        
        lane = Lane(
            id=1,
            left_boundary=left,
            right_boundary=right,
        )
        
        assert lane.id == 1
        assert lane.left_boundary == left
        assert lane.right_boundary == right
    
    def test_width_constant(self):
        """Test lane width calculation for parallel boundaries."""
        left = LaneBoundary(
            coefficients=np.array([-1.75, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        right = LaneBoundary(
            coefficients=np.array([1.75, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        lane = Lane(id=1, left_boundary=left, right_boundary=right)
        
        width = lane.get_width(y=10.0)
        assert np.isclose(width, 3.5)
    
    def test_center_position(self):
        """Test lane center calculation."""
        left = LaneBoundary(
            coefficients=np.array([-2.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        right = LaneBoundary(
            coefficients=np.array([2.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        lane = Lane(id=1, left_boundary=left, right_boundary=right)
        
        center_x = lane.get_center_x(y=10.0)
        assert np.isclose(center_x, 0.0)


class TestLaneCenterline:
    """Tests for LaneCenterline class."""
    
    def test_creation(self):
        """Test LaneCenterline creation."""
        points = np.array([
            [0.0, 0.0],
            [0.1, 5.0],
            [0.2, 10.0],
            [0.3, 15.0],
        ])
        
        centerline = LaneCenterline(
            points=points,
            coefficients=np.array([0.0, 0.02, 0.0, 0.0]),
            curvature=0.001,
        )
        
        assert np.allclose(centerline.points, points)
        assert centerline.curvature == 0.001
    
    def test_interpolation(self):
        """Test centerline interpolation."""
        points = np.array([
            [0.0, 0.0],
            [0.0, 10.0],
            [0.0, 20.0],
        ])
        
        centerline = LaneCenterline(
            points=points,
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        # Interpolate at y=5.0
        x = centerline.interpolate(y=5.0)
        assert np.isclose(x, 0.0)


class TestLaneType:
    """Tests for LaneType enum."""
    
    def test_enum_values(self):
        """Test enum values exist."""
        assert LaneType.SOLID is not None
        assert LaneType.DASHED is not None
        assert LaneType.DOUBLE_SOLID is not None
        assert LaneType.DOUBLE_DASHED is not None
        assert LaneType.SOLID_DASHED is not None
        assert LaneType.DASHED_SOLID is not None
        assert LaneType.UNKNOWN is not None


class TestLaneDetectionResult:
    """Tests for LaneDetectionResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        result = LaneDetectionResult(
            lanes=[],
            centerline=None,
            guidance=None,
            timestamp=0.0,
            frame_id=0,
            processing_time_ms=10.0,
        )
        
        assert len(result.lanes) == 0
        assert result.centerline is None
        assert result.processing_time_ms == 10.0


class TestKeypoints:
    """Tests for keypoint representation."""
    
    def test_keypoints_to_polynomial(self):
        """Test fitting polynomial to keypoints."""
        # Create points along a line: x = 0.5 * y
        y_points = np.linspace(0, 20, 10)
        x_points = 0.5 * y_points
        
        # Fit polynomial
        coeffs = np.polyfit(y_points, x_points, 3)
        coeffs = coeffs[::-1]  # Reverse to [c0, c1, c2, c3]
        
        # Create boundary from coefficients
        boundary = LaneBoundary(
            coefficients=coeffs,
            confidence=0.95,
            lane_type=LaneType.SOLID,
        )
        
        # Evaluate at original points
        x_fitted = boundary.evaluate(y_points)
        
        # Should be close to original
        assert np.allclose(x_points, x_fitted, atol=0.01)
    
    def test_keypoints_curvature(self):
        """Test curvature calculation from keypoints."""
        # Create curved lane: x = 0.01 * y^2
        y_points = np.linspace(0, 20, 20)
        x_points = 0.01 * y_points ** 2
        
        # Compute curvature (second derivative)
        # For x = a*y^2, dx/dy = 2ay, d2x/dy2 = 2a
        # Curvature ≈ d2x/dy2 / (1 + (dx/dy)^2)^(3/2)
        
        # Fit polynomial
        coeffs = np.polyfit(y_points, x_points, 3)
        
        # c2 coefficient (quadratic term) ≈ 0.01
        assert np.isclose(coeffs[1], 0.01, atol=0.001)
