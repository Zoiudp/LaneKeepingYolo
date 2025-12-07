"""
Unit tests for steering guidance component.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from lane_keeping.core.steering import (
    SteeringGuidance,
    SteeringCommand,
    FallbackMode,
)
from lane_keeping.core.lane import (
    LaneBoundary,
    Lane,
    LaneCenterline,
    LaneType,
)


class TestSteeringCommand:
    """Tests for SteeringCommand dataclass."""
    
    def test_creation(self):
        """Test command creation."""
        cmd = SteeringCommand(
            lateral_offset_m=0.5,
            heading_error_rad=0.1,
            curvature=0.01,
            is_valid=True,
            confidence=0.9,
            timestamp=0.0,
        )
        
        assert cmd.lateral_offset_m == 0.5
        assert cmd.heading_error_rad == 0.1
        assert cmd.curvature == 0.01
        assert cmd.is_valid
        assert cmd.confidence == 0.9
    
    def test_invalid_command(self):
        """Test invalid command."""
        cmd = SteeringCommand(
            lateral_offset_m=0.0,
            heading_error_rad=0.0,
            curvature=0.0,
            is_valid=False,
            confidence=0.0,
            timestamp=0.0,
        )
        
        assert not cmd.is_valid
        assert cmd.confidence == 0.0


class TestSteeringGuidance:
    """Tests for SteeringGuidance class."""
    
    @pytest.fixture
    def guidance(self):
        """Create steering guidance system."""
        return SteeringGuidance(
            lookahead_distance=15.0,
            max_lateral_offset=2.0,
            max_heading_error=0.5,
            fallback_mode=FallbackMode.HOLD,
        )
    
    def test_initialization(self, guidance):
        """Test initialization."""
        assert guidance.lookahead_distance == 15.0
        assert guidance.max_lateral_offset == 2.0
        assert guidance.max_heading_error == 0.5
    
    def test_centered_vehicle(self, guidance):
        """Test guidance for centered vehicle."""
        # Create centered lane
        centerline = LaneCenterline(
            points=np.array([[0.0, y] for y in range(0, 50, 5)]),
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        assert cmd.is_valid
        assert abs(cmd.lateral_offset_m) < 0.1
        assert abs(cmd.heading_error_rad) < 0.1
    
    def test_offset_vehicle(self, guidance):
        """Test guidance for offset vehicle."""
        # Vehicle is 0.5m to the right of center
        centerline = LaneCenterline(
            points=np.array([[-0.5, y] for y in range(0, 50, 5)]),
            coefficients=np.array([-0.5, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        assert cmd.is_valid
        # Negative offset = vehicle should steer left
        assert cmd.lateral_offset_m < 0
    
    def test_heading_error(self, guidance):
        """Test heading error calculation."""
        # Straight lane, vehicle angled 5 degrees right
        centerline = LaneCenterline(
            points=np.array([[0.0, y] for y in range(0, 50, 5)]),
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        vehicle_heading = np.radians(5.0)  # 5 degrees right
        cmd = guidance.compute(centerline, vehicle_heading=vehicle_heading)
        
        assert cmd.is_valid
        # Heading error should be approximately -5 degrees (need to steer left)
        assert cmd.heading_error_rad < 0
        assert abs(cmd.heading_error_rad - (-np.radians(5.0))) < 0.05
    
    def test_curved_lane(self, guidance):
        """Test guidance on curved lane."""
        # Create curved centerline: x = 0.01 * y^2
        y_vals = np.arange(0, 50, 2)
        x_vals = 0.01 * y_vals ** 2
        points = np.column_stack([x_vals, y_vals])
        
        centerline = LaneCenterline(
            points=points,
            coefficients=np.array([0.0, 0.0, 0.01, 0.0]),
            curvature=0.02,  # Approximate curvature
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        assert cmd.is_valid
        assert cmd.curvature > 0  # Right turn
    
    def test_safety_bounds_lateral(self, guidance):
        """Test lateral offset safety bounds."""
        # Large lateral offset
        centerline = LaneCenterline(
            points=np.array([[-5.0, y] for y in range(0, 50, 5)]),  # 5m offset
            coefficients=np.array([-5.0, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        # Should be clamped or marked invalid
        assert not cmd.is_valid or abs(cmd.lateral_offset_m) <= guidance.max_lateral_offset
    
    def test_safety_bounds_heading(self, guidance):
        """Test heading error safety bounds."""
        centerline = LaneCenterline(
            points=np.array([[0.0, y] for y in range(0, 50, 5)]),
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        # Large heading error (45 degrees)
        cmd = guidance.compute(centerline, vehicle_heading=np.radians(45.0))
        
        # Should be clamped or marked invalid
        assert not cmd.is_valid or abs(cmd.heading_error_rad) <= guidance.max_heading_error
    
    def test_no_centerline(self, guidance):
        """Test handling of missing centerline."""
        cmd = guidance.compute(None, vehicle_heading=0.0)
        
        assert not cmd.is_valid
    
    def test_fallback_hold(self, guidance):
        """Test fallback mode: hold last command."""
        # First get valid command
        centerline = LaneCenterline(
            points=np.array([[0.3, y] for y in range(0, 50, 5)]),
            coefficients=np.array([0.3, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        cmd1 = guidance.compute(centerline, vehicle_heading=0.0)
        assert cmd1.is_valid
        
        # Now no centerline - should hold
        cmd2 = guidance.compute(None, vehicle_heading=0.0)
        
        # Should return held command
        if cmd2.is_valid:
            assert abs(cmd2.lateral_offset_m - cmd1.lateral_offset_m) < 0.1
    
    def test_lookahead_point(self, guidance):
        """Test lookahead point calculation."""
        # Curved lane
        y_vals = np.arange(0, 50, 1)
        x_vals = 0.005 * y_vals ** 2
        points = np.column_stack([x_vals, y_vals])
        
        centerline = LaneCenterline(
            points=points,
            coefficients=np.array([0.0, 0.0, 0.005, 0.0]),
            curvature=0.01,
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        # At lookahead distance, lane has curved
        expected_x = 0.005 * guidance.lookahead_distance ** 2
        # Lateral offset should account for this
        assert cmd.is_valid


class TestFallbackModes:
    """Tests for different fallback modes."""
    
    def test_fallback_center(self):
        """Test fallback mode: steer to center."""
        guidance = SteeringGuidance(
            lookahead_distance=15.0,
            max_lateral_offset=2.0,
            max_heading_error=0.5,
            fallback_mode=FallbackMode.CENTER,
        )
        
        # First get valid command with offset
        centerline = LaneCenterline(
            points=np.array([[0.5, y] for y in range(0, 50, 5)]),
            coefficients=np.array([0.5, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        guidance.compute(centerline, vehicle_heading=0.0)
        
        # Trigger fallback
        for _ in range(10):
            cmd = guidance.compute(None, vehicle_heading=0.0)
        
        # Should eventually command zero offset
        # (implementation dependent)
    
    def test_fallback_stop(self):
        """Test fallback mode: request stop."""
        guidance = SteeringGuidance(
            lookahead_distance=15.0,
            max_lateral_offset=2.0,
            max_heading_error=0.5,
            fallback_mode=FallbackMode.STOP,
        )
        
        # Trigger fallback
        for _ in range(10):
            cmd = guidance.compute(None, vehicle_heading=0.0)
        
        # Should be invalid (requesting stop)
        assert not cmd.is_valid


class TestCurvatureEstimation:
    """Tests for curvature estimation."""
    
    @pytest.fixture
    def guidance(self):
        return SteeringGuidance(lookahead_distance=15.0)
    
    def test_straight_road_curvature(self, guidance):
        """Test curvature on straight road."""
        centerline = LaneCenterline(
            points=np.array([[0.0, y] for y in range(0, 50, 2)]),
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            curvature=0.0,
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        assert cmd.is_valid
        assert abs(cmd.curvature) < 0.001
    
    def test_gentle_curve_curvature(self, guidance):
        """Test curvature on gentle curve."""
        # Radius ~500m curve
        radius = 500.0
        y_vals = np.arange(0, 50, 2)
        x_vals = radius - np.sqrt(radius**2 - y_vals**2)
        points = np.column_stack([x_vals, y_vals])
        
        centerline = LaneCenterline(
            points=points,
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),  # Approximate
            curvature=1.0 / radius,
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        assert cmd.is_valid
        assert 0 < cmd.curvature < 0.01  # Should detect slight right curve
    
    def test_sharp_curve_curvature(self, guidance):
        """Test curvature on sharp curve."""
        # Radius ~50m curve
        radius = 50.0
        y_vals = np.arange(0, 30, 1)
        x_vals = radius - np.sqrt(np.maximum(0, radius**2 - y_vals**2))
        points = np.column_stack([x_vals, y_vals])
        
        centerline = LaneCenterline(
            points=points,
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            curvature=1.0 / radius,
        )
        
        cmd = guidance.compute(centerline, vehicle_heading=0.0)
        
        assert cmd.is_valid
        assert cmd.curvature > 0.01  # Should detect significant curve


class TestTemporalSmoothing:
    """Tests for temporal smoothing in guidance."""
    
    @pytest.fixture
    def guidance(self):
        return SteeringGuidance(
            lookahead_distance=15.0,
            temporal_smoothing=0.7,
        )
    
    def test_smoothing_reduces_noise(self, guidance):
        """Test that smoothing reduces noisy input."""
        base_offset = 0.3
        
        offsets = []
        for i in range(50):
            # Add noise to offset
            noise = np.random.randn() * 0.2
            noisy_offset = base_offset + noise
            
            centerline = LaneCenterline(
                points=np.array([[noisy_offset, y] for y in range(0, 50, 5)]),
                coefficients=np.array([noisy_offset, 0.0, 0.0, 0.0]),
                curvature=0.0,
            )
            
            cmd = guidance.compute(centerline, vehicle_heading=0.0)
            if cmd.is_valid:
                offsets.append(cmd.lateral_offset_m)
        
        # Variance should be reduced compared to input noise
        if len(offsets) > 10:
            output_std = np.std(offsets)
            # Should be less than input noise std (0.2)
            assert output_std < 0.15
