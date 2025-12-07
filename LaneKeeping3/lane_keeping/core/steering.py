"""
Steering Guidance - Lateral offset and heading error computation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple
import numpy as np

from lane_keeping.core.lane import Lane, LaneCenter, PolynomialCoeffs


class GuidanceStatus(Enum):
    """Status of steering guidance output."""
    VALID = "valid"
    LOW_CONFIDENCE = "low_confidence"
    NO_LANE = "no_lane"
    OUT_OF_RANGE = "out_of_range"
    FALLBACK = "fallback"
    DISABLED = "disabled"


@dataclass
class SteeringOutput:
    """
    Steering guidance output for lateral controller.
    
    All values are in vehicle coordinates:
    - Positive lateral_offset: vehicle is to the right of lane center
    - Positive heading_error: vehicle heading to the right of lane direction
    """
    # Primary outputs
    lateral_offset: float = 0.0  # meters, positive = right of center
    heading_error: float = 0.0  # radians, positive = heading right
    
    # Lookahead point (vehicle coordinates)
    lookahead_x: float = 0.0  # lateral position at lookahead
    lookahead_y: float = 0.0  # forward position (lookahead distance)
    
    # Curvature information
    path_curvature: float = 0.0  # 1/m, positive = curving right
    curvature_rate: float = 0.0  # rate of change of curvature
    
    # Lane width at vehicle position
    lane_width: float = 3.7  # meters
    
    # Confidence and status
    confidence: float = 0.0  # 0-1
    status: GuidanceStatus = GuidanceStatus.VALID
    
    # Safety flags
    is_safe: bool = True
    fallback_active: bool = False
    fallback_reason: str = ""
    
    # Timestamp
    timestamp: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if guidance is valid for use."""
        return self.status == GuidanceStatus.VALID and self.is_safe


@dataclass
class SafetyLimits:
    """Safety limits for steering guidance."""
    max_lateral_offset: float = 1.5  # meters
    max_heading_error: float = 0.5  # radians (~29 degrees)
    max_curvature: float = 0.1  # 1/m (10m radius)
    min_confidence: float = 0.3
    max_offset_rate: float = 1.0  # m/s
    max_heading_rate: float = 0.5  # rad/s


class SteeringGuidance:
    """
    Computes steering guidance signals from lane detection.
    
    Outputs lateral offset and heading error for a lateral controller
    to maintain lane centering.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        """
        Initialize steering guidance.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Lookahead configuration
        self.lookahead_distance = self.config.get('lookahead_distance', 10.0)  # meters
        self.lookahead_time = self.config.get('lookahead_time', None)  # seconds (optional)
        self.min_lookahead = self.config.get('min_lookahead', 5.0)
        self.max_lookahead = self.config.get('max_lookahead', 30.0)
        
        # Speed-dependent lookahead
        self.speed_gain = self.config.get('speed_gain', 0.5)  # lookahead = base + gain * speed
        
        # Smoothing
        self.output_smoothing = self.config.get('output_smoothing', 0.3)
        
        # Safety limits
        self.safety_limits = SafetyLimits(
            max_lateral_offset=self.config.get('max_lateral_offset', 1.5),
            max_heading_error=self.config.get('max_heading_error', 0.5),
            max_curvature=self.config.get('max_curvature', 0.1),
            min_confidence=self.config.get('min_confidence', 0.3),
        )
        
        # State
        self.previous_output: Optional[SteeringOutput] = None
        self.ego_speed: float = 0.0
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'lookahead_distance': 10.0,
            'lookahead_time': None,
            'min_lookahead': 5.0,
            'max_lookahead': 30.0,
            'speed_gain': 0.5,
            'output_smoothing': 0.3,
            'max_lateral_offset': 1.5,
            'max_heading_error': 0.5,
            'max_curvature': 0.1,
            'min_confidence': 0.3,
        }
    
    def update_speed(self, speed: float) -> None:
        """
        Update ego vehicle speed.
        
        Args:
            speed: Vehicle speed in m/s
        """
        self.ego_speed = max(0.0, speed)
    
    def get_lookahead_distance(self) -> float:
        """Get current lookahead distance based on speed."""
        if self.lookahead_time is not None:
            # Time-based lookahead
            distance = self.lookahead_time * self.ego_speed
        else:
            # Speed-dependent lookahead
            distance = self.lookahead_distance + self.speed_gain * self.ego_speed
        
        return np.clip(distance, self.min_lookahead, self.max_lookahead)
    
    def compute(
        self,
        ego_lane: Optional[Lane],
        timestamp: float = 0.0,
    ) -> SteeringOutput:
        """
        Compute steering guidance from ego lane.
        
        Args:
            ego_lane: Current ego lane with centerline
            timestamp: Current timestamp
            
        Returns:
            SteeringOutput with lateral offset and heading error
        """
        output = SteeringOutput(timestamp=timestamp)
        
        # Check for valid lane
        if ego_lane is None or ego_lane.centerline is None:
            output.status = GuidanceStatus.NO_LANE
            output.is_safe = False
            output.fallback_active = True
            output.fallback_reason = "No ego lane detected"
            return self._apply_fallback(output)
        
        centerline = ego_lane.centerline
        
        # Check confidence
        if centerline.confidence < self.safety_limits.min_confidence:
            output.status = GuidanceStatus.LOW_CONFIDENCE
            output.confidence = centerline.confidence
            output.fallback_active = True
            output.fallback_reason = f"Low confidence: {centerline.confidence:.2f}"
            return self._apply_fallback(output)
        
        # Get lookahead distance
        lookahead = self.get_lookahead_distance()
        output.lookahead_y = lookahead
        
        # Check if lookahead is within valid range
        if lookahead < centerline.valid_y_min or lookahead > centerline.valid_y_max:
            output.status = GuidanceStatus.OUT_OF_RANGE
            output.fallback_active = True
            output.fallback_reason = f"Lookahead {lookahead:.1f}m outside valid range"
            return self._apply_fallback(output)
        
        # Compute lateral offset at vehicle position (y=0)
        # Positive offset means vehicle is to the right of center
        center_at_vehicle = centerline.get_lateral_position(0.0)
        if center_at_vehicle is not None:
            output.lateral_offset = -center_at_vehicle  # Negate: if center is left, offset is positive
        
        # Compute lateral position at lookahead
        center_at_lookahead = centerline.get_lateral_position(lookahead)
        if center_at_lookahead is not None:
            output.lookahead_x = center_at_lookahead
        
        # Compute heading error
        # Lane heading at vehicle position vs vehicle heading (assumed 0)
        lane_heading = centerline.get_heading(0.0)
        if lane_heading is not None:
            output.heading_error = -lane_heading  # Negate: if lane curves left, error is positive
        
        # Compute curvature
        curvature = centerline.get_curvature(lookahead)
        if curvature is not None:
            output.path_curvature = curvature
            
            # Estimate curvature rate from polynomial
            if centerline.polynomial is not None:
                output.curvature_rate = 6 * centerline.polynomial.a3
        
        # Lane width
        output.lane_width = centerline.lane_width
        
        # Confidence
        output.confidence = centerline.confidence
        
        # Apply safety checks
        output = self._check_safety(output)
        
        # Apply smoothing
        output = self._apply_smoothing(output)
        
        # Update previous output
        self.previous_output = output
        
        return output
    
    def _check_safety(self, output: SteeringOutput) -> SteeringOutput:
        """Check and enforce safety limits."""
        limits = self.safety_limits
        
        # Check lateral offset
        if abs(output.lateral_offset) > limits.max_lateral_offset:
            output.is_safe = False
            output.fallback_active = True
            output.fallback_reason = f"Lateral offset {output.lateral_offset:.2f}m exceeds limit"
        
        # Check heading error
        if abs(output.heading_error) > limits.max_heading_error:
            output.is_safe = False
            output.fallback_active = True
            output.fallback_reason = f"Heading error {np.degrees(output.heading_error):.1f}° exceeds limit"
        
        # Check curvature
        if abs(output.path_curvature) > limits.max_curvature:
            output.is_safe = False
            output.fallback_active = True
            output.fallback_reason = f"Path curvature {output.path_curvature:.3f}/m exceeds limit"
        
        # Check rate limits (if previous output available)
        if self.previous_output is not None:
            dt = output.timestamp - self.previous_output.timestamp
            if dt > 0:
                offset_rate = abs(output.lateral_offset - self.previous_output.lateral_offset) / dt
                heading_rate = abs(output.heading_error - self.previous_output.heading_error) / dt
                
                if offset_rate > limits.max_offset_rate:
                    output.fallback_active = True
                    output.fallback_reason = f"Offset rate {offset_rate:.2f}m/s exceeds limit"
                
                if heading_rate > limits.max_heading_rate:
                    output.fallback_active = True
                    output.fallback_reason = f"Heading rate {np.degrees(heading_rate):.1f}°/s exceeds limit"
        
        if output.fallback_active and output.status == GuidanceStatus.VALID:
            output.status = GuidanceStatus.FALLBACK
        
        return output
    
    def _apply_smoothing(self, output: SteeringOutput) -> SteeringOutput:
        """Apply output smoothing."""
        if self.previous_output is None:
            return output
        
        alpha = self.output_smoothing
        
        # Only smooth if both outputs are valid
        if output.status == GuidanceStatus.VALID and \
           self.previous_output.status == GuidanceStatus.VALID:
            output.lateral_offset = (
                alpha * output.lateral_offset +
                (1 - alpha) * self.previous_output.lateral_offset
            )
            output.heading_error = (
                alpha * output.heading_error +
                (1 - alpha) * self.previous_output.heading_error
            )
            output.path_curvature = (
                alpha * output.path_curvature +
                (1 - alpha) * self.previous_output.path_curvature
            )
        
        return output
    
    def _apply_fallback(self, output: SteeringOutput) -> SteeringOutput:
        """Apply fallback behavior when lane detection fails."""
        if self.previous_output is not None:
            # Use previous output with decayed confidence
            decay = 0.8
            output.lateral_offset = self.previous_output.lateral_offset
            output.heading_error = self.previous_output.heading_error
            output.path_curvature = self.previous_output.path_curvature
            output.confidence = self.previous_output.confidence * decay
            output.lane_width = self.previous_output.lane_width
            output.lookahead_x = self.previous_output.lookahead_x
            output.lookahead_y = self.previous_output.lookahead_y
        
        return output
    
    def compute_pure_pursuit(
        self,
        ego_lane: Optional[Lane],
        wheelbase: float = 2.7,
    ) -> Tuple[float, float]:
        """
        Compute pure pursuit steering angle and target curvature.
        
        Args:
            ego_lane: Current ego lane
            wheelbase: Vehicle wheelbase in meters
            
        Returns:
            Tuple of (steering_angle, target_curvature)
        """
        output = self.compute(ego_lane)
        
        if not output.is_valid():
            return 0.0, 0.0
        
        lookahead = output.lookahead_y
        lateral_error = output.lookahead_x - output.lateral_offset
        
        # Pure pursuit curvature: kappa = 2 * sin(alpha) / L
        # where alpha is angle to lookahead point
        alpha = np.arctan2(lateral_error, lookahead)
        curvature = 2 * np.sin(alpha) / lookahead
        
        # Steering angle: delta = arctan(L * kappa)
        steering_angle = np.arctan(wheelbase * curvature)
        
        return steering_angle, curvature
    
    def compute_stanley(
        self,
        ego_lane: Optional[Lane],
        k_e: float = 1.0,
        k_soft: float = 1.0,
    ) -> float:
        """
        Compute Stanley controller steering angle.
        
        Args:
            ego_lane: Current ego lane
            k_e: Cross-track error gain
            k_soft: Softening constant
            
        Returns:
            Steering angle in radians
        """
        output = self.compute(ego_lane)
        
        if not output.is_valid():
            return 0.0
        
        # Stanley: delta = heading_error + arctan(k_e * e / (k_soft + v))
        cross_track_term = np.arctan(
            k_e * output.lateral_offset / (k_soft + self.ego_speed)
        )
        
        steering_angle = output.heading_error + cross_track_term
        
        return steering_angle
    
    def reset(self) -> None:
        """Reset internal state."""
        self.previous_output = None


# Alias for backward compatibility
SteeringCommand = SteeringOutput