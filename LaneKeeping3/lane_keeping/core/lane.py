"""
Lane Data Structures - Core representations for lane boundaries and centerlines.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np


class LaneType(Enum):
    """Lane marking type classification."""
    SOLID = "solid"
    DASHED = "dashed"
    DOUBLE_SOLID = "double_solid"
    DOUBLE_DASHED = "double_dashed"
    SOLID_DASHED = "solid_dashed"  # Solid on left, dashed on right
    DASHED_SOLID = "dashed_solid"  # Dashed on left, solid on right
    BOTTS_DOTS = "botts_dots"
    UNKNOWN = "unknown"


class LaneColor(Enum):
    """Lane marking color."""
    WHITE = "white"
    YELLOW = "yellow"
    BLUE = "blue"  # Some regions use blue
    UNKNOWN = "unknown"


@dataclass
class PolynomialCoeffs:
    """
    3rd-order polynomial coefficients for lane representation in ground coordinates.
    
    Lane equation: x = a0 + a1*y + a2*y^2 + a3*y^3
    Where y is the forward distance (longitudinal) and x is lateral position.
    """
    a0: float = 0.0  # Lateral offset at y=0 (meters)
    a1: float = 0.0  # Heading angle tan approximation
    a2: float = 0.0  # Curvature term
    a3: float = 0.0  # Rate of curvature change
    
    def evaluate(self, y: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at given y values."""
        return self.a0 + self.a1 * y + self.a2 * y**2 + self.a3 * y**3
    
    def derivative(self, y: np.ndarray) -> np.ndarray:
        """First derivative (heading) at given y values."""
        return self.a1 + 2 * self.a2 * y + 3 * self.a3 * y**2
    
    def curvature(self, y: np.ndarray) -> np.ndarray:
        """Curvature at given y values (approximation for small angles)."""
        return 2 * self.a2 + 6 * self.a3 * y
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.a0, self.a1, self.a2, self.a3])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PolynomialCoeffs":
        """Create from numpy array."""
        return cls(a0=arr[0], a1=arr[1], a2=arr[2], a3=arr[3])
    
    @classmethod
    def fit_from_points(cls, x: np.ndarray, y: np.ndarray) -> "PolynomialCoeffs":
        """Fit polynomial coefficients from point data."""
        if len(x) < 4:
            # Use lower order fit if insufficient points
            coeffs = np.polyfit(y, x, min(len(x) - 1, 3))
            # Pad with zeros if needed
            coeffs = np.pad(coeffs, (4 - len(coeffs), 0), 'constant')
        else:
            coeffs = np.polyfit(y, x, 3)
        # Reverse order: polyfit returns highest order first
        return cls(a0=coeffs[3], a1=coeffs[2], a2=coeffs[1], a3=coeffs[0])


@dataclass
class LaneBoundary:
    """
    Detected lane boundary representation.
    
    Stores both image-space keypoints and ground-coordinate polynomial.
    """
    # Unique ID for tracking
    id: int = -1
    
    # Image-space representation (N x 2 array of pixel coordinates)
    image_points: Optional[np.ndarray] = None
    
    # Ground-coordinate polynomial representation
    polynomial: Optional[PolynomialCoeffs] = None
    
    # Ground-coordinate points (N x 2 array: [x_lateral, y_forward] in meters)
    ground_points: Optional[np.ndarray] = None
    
    # Classification
    lane_type: LaneType = LaneType.UNKNOWN
    lane_color: LaneColor = LaneColor.UNKNOWN
    
    # Confidence scores
    detection_confidence: float = 0.0
    type_confidence: float = 0.0
    
    # Validity flags
    is_occluded: bool = False
    is_partial: bool = False  # Only partially visible
    is_valid: bool = True
    
    # Valid range in ground coordinates (meters)
    valid_y_min: float = 0.0
    valid_y_max: float = 50.0
    
    # Timestamp
    timestamp: float = 0.0
    
    def get_lateral_position(self, y_forward: float) -> Optional[float]:
        """Get lateral position at specified forward distance."""
        if self.polynomial is None:
            return None
        if y_forward < self.valid_y_min or y_forward > self.valid_y_max:
            return None
        return float(self.polynomial.evaluate(np.array([y_forward]))[0])
    
    def get_heading(self, y_forward: float) -> Optional[float]:
        """Get heading angle (rad) at specified forward distance."""
        if self.polynomial is None:
            return None
        derivative = self.polynomial.derivative(np.array([y_forward]))[0]
        return float(np.arctan(derivative))
    
    def sample_points(self, y_values: np.ndarray) -> np.ndarray:
        """Sample ground coordinate points along the lane boundary."""
        if self.polynomial is None:
            return np.array([])
        x_values = self.polynomial.evaluate(y_values)
        return np.column_stack([x_values, y_values])


@dataclass
class LaneCenter:
    """
    Computed lane centerline from left and right boundaries.
    """
    # Polynomial representation of centerline
    polynomial: Optional[PolynomialCoeffs] = None
    
    # Sampled centerline points in ground coordinates
    ground_points: Optional[np.ndarray] = None
    
    # Image-space centerline (for visualization)
    image_points: Optional[np.ndarray] = None
    
    # Lane width estimate (meters)
    lane_width: float = 3.7  # Default US highway lane width
    lane_width_std: float = 0.0  # Standard deviation of width
    
    # Overall confidence
    confidence: float = 0.0
    
    # Which boundaries were used
    left_boundary_id: int = -1
    right_boundary_id: int = -1
    has_left: bool = False
    has_right: bool = False
    
    # Valid range
    valid_y_min: float = 0.0
    valid_y_max: float = 50.0
    
    # Timestamp
    timestamp: float = 0.0
    
    def get_lateral_position(self, y_forward: float) -> Optional[float]:
        """Get centerline lateral position at forward distance."""
        if self.polynomial is None:
            return None
        return float(self.polynomial.evaluate(np.array([y_forward]))[0])
    
    def get_heading(self, y_forward: float) -> Optional[float]:
        """Get centerline heading at forward distance."""
        if self.polynomial is None:
            return None
        derivative = self.polynomial.derivative(np.array([y_forward]))[0]
        return float(np.arctan(derivative))
    
    def get_curvature(self, y_forward: float) -> Optional[float]:
        """Get curvature at forward distance."""
        if self.polynomial is None:
            return None
        return float(self.polynomial.curvature(np.array([y_forward]))[0])


@dataclass
class Lane:
    """
    Complete lane representation with both boundaries and centerline.
    """
    # Unique lane ID for tracking
    id: int = -1
    
    # Left and right boundaries
    left_boundary: Optional[LaneBoundary] = None
    right_boundary: Optional[LaneBoundary] = None
    
    # Computed centerline
    centerline: Optional[LaneCenter] = None
    
    # Is this the ego lane (vehicle is currently in this lane)
    is_ego_lane: bool = False
    
    # Lane position relative to ego (-1 = left, 0 = ego, 1 = right)
    relative_position: int = 0
    
    # Overall confidence
    confidence: float = 0.0
    
    # Valid flags
    is_valid: bool = True
    has_both_boundaries: bool = False
    
    # Timestamp
    timestamp: float = 0.0
    
    def compute_centerline(
        self,
        default_width: float = 3.7,
        y_samples: Optional[np.ndarray] = None
    ) -> None:
        """
        Compute centerline from boundaries.
        
        If only one boundary is available, estimates center using default width.
        """
        if y_samples is None:
            y_samples = np.linspace(0, 50, 50)
        
        self.centerline = LaneCenter(timestamp=self.timestamp)
        
        has_left = self.left_boundary is not None and self.left_boundary.is_valid
        has_right = self.right_boundary is not None and self.right_boundary.is_valid
        
        self.centerline.has_left = has_left
        self.centerline.has_right = has_right
        self.has_both_boundaries = has_left and has_right
        
        if has_left and has_right:
            # Both boundaries available - compute midpoint
            left_x = self.left_boundary.polynomial.evaluate(y_samples)
            right_x = self.right_boundary.polynomial.evaluate(y_samples)
            center_x = (left_x + right_x) / 2
            
            # Compute lane width statistics
            widths = np.abs(right_x - left_x)
            self.centerline.lane_width = float(np.mean(widths))
            self.centerline.lane_width_std = float(np.std(widths))
            
            # Fit polynomial to centerline
            self.centerline.polynomial = PolynomialCoeffs.fit_from_points(center_x, y_samples)
            self.centerline.ground_points = np.column_stack([center_x, y_samples])
            
            # Combined confidence
            self.centerline.confidence = min(
                self.left_boundary.detection_confidence,
                self.right_boundary.detection_confidence
            )
            self.centerline.left_boundary_id = self.left_boundary.id
            self.centerline.right_boundary_id = self.right_boundary.id
            
        elif has_left:
            # Only left boundary - estimate center using default width
            left_x = self.left_boundary.polynomial.evaluate(y_samples)
            center_x = left_x + default_width / 2
            
            self.centerline.polynomial = PolynomialCoeffs.fit_from_points(center_x, y_samples)
            self.centerline.ground_points = np.column_stack([center_x, y_samples])
            self.centerline.lane_width = default_width
            self.centerline.confidence = self.left_boundary.detection_confidence * 0.7  # Reduced
            self.centerline.left_boundary_id = self.left_boundary.id
            
        elif has_right:
            # Only right boundary - estimate center using default width
            right_x = self.right_boundary.polynomial.evaluate(y_samples)
            center_x = right_x - default_width / 2
            
            self.centerline.polynomial = PolynomialCoeffs.fit_from_points(center_x, y_samples)
            self.centerline.ground_points = np.column_stack([center_x, y_samples])
            self.centerline.lane_width = default_width
            self.centerline.confidence = self.right_boundary.detection_confidence * 0.7
            self.centerline.right_boundary_id = self.right_boundary.id
            
        else:
            # No boundaries - invalid
            self.centerline.confidence = 0.0
            self.is_valid = False
        
        # Set valid range
        if has_left and has_right:
            self.centerline.valid_y_min = max(
                self.left_boundary.valid_y_min,
                self.right_boundary.valid_y_min
            )
            self.centerline.valid_y_max = min(
                self.left_boundary.valid_y_max,
                self.right_boundary.valid_y_max
            )
        elif has_left:
            self.centerline.valid_y_min = self.left_boundary.valid_y_min
            self.centerline.valid_y_max = self.left_boundary.valid_y_max
        elif has_right:
            self.centerline.valid_y_min = self.right_boundary.valid_y_min
            self.centerline.valid_y_max = self.right_boundary.valid_y_max


@dataclass
class LaneDetectionResult:
    """
    Complete lane detection result for a single frame.
    """
    # Frame info
    frame_id: int = 0
    timestamp: float = 0.0
    
    # Detected lane boundaries
    boundaries: List[LaneBoundary] = field(default_factory=list)
    
    # Constructed lanes
    lanes: List[Lane] = field(default_factory=list)
    
    # Ego lane (if identified)
    ego_lane: Optional[Lane] = None
    ego_lane_index: int = -1
    
    # Processing info
    inference_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Status flags
    is_valid: bool = True
    fallback_active: bool = False
    fallback_reason: str = ""
    
    # Raw model outputs (for debugging)
    raw_keypoints: Optional[np.ndarray] = None
    raw_confidences: Optional[np.ndarray] = None
    raw_segmentation: Optional[np.ndarray] = None
    
    @property
    def num_boundaries(self) -> int:
        return len(self.boundaries)
    
    @property
    def num_lanes(self) -> int:
        return len(self.lanes)
    
    @property
    def has_ego_lane(self) -> bool:
        return self.ego_lane is not None and self.ego_lane.is_valid


# Alias for backward compatibility
LaneCenterline = LaneCenter