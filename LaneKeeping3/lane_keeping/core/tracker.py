"""
Lane Tracker - Temporal tracking and smoothing for lane boundaries.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from filterpy.kalman import KalmanFilter

from lane_keeping.core.lane import (
    Lane,
    LaneBoundary,
    LaneCenter,
    LaneDetectionResult,
    PolynomialCoeffs,
)


@dataclass
class TrackState:
    """State for tracking a single lane boundary."""
    id: int
    kalman_filter: KalmanFilter
    polynomial: PolynomialCoeffs
    age: int = 0  # Frames since creation
    hits: int = 0  # Successful detections
    misses: int = 0  # Consecutive missed detections
    confidence: float = 0.0
    last_update_time: float = 0.0
    is_confirmed: bool = False  # Track confirmed after min_hits


class LaneTracker:
    """
    Multi-object tracker for lane boundaries with Kalman filtering.
    
    Tracks lane boundary polynomial coefficients over time to provide
    temporally consistent lane estimates.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        """
        Initialize tracker.
        
        Args:
            config: Tracker configuration
        """
        self.config = config or self._default_config()
        
        # Tracking parameters
        self.max_age = self.config.get('max_age', 10)  # Max frames without detection
        self.min_hits = self.config.get('min_hits', 3)  # Min hits to confirm track
        self.iou_threshold = self.config.get('iou_threshold', 0.3)
        self.distance_threshold = self.config.get('distance_threshold', 0.5)  # meters
        
        # Kalman filter parameters
        self.process_noise = self.config.get('process_noise', 0.01)
        self.measurement_noise = self.config.get('measurement_noise', 0.1)
        
        # Smoothing parameters
        self.smoothing_alpha = self.config.get('smoothing_alpha', 0.3)  # EMA alpha
        self.use_kalman = self.config.get('use_kalman', True)
        
        # State
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 0
        self.frame_count = 0
        
        # History for temporal analysis
        self.history_length = self.config.get('history_length', 30)
        self.detection_history: deque = deque(maxlen=self.history_length)
        
        # Ego motion (for motion compensation)
        self.ego_speed: float = 0.0  # m/s
        self.ego_yaw_rate: float = 0.0  # rad/s
        self.dt: float = 0.033  # Frame interval (30 FPS default)
    
    def _default_config(self) -> Dict:
        """Default tracker configuration."""
        return {
            'max_age': 10,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'distance_threshold': 0.5,
            'process_noise': 0.01,
            'measurement_noise': 0.1,
            'smoothing_alpha': 0.3,
            'use_kalman': True,
            'history_length': 30,
        }
    
    def update_ego_motion(
        self,
        speed: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> None:
        """
        Update ego vehicle motion for motion compensation.
        
        Args:
            speed: Vehicle speed in m/s
            yaw_rate: Yaw rate in rad/s
            dt: Time interval since last frame
        """
        if speed is not None:
            self.ego_speed = speed
        if yaw_rate is not None:
            self.ego_yaw_rate = yaw_rate
        if dt is not None:
            self.dt = dt
    
    def _create_kalman_filter(self, polynomial: PolynomialCoeffs) -> KalmanFilter:
        """
        Create Kalman filter for tracking polynomial coefficients.
        
        State: [a0, a1, a2, a3, da0, da1, da2, da3]
        Measurement: [a0, a1, a2, a3]
        """
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        dt = self.dt
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # Process noise
        q = self.process_noise
        kf.Q = np.diag([q, q, q, q, q*10, q*10, q*10, q*10])
        
        # Scale process noise by speed (faster = more change expected)
        speed_factor = 1.0 + self.ego_speed / 20.0  # Normalize by 20 m/s
        kf.Q *= speed_factor
        
        # Measurement noise
        r = self.measurement_noise
        kf.R = np.diag([r, r*2, r*4, r*8])  # Higher order = more noise
        
        # Initial state
        coeffs = polynomial.to_array()
        kf.x = np.array([coeffs[0], coeffs[1], coeffs[2], coeffs[3], 0, 0, 0, 0])
        
        # Initial covariance
        kf.P = np.diag([r, r, r, r, r*10, r*10, r*10, r*10])
        
        return kf
    
    def _predict_track(self, track: TrackState) -> PolynomialCoeffs:
        """Predict track state forward one timestep."""
        if self.use_kalman:
            track.kalman_filter.predict()
            state = track.kalman_filter.x
            return PolynomialCoeffs(
                a0=state[0],
                a1=state[1],
                a2=state[2],
                a3=state[3],
            )
        else:
            # Simple persistence (no prediction)
            return track.polynomial
    
    def _update_track(
        self,
        track: TrackState,
        measurement: LaneBoundary,
    ) -> None:
        """Update track with new measurement."""
        if measurement.polynomial is None:
            return
        
        coeffs = measurement.polynomial.to_array()
        
        if self.use_kalman:
            track.kalman_filter.update(coeffs)
            state = track.kalman_filter.x
            track.polynomial = PolynomialCoeffs(
                a0=state[0],
                a1=state[1],
                a2=state[2],
                a3=state[3],
            )
        else:
            # Exponential moving average
            alpha = self.smoothing_alpha
            old_coeffs = track.polynomial.to_array()
            new_coeffs = alpha * coeffs + (1 - alpha) * old_coeffs
            track.polynomial = PolynomialCoeffs.from_array(new_coeffs)
        
        track.confidence = measurement.detection_confidence
        track.last_update_time = time.time()
        track.hits += 1
        track.misses = 0
        track.age += 1
        
        if track.hits >= self.min_hits:
            track.is_confirmed = True
    
    def _compute_cost_matrix(
        self,
        tracks: List[TrackState],
        detections: List[LaneBoundary],
    ) -> np.ndarray:
        """
        Compute cost matrix for track-detection association.
        
        Uses polynomial coefficient distance and point-wise distance.
        """
        num_tracks = len(tracks)
        num_dets = len(detections)
        
        if num_tracks == 0 or num_dets == 0:
            return np.empty((0, 0))
        
        cost_matrix = np.zeros((num_tracks, num_dets))
        
        # Sample points for distance computation
        y_samples = np.linspace(5, 40, 10)
        
        for i, track in enumerate(tracks):
            track_poly = track.polynomial
            track_points = track_poly.evaluate(y_samples)
            
            for j, det in enumerate(detections):
                if det.polynomial is None:
                    cost_matrix[i, j] = float('inf')
                    continue
                
                det_poly = det.polynomial
                det_points = det_poly.evaluate(y_samples)
                
                # Point-wise distance
                point_dist = np.mean(np.abs(track_points - det_points))
                
                # Coefficient distance (weighted)
                coeff_dist = np.sum(np.abs(track_poly.to_array() - det_poly.to_array()) * 
                                   [1, 2, 4, 8])  # Weight higher orders more
                
                cost_matrix[i, j] = point_dist + 0.1 * coeff_dist
        
        return cost_matrix
    
    def _associate(
        self,
        tracks: List[TrackState],
        detections: List[LaneBoundary],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using Hungarian algorithm.
        
        Returns:
            matches: List of (track_idx, detection_idx) pairs
            unmatched_tracks: Indices of unmatched tracks
            unmatched_detections: Indices of unmatched detections
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        cost_matrix = self._compute_cost_matrix(tracks, detections)
        
        # Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < self.distance_threshold:
                matches.append((track_idx, det_idx))
                unmatched_tracks.discard(track_idx)
                unmatched_detections.discard(det_idx)
        
        return matches, list(unmatched_tracks), list(unmatched_detections)
    
    def update(
        self,
        detection_result: LaneDetectionResult,
    ) -> LaneDetectionResult:
        """
        Update tracker with new detection result.
        
        Args:
            detection_result: Raw detection result
            
        Returns:
            Updated detection result with tracked, smoothed lanes
        """
        self.frame_count += 1
        
        # Get current detections
        detections = detection_result.boundaries
        
        # Predict all tracks forward
        track_list = list(self.tracks.values())
        for track in track_list:
            self._predict_track(track)
        
        # Associate tracks with detections
        matches, unmatched_tracks, unmatched_dets = self._associate(
            track_list, detections
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = track_list[track_idx]
            detection = detections[det_idx]
            self._update_track(track, detection)
        
        # Handle unmatched tracks (potential occlusion)
        for track_idx in unmatched_tracks:
            track = track_list[track_idx]
            track.misses += 1
            track.age += 1
            
            if track.misses > self.max_age:
                del self.tracks[track.id]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            if detection.polynomial is None:
                continue
            
            new_track = TrackState(
                id=self.next_id,
                kalman_filter=self._create_kalman_filter(detection.polynomial),
                polynomial=detection.polynomial,
                confidence=detection.detection_confidence,
                last_update_time=time.time(),
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # Create updated detection result with smoothed lanes
        tracked_result = self._create_tracked_result(detection_result)
        
        # Store in history
        self.detection_history.append(tracked_result)
        
        return tracked_result
    
    def _create_tracked_result(
        self,
        original_result: LaneDetectionResult,
    ) -> LaneDetectionResult:
        """Create detection result from tracked state."""
        result = LaneDetectionResult(
            frame_id=self.frame_count,
            timestamp=original_result.timestamp,
            inference_time_ms=original_result.inference_time_ms,
            postprocess_time_ms=original_result.postprocess_time_ms,
            total_time_ms=original_result.total_time_ms,
        )
        
        # Convert tracks to boundaries
        tracked_boundaries = []
        for track in self.tracks.values():
            if not track.is_confirmed:
                continue
            
            boundary = LaneBoundary(
                id=track.id,
                polynomial=track.polynomial,
                detection_confidence=track.confidence,
                timestamp=original_result.timestamp,
                is_valid=track.misses == 0,  # Mark as uncertain if missed
            )
            
            # Set valid range
            boundary.valid_y_min = 0.0
            boundary.valid_y_max = 50.0
            
            tracked_boundaries.append(boundary)
        
        result.boundaries = tracked_boundaries
        
        # Reconstruct lanes
        result.lanes = self._construct_tracked_lanes(tracked_boundaries)
        
        # Identify ego lane
        if result.lanes:
            result.ego_lane, result.ego_lane_index = self._identify_ego_lane(result.lanes)
        
        result.is_valid = len(tracked_boundaries) > 0
        if not result.is_valid and len(original_result.boundaries) == 0:
            result.fallback_active = True
            result.fallback_reason = "No tracked or detected lanes"
        
        return result
    
    def _construct_tracked_lanes(
        self,
        boundaries: List[LaneBoundary],
    ) -> List[Lane]:
        """Construct lanes from tracked boundaries."""
        if len(boundaries) < 1:
            return []
        
        lanes = []
        
        # Sort by lateral position
        sorted_boundaries = sorted(
            boundaries,
            key=lambda b: b.get_lateral_position(5.0) or 0.0
        )
        
        # Pair consecutive boundaries
        for i in range(len(sorted_boundaries) - 1):
            left = sorted_boundaries[i]
            right = sorted_boundaries[i + 1]
            
            left_x = left.get_lateral_position(10.0)
            right_x = right.get_lateral_position(10.0)
            
            if left_x is not None and right_x is not None:
                width = abs(right_x - left_x)
                if 2.5 < width < 5.0:
                    lane = Lane(
                        id=len(lanes),
                        left_boundary=left,
                        right_boundary=right,
                        timestamp=left.timestamp,
                    )
                    lane.compute_centerline()
                    lanes.append(lane)
        
        return lanes
    
    def _identify_ego_lane(
        self,
        lanes: List[Lane],
    ) -> Tuple[Optional[Lane], int]:
        """Identify ego lane from tracked lanes."""
        if not lanes:
            return None, -1
        
        ego_lane = None
        ego_idx = -1
        min_offset = float('inf')
        
        for idx, lane in enumerate(lanes):
            if lane.centerline is None:
                continue
            
            offset = abs(lane.centerline.get_lateral_position(5.0) or float('inf'))
            
            if offset < min_offset and offset < 2.0:  # Must be within 2m
                min_offset = offset
                ego_lane = lane
                ego_idx = idx
        
        if ego_lane:
            ego_lane.is_ego_lane = True
            ego_lane.relative_position = 0
        
        return ego_lane, ego_idx
    
    def get_smoothed_centerline(
        self,
        lookahead_distance: float = 30.0,
        num_points: int = 30,
    ) -> Optional[np.ndarray]:
        """
        Get smoothed centerline points from tracking history.
        
        Uses weighted average of recent detections for extra smoothing.
        """
        if len(self.detection_history) == 0:
            return None
        
        # Get recent ego lane centerlines
        centerlines = []
        weights = []
        
        for i, result in enumerate(reversed(self.detection_history)):
            if result.ego_lane is None or result.ego_lane.centerline is None:
                continue
            
            weight = 0.9 ** i  # Exponential decay
            weights.append(weight)
            
            y_samples = np.linspace(0, lookahead_distance, num_points)
            x_samples = result.ego_lane.centerline.polynomial.evaluate(y_samples)
            centerlines.append(np.column_stack([x_samples, y_samples]))
        
        if not centerlines:
            return None
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        smoothed = np.zeros_like(centerlines[0])
        
        for cl, w in zip(centerlines, weights):
            smoothed += cl * w
        
        return smoothed
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.detection_history.clear()
        self.frame_count = 0
        self.next_id = 0
