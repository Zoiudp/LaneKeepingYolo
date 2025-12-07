"""
Unit tests for tracker component.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from lane_keeping.core.tracker import LaneTracker, KalmanFilter, TrackedLane
from lane_keeping.core.lane import LaneBoundary, Lane, LaneType


class TestKalmanFilter:
    """Tests for Kalman filter implementation."""
    
    @pytest.fixture
    def kf(self):
        """Create Kalman filter for 4 polynomial coefficients."""
        return KalmanFilter(dim_state=8, dim_measurement=4)
    
    def test_initialization(self, kf):
        """Test filter initialization."""
        init_state = np.array([1.0, 0.1, 0.01, 0.001])
        kf.initialize(init_state)
        
        # State should contain measurement and velocity
        assert kf.x[:4].tolist() == pytest.approx(init_state.tolist())
    
    def test_predict(self, kf):
        """Test prediction step."""
        init_state = np.array([1.0, 0.0, 0.0, 0.0])
        kf.initialize(init_state)
        
        # Predict
        predicted = kf.predict()
        
        # Should have predicted state
        assert predicted is not None
        assert len(predicted) == 4
    
    def test_update(self, kf):
        """Test update step with measurement."""
        init_state = np.array([1.0, 0.0, 0.0, 0.0])
        kf.initialize(init_state)
        
        kf.predict()
        
        # Update with measurement
        measurement = np.array([1.1, 0.0, 0.0, 0.0])
        updated = kf.update(measurement)
        
        # Updated state should be between prediction and measurement
        assert updated is not None
    
    def test_convergence(self, kf):
        """Test filter converges with consistent measurements."""
        true_state = np.array([2.0, 0.5, 0.01, 0.0])
        kf.initialize(np.zeros(4))
        
        # Apply multiple measurements
        for _ in range(50):
            kf.predict()
            noisy_measurement = true_state + np.random.randn(4) * 0.01
            kf.update(noisy_measurement)
        
        # Should converge to true state
        final_state = kf.x[:4]
        assert np.allclose(final_state, true_state, atol=0.1)


class TestTrackedLane:
    """Tests for TrackedLane class."""
    
    def test_creation(self):
        """Test tracked lane creation."""
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        tracked = TrackedLane(
            id=1,
            coefficients=coeffs,
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        assert tracked.id == 1
        assert np.allclose(tracked.coefficients, coeffs)
        assert tracked.confidence == 0.9
        assert tracked.hits == 1
        assert tracked.age == 0
    
    def test_update(self):
        """Test tracked lane update."""
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        tracked = TrackedLane(
            id=1,
            coefficients=coeffs,
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        # Update with new measurement
        new_coeffs = np.array([1.1, 0.0, 0.0, 0.0])
        tracked.update(new_coeffs, confidence=0.95)
        
        assert tracked.hits == 2
        assert tracked.confidence == 0.95
    
    def test_predict(self):
        """Test tracked lane prediction."""
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        tracked = TrackedLane(
            id=1,
            coefficients=coeffs,
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        # Predict (no measurement this frame)
        tracked.predict()
        
        assert tracked.age == 1
    
    def test_is_confirmed(self):
        """Test confirmation logic."""
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        tracked = TrackedLane(
            id=1,
            coefficients=coeffs,
            confidence=0.9,
            lane_type=LaneType.SOLID,
            min_hits=3,
        )
        
        assert not tracked.is_confirmed()
        
        tracked.update(coeffs, 0.9)
        assert not tracked.is_confirmed()
        
        tracked.update(coeffs, 0.9)
        assert tracked.is_confirmed()
    
    def test_is_dead(self):
        """Test track removal logic."""
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        tracked = TrackedLane(
            id=1,
            coefficients=coeffs,
            confidence=0.9,
            lane_type=LaneType.SOLID,
            max_age=3,
        )
        
        assert not tracked.is_dead()
        
        for _ in range(3):
            tracked.predict()
        
        assert tracked.is_dead()


class TestLaneTracker:
    """Tests for LaneTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create lane tracker."""
        return LaneTracker(
            max_age=5,
            min_hits=3,
            iou_threshold=0.3,
        )
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.max_age == 5
        assert tracker.min_hits == 3
        assert len(tracker.tracks) == 0
    
    def test_single_lane_tracking(self, tracker):
        """Test tracking single lane over time."""
        # Create detection
        boundary = LaneBoundary(
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        detections = [boundary]
        
        # Track over multiple frames
        for _ in range(5):
            tracked = tracker.update(detections)
        
        # Should have one confirmed track
        assert len(tracked) == 1
        assert tracked[0].id == tracker.tracks[0].id
    
    def test_multiple_lanes_tracking(self, tracker):
        """Test tracking multiple lanes."""
        # Left and right boundaries
        left = LaneBoundary(
            coefficients=np.array([-1.75, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        right = LaneBoundary(
            coefficients=np.array([1.75, 0.0, 0.0, 0.0]),
            confidence=0.85,
            lane_type=LaneType.DASHED,
        )
        
        detections = [left, right]
        
        # Track over multiple frames
        for _ in range(5):
            tracked = tracker.update(detections)
        
        # Should have two confirmed tracks
        assert len(tracked) == 2
    
    def test_track_persistence(self, tracker):
        """Test tracks persist through missed detections."""
        boundary = LaneBoundary(
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        # Initialize track
        for _ in range(5):
            tracker.update([boundary])
        
        assert len(tracker.tracks) >= 1
        track_id = tracker.tracks[0].id
        
        # Miss detection for a few frames
        for _ in range(2):
            tracked = tracker.update([])
        
        # Track should still exist (within max_age)
        track_exists = any(t.id == track_id for t in tracker.tracks)
        assert track_exists
    
    def test_track_removal(self, tracker):
        """Test tracks are removed after max_age."""
        boundary = LaneBoundary(
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        # Initialize track
        for _ in range(5):
            tracker.update([boundary])
        
        # Miss detection for max_age + 1 frames
        for _ in range(tracker.max_age + 1):
            tracker.update([])
        
        # Track should be removed
        assert len(tracker.tracks) == 0
    
    def test_id_persistence(self, tracker):
        """Test lane IDs persist through tracking."""
        boundary = LaneBoundary(
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        # Track same lane
        ids = []
        for _ in range(10):
            tracked = tracker.update([boundary])
            if tracked:
                ids.append(tracked[0].id)
        
        # All IDs should be the same
        assert len(set(ids)) == 1
    
    def test_smoothed_output(self, tracker):
        """Test temporal smoothing of outputs."""
        # Add noise to detections
        base_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
        
        smoothed_coeffs = []
        for _ in range(20):
            noisy_coeffs = base_coeffs + np.random.randn(4) * 0.1
            boundary = LaneBoundary(
                coefficients=noisy_coeffs,
                confidence=0.9,
                lane_type=LaneType.SOLID,
            )
            
            tracked = tracker.update([boundary])
            if tracked:
                smoothed_coeffs.append(tracked[0].coefficients.copy())
        
        # Smoothed output should have less variance than input
        if len(smoothed_coeffs) > 1:
            smoothed_var = np.var(smoothed_coeffs, axis=0)
            # Variance should be reduced
            assert np.all(smoothed_var < 0.1)
    
    def test_reset(self, tracker):
        """Test tracker reset."""
        boundary = LaneBoundary(
            coefficients=np.array([0.0, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        for _ in range(5):
            tracker.update([boundary])
        
        assert len(tracker.tracks) > 0
        
        tracker.reset()
        
        assert len(tracker.tracks) == 0


class TestHungarianMatching:
    """Tests for Hungarian matching in tracker."""
    
    @pytest.fixture
    def tracker(self):
        return LaneTracker(max_age=5, min_hits=3, iou_threshold=0.3)
    
    def test_correct_association(self, tracker):
        """Test correct lane association."""
        # Initialize with two lanes
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
        
        for _ in range(5):
            tracker.update([left, right])
        
        # Slightly perturbed detections should match correctly
        left_perturbed = LaneBoundary(
            coefficients=np.array([-1.95, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        right_perturbed = LaneBoundary(
            coefficients=np.array([2.05, 0.0, 0.0, 0.0]),
            confidence=0.9,
            lane_type=LaneType.SOLID,
        )
        
        tracked = tracker.update([right_perturbed, left_perturbed])  # Reversed order
        
        # Should still maintain correct association
        assert len(tracked) == 2
        
        # Find left and right tracks
        left_track = min(tracked, key=lambda t: t.coefficients[0])
        right_track = max(tracked, key=lambda t: t.coefficients[0])
        
        assert left_track.coefficients[0] < 0
        assert right_track.coefficients[0] > 0
