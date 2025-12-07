"""
Evaluation Metrics for Lane Detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class LaneMetrics:
    """
    Aggregated metrics for lane detection evaluation.
    """
    # Detection metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Geometric metrics
    mean_lateral_error: float = 0.0  # meters
    p95_lateral_error: float = 0.0
    mean_lateral_error_10m: float = 0.0  # at 10m
    mean_lateral_error_30m: float = 0.0  # at 30m
    
    # IoU metrics
    mean_iou: float = 0.0
    
    # Point-to-curve metrics
    mean_point_distance: float = 0.0
    rmse_point_distance: float = 0.0
    
    # Temporal metrics
    lane_persistence: float = 0.0  # ID continuity rate
    lateral_jitter: float = 0.0  # std of lateral offset
    
    # Per-scenario metrics
    scenario_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Counts
    num_true_positives: int = 0
    num_false_positives: int = 0
    num_false_negatives: int = 0
    num_samples: int = 0
    
    def update_detection(
        self,
        tp: int,
        fp: int,
        fn: int,
    ) -> None:
        """Update detection metrics."""
        self.num_true_positives += tp
        self.num_false_positives += fp
        self.num_false_negatives += fn
        
        total_pred = self.num_true_positives + self.num_false_positives
        total_gt = self.num_true_positives + self.num_false_negatives
        
        self.precision = self.num_true_positives / max(total_pred, 1)
        self.recall = self.num_true_positives / max(total_gt, 1)
        
        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mean_lateral_error': self.mean_lateral_error,
            'p95_lateral_error': self.p95_lateral_error,
            'mean_lateral_error_10m': self.mean_lateral_error_10m,
            'mean_lateral_error_30m': self.mean_lateral_error_30m,
            'mean_iou': self.mean_iou,
            'mean_point_distance': self.mean_point_distance,
            'rmse_point_distance': self.rmse_point_distance,
            'lane_persistence': self.lane_persistence,
            'lateral_jitter': self.lateral_jitter,
            'num_samples': self.num_samples,
        }


def compute_lane_f1(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    threshold: float = 0.5,
    image_height: int = 720,
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute F1 score for lane detection.
    
    Args:
        pred_lanes: List of predicted lane point arrays
        gt_lanes: List of ground truth lane point arrays
        threshold: IoU/distance threshold for matching
        image_height: Image height for normalization
        
    Returns:
        Tuple of (precision, recall, f1, tp, fp, fn)
    """
    num_pred = len(pred_lanes)
    num_gt = len(gt_lanes)
    
    if num_pred == 0 and num_gt == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if num_pred == 0:
        return 0.0, 0.0, 0.0, 0, 0, num_gt
    if num_gt == 0:
        return 0.0, 0.0, 0.0, 0, num_pred, 0
    
    # Compute pairwise distances
    distances = np.zeros((num_pred, num_gt))
    
    for i, pred in enumerate(pred_lanes):
        for j, gt in enumerate(gt_lanes):
            distances[i, j] = _lane_distance(pred, gt, image_height)
    
    # Match using Hungarian algorithm (simple greedy here)
    matched_pred = set()
    matched_gt = set()
    
    # Sort all pairs by distance
    pairs = []
    for i in range(num_pred):
        for j in range(num_gt):
            pairs.append((distances[i, j], i, j))
    pairs.sort()
    
    # Greedy matching
    for dist, i, j in pairs:
        if i not in matched_pred and j not in matched_gt:
            if dist < threshold:
                matched_pred.add(i)
                matched_gt.add(j)
    
    tp = len(matched_pred)
    fp = num_pred - tp
    fn = num_gt - len(matched_gt)
    
    precision = tp / max(num_pred, 1)
    recall = tp / max(num_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    return precision, recall, f1, tp, fp, fn


def _lane_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    image_height: int,
) -> float:
    """
    Compute distance between two lanes.
    
    Uses point-to-curve distance normalized by image height.
    """
    if len(pred) == 0 or len(gt) == 0:
        return float('inf')
    
    # Sample at common y values
    y_min = max(pred[:, 1].min(), gt[:, 1].min())
    y_max = min(pred[:, 1].max(), gt[:, 1].max())
    
    if y_min >= y_max:
        return float('inf')
    
    y_samples = np.linspace(y_min, y_max, 20)
    
    # Interpolate x at sample points
    pred_x = np.interp(y_samples, pred[:, 1], pred[:, 0])
    gt_x = np.interp(y_samples, gt[:, 1], gt[:, 0])
    
    # Compute mean absolute distance normalized by image height
    distance = np.mean(np.abs(pred_x - gt_x)) / image_height
    
    return distance


def compute_lateral_error(
    pred_polynomial: np.ndarray,
    gt_polynomial: np.ndarray,
    y_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute lateral error metrics in ground coordinates.
    
    Args:
        pred_polynomial: Predicted polynomial coefficients [4]
        gt_polynomial: Ground truth polynomial coefficients [4]
        y_values: Y values (forward distances) to evaluate at
        
    Returns:
        Dictionary with error metrics
    """
    # Evaluate polynomials
    pred_x = np.polyval(pred_polynomial[::-1], y_values)
    gt_x = np.polyval(gt_polynomial[::-1], y_values)
    
    # Compute errors
    errors = np.abs(pred_x - gt_x)
    
    # Find errors at specific distances
    y_10m_idx = np.argmin(np.abs(y_values - 10))
    y_30m_idx = np.argmin(np.abs(y_values - 30))
    
    return {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'p95_error': float(np.percentile(errors, 95)),
        'max_error': float(np.max(errors)),
        'error_10m': float(errors[y_10m_idx]),
        'error_30m': float(errors[y_30m_idx]),
    }


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Compute Intersection over Union for lane segmentation.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_point_to_curve_rmse(
    pred_points: np.ndarray,
    gt_curve: np.ndarray,
) -> float:
    """
    Compute RMSE of point-to-curve distances.
    
    Args:
        pred_points: Predicted points [N, 2]
        gt_curve: Ground truth curve points [M, 2]
        
    Returns:
        RMSE distance
    """
    if len(pred_points) == 0 or len(gt_curve) == 0:
        return float('inf')
    
    # For each predicted point, find nearest point on GT curve
    distances = []
    
    for point in pred_points:
        dists = np.linalg.norm(gt_curve - point, axis=1)
        distances.append(np.min(dists))
    
    rmse = np.sqrt(np.mean(np.array(distances) ** 2))
    
    return float(rmse)


class LaneEvaluator:
    """
    Evaluator for lane detection models.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        lateral_thresholds: Tuple[float, ...] = (10.0, 30.0, 50.0),
    ):
        """
        Initialize evaluator.
        
        Args:
            iou_threshold: Threshold for lane matching
            lateral_thresholds: Forward distances for lateral error evaluation
        """
        self.iou_threshold = iou_threshold
        self.lateral_thresholds = lateral_thresholds
        
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.metrics = LaneMetrics()
        self.lateral_errors = []
        self.lateral_errors_by_distance = {d: [] for d in self.lateral_thresholds}
        self.ious = []
        self.point_distances = []
    
    def update(
        self,
        pred_lanes: List[np.ndarray],
        gt_lanes: List[np.ndarray],
        pred_polynomials: Optional[List[np.ndarray]] = None,
        gt_polynomials: Optional[List[np.ndarray]] = None,
        pred_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update metrics with a single sample.
        
        Args:
            pred_lanes: Predicted lane points
            gt_lanes: Ground truth lane points
            pred_polynomials: Predicted polynomial coefficients
            gt_polynomials: Ground truth polynomial coefficients
            pred_mask: Predicted segmentation mask
            gt_mask: Ground truth segmentation mask
        """
        self.metrics.num_samples += 1
        
        # Detection metrics
        _, _, _, tp, fp, fn = compute_lane_f1(
            pred_lanes, gt_lanes, self.iou_threshold
        )
        self.metrics.update_detection(tp, fp, fn)
        
        # Lateral error metrics (if polynomials available)
        if pred_polynomials and gt_polynomials:
            for pred_poly, gt_poly in zip(pred_polynomials, gt_polynomials):
                y_values = np.linspace(0, 50, 50)
                errors = compute_lateral_error(pred_poly, gt_poly, y_values)
                
                self.lateral_errors.append(errors['mean_error'])
                
                for d in self.lateral_thresholds:
                    y_idx = np.argmin(np.abs(y_values - d))
                    pred_x = np.polyval(pred_poly[::-1], [d])[0]
                    gt_x = np.polyval(gt_poly[::-1], [d])[0]
                    self.lateral_errors_by_distance[d].append(abs(pred_x - gt_x))
        
        # IoU metrics
        if pred_mask is not None and gt_mask is not None:
            iou = compute_iou(pred_mask, gt_mask)
            self.ious.append(iou)
        
        # Point distance metrics
        for pred, gt in zip(pred_lanes, gt_lanes):
            if len(pred) > 0 and len(gt) > 0:
                rmse = compute_point_to_curve_rmse(pred, gt)
                self.point_distances.append(rmse)
    
    def compute(self) -> LaneMetrics:
        """
        Compute final metrics.
        
        Returns:
            LaneMetrics with all computed values
        """
        # Lateral errors
        if self.lateral_errors:
            self.metrics.mean_lateral_error = float(np.mean(self.lateral_errors))
            self.metrics.p95_lateral_error = float(np.percentile(self.lateral_errors, 95))
        
        if self.lateral_errors_by_distance.get(10.0):
            self.metrics.mean_lateral_error_10m = float(
                np.mean(self.lateral_errors_by_distance[10.0])
            )
        
        if self.lateral_errors_by_distance.get(30.0):
            self.metrics.mean_lateral_error_30m = float(
                np.mean(self.lateral_errors_by_distance[30.0])
            )
        
        # IoU
        if self.ious:
            self.metrics.mean_iou = float(np.mean(self.ious))
        
        # Point distances
        if self.point_distances:
            self.metrics.mean_point_distance = float(np.mean(self.point_distances))
            self.metrics.rmse_point_distance = float(
                np.sqrt(np.mean(np.array(self.point_distances) ** 2))
            )
        
        return self.metrics
