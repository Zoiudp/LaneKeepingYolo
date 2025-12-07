"""
Debug Dashboard for Lane Detection Analysis.
"""

from typing import Dict, List, Optional
import numpy as np


class DebugDashboard:
    """
    Interactive debug dashboard for lane detection analysis.
    
    Provides:
    - Real-time parameter tuning
    - Performance monitoring
    - Failure case analysis
    - Scenario filtering
    """
    
    def __init__(
        self,
        history_length: int = 1000,
    ):
        """
        Initialize dashboard.
        
        Args:
            history_length: Number of frames to keep in history
        """
        self.history_length = history_length
        
        # Performance history
        self.timing_history = []
        self.accuracy_history = []
        self.failure_cases = []
        
        # Current state
        self.current_frame = 0
        self.is_paused = False
        
        # Tunable parameters
        self.params = {
            'smoothing_alpha': 0.3,
            'lookahead_distance': 10.0,
            'confidence_threshold': 0.5,
            'fallback_timeout': 5,
        }
    
    def update(
        self,
        frame_id: int,
        timing_ms: float,
        confidence: float,
        lateral_error: Optional[float] = None,
        is_fallback: bool = False,
        failure_reason: str = "",
    ) -> None:
        """
        Update dashboard with new frame data.
        
        Args:
            frame_id: Frame identifier
            timing_ms: Processing time in milliseconds
            confidence: Detection confidence
            lateral_error: Lateral offset error (if GT available)
            is_fallback: Whether fallback mode is active
            failure_reason: Reason for failure (if any)
        """
        self.current_frame = frame_id
        
        # Update timing history
        self.timing_history.append({
            'frame_id': frame_id,
            'timing_ms': timing_ms,
        })
        
        # Update accuracy history
        self.accuracy_history.append({
            'frame_id': frame_id,
            'confidence': confidence,
            'lateral_error': lateral_error,
            'is_fallback': is_fallback,
        })
        
        # Record failure cases
        if is_fallback or failure_reason:
            self.failure_cases.append({
                'frame_id': frame_id,
                'reason': failure_reason,
                'confidence': confidence,
            })
        
        # Trim history
        if len(self.timing_history) > self.history_length:
            self.timing_history.pop(0)
        if len(self.accuracy_history) > self.history_length:
            self.accuracy_history.pop(0)
    
    def get_statistics(self) -> Dict:
        """Get current performance statistics."""
        if not self.timing_history:
            return {}
        
        timings = [h['timing_ms'] for h in self.timing_history]
        confidences = [h['confidence'] for h in self.accuracy_history if h['confidence'] is not None]
        errors = [h['lateral_error'] for h in self.accuracy_history if h['lateral_error'] is not None]
        fallbacks = [h['is_fallback'] for h in self.accuracy_history]
        
        stats = {
            'current_frame': self.current_frame,
            'total_frames': len(self.timing_history),
            
            # Timing
            'mean_timing_ms': float(np.mean(timings)),
            'p95_timing_ms': float(np.percentile(timings, 95)),
            'max_timing_ms': float(np.max(timings)),
            'fps': 1000.0 / np.mean(timings) if np.mean(timings) > 0 else 0,
            
            # Confidence
            'mean_confidence': float(np.mean(confidences)) if confidences else 0,
            'min_confidence': float(np.min(confidences)) if confidences else 0,
            
            # Accuracy
            'mean_lateral_error': float(np.mean(errors)) if errors else 0,
            'max_lateral_error': float(np.max(errors)) if errors else 0,
            
            # Reliability
            'fallback_rate': sum(fallbacks) / len(fallbacks) if fallbacks else 0,
            'num_failures': len(self.failure_cases),
        }
        
        return stats
    
    def get_failure_analysis(self) -> Dict:
        """Analyze failure cases."""
        if not self.failure_cases:
            return {'total_failures': 0}
        
        # Group by reason
        reasons = {}
        for case in self.failure_cases:
            reason = case['reason'] or 'Unknown'
            if reason not in reasons:
                reasons[reason] = 0
            reasons[reason] += 1
        
        # Sort by frequency
        sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_failures': len(self.failure_cases),
            'failure_reasons': dict(sorted_reasons),
            'recent_failures': self.failure_cases[-10:],
        }
    
    def update_param(self, name: str, value: float) -> None:
        """Update a tunable parameter."""
        if name in self.params:
            self.params[name] = value
    
    def get_params(self) -> Dict:
        """Get current parameter values."""
        return self.params.copy()
    
    def reset(self) -> None:
        """Reset dashboard state."""
        self.timing_history.clear()
        self.accuracy_history.clear()
        self.failure_cases.clear()
        self.current_frame = 0
    
    def export_report(self, filepath: str) -> None:
        """Export analysis report to file."""
        import json
        
        report = {
            'statistics': self.get_statistics(),
            'failure_analysis': self.get_failure_analysis(),
            'parameters': self.params,
            'timing_history': self.timing_history[-100:],  # Last 100 frames
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
