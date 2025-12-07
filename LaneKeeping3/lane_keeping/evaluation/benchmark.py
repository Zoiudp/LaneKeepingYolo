"""
Benchmark suite for lane detection models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import numpy as np
import torch

from lane_keeping.core.detector import LaneDetector
from lane_keeping.evaluation.metrics import LaneMetrics, LaneEvaluator


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    # Test scenarios
    scenarios: List[str] = None
    
    # Timing
    warmup_iterations: int = 10
    timing_iterations: int = 100
    
    # Accuracy thresholds (pass/fail criteria)
    min_precision: float = 0.95
    min_recall: float = 0.95
    max_lateral_error_10m: float = 0.10  # meters
    max_lateral_error_30m: float = 0.15  # meters
    max_inference_time_ms: float = 40.0
    
    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = [
                'day_clear',
                'day_shadow',
                'night',
                'rain',
                'fog',
                'curve',
                'merge',
                'construction',
            ]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    scenario: str
    metrics: LaneMetrics
    timing: Dict[str, float]
    passed: bool
    failures: List[str]


class LaneBenchmark:
    """
    Benchmark suite for evaluating lane detection models.
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.evaluator = LaneEvaluator()
    
    def run(
        self,
        model: LaneDetector,
        test_data_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run full benchmark suite.
        
        Args:
            model: Lane detector model to evaluate
            test_data_dir: Directory containing test data
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}
        
        for scenario in self.config.scenarios:
            scenario_dir = test_data_dir / scenario
            
            if not scenario_dir.exists():
                print(f"Skipping scenario {scenario}: directory not found")
                continue
            
            result = self._run_scenario(model, scenario, scenario_dir)
            results[scenario] = result
            
            self._print_result(result)
        
        # Compute aggregate results
        aggregate = self._aggregate_results(results)
        results['aggregate'] = aggregate
        
        # Save results
        if output_dir:
            self._save_results(results, output_dir)
        
        return results
    
    def _run_scenario(
        self,
        model: LaneDetector,
        scenario: str,
        data_dir: Path,
    ) -> BenchmarkResult:
        """Run benchmark for a single scenario."""
        import cv2
        
        self.evaluator.reset()
        
        # Load test images
        image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
        
        if not image_files:
            return BenchmarkResult(
                scenario=scenario,
                metrics=LaneMetrics(),
                timing={},
                passed=False,
                failures=['No test images found'],
            )
        
        # Warmup
        if image_files:
            warmup_img = cv2.imread(str(image_files[0]))
            model.warmup(self.config.warmup_iterations)
        
        # Timing measurements
        inference_times = []
        
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Load ground truth
            gt_path = img_path.with_suffix('.json')
            gt_data = self._load_ground_truth(gt_path)
            
            # Run inference with timing
            start = time.perf_counter()
            result = model.detect(image)
            inference_times.append((time.perf_counter() - start) * 1000)
            
            # Extract predictions
            pred_lanes = [b.image_points for b in result.boundaries if b.image_points is not None]
            pred_polys = [b.polynomial.to_array() if b.polynomial else None 
                        for b in result.boundaries]
            pred_polys = [p for p in pred_polys if p is not None]
            
            # Update evaluator
            self.evaluator.update(
                pred_lanes=pred_lanes,
                gt_lanes=gt_data.get('lanes', []),
                pred_polynomials=pred_polys,
                gt_polynomials=gt_data.get('polynomials', []),
            )
        
        # Compute metrics
        metrics = self.evaluator.compute()
        
        # Compute timing stats
        timing = {
            'mean_ms': float(np.mean(inference_times)) if inference_times else 0,
            'std_ms': float(np.std(inference_times)) if inference_times else 0,
            'p50_ms': float(np.percentile(inference_times, 50)) if inference_times else 0,
            'p95_ms': float(np.percentile(inference_times, 95)) if inference_times else 0,
            'p99_ms': float(np.percentile(inference_times, 99)) if inference_times else 0,
            'max_ms': float(np.max(inference_times)) if inference_times else 0,
        }
        
        # Check pass/fail criteria
        passed, failures = self._check_criteria(metrics, timing)
        
        return BenchmarkResult(
            scenario=scenario,
            metrics=metrics,
            timing=timing,
            passed=passed,
            failures=failures,
        )
    
    def _load_ground_truth(self, gt_path: Path) -> Dict:
        """Load ground truth annotations."""
        if not gt_path.exists():
            return {'lanes': [], 'polynomials': []}
        
        with open(gt_path, 'r') as f:
            data = json.load(f)
        
        lanes = []
        polynomials = []
        
        for lane in data.get('lanes', []):
            if 'points' in lane:
                lanes.append(np.array(lane['points']))
            if 'polynomial' in lane:
                polynomials.append(np.array(lane['polynomial']))
        
        return {'lanes': lanes, 'polynomials': polynomials}
    
    def _check_criteria(
        self,
        metrics: LaneMetrics,
        timing: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """Check if metrics meet pass criteria."""
        failures = []
        
        if metrics.precision < self.config.min_precision:
            failures.append(
                f"Precision {metrics.precision:.3f} < {self.config.min_precision}"
            )
        
        if metrics.recall < self.config.min_recall:
            failures.append(
                f"Recall {metrics.recall:.3f} < {self.config.min_recall}"
            )
        
        if metrics.mean_lateral_error_10m > self.config.max_lateral_error_10m:
            failures.append(
                f"Lateral error @10m {metrics.mean_lateral_error_10m:.3f}m > "
                f"{self.config.max_lateral_error_10m}m"
            )
        
        if metrics.mean_lateral_error_30m > self.config.max_lateral_error_30m:
            failures.append(
                f"Lateral error @30m {metrics.mean_lateral_error_30m:.3f}m > "
                f"{self.config.max_lateral_error_30m}m"
            )
        
        if timing.get('p95_ms', 0) > self.config.max_inference_time_ms:
            failures.append(
                f"P95 inference time {timing['p95_ms']:.1f}ms > "
                f"{self.config.max_inference_time_ms}ms"
            )
        
        return len(failures) == 0, failures
    
    def _aggregate_results(
        self,
        results: Dict[str, BenchmarkResult],
    ) -> BenchmarkResult:
        """Aggregate results across all scenarios."""
        all_metrics = LaneMetrics()
        all_timing = {
            'mean_ms': [],
            'p95_ms': [],
        }
        all_failures = []
        
        for scenario, result in results.items():
            if scenario == 'aggregate':
                continue
            
            # Aggregate detection metrics
            all_metrics.num_true_positives += result.metrics.num_true_positives
            all_metrics.num_false_positives += result.metrics.num_false_positives
            all_metrics.num_false_negatives += result.metrics.num_false_negatives
            all_metrics.num_samples += result.metrics.num_samples
            
            # Aggregate timing
            if result.timing:
                all_timing['mean_ms'].append(result.timing.get('mean_ms', 0))
                all_timing['p95_ms'].append(result.timing.get('p95_ms', 0))
            
            # Collect failures
            for failure in result.failures:
                all_failures.append(f"[{scenario}] {failure}")
        
        # Compute aggregate precision/recall
        total_pred = all_metrics.num_true_positives + all_metrics.num_false_positives
        total_gt = all_metrics.num_true_positives + all_metrics.num_false_negatives
        
        all_metrics.precision = all_metrics.num_true_positives / max(total_pred, 1)
        all_metrics.recall = all_metrics.num_true_positives / max(total_gt, 1)
        
        if all_metrics.precision + all_metrics.recall > 0:
            all_metrics.f1_score = (
                2 * all_metrics.precision * all_metrics.recall /
                (all_metrics.precision + all_metrics.recall)
            )
        
        # Aggregate timing
        timing = {
            'mean_ms': float(np.mean(all_timing['mean_ms'])) if all_timing['mean_ms'] else 0,
            'p95_ms': float(np.max(all_timing['p95_ms'])) if all_timing['p95_ms'] else 0,
        }
        
        return BenchmarkResult(
            scenario='aggregate',
            metrics=all_metrics,
            timing=timing,
            passed=len(all_failures) == 0,
            failures=all_failures,
        )
    
    def _print_result(self, result: BenchmarkResult) -> None:
        """Print benchmark result."""
        status = "PASS" if result.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"Scenario: {result.scenario} [{status}]")
        print(f"{'='*60}")
        
        print(f"  Precision: {result.metrics.precision:.4f}")
        print(f"  Recall: {result.metrics.recall:.4f}")
        print(f"  F1 Score: {result.metrics.f1_score:.4f}")
        
        if result.metrics.mean_lateral_error > 0:
            print(f"  Mean Lateral Error: {result.metrics.mean_lateral_error:.3f}m")
        
        if result.timing:
            print(f"  Inference Time (P95): {result.timing.get('p95_ms', 0):.1f}ms")
        
        if result.failures:
            print("  Failures:")
            for failure in result.failures:
                print(f"    - {failure}")
    
    def _save_results(
        self,
        results: Dict[str, BenchmarkResult],
        output_dir: Path,
    ) -> None:
        """Save benchmark results to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        output = {}
        for scenario, result in results.items():
            output[scenario] = {
                'passed': result.passed,
                'metrics': result.metrics.to_dict(),
                'timing': result.timing,
                'failures': result.failures,
            }
        
        output_path = output_dir / 'benchmark_results.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
