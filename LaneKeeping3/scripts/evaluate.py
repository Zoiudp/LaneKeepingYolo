"""
Evaluation script for lane detection model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
import torch
from tqdm import tqdm

from lane_keeping.models.lane_yolo import LaneYOLO
from lane_keeping.data.datasets import TuSimpleDataset, CULaneDataset
from lane_keeping.data.dataloader import create_dataloader
from lane_keeping.evaluation.metrics import LaneEvaluator, LaneMetrics
from lane_keeping.evaluation.benchmark import LaneBenchmark


def evaluate(config: Dict, weights_path: str, device: str = 'cuda') -> Dict:
    """
    Evaluate lane detection model.
    
    Args:
        config: Evaluation configuration
        weights_path: Path to model weights
        device: Device to use
    
    Returns:
        Evaluation metrics dictionary
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LaneYOLO(
        num_keypoints=config['model']['num_keypoints'],
        num_lanes=config['model']['num_lanes'],
        use_polynomial=config['model']['use_polynomial'],
        backbone_variant=config['model']['backbone_variant'],
    )
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create dataset
    test_dataset = create_dataset(config['data']['test_dir'], 'test', config)
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
    )
    
    # Create evaluator
    evaluator = LaneEvaluator(
        lateral_threshold=config.get('lateral_threshold', 0.15),
    )
    
    # Evaluate
    latencies = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            
            # Measure latency
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            outputs = model(images)
            end.record()
            
            torch.cuda.synchronize()
            latency = start.elapsed_time(end)
            latencies.append(latency)
            
            # Update evaluator
            batch_size = images.shape[0]
            for i in range(batch_size):
                pred_kpts = outputs['keypoints'][i].cpu().numpy()
                gt_kpts = batch['keypoints'][i].numpy()
                pred_conf = outputs['confidences'][i].cpu().numpy()
                gt_exists = batch['lane_exists'][i].numpy()
                
                pred_lanes = [pred_kpts[j] for j in range(len(pred_conf)) if pred_conf[j] > 0.5]
                gt_lanes = [gt_kpts[j] for j in range(len(gt_exists)) if gt_exists[j] > 0.5]
                
                evaluator.update(pred_lanes, gt_lanes)
    
    # Compute metrics
    metrics = evaluator.compute()
    
    # Latency stats
    latencies = np.array(latencies)
    
    results = {
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1_score': metrics.f1_score,
        'mean_lateral_error_m': metrics.mean_lateral_error,
        'latency_ms': {
            'mean': float(np.mean(latencies)),
            'std': float(np.std(latencies)),
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'p99': float(np.percentile(latencies, 99)),
        },
        'fps': float(1000.0 / np.mean(latencies)),
    }
    
    return results


def evaluate_scenarios(config: Dict, weights_path: str, device: str = 'cuda') -> Dict:
    """
    Evaluate model on different scenarios.
    
    Args:
        config: Evaluation configuration
        weights_path: Path to model weights
        device: Device to use
    
    Returns:
        Per-scenario metrics dictionary
    """
    benchmark = LaneBenchmark(config)
    
    # Load model
    model = LaneYOLO(
        num_keypoints=config['model']['num_keypoints'],
        num_lanes=config['model']['num_lanes'],
        use_polynomial=config['model']['use_polynomial'],
        backbone_variant=config['model']['backbone_variant'],
    )
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Run benchmark
    results = benchmark.run(model, device)
    
    return results


def create_dataset(data_dir: str, split: str, config: Dict):
    """Create dataset based on configuration."""
    dataset_type = config['data'].get('dataset_type', 'tusimple')
    
    if dataset_type == 'tusimple':
        return TuSimpleDataset(
            root_dir=data_dir,
            split=split,
            transform=None,
            num_keypoints=config['model']['num_keypoints'],
            num_lanes=config['model']['num_lanes'],
            image_size=tuple(config['model']['input_size']),
        )
    elif dataset_type == 'culane':
        return CULaneDataset(
            root_dir=data_dir,
            split=split,
            transform=None,
            num_keypoints=config['model']['num_keypoints'],
            num_lanes=config['model']['num_lanes'],
            image_size=tuple(config['model']['input_size']),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def print_results(results: Dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\nPrecision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Mean Lateral Error: {results['mean_lateral_error_m']:.4f} m")
    
    print(f"\nLatency:")
    print(f"  Mean: {results['latency_ms']['mean']:.2f} ms")
    print(f"  P50: {results['latency_ms']['p50']:.2f} ms")
    print(f"  P95: {results['latency_ms']['p95']:.2f} ms")
    print(f"  P99: {results['latency_ms']['p99']:.2f} ms")
    print(f"  FPS: {results['fps']:.1f}")
    
    # Check acceptance criteria
    print("\n" + "-" * 50)
    print("ACCEPTANCE CRITERIA CHECK")
    print("-" * 50)
    
    criteria = [
        ("Precision >= 0.95", results['precision'] >= 0.95),
        ("Recall >= 0.95", results['recall'] >= 0.95),
        ("F1 Score >= 0.95", results['f1_score'] >= 0.95),
        ("Lateral Error <= 0.15m", results['mean_lateral_error_m'] <= 0.15),
        ("Latency <= 40ms", results['latency_ms']['p95'] <= 40),
    ]
    
    all_pass = True
    for name, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
        all_pass = all_pass and passed
    
    print("-" * 50)
    if all_pass:
        print("All acceptance criteria PASSED!")
    else:
        print("Some acceptance criteria FAILED.")


def main():
    parser = argparse.ArgumentParser(description='Evaluate lane detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--scenarios', action='store_true', help='Evaluate on scenarios')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run evaluation
    if args.scenarios:
        results = evaluate_scenarios(config, args.weights, args.device)
    else:
        results = evaluate(config, args.weights, args.device)
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
