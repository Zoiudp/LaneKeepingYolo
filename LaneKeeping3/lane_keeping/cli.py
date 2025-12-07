"""
Lane Keeping CLI - Command line interface for lane detection system.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Lane Keeping System - Production-grade lane detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  lane-keeping train --config configs/train_tusimple.yaml
  
  # Run inference on video
  lane-keeping infer --config configs/base.yaml --weights best.pt --source video.mp4
  
  # Evaluate model
  lane-keeping eval --config configs/base.yaml --weights best.pt
  
  # Export model
  lane-keeping export --config configs/base.yaml --weights best.pt --format onnx
  
  # Run live camera
  lane-keeping live --config configs/deploy_jetson.yaml --weights best.engine
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train lane detection model')
    train_parser.add_argument('--config', type=str, required=True, help='Config file')
    train_parser.add_argument('--output', type=str, default='outputs/train', help='Output dir')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--config', type=str, required=True, help='Config file')
    infer_parser.add_argument('--weights', type=str, required=True, help='Model weights')
    infer_parser.add_argument('--source', type=str, required=True, help='Input source')
    infer_parser.add_argument('--output', type=str, default=None, help='Output path')
    infer_parser.add_argument('--device', type=str, default='cuda', help='Device')
    infer_parser.add_argument('--tensorrt', action='store_true', help='Use TensorRT')
    infer_parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--config', type=str, required=True, help='Config file')
    eval_parser.add_argument('--weights', type=str, required=True, help='Model weights')
    eval_parser.add_argument('--output', type=str, default='eval_results.json', help='Output')
    eval_parser.add_argument('--device', type=str, default='cuda', help='Device')
    eval_parser.add_argument('--scenarios', action='store_true', help='Run scenario tests')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--config', type=str, required=True, help='Config file')
    export_parser.add_argument('--weights', type=str, required=True, help='Model weights')
    export_parser.add_argument('--output', type=str, required=True, help='Output path')
    export_parser.add_argument('--format', type=str, default='onnx',
                               choices=['onnx', 'torchscript', 'tensorrt'])
    export_parser.add_argument('--fp16', action='store_true', help='FP16 precision')
    export_parser.add_argument('--int8', action='store_true', help='INT8 precision')
    export_parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    
    # Live command
    live_parser = subparsers.add_parser('live', help='Run live camera inference')
    live_parser.add_argument('--config', type=str, required=True, help='Config file')
    live_parser.add_argument('--weights', type=str, required=True, help='Model weights')
    live_parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    live_parser.add_argument('--output', type=str, default=None, help='Save video')
    live_parser.add_argument('--tensorrt', action='store_true', help='Use TensorRT')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    bench_parser.add_argument('--config', type=str, required=True, help='Config file')
    bench_parser.add_argument('--weights', type=str, required=True, help='Model weights')
    bench_parser.add_argument('--iterations', type=int, default=100, help='Iterations')
    bench_parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate handler
    if args.command == 'train':
        from scripts.train import main as train_main
        sys.argv = ['train', '--config', args.config, '--output', args.output, '--device', args.device]
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        train_main()
    
    elif args.command == 'infer':
        from scripts.inference import main as infer_main
        sys.argv = ['inference', '--config', args.config, '--weights', args.weights,
                   '--source', args.source, '--device', args.device]
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.tensorrt:
            sys.argv.append('--tensorrt')
        if args.no_display:
            sys.argv.append('--no-display')
        infer_main()
    
    elif args.command == 'eval':
        from scripts.evaluate import main as eval_main
        sys.argv = ['evaluate', '--config', args.config, '--weights', args.weights,
                   '--output', args.output, '--device', args.device]
        if args.scenarios:
            sys.argv.append('--scenarios')
        eval_main()
    
    elif args.command == 'export':
        from scripts.export_model import main as export_main
        sys.argv = ['export', '--config', args.config, '--weights', args.weights,
                   '--output', args.output, '--format', args.format,
                   '--batch-size', str(args.batch_size)]
        if args.fp16:
            sys.argv.append('--fp16')
        if args.int8:
            sys.argv.append('--int8')
        export_main()
    
    elif args.command == 'live':
        run_live(args)
    
    elif args.command == 'benchmark':
        run_benchmark(args)


def run_live(args):
    """Run live camera inference."""
    import yaml
    from lane_keeping.core.system import LaneKeepingSystem
    from lane_keeping.visualization.overlay import LaneOverlay
    from scripts.inference import run_inference_camera
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    system = LaneKeepingSystem(config)
    
    if args.tensorrt:
        system.load_tensorrt_engine(args.weights)
    else:
        system.load_weights(args.weights)
    
    overlay = LaneOverlay()
    
    run_inference_camera(
        system, overlay,
        camera_id=args.camera,
        output_path=args.output,
    )


def run_benchmark(args):
    """Run performance benchmark."""
    import yaml
    import torch
    import numpy as np
    import time
    
    from lane_keeping.models.lane_yolo import LaneYOLO
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = LaneYOLO(
        num_keypoints=config['model']['num_keypoints'],
        num_lanes=config['model']['num_lanes'],
        use_polynomial=config['model']['use_polynomial'],
        backbone_variant=config['model']['backbone_variant'],
    )
    
    # Load weights
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Benchmark
    input_size = config['model']['input_size']
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Warm up
    print("Warming up...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {args.iterations} iterations...")
    latencies = []
    
    for _ in range(args.iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
    
    latencies = np.array(latencies)
    
    print("\n=== Benchmark Results ===")
    print(f"Device: {device}")
    print(f"Input size: {input_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Mean latency: {np.mean(latencies):.2f} ms")
    print(f"Std latency: {np.std(latencies):.2f} ms")
    print(f"Min latency: {np.min(latencies):.2f} ms")
    print(f"Max latency: {np.max(latencies):.2f} ms")
    print(f"P50 latency: {np.percentile(latencies, 50):.2f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 latency: {np.percentile(latencies, 99):.2f} ms")
    print(f"Throughput: {1000.0 / np.mean(latencies):.1f} FPS")
    
    # Check against target
    target_latency = 40.0
    if np.percentile(latencies, 95) <= target_latency:
        print(f"\n✓ Meets latency target (<= {target_latency} ms @ P95)")
    else:
        print(f"\n✗ Does not meet latency target (<= {target_latency} ms @ P95)")


if __name__ == '__main__':
    main()
