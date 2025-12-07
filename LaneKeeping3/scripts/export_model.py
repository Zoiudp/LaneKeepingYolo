"""
Model export script for lane detection.
"""

import argparse
from pathlib import Path
import yaml
import torch

from lane_keeping.models.lane_yolo import LaneYOLO
from lane_keeping.deployment.export import ModelExporter


def main():
    parser = argparse.ArgumentParser(description='Export lane detection model')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx', 'torchscript', 'tensorrt'],
                        help='Export format')
    parser.add_argument('--fp16', action='store_true', help='Export with FP16 precision')
    parser.add_argument('--int8', action='store_true', help='Export with INT8 precision')
    parser.add_argument('--calibration-data', type=str, default=None,
                        help='Path to calibration data for INT8 quantization')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for export')
    parser.add_argument('--dynamic-batch', action='store_true', help='Enable dynamic batch size')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    print("Creating model...")
    model = LaneYOLO(
        num_keypoints=config['model']['num_keypoints'],
        num_lanes=config['model']['num_lanes'],
        use_polynomial=config['model']['use_polynomial'],
        backbone_variant=config['model']['backbone_variant'],
    )
    
    # Load weights
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get input size
    input_size = tuple(config['model']['input_size'])
    
    # Create exporter
    exporter = ModelExporter(model, input_size=input_size)
    
    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'onnx':
        print(f"Exporting to ONNX (opset {args.opset})...")
        exporter.export_onnx(
            str(output_path),
            opset_version=args.opset,
            dynamic_batch=args.dynamic_batch,
            simplify=True,
        )
        
        # Verify
        exporter.verify_onnx(str(output_path))
        
    elif args.format == 'torchscript':
        print("Exporting to TorchScript...")
        exporter.export_torchscript(str(output_path))
        
    elif args.format == 'tensorrt':
        print("Exporting to TensorRT...")
        
        # Determine precision
        precision = 'fp32'
        if args.fp16:
            precision = 'fp16'
        elif args.int8:
            precision = 'int8'
        
        # Build engine
        onnx_path = str(output_path).replace('.engine', '.onnx')
        
        # First export to ONNX
        exporter.export_onnx(onnx_path, dynamic_batch=False)
        
        # Then convert to TensorRT
        exporter.export_tensorrt(
            onnx_path,
            str(output_path),
            precision=precision,
            calibration_data=args.calibration_data,
            max_batch_size=args.batch_size,
        )
    
    print(f"Export complete: {output_path}")
    
    # Print model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6:.2f}M")
    
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
