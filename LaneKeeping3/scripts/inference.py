"""
Inference script for lane detection.
"""

import argparse
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import yaml
import torch

from lane_keeping.core.system import LaneKeepingSystem
from lane_keeping.visualization.overlay import LaneOverlay


def run_inference_video(
    system: LaneKeepingSystem,
    overlay: LaneOverlay,
    video_path: str,
    output_path: Optional[str] = None,
    show_display: bool = True,
    target_fps: float = 30.0,
) -> dict:
    """
    Run inference on video file.
    
    Args:
        system: Lane keeping system
        overlay: Visualization overlay
        video_path: Path to input video
        output_path: Optional path to save output video
        show_display: Whether to show live display
        target_fps: Target FPS for playback
    
    Returns:
        Dictionary with inference statistics
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics
    latencies = []
    frame_count = 0
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_time = time.perf_counter()
        result = system.process_frame(frame_rgb)
        latency = (time.perf_counter() - start_time) * 1000  # ms
        latencies.append(latency)
        
        # Create visualization
        vis_frame = overlay.render(
            frame_rgb,
            result.lanes,
            result.centerline,
            result.guidance,
        )
        
        # Add stats overlay
        vis_frame = add_stats_overlay(vis_frame, result, latency)
        
        # Convert back to BGR for OpenCV
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Save frame
        if writer:
            writer.write(vis_frame_bgr)
        
        # Display
        if show_display:
            cv2.imshow('Lane Detection', vis_frame_bgr)
            key = cv2.waitKey(int(1000 / target_fps))
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)  # Pause
        
        frame_count += 1
        
        # Progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Compute statistics
    latencies = np.array(latencies)
    stats = {
        'total_frames': frame_count,
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99)),
        'fps': float(1000.0 / np.mean(latencies)),
    }
    
    print("\n=== Inference Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    return stats


def run_inference_camera(
    system: LaneKeepingSystem,
    overlay: LaneOverlay,
    camera_id: int = 0,
    output_path: Optional[str] = None,
) -> None:
    """
    Run inference on camera stream.
    
    Args:
        system: Lane keeping system
        overlay: Visualization overlay
        camera_id: Camera device ID
        output_path: Optional path to save output video
    """
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera: {camera_id}")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera: {width}x{height}")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    print("Press 'q' to quit, 'r' to reset tracker")
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start_time = time.perf_counter()
        result = system.process_frame(frame_rgb)
        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        # Create visualization
        vis_frame = overlay.render(
            frame_rgb,
            result.lanes,
            result.centerline,
            result.guidance,
        )
        
        # Add stats overlay
        vis_frame = add_stats_overlay(vis_frame, result, latency)
        
        # Convert back to BGR for OpenCV
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Save frame
        if writer:
            writer.write(vis_frame_bgr)
        
        # Display
        cv2.imshow('Lane Detection - Live', vis_frame_bgr)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            system.reset()
            print("Tracker reset")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def run_inference_image(
    system: LaneKeepingSystem,
    overlay: LaneOverlay,
    image_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run inference on single image.
    
    Args:
        system: Lane keeping system
        overlay: Visualization overlay
        image_path: Path to input image
        output_path: Optional path to save output image
    
    Returns:
        Detection result dictionary
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    start_time = time.perf_counter()
    result = system.process_frame(image_rgb)
    latency = (time.perf_counter() - start_time) * 1000  # ms
    
    print(f"Inference time: {latency:.2f} ms")
    print(f"Detected {len(result.lanes)} lanes")
    
    if result.guidance:
        print(f"Lateral offset: {result.guidance.lateral_offset_m:.3f} m")
        print(f"Heading error: {result.guidance.heading_error_rad * 180 / np.pi:.2f} deg")
    
    # Create visualization
    vis_frame = overlay.render(
        image_rgb,
        result.lanes,
        result.centerline,
        result.guidance,
    )
    
    # Save output
    if output_path:
        vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_frame_bgr)
        print(f"Saved output to {output_path}")
    
    return {
        'lanes': result.lanes,
        'centerline': result.centerline,
        'guidance': result.guidance,
        'latency_ms': latency,
    }


def add_stats_overlay(
    frame: np.ndarray,
    result,
    latency_ms: float,
) -> np.ndarray:
    """Add statistics overlay to frame."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Convert to PIL
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Stats text
    lines = [
        f"FPS: {1000.0 / latency_ms:.1f}",
        f"Latency: {latency_ms:.1f} ms",
        f"Lanes: {len(result.lanes)}",
    ]
    
    if result.guidance:
        lines.extend([
            f"Lat. Offset: {result.guidance.lateral_offset_m:.2f} m",
            f"Heading Err: {result.guidance.heading_error_rad * 180 / np.pi:.1f}°",
        ])
        
        if not result.guidance.is_valid:
            lines.append("⚠ INVALID")
    
    # Draw background
    y = 10
    for line in lines:
        bbox = draw.textbbox((10, y), line, font=font)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill=(0, 0, 0, 180),
        )
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += 20
    
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description='Lane detection inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Input source (image/video path or camera ID)')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--tensorrt', action='store_true', help='Use TensorRT engine')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create system
    system = LaneKeepingSystem(config)
    
    # Load weights
    if args.tensorrt:
        system.load_tensorrt_engine(args.weights)
    else:
        system.load_weights(args.weights, device=args.device)
    
    # Create overlay
    overlay = LaneOverlay()
    
    # Determine source type
    source = args.source
    
    if source.isdigit():
        # Camera
        run_inference_camera(
            system, overlay,
            camera_id=int(source),
            output_path=args.output,
        )
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video
        run_inference_video(
            system, overlay,
            video_path=source,
            output_path=args.output,
            show_display=not args.no_display,
        )
    else:
        # Image
        run_inference_image(
            system, overlay,
            image_path=source,
            output_path=args.output,
        )


if __name__ == '__main__':
    main()
