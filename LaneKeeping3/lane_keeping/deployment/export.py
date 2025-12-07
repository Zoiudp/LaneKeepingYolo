"""
Model Export utilities for deployment.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class ModelExporter:
    """
    Export lane detection models to various deployment formats.
    
    Supported formats:
    - ONNX
    - TensorRT (via ONNX)
    - TorchScript
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (640, 640),
        device: str = 'cuda',
    ):
        """
        Initialize exporter.
        
        Args:
            model: PyTorch model to export
            input_size: Model input size (W, H)
            device: Device for export
        """
        self.model = model
        self.input_size = input_size
        self.device = device
        
        # Move model to device and eval mode
        self.model.to(device)
        self.model.eval()
    
    def export_onnx(
        self,
        output_path: Union[str, Path],
        opset_version: int = 17,
        dynamic_axes: Optional[Dict] = None,
        simplify: bool = True,
    ) -> Path:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output file path
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            simplify: Whether to simplify ONNX model
            
        Returns:
            Path to exported model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, self.input_size[1], self.input_size[0],
            device=self.device,
        )
        
        # Default dynamic axes for batch size
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'keypoints': {0: 'batch_size'},
                'confidences': {0: 'batch_size'},
                'lane_types': {0: 'batch_size'},
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['keypoints', 'confidences', 'lane_types', 'polynomials'],
            dynamic_axes=dynamic_axes,
        )
        
        # Simplify ONNX model
        if simplify:
            self._simplify_onnx(output_path)
        
        # Verify export
        self._verify_onnx(output_path, dummy_input)
        
        print(f"ONNX model exported to: {output_path}")
        return output_path
    
    def _simplify_onnx(self, model_path: Path) -> None:
        """Simplify ONNX model using onnxsim."""
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(str(model_path))
            model_simp, check = simplify(model)
            
            if check:
                onnx.save(model_simp, str(model_path))
                print("ONNX model simplified successfully")
            else:
                print("Warning: ONNX simplification check failed")
                
        except ImportError:
            print("Warning: onnxsim not installed, skipping simplification")
    
    def _verify_onnx(
        self,
        model_path: Path,
        dummy_input: torch.Tensor,
    ) -> None:
        """Verify ONNX export by comparing outputs."""
        import onnxruntime as ort
        
        # PyTorch inference
        with torch.no_grad():
            torch_outputs = self.model(dummy_input)
        
        # ONNX inference
        ort_session = ort.InferenceSession(str(model_path))
        input_name = ort_session.get_inputs()[0].name
        ort_outputs = ort_session.run(None, {input_name: dummy_input.cpu().numpy()})
        
        # Compare outputs
        if isinstance(torch_outputs, dict):
            torch_keypoints = torch_outputs['keypoints'].cpu().numpy()
        else:
            torch_keypoints = torch_outputs[0].cpu().numpy()
        
        onnx_keypoints = ort_outputs[0]
        
        max_diff = np.abs(torch_keypoints - onnx_keypoints).max()
        print(f"ONNX verification - max output difference: {max_diff:.6f}")
        
        if max_diff > 1e-4:
            print("Warning: ONNX output differs from PyTorch by more than 1e-4")
    
    def export_tensorrt(
        self,
        output_path: Union[str, Path],
        fp16: bool = True,
        int8: bool = False,
        calibration_data: Optional[np.ndarray] = None,
        workspace_size: int = 4,
    ) -> Path:
        """
        Export model to TensorRT format.
        
        Args:
            output_path: Output file path
            fp16: Enable FP16 precision
            int8: Enable INT8 precision (requires calibration data)
            calibration_data: Calibration dataset for INT8
            workspace_size: TensorRT workspace size in GB
            
        Returns:
            Path to exported engine
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT not installed. Please install tensorrt package.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # First export to ONNX
        onnx_path = output_path.with_suffix('.onnx')
        self.export_onnx(onnx_path, simplify=True)
        
        # Build TensorRT engine
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX parse error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size * (1 << 30)
        )
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Enabled FP16 precision")
        
        if int8:
            if calibration_data is None:
                raise ValueError("INT8 requires calibration data")
            
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = Int8Calibrator(calibration_data, cache_file='calibration.cache')
            config.int8_calibrator = calibrator
            print("Enabled INT8 precision")
        
        # Build engine
        print("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"TensorRT engine exported to: {output_path}")
        return output_path
    
    def export_torchscript(
        self,
        output_path: Union[str, Path],
        method: str = 'trace',
    ) -> Path:
        """
        Export model to TorchScript format.
        
        Args:
            output_path: Output file path
            method: Export method ('trace' or 'script')
            
        Returns:
            Path to exported model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_input = torch.randn(
            1, 3, self.input_size[1], self.input_size[0],
            device=self.device,
        )
        
        if method == 'trace':
            traced = torch.jit.trace(self.model, dummy_input)
        else:
            traced = torch.jit.script(self.model)
        
        traced.save(str(output_path))
        
        print(f"TorchScript model exported to: {output_path}")
        return output_path
    
    def benchmark(
        self,
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            num_iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            
        Returns:
            Benchmark results
        """
        import time
        
        dummy_input = torch.randn(
            1, 3, self.input_size[1], self.input_size[0],
            device=self.device,
        )
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        timings = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            timings.append((time.perf_counter() - start) * 1000)
        
        return {
            'mean_ms': float(np.mean(timings)),
            'std_ms': float(np.std(timings)),
            'min_ms': float(np.min(timings)),
            'max_ms': float(np.max(timings)),
            'p50_ms': float(np.percentile(timings, 50)),
            'p95_ms': float(np.percentile(timings, 95)),
            'p99_ms': float(np.percentile(timings, 99)),
            'fps': 1000 / np.mean(timings),
        }


class Int8Calibrator:
    """INT8 calibrator for TensorRT."""
    
    def __init__(
        self,
        calibration_data: np.ndarray,
        cache_file: str = 'calibration.cache',
        batch_size: int = 1,
    ):
        """
        Initialize calibrator.
        
        Args:
            calibration_data: Calibration dataset [N, C, H, W]
            cache_file: Path to calibration cache
            batch_size: Batch size for calibration
        """
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data = calibration_data
        self.current_index = 0
        
        # Allocate device memory
        self.device_input = None
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None
        
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        # Copy to device
        import pycuda.driver as cuda
        
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch.nbytes)
        
        cuda.memcpy_htod(self.device_input, batch.astype(np.float32).ravel())
        
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
