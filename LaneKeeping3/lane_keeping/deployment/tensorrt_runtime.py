"""
TensorRT Runtime for optimized inference.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class TensorRTRunner:
    """
    TensorRT engine runner for optimized inference.
    """
    
    def __init__(
        self,
        engine_path: str,
    ):
        """
        Initialize TensorRT runner.
        
        Args:
            engine_path: Path to TensorRT engine file
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT/PyCUDA not installed")
        
        self.engine_path = Path(engine_path)
        
        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self._allocate_buffers()
    
    def _allocate_buffers(self) -> None:
        """Allocate input/output buffers."""
        import pycuda.driver as cuda
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            # Convert dtype
            if dtype == self.engine.get_tensor_dtype(name):
                np_dtype = np.float32
            else:
                np_dtype = np.float32
            
            # Calculate size
            size = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(size)
            
            # Create host buffer
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), np_dtype)
            
            self.bindings.append(int(device_mem))
            
            binding_info = {
                'name': name,
                'shape': shape,
                'dtype': np_dtype,
                'device': device_mem,
                'host': host_mem,
            }
            
            if self.engine.get_tensor_mode(name) == self.engine.get_tensor_mode(name):
                self.inputs.append(binding_info)
            else:
                self.outputs.append(binding_info)
    
    def infer(
        self,
        input_array: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on input array.
        
        Args:
            input_array: Input numpy array [B, C, H, W]
            
        Returns:
            Dictionary of output tensors
        """
        import pycuda.driver as cuda
        
        # Copy input to host buffer
        np.copyto(self.inputs[0]['host'], input_array.ravel())
        
        # Copy input to device
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream,
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        
        # Copy outputs from device
        outputs = {}
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output['host'],
                output['device'],
                self.stream,
            )
        
        # Synchronize
        self.stream.synchronize()
        
        # Format outputs
        for output in self.outputs:
            shape = output['shape']
            outputs[output['name']] = output['host'].reshape(shape)
        
        return outputs
    
    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark TensorRT engine performance.
        
        Args:
            input_shape: Input tensor shape
            num_iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            
        Returns:
            Benchmark results
        """
        import time
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            _ = self.infer(dummy_input)
        
        # Benchmark
        timings = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.infer(dummy_input)
            timings.append((time.perf_counter() - start) * 1000)
        
        return {
            'mean_ms': float(np.mean(timings)),
            'std_ms': float(np.std(timings)),
            'min_ms': float(np.min(timings)),
            'max_ms': float(np.max(timings)),
            'p50_ms': float(np.percentile(timings, 50)),
            'p95_ms': float(np.percentile(timings, 95)),
            'fps': 1000 / np.mean(timings),
        }
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
