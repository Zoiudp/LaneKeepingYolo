# Lane Keeping Docker Image
# Multi-stage build for production deployment

# Stage 1: Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /app/wheels

# Install Python packages
RUN pip install --no-cache-dir /app/wheels/* && \
    rm -rf /app/wheels

# Copy application code
COPY lane_keeping/ ./lane_keeping/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY pyproject.toml .
COPY README.md .

# Install application
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port for any web interfaces
EXPOSE 8080

# Default command
CMD ["python", "-m", "lane_keeping.cli", "--help"]

# Stage 3: TensorRT runtime (for Jetson deployment)
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime as tensorrt

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (minimal for inference)
COPY requirements.txt .

# Install minimal dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python-headless \
    pyyaml \
    pycuda \
    tensorrt

# Copy application
COPY lane_keeping/ ./lane_keeping/
COPY configs/ ./configs/

# Create models directory for mounting
RUN mkdir -p /app/models

# Environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/lane_model.engine

# Default command for inference
CMD ["python3", "-c", "from lane_keeping.deployment.tensorrt_runtime import TensorRTRunner; print('TensorRT runtime ready')"]
