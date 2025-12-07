"""
Test configuration for pytest.
"""

import pytest
import numpy as np
import torch


@pytest.fixture(scope="session")
def device():
    """Get torch device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    return 42


@pytest.fixture
def sample_frame():
    """Create sample RGB frame for testing."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_small():
    """Create small sample frame for faster testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor():
    """Create sample tensor for model testing."""
    return torch.randn(1, 3, 640, 640)


@pytest.fixture
def base_config():
    """Create base configuration for testing."""
    return {
        'model': {
            'backbone_variant': 'n',
            'input_size': [320, 320],
            'num_keypoints': 36,
            'num_lanes': 4,
            'use_polynomial': True,
            'confidence_threshold': 0.5,
        },
        'tracker': {
            'max_age': 5,
            'min_hits': 2,
            'iou_threshold': 0.3,
        },
        'steering': {
            'lookahead_distance': 15.0,
            'max_lateral_offset': 2.0,
            'max_heading_error': 0.5,
        },
        'camera': {
            'width': 640,
            'height': 480,
            'fx': 500.0,
            'fy': 500.0,
            'cx': 320.0,
            'cy': 240.0,
            'camera_height': 1.5,
        },
        'loss': {
            'keypoint_weight': 1.0,
            'polynomial_weight': 1.0,
            'classification_weight': 0.5,
            'confidence_weight': 1.0,
        },
    }


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
