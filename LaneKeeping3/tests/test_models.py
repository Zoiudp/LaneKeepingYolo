"""
Unit tests for lane detection model components.
"""

import pytest
import torch
import numpy as np

from lane_keeping.models.lane_yolo import LaneYOLO
from lane_keeping.models.backbone import LaneBackbone
from lane_keeping.models.heads import (
    LaneRegressionHead,
    LaneSegmentationHead,
    LaneClassificationHead,
)
from lane_keeping.models.losses import (
    PolynomialRegressionLoss,
    LaneSegmentationLoss,
    LaneConfidenceLoss,
    LaneLoss,
)


class TestLaneBackbone:
    """Tests for LaneBackbone."""
    
    @pytest.fixture
    def backbone_nano(self):
        """Create nano backbone."""
        return LaneBackbone(variant='n')
    
    @pytest.fixture
    def backbone_small(self):
        """Create small backbone."""
        return LaneBackbone(variant='s')
    
    def test_backbone_output_shapes(self, backbone_nano):
        """Test backbone output feature map shapes."""
        x = torch.randn(2, 3, 640, 640)
        features = backbone_nano(x)
        
        # Should return multi-scale features
        assert isinstance(features, (list, tuple))
        assert len(features) >= 3
        
        # Check spatial dimensions decrease
        prev_size = x.shape[-1]
        for feat in features:
            assert feat.shape[-1] <= prev_size
            prev_size = feat.shape[-1]
    
    def test_backbone_variants(self):
        """Test different backbone variants."""
        variants = ['n', 's', 'm']
        
        for variant in variants:
            backbone = LaneBackbone(variant=variant)
            x = torch.randn(1, 3, 320, 320)
            features = backbone(x)
            
            assert len(features) > 0
    
    def test_backbone_channels(self, backbone_nano):
        """Test output channel counts."""
        x = torch.randn(1, 3, 640, 640)
        features = backbone_nano(x)
        
        # All features should have batch dimension of 1
        for feat in features:
            assert feat.shape[0] == 1
            assert feat.dim() == 4  # BCHW format


class TestLaneRegressionHead:
    """Tests for LaneRegressionHead."""
    
    @pytest.fixture
    def head(self):
        """Create regression head."""
        return LaneRegressionHead(
            in_channels=256,
            num_lanes=4,
            num_keypoints=72,
            num_poly_coeffs=4,
        )
    
    def test_output_shape(self, head):
        """Test output shape."""
        x = torch.randn(2, 256, 20, 20)
        output = head(x)
        
        assert 'keypoints' in output
        assert 'confidences' in output
        
        # Keypoints: [B, num_lanes, num_keypoints, 2]
        assert output['keypoints'].shape == (2, 4, 72, 2)
        
        # Confidences: [B, num_lanes]
        assert output['confidences'].shape == (2, 4)
    
    def test_confidence_range(self, head):
        """Test confidence values are in [0, 1]."""
        x = torch.randn(2, 256, 20, 20)
        output = head(x)
        
        confs = output['confidences']
        assert torch.all(confs >= 0.0)
        assert torch.all(confs <= 1.0)


class TestLaneSegmentationHead:
    """Tests for LaneSegmentationHead."""
    
    @pytest.fixture
    def head(self):
        """Create segmentation head."""
        return LaneSegmentationHead(
            in_channels=256,
            num_classes=5,  # 4 lanes + background
        )
    
    def test_output_shape(self, head):
        """Test output shape."""
        x = torch.randn(2, 256, 80, 80)
        output = head(x)
        
        # Should be [B, num_classes, H, W]
        assert output.shape[0] == 2
        assert output.shape[1] == 5
        assert output.shape[2] == 80 or output.shape[2] > 80  # May upsample
    
    def test_softmax_output(self, head):
        """Test softmax output sums to 1."""
        x = torch.randn(2, 256, 40, 40)
        logits = head(x)
        probs = torch.softmax(logits, dim=1)
        
        # Sum across classes should be 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestLaneClassificationHead:
    """Tests for LaneClassificationHead."""
    
    @pytest.fixture
    def head(self):
        """Create classification head."""
        return LaneClassificationHead(
            in_channels=256,
            num_lanes=4,
            num_types=7,  # Lane types
        )
    
    def test_output_shape(self, head):
        """Test output shape."""
        x = torch.randn(2, 256, 20, 20)
        output = head(x)
        
        # Should be [B, num_lanes, num_types]
        assert output.shape == (2, 4, 7)


class TestLaneYOLO:
    """Tests for full LaneYOLO model."""
    
    @pytest.fixture
    def model(self):
        """Create full model."""
        return LaneYOLO(
            num_keypoints=72,
            num_lanes=4,
            use_polynomial=True,
            backbone_variant='n',
        )
    
    def test_forward_pass(self, model):
        """Test forward pass."""
        x = torch.randn(2, 3, 640, 640)
        output = model(x)
        
        assert 'keypoints' in output
        assert 'confidences' in output
    
    def test_output_shapes(self, model):
        """Test all output shapes."""
        x = torch.randn(1, 3, 640, 640)
        output = model(x)
        
        # Keypoints
        assert output['keypoints'].shape == (1, 4, 72, 2)
        
        # Confidences
        assert output['confidences'].shape == (1, 4)
    
    def test_eval_mode(self, model):
        """Test model in evaluation mode."""
        model.eval()
        x = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = model(x)
        
        assert output['keypoints'].shape[0] == 1
    
    def test_batch_independence(self, model):
        """Test that batch samples are independent."""
        model.eval()
        
        x1 = torch.randn(1, 3, 640, 640)
        x2 = torch.randn(1, 3, 640, 640)
        x_batch = torch.cat([x1, x2], dim=0)
        
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
            out_batch = model(x_batch)
        
        # Outputs should match when processed individually vs batch
        assert torch.allclose(out1['keypoints'], out_batch['keypoints'][:1], atol=1e-5)
        assert torch.allclose(out2['keypoints'], out_batch['keypoints'][1:], atol=1e-5)


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_polynomial_regression_loss(self):
        """Test polynomial regression loss."""
        loss_fn = PolynomialRegressionLoss()
        
        pred = torch.randn(2, 4, 4)  # [B, num_lanes, num_coeffs]
        target = torch.randn(2, 4, 4)
        mask = torch.ones(2, 4)  # All lanes valid
        
        loss = loss_fn(pred, target, mask)
        
        assert loss.dim() == 0  # Scalar
        assert loss >= 0
    
    def test_segmentation_loss(self):
        """Test segmentation loss."""
        loss_fn = LaneSegmentationLoss()
        
        pred = torch.randn(2, 5, 160, 160)  # [B, C, H, W]
        target = torch.randint(0, 5, (2, 160, 160))  # [B, H, W]
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_confidence_loss(self):
        """Test confidence loss."""
        loss_fn = LaneConfidenceLoss()
        
        pred = torch.sigmoid(torch.randn(2, 4))  # [B, num_lanes]
        target = torch.randint(0, 2, (2, 4)).float()
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0
        assert loss >= 0
    
    def test_combined_loss(self):
        """Test combined loss function."""
        loss_fn = LaneLoss(
            keypoint_weight=1.0,
            polynomial_weight=1.0,
            classification_weight=0.5,
            confidence_weight=1.0,
        )
        
        # Mock outputs and targets
        outputs = {
            'keypoints': torch.randn(2, 4, 72, 2),
            'confidences': torch.sigmoid(torch.randn(2, 4)),
            'polynomials': torch.randn(2, 4, 4),
        }
        
        targets = {
            'keypoints': torch.randn(2, 4, 72, 2),
            'lane_exists': torch.randint(0, 2, (2, 4)).float(),
            'polynomials': torch.randn(2, 4, 4),
        }
        
        loss, loss_dict = loss_fn(outputs, targets)
        
        assert loss.dim() == 0
        assert loss >= 0
        assert isinstance(loss_dict, dict)


class TestModelGradients:
    """Tests for gradient flow."""
    
    def test_gradient_flow(self):
        """Test that gradients flow through model."""
        model = LaneYOLO(
            num_keypoints=72,
            num_lanes=4,
            use_polynomial=True,
            backbone_variant='n',
        )
        
        x = torch.randn(1, 3, 320, 320, requires_grad=True)
        output = model(x)
        
        # Compute dummy loss
        loss = output['keypoints'].sum() + output['confidences'].sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelExport:
    """Tests for model export compatibility."""
    
    def test_torchscript_trace(self):
        """Test TorchScript tracing."""
        model = LaneYOLO(
            num_keypoints=72,
            num_lanes=4,
            use_polynomial=True,
            backbone_variant='n',
        )
        model.eval()
        
        x = torch.randn(1, 3, 320, 320)
        
        # Should be traceable
        with torch.no_grad():
            traced = torch.jit.trace(model, x)
        
        # Verify traced model works
        out_orig = model(x)
        out_traced = traced(x)
        
        assert torch.allclose(
            out_orig['keypoints'],
            out_traced['keypoints'],
            atol=1e-5
        )
