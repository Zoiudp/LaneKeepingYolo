"""
Training script for lane detection model.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lane_keeping.models.lane_yolo import LaneYOLO
from lane_keeping.models.losses import LaneLoss
from lane_keeping.data.datasets import TuSimpleDataset, CULaneDataset
from lane_keeping.data.dataloader import create_dataloader
from lane_keeping.evaluation.metrics import LaneEvaluator


def train(config: Dict) -> None:
    """
    Train lane detection model.
    
    Args:
        config: Training configuration dictionary
    """
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create model
    model = LaneYOLO(
        num_keypoints=config['model']['num_keypoints'],
        num_lanes=config['model']['num_lanes'],
        use_polynomial=config['model']['use_polynomial'],
        backbone_variant=config['model']['backbone_variant'],
    )
    model = model.to(device)
    
    # Load pretrained weights if specified
    if config.get('pretrained_weights'):
        checkpoint = torch.load(config['pretrained_weights'], map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded pretrained weights from {config['pretrained_weights']}")
    
    # Create datasets
    train_dataset = create_dataset(config['data']['train_dir'], 'train', config)
    val_dataset = create_dataset(config['data']['val_dir'], 'val', config)
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
    )
    
    # Create loss function
    criterion = LaneLoss(
        keypoint_weight=config['loss']['keypoint_weight'],
        polynomial_weight=config['loss']['polynomial_weight'],
        classification_weight=config['loss']['classification_weight'],
        confidence_weight=config['loss']['confidence_weight'],
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config['training'])
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config['training'], len(train_loader))
    
    # Create evaluator
    evaluator = LaneEvaluator()
    
    # Create tensorboard writer
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    best_f1 = 0.0
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, evaluator, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
        writer.add_scalar('Metrics/f1', val_metrics['f1'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'config': config,
            'metrics': val_metrics,
        }
        
        torch.save(checkpoint, output_dir / 'last.pt')
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"New best F1: {best_f1:.4f}")
    
    writer.close()
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != 'image'}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    evaluator: LaneEvaluator,
    device: torch.device,
) -> Dict:
    """Validate model."""
    model.eval()
    evaluator.reset()
    
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != 'image'}
            
            outputs = model(images)
            
            # Compute loss
            loss, _ = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Update evaluator
            batch_size = images.shape[0]
            for i in range(batch_size):
                pred_kpts = outputs['keypoints'][i].cpu().numpy()
                gt_kpts = targets['keypoints'][i].cpu().numpy()
                pred_conf = outputs['confidences'][i].cpu().numpy()
                gt_exists = targets['lane_exists'][i].cpu().numpy()
                
                # Convert to lane format
                pred_lanes = [pred_kpts[j] for j in range(len(pred_conf)) if pred_conf[j] > 0.5]
                gt_lanes = [gt_kpts[j] for j in range(len(gt_exists)) if gt_exists[j] > 0.5]
                
                evaluator.update(pred_lanes, gt_lanes)
    
    metrics = evaluator.compute()
    
    return {
        'loss': total_loss / len(dataloader),
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1': metrics.f1_score,
    }


def create_dataset(data_dir: str, split: str, config: Dict):
    """Create dataset based on configuration."""
    from lane_keeping.processing.augmentation import LaneAugmentation
    
    transform = LaneAugmentation(
        config=config.get('augmentation', {}),
        mode=split,
    )
    
    dataset_type = config['data'].get('dataset_type', 'tusimple')
    
    if dataset_type == 'tusimple':
        return TuSimpleDataset(
            root_dir=data_dir,
            split=split,
            transform=transform,
            num_keypoints=config['model']['num_keypoints'],
            num_lanes=config['model']['num_lanes'],
            image_size=tuple(config['model']['input_size']),
        )
    elif dataset_type == 'culane':
        return CULaneDataset(
            root_dir=data_dir,
            split=split,
            transform=transform,
            num_keypoints=config['model']['num_keypoints'],
            num_lanes=config['model']['num_lanes'],
            image_size=tuple(config['model']['input_size']),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer."""
    optimizer_type = config.get('optimizer', 'adamw')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer, config: Dict, steps_per_epoch: int):
    """Create learning rate scheduler."""
    scheduler_type = config.get('scheduler', 'cosine')
    epochs = config.get('epochs', 100)
    
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * steps_per_epoch,
            eta_min=config.get('min_lr', 1e-6),
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30) * steps_per_epoch,
            gamma=config.get('gamma', 0.1),
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def main():
    parser = argparse.ArgumentParser(description='Train lane detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='outputs/train', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line args
    config['output_dir'] = args.output
    config['device'] = args.device
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
