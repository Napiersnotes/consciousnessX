"""
Demo: Complete Training Pipeline

Demonstrates the full training pipeline with checkpointing, scheduling, and evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.training.trainer import Trainer
from src.training.checkpoint import CheckpointManager
from src.training.scheduler import CosineAnnealingLR, ReduceLROnPlateau
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.performance_tracker import PerformanceTracker
from src.utils.config_manager import ConfigManager
from src.utils.logging_utils import setup_logger, TrainingLogger
from src.utils.data_loader import DataLoader as CustomDataLoader


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class MockDataset:
    """Mock dataset for demonstration."""
    
    def __init__(self, num_samples=1000, input_dim=784, num_classes=10):
        self.X = np.random.randn(num_samples, input_dim).astype(np.float32)
        self.y = np.random.randint(0, num_classes, num_samples)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    """Run training pipeline demonstration."""
    
    # Setup logging
    logger = setup_logger('training_pipeline_demo')
    training_logger = TrainingLogger(log_dir='logs', log_to_file=True)
    logger.info("Starting Training Pipeline Demo")
    
    # Load configuration
    logger.info("Loading configuration...")
    config = ConfigManager()
    config.update(
        neural={
            'num_neurons': 1000,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        },
        training={
            'max_steps': 10000,
            'save_interval': 1000,
            'eval_interval': 500
        }
    )
    config.validate()
    logger.info(f"Configuration loaded: {config}")
    
    # Create model
    logger.info("Creating model...")
    model = SimpleModel(
        input_dim=784,
        hidden_dim=config.get('neural.hidden_dim', 256),
        output_dim=10
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('neural.learning_rate', 0.001)
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.get('training.max_steps', 10000),
        eta_min=0.00001
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    train_data = MockDataset(num_samples=1000)
    val_data = MockDataset(num_samples=200)
    
    # Create PyTorch data loaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_data.X), torch.tensor(train_data.y)),
        batch_size=config.get('neural.batch_size', 32),
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_data.X), torch.tensor(val_data.y)),
        batch_size=32,
        shuffle=False
    )
    
    # Create checkpoint manager
    logger.info("Setting up checkpoint manager...")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir='checkpoints',
        save_best=True,
        save_interval=2
    )
    
    # Create performance tracker
    tracker = PerformanceTracker(window_size=100, store_history=True)
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Training loop
    logger.info("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.get('neural.epochs', 10)):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        tracker.start_timer('epoch')
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            batch_total = batch_y.size(0)
            batch_correct = (predicted == batch_y).sum().item()
            
            # Track metrics
            epoch_loss += loss.item() * batch_total
            epoch_correct += batch_correct
            epoch_total += batch_total
            
            global_step += 1
            
            # Log training step
            if global_step % 10 == 0:
                tracker.update({
                    'loss': loss.item(),
                    'accuracy': batch_correct / batch_total,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
                
                training_logger.log_step(
                    step=global_step,
                    loss=loss.item(),
                    metric=batch_correct / batch_total,
                    lr=optimizer.param_groups[0]['lr']
                )
        
        # Calculate epoch metrics
        epoch_loss = epoch_loss / epoch_total
        epoch_accuracy = epoch_correct / epoch_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        tracker.start_timer('validation')
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_y.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total
        
        tracker.stop_timer('validation')
        epoch_time = tracker.stop_timer('epoch')
        
        # Update tracker with epoch metrics
        tracker.update({
            'epoch_loss': epoch_loss,
            'epoch_accuracy': epoch_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        # Log epoch
        training_logger.log_epoch(epoch, epoch_loss, val_loss, epoch_time)
        
        logger.info(
            f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, "
            f"Train Acc = {epoch_accuracy:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Acc = {val_accuracy:.4f}"
        )
        
        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'train_accuracy': epoch_accuracy,
            'val_accuracy': val_accuracy,
            'global_step': global_step
        }
        
        checkpoint_manager.save_checkpoint(
            checkpoint_data,
            metric=val_loss,
            is_best=(val_loss < best_val_loss)
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Final evaluation
    logger.info("\n=== Final Evaluation ===")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final training accuracy: {tracker.get_average('epoch_accuracy'):.4f}")
    logger.info(f"Final validation accuracy: {tracker.get_average('val_accuracy'):.4f}")
    
    # Load best checkpoint
    best_checkpoint = checkpoint_manager.load_best_checkpoint()
    logger.info(f"Best checkpoint loaded from epoch {best_checkpoint['epoch']}")
    
    # Get training summary
    summary = tracker.get_summary()
    logger.info(f"\n=== Training Summary ===")
    logger.info(f"Total steps: {summary['step_count']}")
    logger.info(f"Average epoch time: {tracker.get_average_duration('epoch'):.2f}s")
    logger.info(f"Average validation time: {tracker.get_average_duration('validation'):.2f}s")
    
    # Log summary
    training_logger.log_metrics({
        'total_steps': summary['step_count'],
        'best_val_loss': best_val_loss,
        'final_train_acc': summary['best'].get('epoch_accuracy', 0),
        'final_val_acc': summary['best'].get('val_accuracy', 0)
    })
    
    training_logger.log_summary()
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()