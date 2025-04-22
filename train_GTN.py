import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import torch
import yaml
import logging
from pathlib import Path
from datetime import datetime
from models.OutfitGTN import OutfitGTN
from data.data_loader import FashionDataLoader
from trainers.trainer import OutfitTrainer
from utils.gpu_utils import configure_gpu, get_device
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import shutil
import signal
import sys
from torch.cuda.amp import GradScaler, autocast

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TF warnings

# Global variables for checkpoint saving
latest_checkpoint_data = None
checkpoint_path = None

def save_checkpoint_on_exit(sig, frame):
    """Save checkpoint when receiving a termination signal"""
    if latest_checkpoint_data is not None and checkpoint_path is not None:
        print(f"\nReceived signal {sig}. Saving checkpoint before exiting...")
        torch.save(latest_checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, save_checkpoint_on_exit)   # Ctrl+C
signal.signal(signal.SIGTERM, save_checkpoint_on_exit)  # kill command

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    # Test file writing
    try:
        with open(log_file, 'w') as f:
            f.write("Initializing log file\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    # Force immediate flushing
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    
    return logger

def main():
    # Load config
    with open('config/config_GTN.yaml', 'r') as f:
        config = yaml.safe_load(f)

    accumulation_steps = 4 # Number of steps to accumulate gradients before updating weights
    
    # Clear old logs
    log_dir = Path(config['training']['log_dir'])
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Configure GPU
    configure_gpu()
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize model with all parameters from config
    model = OutfitGTN(
        input_dim=config['model']['input_dim'],
        hidden_channels=config['model']['hidden_channels'],
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        residual=config['model']['residual'],
        temperature=config['model']['temperature'],
        enable_drop=config['model']['enable_drop'],
        drop_rate=config['model']['drop_rate']
    )
    logger.info(f"Initialized OutfitGTN model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer with weight decay
    trainer = OutfitTrainer(
        model=model,
        lr=config['training']['learning_rate'],
        device=device,
        margin=config['training']['margin'],
        weight_decay=config['training']['weight_decay'],
        clip_gradients=config['training']['gradient_clipping']['enabled'],
        max_norm=config['training']['gradient_clipping']['max_norm']
    )

    # Initialize learning rate scheduler
    scheduler = None
    if config['training'].get('scheduler', {}).get('type') == 'cosine_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            trainer.optimizer,
            T_0=config['training']['scheduler'].get('T_0', 10),
            T_mult=config['training']['scheduler'].get('T_mult', 2),
            eta_min=config['training']['scheduler'].get('eta_min', 1e-6)
        )
        logger.info(f"Initialized CosineAnnealingWarmRestarts scheduler")
    
    # Initialize data loader with auto_download
    data_loader = FashionDataLoader(
        batch_size=config['training']['batch_size'],
        max_pos_samples=config['training']['max_pos_samples'],
        max_neg_samples=config['training']['max_neg_samples'],
        auto_download=config['data'].get('auto_download', True),  # Default to True
        force_update=config['data'].get('force_update', False)  # Default to False
    ).load_data(
        data_dir=config['data']['data_dir']
    )
    
    # Get train and validation loaders
    train_loader, val_loader = data_loader.get_train_val_dataloaders(
        val_ratio=0.2,  # 80% train, 20% validation
        shuffle=True
    )
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Create tensorboard writer with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/run_GTN_{timestamp}")
    
    # Check if we should resume from checkpoint
    start_epoch = 0
    checkpoint_path = f"{config['training']['checkpoint_dir']}/best_model_GTN.pt"
    resume_path = f"{config['training']['checkpoint_dir']}/latest_checkpoint_GTN.pt"

    # Only resume if enabled in config and checkpoint exists
    if config['training'].get('resume_training', True) and os.path.exists(resume_path):
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load scheduler state if it exists
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Resume from correct epoch
        start_epoch = checkpoint['epoch'] + 1
        # Restore best validation loss
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        logger.info(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.4f}")
    else:
        if not config['training'].get('resume_training', True):
            logger.info("Resume training disabled in config. Starting fresh.")
        elif not os.path.exists(resume_path):
            logger.info("No checkpoint found. Starting training from scratch.")
        
        best_val_loss = float('inf')
        patience_counter = 0
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Training
        train_metrics = {'loss': 0.0, 'avg_pos_sim': 0.0, 'avg_neg_sim': 0.0}
        num_batches = 0
        
        for batch in train_loader:
            # Pass scaler to trainer
            batch_metrics = trainer.train_step(batch, accumulation_steps=accumulation_steps, scaler=scaler)
            if isinstance(batch_metrics, dict):
                for k in train_metrics:
                    if k in batch_metrics:
                        train_metrics[k] += batch_metrics[k]
            else:
                train_metrics['loss'] += batch_metrics
            num_batches += 1
        
        # Add epoch boundary handling right here (after batch loop)
        if trainer.accumulation_counter % accumulation_steps != 0:
            # Force optimizer step for leftover gradients at epoch boundary
            if scaler is not None:
                scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
                scaler.step(trainer.optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
                trainer.optimizer.step()
            trainer.optimizer.zero_grad()

        # Average metrics
        for k in train_metrics:
            train_metrics[k] /= max(num_batches, 1)
        
        # Validation
        val_metrics = trainer.evaluate(val_loader)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss = {train_metrics['loss']:.4f}, "
            f"Val Loss = {val_metrics['val_loss']:.4f}, "
            f"Pos Sim = {train_metrics['avg_pos_sim']:.4f}, "
            f"Neg Sim = {train_metrics['avg_neg_sim']:.4f}"
        )
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Similarity/positive', train_metrics['avg_pos_sim'], epoch)
        writer.add_scalar('Similarity/negative', train_metrics['avg_neg_sim'], epoch)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Learning/rate', current_lr, epoch)
            logger.info(f"Learning rate: {current_lr:.8f}")

        # Save checkpoint if best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            extra_info={
                'epoch': epoch,
                'metrics': train_metrics
            }
            if scheduler is not None:
                extra_info['scheduler_state_dict'] = scheduler.state_dict()

            trainer.save_checkpoint(
                path=f"{config['training']['checkpoint_dir']}/best_model_GTN.pt",
                extra_info=extra_info
            )
        else:
            patience_counter += 1
        
        # Save latest checkpoint for resuming
        latest_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter
        }
        if scheduler is not None:
            latest_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(latest_checkpoint, resume_path)
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # At the end of training loop (after the loop completes)
    if os.path.exists(resume_path):
        # Training completed normally, we can clean up the latest checkpoint
        logger.info(f"Training completed successfully. Removing temporary checkpoint.")
        try:
            os.remove(resume_path)
            logger.info(f"Removed temporary checkpoint: {resume_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary checkpoint: {e}")

    # Keep the final best model
    logger.info(f"Final best model saved at: {config['training']['checkpoint_dir']}/best_model_GTN.pt")
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 