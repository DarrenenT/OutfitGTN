import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict, Any
from torch_geometric.data import Data
from tqdm import tqdm
import logging
from utils.gpu_utils import get_device
from torch.cuda.amp import autocast

class OutfitTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        margin: float = 1.2,
        weight_decay: float = 1e-5,
        clip_gradients: bool = False,
        max_norm: float = 1.0,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The OutfitGAT model
            optimizer: Optional optimizer (defaults to Adam)
            lr: Learning rate if optimizer not provided
            device: Device to run on
            margin: Margin for triplet loss
            weight_decay: L2 regularization
            clip_gradients: Whether to clip gradients
            max_norm: Maximum norm for gradient clipping
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer or optim.Adam(
            model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        self.margin = margin
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'avg_pos_sim': [],
            'avg_neg_sim': [],
            'embedding_norm': []
        }

        self.accumulation_counter = 0
        self.clip_gradients = clip_gradients
        self.max_norm = max_norm
        
    def train_step(self, batch_tuple: tuple, accumulation_steps: int = 4, scaler=None) -> Dict[str, float]:
        """
        Perform one training step with separate local graphs.
        
        Args:
            batch_tuple: Tuple of (query_data, pos_data_list, neg_data_list)
                query_data: Graph Data object for query node
                pos_data_list: List of 1 to max_pos_samples Graph Data objects
                neg_data_list: List of 1 to max_neg_samples Graph Data objects
            accumulation_steps: Number of steps to accumulate gradients before updating weights
            scaler: Optional GradScaler for mixed precision training
        """
        self.model.train()
        # Only zero gradients on first step
        if self.accumulation_counter % accumulation_steps == 0:
            self.optimizer.zero_grad()
        
        # Move batch to device
        query_data = batch_tuple[0].to(self.device)
        pos_data_list = [data.to(self.device) for data in batch_tuple[1]]  # Length ≤ max_pos_samples
        neg_data_list = [data.to(self.device) for data in batch_tuple[2]]  # Length ≤ max_neg_samples
        
        # Use autocast for forward pass
        with autocast(enabled=scaler is not None):
            # Get embeddings from separate local graphs
            query_emb, pos_emb, neg_emb = self.model((query_data, pos_data_list, neg_data_list))
        
            # Calculate loss
            loss = self.compute_triplet_loss(query_emb, pos_emb, neg_emb)
        
            # Scale for gradient accumulation
            scaled_loss = loss / accumulation_steps

        # Use scaler for backward pass if provided
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
            if self.accumulation_counter % accumulation_steps == 0:
                scaler.unscale_(self.optimizer)
                if self.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                scaler.step(self.optimizer)
                scaler.update()
        
        else:
            scaled_loss.backward()
            if self.accumulation_counter % accumulation_steps == 0:
                if self.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()
        
        # Update counter after backward is done
        self.accumulation_counter += 1
        
        # Calculate metrics - reuse tensor reshaping operations
        with torch.no_grad():
            batch_size = query_emb.size(0)
            actual_pos = pos_emb.size(0) // batch_size
            actual_neg = neg_emb.size(0) // batch_size
            
            # Reshape for similarity computation - use same format as loss function
            query_expanded = query_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            pos_emb_reshaped = pos_emb.view(batch_size, actual_pos, -1)  # [batch_size, actual_pos, embedding_dim]
            neg_emb_reshaped = neg_emb.view(batch_size, actual_neg, -1)  # [batch_size, actual_neg, embedding_dim]
            
            # Reuse same computation pattern as in loss function
            pos_sim = F.cosine_similarity(query_expanded, pos_emb_reshaped, dim=-1)
            neg_sim = F.cosine_similarity(query_expanded, neg_emb_reshaped, dim=-1)
            
            metrics = {
                'loss': loss.item(),
                'avg_pos_sim': pos_sim.mean().item(),
                'avg_neg_sim': neg_sim.mean().item(),
                'pos_sim_std': pos_sim.std().item(),
                'neg_sim_std': neg_sim.std().item(),
                'embedding_norm': query_emb.norm(dim=1).mean().item(),
                'num_pos': actual_pos,  # Track actual numbers
                'num_neg': actual_neg
            }
            
            # Update metrics history
            for k, v in metrics.items():
                if k in self.metrics:
                    self.metrics[k].append(v)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate the model on validation set with separate local graphs."""
        self.model.eval()
        total_loss = 0
        total_pos_sim = 0
        total_neg_sim = 0
        num_batches = 0
        
        for batch_tuple in val_loader:
            # Move batch to device
            query_data = batch_tuple[0].to(self.device)
            pos_data_list = [data.to(self.device) for data in batch_tuple[1]]
            neg_data_list = [data.to(self.device) for data in batch_tuple[2]]
            
            # Use autocast for evaluation too
            with autocast(enabled=True):
                # Get embeddings
                query_emb, pos_emb, neg_emb = self.model((query_data, pos_data_list, neg_data_list))
                
                # Calculate loss and similarities
                loss = self.compute_triplet_loss(query_emb, pos_emb, neg_emb)
            
            # These can be outside autocast
            pos_sim = F.cosine_similarity(query_emb.unsqueeze(1), pos_emb, dim=-1)
            neg_sim = F.cosine_similarity(query_emb.unsqueeze(1), neg_emb, dim=-1)
            
            total_loss += loss.item()
            total_pos_sim += pos_sim.mean().item()
            total_neg_sim += neg_sim.mean().item()
            num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_pos_sim': total_pos_sim / num_batches,
            'val_neg_sim': total_neg_sim / num_batches
        }
        
        return metrics
    
    def compute_triplet_loss(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced triplet loss with better temperature scaling and optimized reshaping
        
        Args:
            query: Query embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size * actual_pos_samples, embedding_dim]
            negative: Negative embeddings [batch_size * actual_neg_samples, embedding_dim]
        """
        batch_size = query.size(0)
        actual_pos = positive.size(0) // batch_size  # Might be less than max_pos_samples
        actual_neg = negative.size(0) // batch_size  # Might be less than max_neg_samples
        
        if actual_pos == 0 or actual_neg == 0:
            raise ValueError("Each query must have at least one positive and one negative sample")
        
        # Validate shapes
        assert query.dim() == 2, f"Expected query shape [batch_size, embedding_dim], got {query.shape}"
        assert positive.dim() == 2, f"Expected positive shape [batch_size * num_pos, embedding_dim], got {positive.shape}"
        assert negative.dim() == 2, f"Expected negative shape [batch_size * num_neg, embedding_dim], got {negative.shape}"
        
        # Reshape tensors for comparison - do this once
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        positive_reshaped = positive.view(batch_size, actual_pos, -1)  # [batch_size, actual_pos, embedding_dim]
        negative_reshaped = negative.view(batch_size, actual_neg, -1)  # [batch_size, actual_neg, embedding_dim]
        
        # Compute similarities (using model's temperature)
        pos_sim = F.cosine_similarity(query_expanded, positive_reshaped, dim=-1)  # [batch_size, actual_pos]
        neg_sim = F.cosine_similarity(query_expanded, negative_reshaped, dim=-1)  # [batch_size, actual_neg]
        
        # Use 10-15% of total negatives as top-k 
        k = max(min(int(actual_neg * 0.15), 8), 3)  # Between 3 and 8, targeting ~15% of negatives
        topk_neg_sim, _ = torch.topk(neg_sim, k, dim=1)
        avg_hard_neg_sim = topk_neg_sim.mean(dim=1)  # [batch_size]
        
        # Use this average in the loss calculation
        avg_hard_neg_expanded = avg_hard_neg_sim.unsqueeze(1)
        triplet_loss = F.relu(self.margin - pos_sim + avg_hard_neg_expanded)
        
        # Stronger penalties
        pos_penalty = F.relu(0.55 - pos_sim) # increased from 0.5
        neg_penalty = F.relu(avg_hard_neg_sim + 0.1)
        
        # Weighted loss components - use efficient mean calculation
        loss = (
            triplet_loss.mean() + 
            0.4 * pos_penalty.mean() + # increased from 0.4
            0.3 * neg_penalty.mean()
        )
        
        return loss
    
    def save_checkpoint(self, path: str, extra_info: Optional[Dict[str, Any]] = None):
        """Save model checkpoint with optional extra information."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        if extra_info:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint and return any extra information."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Remove standard keys to get extra info
        standard_keys = {'model_state_dict', 'optimizer_state_dict', 'best_val_loss'}
        extra_info = {k: v for k, v in checkpoint.items() if k not in standard_keys}
        
        self.logger.info(f"Loaded checkpoint from {path}")
        return extra_info