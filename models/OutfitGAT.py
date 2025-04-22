import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import List, Optional

class OutfitGAT(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_channels: List[int],
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        residual: bool = True,
        temperature: float = 0.1,
        enable_drop: bool = False,
        drop_rate: float = 0.2,
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_channels: List of hidden dimensions
            embedding_dim: Final embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            residual: Whether to use residual connections
            temperature: Temperature scaling parameter
            enable_drop: Whether to enable connection dropping during training
            drop_rate: Percentage of connections to drop if enable_drop is True
        """
        super(OutfitGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.residual = residual
        self.temperature = temperature
        self.enable_drop = enable_drop
        self.drop_rate = drop_rate
        
        # Item and outfit encoders
        self.item_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.ReLU(),
            nn.Linear(hidden_channels[0], hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.ReLU(),
        )
        
        self.outfit_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.ReLU(),
            nn.Linear(hidden_channels[0], hidden_channels[0]),
            nn.LayerNorm(hidden_channels[0]),
            nn.ReLU(),
        )
        
        # First GATConv layer: input_dim → hidden_dim
        self.conv_1 = GATConv(
            in_channels=hidden_channels[0],
            out_channels=hidden_channels[1] // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
            bias=True,
            aggr='add',
            edge_dim=1,
            negative_slope=0.2
        )

        # Second GATConv layer: hidden_dim → hidden_dim
        self.conv_2 = GATConv(
            in_channels=hidden_channels[1],
            out_channels=hidden_channels[1] // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
            bias=True,
            aggr='add',
            edge_dim=1,
            negative_slope=0.2
        )
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_channels[1])
        
        # Feature gating for the final features
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_channels[1], hidden_channels[1]),
            nn.LayerNorm(hidden_channels[1]),
            nn.Sigmoid()
        )
        
        # Skip projection for residual connection
        self.skip_proj = nn.Linear(hidden_channels[1], hidden_channels[1])
        
        self.dropout = nn.Dropout(dropout)
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_channels[1], embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Hop attention weights
        self.hop_attention = nn.Parameter(torch.FloatTensor([0.4, 0.6]))
        self.hop_attention = nn.Parameter(F.softmax(self.hop_attention, dim=0))
        
        # Layer-specific attention
        self.layer_weight = nn.Sequential(
            nn.Linear(hidden_channels[0], 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, data_tuple: tuple) -> tuple:
        """Forward pass with batch processing of graphs"""
        query_data, pos_data_list, neg_data_list = data_tuple
        
        # Process query graph
        query_embedding = self._process_single_graph(query_data)  # [num_nodes, embedding_dim]
        query_embedding = query_embedding[query_data.root_idx]  # [batch_size, embedding_dim]
        
        # Batch process positive graphs
        if pos_data_list:
            # Combine graph processing when possible
            if all(hasattr(d, 'batch') for d in pos_data_list):
                # When data is already batched by PyG
                combined_pos_data = self._combine_graphs(pos_data_list)
                pos_embeddings = self._process_single_graph(combined_pos_data)
                pos_root_indices = torch.cat([d.root_idx + self._get_offset(combined_pos_data, i) 
                                             for i, d in enumerate(pos_data_list)])
                pos_embeddings = pos_embeddings[pos_root_indices]
            else:
                # Process in larger chunks when full batching not possible
                chunk_size = min(16, len(pos_data_list))  # Process 16 graphs at once
                pos_embeddings = []
                for i in range(0, len(pos_data_list), chunk_size):
                    if i > 0 and i % 32 == 0:  # Every 32 chunks
                        torch.cuda.empty_cache()  # Free memory during long loops
                    chunk = pos_data_list[i:i+chunk_size]
                    # Process each chunk and concatenate results
                    pos_embs = torch.cat([
                        self._process_single_graph(data)[data.root_idx]
                        for data in chunk
                    ])
                    pos_embeddings.append(pos_embs)
                pos_embeddings = torch.cat(pos_embeddings) if pos_embeddings else torch.empty(
                    (0, query_embedding.size(-1)), device=query_embedding.device)
        else:
            pos_embeddings = torch.empty((0, query_embedding.size(-1)), device=query_embedding.device)
        
        # Batch process negative graphs - same pattern as positives
        if neg_data_list:
            if all(hasattr(d, 'batch') for d in neg_data_list):
                combined_neg_data = self._combine_graphs(neg_data_list)
                neg_embeddings = self._process_single_graph(combined_neg_data)
                neg_root_indices = torch.cat([d.root_idx + self._get_offset(combined_neg_data, i) 
                                             for i, d in enumerate(neg_data_list)])
                neg_embeddings = neg_embeddings[neg_root_indices]
            else:
                chunk_size = min(32, len(neg_data_list))  # Process 32 graphs at once (more than positives)
                neg_embeddings = []
                for i in range(0, len(neg_data_list), chunk_size):
                    if i > 0 and i % 32 == 0:  # Every 32 chunks
                        torch.cuda.empty_cache()  # Free memory during long loops
                    chunk = neg_data_list[i:i+chunk_size]
                    neg_embs = torch.cat([
                        self._process_single_graph(data)[data.root_idx]
                        for data in chunk
                    ])
                    neg_embeddings.append(neg_embs)
                neg_embeddings = torch.cat(neg_embeddings) if neg_embeddings else torch.empty(
                    (0, query_embedding.size(-1)), device=query_embedding.device)
        else:
            neg_embeddings = torch.empty((0, query_embedding.size(-1)), device=query_embedding.device)
        
        # Apply temperature scaling
        query_embedding = query_embedding / self.temperature
        pos_embeddings = pos_embeddings / self.temperature
        neg_embeddings = neg_embeddings / self.temperature
        
        return query_embedding, pos_embeddings, neg_embeddings

    def _process_single_graph(self, data: Data) -> torch.Tensor:
        """Process a single graph with enhanced 2-hop attention"""
        x = data.x.view(-1, self.input_dim)
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        # Initialize features
        out = torch.zeros(x.size(0), self.hidden_channels[0], device=x.device)
        
        # Process different node types
        item_mask = data.node_type == 1
        outfit_mask = data.node_type == 0
        
        if item_mask.any():
            out[item_mask] = self.item_encoder(x[item_mask])
        if outfit_mask.any():
            out[outfit_mask] = self.outfit_encoder(x[outfit_mask])
        
        # Connection dropping during training
        if self.training and self.enable_drop:
            drop_mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_rate
            if not drop_mask.all():  # If any connections were dropped
                edge_index = edge_index[:, drop_mask]
                # If edge attributes exist, also drop those
                if edge_attr is not None:
                    edge_attr = edge_attr[drop_mask]
        
        # 1-hop and 2-hop message passing
        one_hop = self.conv_1(out, edge_index)        # [num_nodes, hidden_channels[1]]
        two_hop = self.conv_2(one_hop, edge_index)    # [num_nodes, hidden_channels[1]]
        
        # Dynamic hop attention: determine how much to value 1-hop vs 2-hop for each node
        hop_weights = self.layer_weight(out)          # [num_nodes, 2]
        out = hop_weights[:, 0].unsqueeze(1) * one_hop + \
              hop_weights[:, 1].unsqueeze(1) * two_hop
        
        # LayerNorm
        out = self.layer_norm(out)
        
        # Feature gating
        gates = self.feature_gate(out)
        out = out * gates
        
        # Activation
        out = F.elu(out)
        
        # Residual connection (from 1-hop to final layer)
        identity = self.skip_proj(one_hop)
        if identity.shape != out.shape:
            identity = identity.view(*out.shape)
        out = self.hop_attention[0] * out + self.hop_attention[1] * identity
        
        out = self.dropout(out)
        
        # Final embedding
        embeddings = self.final_projection(out)
        return F.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
    def get_attention_weights(self, data_tuple: tuple) -> List[torch.Tensor]:
        """Get attention weights for visualization"""
        query_data, _, _ = data_tuple
        x = query_data.x.view(-1, self.input_dim)
        edge_index = query_data.edge_index
        
        # Initialize features
        out = torch.zeros(x.size(0), self.hidden_channels[0], device=x.device)
        
        # Process node types
        item_mask = query_data.node_type == 1
        outfit_mask = query_data.node_type == 0
        
        if item_mask.any():
            out[item_mask] = self.item_encoder(x[item_mask])
        if outfit_mask.any():
            out[outfit_mask] = self.outfit_encoder(x[outfit_mask])
        
        # Get attention weights from both conv layers
        _, attn1 = self.conv_1(out, edge_index, return_attention_weights=True)
        one_hop = self.conv_1(out, edge_index)
        _, attn2 = self.conv_2(one_hop, edge_index, return_attention_weights=True)
        
        return [attn1, attn2]

    def _combine_graphs(self, data_list):
        """Combine multiple PyG Data objects into a single batched graph"""
        from torch_geometric.data import Batch
        return Batch.from_data_list(data_list)

    def _get_offset(self, batched_data, batch_idx):
        """Get node offset for a specific batch index"""
        batch = batched_data.batch
        return (batch == batch_idx).nonzero().min() 