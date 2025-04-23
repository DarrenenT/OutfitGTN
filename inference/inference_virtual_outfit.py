import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import torch
import yaml
import numpy as np
import json
import logging
from pathlib import Path
from models.OutfitGTN import OutfitGTN
from utils.gpu_utils import configure_gpu, get_device
from torch_geometric.data import Data
import argparse

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(config_path, checkpoint_path, device):
    """Load the OutfitGTN model from a checkpoint."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = OutfitGTN(
        input_dim=config['model']['input_dim'],
        hidden_channels=config['model']['hidden_channels'],
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        residual=config['model']['residual'],
        temperature=config['model']['temperature'],
        enable_drop=False,  # Disable connection dropping for inference
        drop_rate=0.0
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def load_embeddings(embeddings_path):
    """Load pre-computed GTN embeddings"""
    data = np.load(embeddings_path)
    item_ids = data['item_ids']
    embeddings = data['embeddings']
    
    # Create a dictionary mapping item IDs to embeddings
    embedding_dict = {int(item_id): embedding for item_id, embedding in zip(item_ids, embeddings)}
    
    return embedding_dict

def construct_virtual_outfit_graph(selected_items, item_embeddings, nodes=None):
    """
    Construct a graph with a virtual outfit node connected to the selected items.
    
    Args:
        selected_items: List of IDs of selected items
        item_embeddings: Dictionary mapping item IDs to embeddings
        nodes: Optional dictionary of all nodes (used if available)
    
    Returns:
        PyTorch Geometric Data object representing the graph
    """
    # Create a virtual outfit node ID (negative to avoid conflicts)
    virtual_outfit_id = -1
    
    # Initialize node list with the virtual outfit node
    node_ids = [virtual_outfit_id] + selected_items
    
    # Create node types (0 for outfit, 1 for item)
    node_types = [0]  # Virtual outfit is type 0
    node_types.extend([1 for _ in selected_items])  # Items are type 1
    
    # Create node features
    features = []
    
    # Virtual outfit feature is the average of selected items
    outfit_feature = np.mean([item_embeddings[item_id] for item_id in selected_items], axis=0)
    features.append(outfit_feature)
    
    # Add selected item features
    for item_id in selected_items:
        features.append(item_embeddings[item_id])
    
    # Create edge indices (connections between nodes)
    edge_index = []
    
    # Connect virtual outfit to all items (bidirectional edges)
    for i in range(1, len(node_ids)):
        edge_index.append([0, i])  # Outfit to item
        edge_index.append([i, 0])  # Item to outfit
    
    # Convert to tensors
    x = torch.tensor(np.array(features), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_type = torch.tensor(node_types, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        node_type=node_type,
        root_idx=torch.tensor([0])  # Virtual outfit is the root node
    )
    
    return data

def compute_outfit_embedding(model, outfit_data, device):
    """Compute embedding for the virtual outfit node."""
    model.eval()
    with torch.no_grad():
        # Move data to device
        outfit_data = outfit_data.to(device)
        # Process the graph and get embeddings
        embeddings = model._process_single_graph(outfit_data)
        # Extract the embedding for the virtual outfit node (root node)
        outfit_embedding = embeddings[0]  # Index 0 is the virtual outfit node
    return outfit_embedding.cpu().numpy()

def find_compatible_items(outfit_embedding, catalog_embeddings, item_ids, excluded_items=None, top_k=10):
    """
    Find the most compatible items to complete an outfit.
    
    Args:
        outfit_embedding: Embedding of the virtual outfit
        catalog_embeddings: Array of embeddings for all catalog items
        item_ids: List of item IDs corresponding to catalog_embeddings
        excluded_items: List of item IDs to exclude (e.g., already selected items)
        top_k: Number of top compatible items to return
    
    Returns:
        Lists of item IDs and similarity scores
    """
    # Calculate cosine similarity
    similarities = np.dot(catalog_embeddings, outfit_embedding) / (
        np.linalg.norm(catalog_embeddings, axis=1) * np.linalg.norm(outfit_embedding)
    )
    
    # Create a list of (item_id, similarity) tuples
    item_similarities = list(zip(item_ids, similarities))
    
    # Filter out excluded items
    if excluded_items:
        excluded_set = set(excluded_items)
        item_similarities = [(item_id, sim) for item_id, sim in item_similarities 
                             if item_id not in excluded_set]
    
    # Sort by similarity (descending)
    item_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-k items
    top_items = item_similarities[:top_k]
    
    # Separate item IDs and similarities
    top_item_ids = [item[0] for item in top_items]
    top_similarities = [item[1] for item in top_items]
    
    return top_item_ids, top_similarities

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-item recommendation with virtual outfit node")
    parser.add_argument("--config", type=str, default="config/config_GTN.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_GTN.pt", help="Path to model checkpoint")
    parser.add_argument("--embeddings", type=str, default="data/embeddings.npz", help="Path to pre-computed embeddings")
    parser.add_argument("--graph_path", type=str, default="data_source/graph.json", help="Path to graph data (optional)")
    parser.add_argument("--items", type=str, required=True, help="Comma-separated list of item IDs")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations to return")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Configure device
    configure_gpu()
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.config, args.checkpoint, device)
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Load pre-computed embeddings
    logger.info(f"Loading embeddings from {args.embeddings}")
    item_embeddings = load_embeddings(args.embeddings)
    logger.info(f"Loaded embeddings for {len(item_embeddings)} items")
    
    # Parse selected items
    selected_items = [int(item_id.strip()) for item_id in args.items.split(',')]
    logger.info(f"Selected items: {selected_items}")
    
    # Check if all selected items have embeddings
    for item_id in selected_items:
        if item_id not in item_embeddings:
            logger.error(f"Item {item_id} not found in embeddings")
            return
    
    # Construct virtual outfit graph
    outfit_graph = construct_virtual_outfit_graph(selected_items, item_embeddings)
    
    # Compute outfit embedding
    outfit_embedding = compute_outfit_embedding(model, outfit_graph, device)
    
    # Find compatible items
    catalog_item_ids = list(item_embeddings.keys())
    catalog_embeddings = np.array([item_embeddings[item_id] for item_id in catalog_item_ids])
    
    # Exclude already selected items from recommendations
    compatible_items, similarities = find_compatible_items(
        outfit_embedding, 
        catalog_embeddings, 
        catalog_item_ids, 
        excluded_items=selected_items, 
        top_k=args.top_k
    )
    
    # Display results
    logger.info("\nRecommended items to complete the outfit:")
    for i, (item_id, similarity) in enumerate(zip(compatible_items, similarities), 1):
        logger.info(f"  {i}. Item {item_id} (compatibility: {similarity:.4f})")

if __name__ == "__main__":
    main() 