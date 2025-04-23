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

def load_context_embeddings(context_path):
    """Load context embeddings (seasonal, demographic, etc.)"""
    with open(context_path, 'r') as f:
        context_data = json.load(f)
    
    # Create a dictionary mapping context names to embeddings
    context_embeddings = {name: np.array(emb) for name, emb in context_data.items()}
    
    return context_embeddings

def construct_contextual_graph(selected_items, item_embeddings, context_tokens, context_embeddings):
    """
    Construct a graph with selected items, virtual outfit node, and context nodes.
    
    Args:
        selected_items: List of IDs of selected items
        item_embeddings: Dictionary mapping item IDs to embeddings
        context_tokens: List of context token names to include
        context_embeddings: Dictionary mapping context token names to embeddings
    
    Returns:
        PyTorch Geometric Data object representing the graph with context
    """
    # Create virtual node IDs (negative to avoid conflicts)
    virtual_outfit_id = -1
    
    # Create node IDs list with virtual outfit first, then context tokens, then items
    node_ids = [virtual_outfit_id]
    
    # Create node types (0 for outfit/context, 1 for item)
    node_types = [0]  # Virtual outfit is type 0
    
    # Create node features
    features = []
    
    # Calculate virtual outfit feature (average of selected items)
    if selected_items:
        outfit_feature = np.mean([item_embeddings[item_id] for item_id in selected_items], axis=0)
    else:
        # If no items selected, use a zero vector with appropriate size
        outfit_feature = np.zeros(next(iter(item_embeddings.values())).shape)
    
    features.append(outfit_feature)
    
    # Add context nodes
    context_ids = []
    for i, token in enumerate(context_tokens, start=1):
        if token in context_embeddings:
            node_ids.append(-i - 1)  # Context node IDs are negative and unique
            node_types.append(0)  # Context nodes are type 0 (like outfits)
            features.append(context_embeddings[token])
            context_ids.append(len(node_ids) - 1)  # Store the index of this context
    
    # Calculate the offset for item nodes
    item_offset = len(node_ids)
    
    # Add selected items
    for item_id in selected_items:
        node_ids.append(item_id)
        node_types.append(1)  # Items are type 1
        features.append(item_embeddings[item_id])
    
    # Create edge indices (connections between nodes)
    edge_index = []
    
    # Connect virtual outfit to all context nodes (bidirectional)
    for ctx_idx in context_ids:
        edge_index.append([0, ctx_idx])  # Outfit to context
        edge_index.append([ctx_idx, 0])  # Context to outfit
    
    # Connect virtual outfit to all items (bidirectional)
    for i in range(len(selected_items)):
        idx = item_offset + i
        edge_index.append([0, idx])  # Outfit to item
        edge_index.append([idx, 0])  # Item to outfit
    
    # Connect context nodes to all items (bidirectional)
    for ctx_idx in context_ids:
        for i in range(len(selected_items)):
            idx = item_offset + i
            edge_index.append([ctx_idx, idx])  # Context to item
            edge_index.append([idx, ctx_idx])  # Item to context
    
    # Convert to tensors
    x = torch.tensor(np.array(features), dtype=torch.float)
    
    # Handle empty edge case
    if not edge_index:
        # Create a self-loop for the outfit node if no other edges
        edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
    else:
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

def compute_contextual_embedding(model, graph_data, device):
    """Compute embedding for the virtual outfit node in a contextual graph."""
    model.eval()
    with torch.no_grad():
        # Move data to device
        graph_data = graph_data.to(device)
        # Process the graph and get embeddings
        embeddings = model._process_single_graph(graph_data)
        # Extract the embedding for the virtual outfit node (root node)
        contextual_embedding = embeddings[0]  # Index 0 is the virtual outfit node
    return contextual_embedding.cpu().numpy()

def find_compatible_items(contextual_embedding, catalog_embeddings, item_ids, excluded_items=None, top_k=10):
    """Find the most compatible items based on contextual embedding."""
    # Calculate cosine similarity
    similarities = np.dot(catalog_embeddings, contextual_embedding) / (
        np.linalg.norm(catalog_embeddings, axis=1) * np.linalg.norm(contextual_embedding)
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
    parser = argparse.ArgumentParser(description="Contextual multi-item recommendation")
    parser.add_argument("--config", type=str, default="config/config_GTN.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_GTN.pt", help="Path to model checkpoint")
    parser.add_argument("--embeddings", type=str, default="data/embeddings.npz", help="Path to pre-computed embeddings")
    parser.add_argument("--context_path", type=str, default="data/context_embeddings.json", help="Path to context embeddings")
    parser.add_argument("--items", type=str, default="", help="Comma-separated list of item IDs (optional)")
    parser.add_argument("--contexts", type=str, required=True, help="Comma-separated list of context tokens")
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
    
    # Load context embeddings
    logger.info(f"Loading context embeddings from {args.context_path}")
    context_embeddings = load_context_embeddings(args.context_path)
    logger.info(f"Loaded {len(context_embeddings)} context embeddings")
    
    # Parse context tokens
    context_tokens = [token.strip() for token in args.contexts.split(',')]
    logger.info(f"Context tokens: {context_tokens}")
    
    # Validate context tokens
    for token in context_tokens:
        if token not in context_embeddings:
            logger.warning(f"Context token '{token}' not found in context embeddings")
    
    # Parse selected items (if any)
    selected_items = []
    if args.items:
        selected_items = [int(item_id.strip()) for item_id in args.items.split(',')]
        logger.info(f"Selected items: {selected_items}")
        
        # Check if all selected items have embeddings
        for item_id in selected_items:
            if item_id not in item_embeddings:
                logger.error(f"Item {item_id} not found in embeddings")
                return
    
    # Construct contextual graph
    contextual_graph = construct_contextual_graph(
        selected_items, 
        item_embeddings, 
        context_tokens, 
        context_embeddings
    )
    
    # Compute contextual embedding
    contextual_embedding = compute_contextual_embedding(model, contextual_graph, device)
    
    # Find compatible items
    catalog_item_ids = list(item_embeddings.keys())
    catalog_embeddings = np.array([item_embeddings[item_id] for item_id in catalog_item_ids])
    
    # Exclude already selected items from recommendations
    compatible_items, similarities = find_compatible_items(
        contextual_embedding, 
        catalog_embeddings, 
        catalog_item_ids, 
        excluded_items=selected_items, 
        top_k=args.top_k
    )
    
    # Display results
    context_str = ", ".join(context_tokens)
    if selected_items:
        logger.info(f"\nRecommended items for context '{context_str}' and selected items {selected_items}:")
    else:
        logger.info(f"\nRecommended items for context '{context_str}':")
        
    for i, (item_id, similarity) in enumerate(zip(compatible_items, similarities), 1):
        logger.info(f"  {i}. Item {item_id} (compatibility: {similarity:.4f})")

def create_sample_context_embeddings():
    """Create sample context embeddings for testing."""
    # Example context tokens
    contexts = {
        "winter": np.random.randn(1536),  # Match input_dim from config
        "summer": np.random.randn(1536),
        "spring": np.random.randn(1536),
        "fall": np.random.randn(1536),
        "casual": np.random.randn(1536),
        "formal": np.random.randn(1536),
        "party": np.random.randn(1536),
        "work": np.random.randn(1536),
        "outdoor": np.random.randn(1536),
        "indoor": np.random.randn(1536),
        "male": np.random.randn(1536),
        "female": np.random.randn(1536)
    }
    
    # Normalize embeddings
    for key in contexts:
        contexts[key] = contexts[key] / np.linalg.norm(contexts[key])
        contexts[key] = contexts[key].tolist()  # Convert to list for JSON serialization
    
    # Save to file
    output_path = "data/context_embeddings.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(contexts, f)
    
    print(f"Created sample context embeddings at {output_path}")

if __name__ == "__main__":
    # Uncomment to create sample context embeddings
    # create_sample_context_embeddings()
    main() 