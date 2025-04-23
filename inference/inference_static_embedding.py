import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import torch
import yaml
import numpy as np
import json
import logging
from pathlib import Path
import sys
# Add the root directory to the path so we can use absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.OutfitGTN import OutfitGTN
from utils.gpu_utils import configure_gpu, get_device
from data.fashion_node import FashionNode
from torch_geometric.data import Data
import argparse
from dotenv import load_dotenv
load_dotenv()

def setup_logging():
    """Setup logging configuration"""
    # Clear any existing handlers to prevent duplicate logs
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
            
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

def construct_synthetic_graph(item_id, item_embedding, similar_items, nodes):
    """
    Construct a synthetic bipartite graph for a new item or item without outfit connections.
    
    Args:
        item_id: ID of the target item
        item_embedding: Feature embedding of the target item
        similar_items: List of IDs of similar items from the training set
        nodes: Dictionary of all nodes (to find outfit connections of similar items)
    
    Returns:
        PyTorch Geometric Data object representing the synthetic graph
    """
    # Create a virtual node ID for the target item (to avoid conflicts)
    virtual_item_id = -1
    
    # Find all outfits connected to the similar items
    outfits = set()
    for sim_id in similar_items:
        if sim_id in nodes:
            outfits.update(nodes[sim_id].neighbors)
    
    # Create node IDs list with the target item first, then outfits, then similar items
    node_ids = [virtual_item_id] + list(outfits) + similar_items
    
    # Create node types (0 for outfit, 1 for item)
    node_types = [1]  # Target item is type 1 (item)
    node_types.extend([0 for _ in outfits])  # Outfits are type 0
    node_types.extend([1 for _ in similar_items])  # Similar items are type 1
    
    # Create node features
    features = [item_embedding]  # Target item embedding
    
    # Add outfit and similar item embeddings from nodes dictionary
    for node_id in node_ids[1:]:
        if node_id in nodes:
            features.append(nodes[node_id].embedding)
        else:
            # If embedding is missing, use a zero vector
            features.append(np.zeros(item_embedding.shape, dtype=np.float32))
    
    # Create edge indices (connections between nodes)
    edge_index = []
    
    # Connect target item to all outfits
    for i, node_id in enumerate(node_ids[1:len(outfits)+1], 1):
        # Bidirectional edge between item and outfit
        edge_index.append([0, i])  # Target item to outfit
        edge_index.append([i, 0])  # Outfit to target item
    
    # Connect outfits to similar items
    outfit_offset = 1
    similar_offset = len(outfits) + 1
    
    for i, sim_id in enumerate(similar_items):
        if sim_id in nodes:
            # Find common outfits between this similar item and all outfits
            for j, outfit_id in enumerate(outfits):
                if outfit_id in nodes[sim_id].neighbors:
                    # Connect outfit to similar item
                    edge_index.append([outfit_offset + j, similar_offset + i])
                    edge_index.append([similar_offset + i, outfit_offset + j])
    
    # Convert to tensors
    x = torch.tensor(np.array(features), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_type = torch.tensor(node_types, dtype=torch.long)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        node_type=node_type,
        root_idx=torch.tensor([0])  # Target item is the root node (idx 0)
    )
    
    return data

def compute_item_embedding(model, item_data, device):
    """Compute embedding for a single item using its synthetic graph."""
    model.eval()
    with torch.no_grad():
        # Move data to device
        item_data = item_data.to(device)
        # Process the graph and get embeddings
        embeddings = model._process_single_graph(item_data)
        # Extract the embedding for the root node (our target item)
        root_idx = item_data.root_idx.item()
        item_embedding = embeddings[root_idx]
    return item_embedding.cpu().numpy()

def find_similar_catalog_items(query_embedding, catalog_embeddings, top_k=10):
    """Find the most similar catalog items to the query item."""
    # Calculate cosine similarity
    similarities = np.dot(catalog_embeddings, query_embedding) / (
        np.linalg.norm(catalog_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get indices of top k similar items
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return top_indices, similarities[top_indices]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate static embeddings for fashion items")
    parser.add_argument("--config", type=str, default="config/config_GTN.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_GTN.pt", help="Path to model checkpoint")
    parser.add_argument("--training_graph_path", type=str, required=True, help="Path to original training graph data")
    parser.add_argument("--ecomm_graph_path", type=str, required=True, help="Path to e-commerce graph data")
    parser.add_argument("--output_path", type=str, default="data/embeddings.npz", help="Path to save embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num_similar", type=int, default=3, help="Number of similar items to use for synthetic graphs")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Configure device
    configure_gpu()
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from config {args.config}")
    model, config = load_model(args.config, args.checkpoint, device)
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Load TRAINING graph data (for finding similar items and their outfit connections)
    logger.info(f"Loading training graph data from {args.training_graph_path}")
    with open(args.training_graph_path, 'r') as f:
        training_graph_data = json.load(f)
    
    logger.info(f"Training graph data loaded. Contains {len(training_graph_data.get('nodes', []))} nodes total")
    
    # Process training nodes
    training_nodes = {}
    for node_data in training_graph_data.get('nodes', []):
        try:
            node_id = int(node_data['id'])
            training_nodes[node_id] = FashionNode(node_data)
            # logger.debug(f"Processed training node {node_id} of type {training_nodes[node_id].node_type}")
        except Exception as e:
            logger.warning(f"Error processing training node {node_data.get('id')}: {e}")
    
    logger.info(f"Processed {len(training_nodes)} training nodes successfully")
    
    # Identify training item nodes 
    training_item_nodes = {nid: node for nid, node in training_nodes.items() if node.node_type == "item"}
    logger.info(f"Found {len(training_item_nodes)} items in the training graph")
    
    # Exit early if no training items found
    if len(training_item_nodes) == 0:
        logger.error("No items found in training graph. Cannot proceed with embedding generation.")
        logger.info("Please check the format of your training graph JSON file.")
        logger.info("Each node should have: 'id', 'type' ('item' or 'outfit'), 'embedding', and 'neighbors' fields.")
        sys.exit(1)
    
    # Create a feature matrix for similarity search from training items
    training_item_ids = list(training_item_nodes.keys())
    training_feature_matrix = np.array([training_nodes[nid].embedding for nid in training_item_ids])
    logger.info(f"Created training feature matrix with shape {training_feature_matrix.shape}")
    
    # Load E-COMMERCE graph data (for items that need embeddings)
    logger.info(f"Loading e-commerce graph data from {args.ecomm_graph_path}")
    with open(args.ecomm_graph_path, 'r') as f:
        ecomm_graph_data = json.load(f)
    
    logger.info(f"E-commerce graph data loaded. Contains {len(ecomm_graph_data.get('nodes', []))} nodes total")
    
    # Process e-commerce nodes
    ecomm_nodes = {}
    for node_data in ecomm_graph_data.get('nodes', []):
        try:
            node_id = node_data['id']
            ecomm_nodes[node_id] = FashionNode(node_data)
            # logger.debug(f"Processed e-commerce node {node_id}")
        except Exception as e:
            logger.warning(f"Error processing e-commerce node {node_data.get('id')}: {e}")
    
    logger.info(f"Processed {len(ecomm_nodes)} e-commerce nodes successfully")
    
    # Process all e-commerce items
    embeddings = {}
    ecomm_item_ids = list(ecomm_nodes.keys())
    logger.info(f"Processing {len(ecomm_item_ids)} e-commerce items")
    
    for i, item_id in enumerate(ecomm_item_ids):
        if i % 100 == 0:
            logger.info(f"Processing item {i+1}/{len(ecomm_item_ids)}")
            
        # Get item features
        item_embedding = ecomm_nodes[item_id].embedding
        logger.debug(f"Item {item_id} has embedding of shape {item_embedding.shape}")
        
        # Find similar items FROM THE TRAINING SET (important!)
        logger.debug(f"Finding {args.num_similar} similar items for {item_id} from training set")
        similarities = np.dot(training_feature_matrix, item_embedding) / (
            np.linalg.norm(training_feature_matrix, axis=1) * np.linalg.norm(item_embedding)
        )
        similar_indices = np.argsort(similarities)[-args.num_similar:][::-1]
        similar_items = [training_item_ids[idx] for idx in similar_indices]
        logger.debug(f"Similar items for {item_id}: {similar_items}")
        
        # Construct synthetic graph using training items and their outfit connections
        logger.debug(f"Constructing synthetic graph for {item_id}")
        item_graph = construct_synthetic_graph(item_id, item_embedding, similar_items, training_nodes)
        logger.debug(f"Graph for {item_id} has {item_graph.num_nodes} nodes and {item_graph.num_edges} edges")
        
        # Compute embedding
        logger.debug(f"Computing OutfitGTN embedding for {item_id}")
        new_embedding = compute_item_embedding(model, item_graph, device)
        logger.debug(f"Generated embedding of shape {new_embedding.shape} for {item_id}")
        
        # Store embedding
        embeddings[item_id] = new_embedding
        
    # Save embeddings
    logger.info(f"Saving {len(embeddings)} embeddings to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez_compressed(
        args.output_path,
        embeddings=np.array([embeddings[item_id] for item_id in ecomm_item_ids]),
        item_ids=np.array(ecomm_item_ids)
    )
    logger.info(f"Saved {len(embeddings)} embeddings to {args.output_path}")
    
    # Example of using the embeddings for similarity search
    logger.info("\nExample similarity search:")
    
    # Choose a random item as query
    query_idx = np.random.randint(len(ecomm_item_ids))
    query_id = ecomm_item_ids[query_idx]
    query_embedding = embeddings[query_id]
    
    # Find similar items
    all_embeddings = np.array([embeddings[item_id] for item_id in ecomm_item_ids])
    similar_indices, similarities = find_similar_catalog_items(query_embedding, all_embeddings, top_k=5)
    
    logger.info(f"Query item: {query_id}")
    for i, idx in enumerate(similar_indices):
        similar_id = ecomm_item_ids[idx]
        logger.info(f"  Similar item {i+1}: {similar_id} (similarity: {similarities[i]:.4f})")

if __name__ == "__main__":
    main() 