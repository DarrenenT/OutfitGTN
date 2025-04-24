import json
import numpy as np
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_average_outfit_embedding():
    # Path to training graph
    graph_path = "/home/OutfitGTN/data_source/graph_2025-04-12_19-53-08.json"
    
    # Load training graph
    logger.info(f"Loading training graph from {graph_path}")
    with open(graph_path, 'r') as f:
        training_graph = json.load(f)
    
    logger.info(f"Graph loaded. Contains {len(training_graph.get('nodes', []))} nodes total")
    
    # Filter outfit nodes
    outfit_nodes = [node for node in training_graph.get('nodes', []) if node.get('type') == 'outfit']
    logger.info(f"Found {len(outfit_nodes)} outfit nodes")
    
    if not outfit_nodes:
        logger.error("No outfit nodes found in the training graph")
        return None
    
    # Extract embeddings
    outfit_embeddings = []
    for node in outfit_nodes:
        if 'embedding' in node:
            outfit_embeddings.append(node['embedding'])
    
    logger.info(f"Extracted embeddings from {len(outfit_embeddings)} outfit nodes")
    
    if not outfit_embeddings:
        logger.error("No outfit embeddings found")
        return None
    
    # Calculate average embedding
    average_embedding = np.mean(outfit_embeddings, axis=0)
    embedding_dim = average_embedding.shape[0]
    logger.info(f"Calculated average embedding with dimension {embedding_dim}")
    
    # Save average embedding
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_source")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "average_outfit_embedding.npz")
    
    np.savez_compressed(
        output_path,
        average_outfit_embedding=average_embedding
    )
    logger.info(f"Saved average outfit embedding to {output_path}")
    
    return average_embedding

if __name__ == "__main__":
    avg_embedding = calculate_average_outfit_embedding()
    if avg_embedding is not None:
        # Print summary statistics
        logger.info(f"Average embedding stats: mean={np.mean(avg_embedding)}, std={np.std(avg_embedding)}")
        logger.info(f"Norm of average embedding: {np.linalg.norm(avg_embedding)}")