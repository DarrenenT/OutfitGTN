import os
import torch
import numpy as np
import logging
import time
from tqdm import tqdm
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor
from openai import AzureOpenAI
from typing import List, Optional, Dict, Any
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

def create_openai_client():
    """Create a new OpenAI client instance"""
    return AzureOpenAI(
        azure_endpoint = "https://facciopenai.openai.azure.com/", 
        api_key = os.getenv('OPENAI_API_KEY'),  
        api_version = "2025-01-01-preview"
    )

def retry_api_call(api_call, *args, **kwargs):
    max_retries = 5
    initial_delay = 1  # Start with a 1-second delay
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return api_call(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):  # Check for rate limit error
                logger.warning(f"Rate limit hit, retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"Error during API call: {e}")
                raise e
    raise Exception("Max retries exceeded")

def get_text_embeddings(item_descriptions: List[str], outfit_description: Optional[str] = None) -> Dict[str, Any]:
    """Get embeddings for text descriptions in parallel using OpenAI"""
    client = create_openai_client()
    
    # Use ThreadPoolExecutor for parallel API calls
    def get_single_embedding(description):
        response = retry_api_call(
            client.embeddings.create,
            input=description,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)
    
    # Combine all descriptions for parallel processing
    all_descriptions = item_descriptions.copy()
    has_outfit = False
    if outfit_description:
        has_outfit = True
        all_descriptions.append(outfit_description)
    
    # Process all in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        all_embeddings = list(executor.map(get_single_embedding, all_descriptions))
    
    # Split the results
    if has_outfit:
        item_embeddings = all_embeddings[:-1]
        outfit_embedding = all_embeddings[-1]
        return {
            "item_embeddings": item_embeddings,
            "outfit_embedding": outfit_embedding
        }
    else:
        return {
            "item_embeddings": all_embeddings,
            "outfit_embedding": None
        }

def load_average_outfit_embedding():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                       "data_source/average_outfit_embedding.npz")
    data = np.load(path)
    return data['average_outfit_embedding']

def construct_outfit_graph(item_embeddings: List[np.ndarray], outfit_embedding: Optional[np.ndarray] = None) -> Data:
    """
    Construct a graph with a virtual outfit node connected to item nodes
    
    Args:
        item_embeddings: List of item embeddings
        outfit_embedding: Optional outfit embedding (if None, use average of items)
        
    Returns:
        PyTorch Geometric Data object
    """
    # Create outfit embedding if not provided
    if outfit_embedding is None:
        outfit_embedding = load_average_outfit_embedding()
    
    # Create node features with outfit node first, then item nodes
    features = [outfit_embedding] + item_embeddings
    
    # Create node types (0 for outfit, 1 for items)
    node_types = [0] + [1] * len(item_embeddings)
    
    # Create edges (bidirectional connections between outfit and each item)
    edge_index = []
    for i in range(1, len(features)):
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
        root_idx=torch.tensor([0])  # Outfit is the root node
    )
    
    return data

def generate_outfit_embedding_from_text(
    model: Any, 
    device: torch.device,
    item_descriptions: List[str], 
    outfit_description: Optional[str] = None
) -> np.ndarray:
    """
    Generate OutfitGTN embedding from text descriptions
    
    Args:
        model: The OutfitGTN model
        device: The device to run inference on
        item_descriptions: List of item descriptions
        outfit_description: Optional outfit description
        
    Returns:
        OutfitGTN embedding for the outfit
    """
    logger = setup_logging()
    
    # Get text embeddings for items and outfit in parallel
    logger.info(f"Generating embeddings for {len(item_descriptions)} items" + 
                (f" and outfit description" if outfit_description else ""))
    
    embeddings_result = get_text_embeddings(item_descriptions, outfit_description)
    item_embeddings = embeddings_result["item_embeddings"]
    outfit_embedding = embeddings_result["outfit_embedding"]
    
    # Construct graph
    logger.info("Constructing outfit graph")
    outfit_graph = construct_outfit_graph(item_embeddings, outfit_embedding)
    
    # Run inference
    logger.info("Computing OutfitGTN embedding")
    model.eval()
    with torch.no_grad():
        # Move data to device
        outfit_graph = outfit_graph.to(device)
        # Process the graph and get embeddings
        embeddings = model._process_single_graph(outfit_graph)
        # Extract the outfit embedding (root node)
        outfit_embedding = embeddings[0].cpu().numpy()
    
    return outfit_embedding