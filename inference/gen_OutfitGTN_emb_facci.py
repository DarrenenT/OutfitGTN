import io
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import random
from tqdm import tqdm
import logging
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import json
import concurrent.futures
import sys

# Add the root directory to the path so we can use absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_source.download_data_source import ensure_latest_data

load_dotenv()

# Set up logging - clear any existing handlers first to prevent duplicates
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Thread-safe print lock
print_lock = threading.Lock()

# MongoDB setup
mongo_client = MongoClient(os.getenv("ATLAS_Ecomm_MONGODB_PROD_URI"))
db = mongo_client["facci-v2"]
items_collection = db["items"]
embeddings_collection = db["item_embeddings"]  # New collection for embeddings

# OutfitGTN model setup
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/config_GTN.yaml")
checkpoint_tuple = ensure_latest_data(os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints"), force_update=True, file_type="best_model_GTN")
checkpoint_path = checkpoint_tuple[0]  # Get first element of tuple

training_tuple = ensure_latest_data(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_source"), force_update=True, file_type="graph")
training_graph_path = training_tuple[0]  # Get first element of tuple

def process_item(item):
    try:
        item_nodes = []
        if not item:
            return []
            
        if "colorVariants" not in item or not item["colorVariants"]:
            logger.info(f"Item {item.get('_id', 'unknown')} has no color variants")
            return []
            
        for color_variant in item.get("colorVariants", []):
            if "item_embedding" in color_variant:
                # Create a node for each color variant
                try:
                    node = {
                        "id": f"{item['_id']}_{color_variant.get('color', 'Unknown')}",
                        "type": "item",
                        "embedding": color_variant["item_embedding"],
                        "neighbors": []
                    }
                    item_nodes.append(node)
                except Exception as e:
                    logger.error(f"Error creating node for item {item['_id']}, color {color_variant.get('color', 'Unknown')}: {e}")
                    continue
            else:
                logger.info(f"Item {item['_id']} color {color_variant.get('color', 'Unknown')} has no item_embedding")
        return item_nodes
    except Exception as e:
        logger.error(f"Error processing item {item.get('_id', 'unknown')}: {e}")
        return []
    
def create_graph_json_for_ecommerce():
    # Only process items that have an item_embedding and either no generated_outfitgtn_embedding or a generated_outfitgtn_embedding that is False
    items_cursor = items_collection.find(
        {"colorVariants.item_embedding": {"$exists": True}, "$or": [{"colorVariants.generated_outfitgtn_embedding": False}, {"colorVariants.generated_outfitgtn_embedding": {"$exists": False}}]}, 
        {"_id": 1, "colorVariants": 1}
    )
    
    items = list(items_cursor)
    logger.info(f"Found {len(items)} items with embeddings")
    
    if not items:
        logger.error("No items with embeddings found in database")
        return None
        
    nodes = []
    
    # Process items in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for item in items:
            futures.append(executor.submit(process_item, item))
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            if result:
                nodes.extend(result)
    
    logger.info(f"Created graph with {len(nodes)} nodes")
    if not nodes:
        logger.error("No valid nodes generated for graph")
        return None
                
    # Create data directory if it doesn't exist
    data_source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_source")
    os.makedirs(data_source_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = os.path.join(data_source_dir, f"facci_graph_prod_{timestamp}.json")
    
    with open(graph_path, "w") as f:
        json.dump({"nodes": nodes}, f)
    
    logger.info(f"Saved graph to {graph_path}")
    return graph_path

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graph file from MongoDB data
    ecomm_graph_path = create_graph_json_for_ecommerce()
    if not ecomm_graph_path:
        logger.error("Failed to create e-commerce graph. Exiting.")
        return
    
    # Get the script path
    inference_script = os.path.join(os.path.dirname(__file__), "inference_static_embedding.py")
    
    # Output path for embeddings
    output_path = os.path.join(output_dir, "ecommerce_embeddings.npz")
    
    # Run the inference script with the right parameters
    logger.info("Running inference...")
    logger.info(f"Using config: {config_path}")
    logger.info(f"Using checkpoint: {checkpoint_path}")
    logger.info(f"Using training graph: {training_graph_path}")
    logger.info(f"Using e-commerce graph: {ecomm_graph_path}")
    logger.info(f"Output will be saved to: {output_path}")
    
    import subprocess
    cmd = [
        sys.executable,  # Use the current Python interpreter
        inference_script,
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--training_graph_path", training_graph_path,
        "--ecomm_graph_path", ecomm_graph_path,
        "--output_path", output_path,
        "--batch_size", "128",
        "--num_similar", "3"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Log subprocess output
    if process.stdout:
        logger.info(f"Subprocess stdout: {process.stdout}")
    if process.stderr:
        logger.error(f"Subprocess stderr: {process.stderr}")
        
    if process.returncode != 0:
        logger.error(f"Inference failed with return code {process.returncode}")
        return
        
    logger.info("Inference completed")
    
    # Check if the output file exists
    if not os.path.exists(output_path):
        logger.error(f"Output file {output_path} not found")
        return
        
    # Load generated embeddings
    try:
        data = np.load(output_path)
        item_ids = data['item_ids']
        embeddings = data['embeddings']
        
        logger.info(f"Loaded embeddings for {len(item_ids)} items")
        
        # Create a dictionary mapping item_ids to embeddings
        embedding_dict = {str(item_id): embedding for item_id, embedding in zip(item_ids, embeddings)}
        
        # Update embeddings collection instead of modifying items 
        updates = 0
        inserts = 0
        
        # First, let's parse the item IDs to extract original item ID and color
        for variant_id, embedding in embedding_dict.items():
            try:
                # Parse the variant ID (format: "item_id_color")
                parts = variant_id.split('_')
                if len(parts) < 2:
                    logger.warning(f"Invalid variant ID format: {variant_id}")
                    continue
                    
                item_id = parts[0]
                color = '_'.join(parts[1:])  # Handle colors with underscores
                
                # Get item metadata
                item = items_collection.find_one(
                    {"_id": item_id, "colorVariants.color": color},
                )
                
                if not item:
                    logger.warning(f"Could not find item {item_id} with color {color}")
                    continue
                
                # Check if an entry already exists
                existing = embeddings_collection.find_one({
                    "item_id": item_id,
                    "color": color
                })
                
                # Prepare embedding document
                embedding_doc = {
                    "item_id": item_id,
                    "color": color, 
                    "outfitgtn_embedding": embedding.tolist(),
                    "updatedAt": datetime.now()
                }
                
                # Insert or update
                if existing:
                    embeddings_collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": embedding_doc}
                    )
                    updates += 1
                else:
                    embeddings_collection.insert_one(embedding_doc)
                    inserts += 1
                
                # Update the items collection to mark this color variant as processed
                items_collection.update_one(
                    {"_id": item_id, "colorVariants.color": color},
                    {"$set": {"colorVariants.$.generated_outfitgtn_embedding": True}}
                )
                
            except Exception as e:
                logger.error(f"Error processing embedding for {variant_id}: {e}")
        
        logger.info(f"Embeddings collection updated: {inserts} inserts, {updates} updates")
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")

if __name__ == "__main__":
    main()

