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
import orjson
import concurrent.futures
import sys
from bson.objectid import ObjectId
from pymongo.operations import UpdateOne, InsertOne

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
    
    with open(graph_path, "wb") as f:  # Note: "wb" for binary mode
        f.write(orjson.dumps({"nodes": nodes})) # orjson for faster json dumping
    
    logger.info(f"Saved graph to {graph_path}")
    return graph_path

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference_results")
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
        "--batch_size", "256",
        "--num_similar", "3"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=False, text=True)
        
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
        
        # Prepare bulk operations
        embedding_bulk_operations = []
        items_bulk_operations = []
        
        # First pass: parse IDs and create operation lists
        variant_data = []
        for variant_id, embedding in embedding_dict.items():
            try:
                # Parse the variant ID (format: "item_id_color")
                parts = variant_id.split('_')
                if len(parts) < 2:
                    logger.warning(f"Invalid variant ID format: {variant_id}")
                    continue
                    
                item_id = parts[0]
                color = '_'.join(parts[1:])  # Handle colors with underscores
                
                # Add to our processing list
                variant_data.append({
                    'item_id': item_id,
                    'color': color,
                    'embedding': embedding
                })
                
            except Exception as e:
                logger.error(f"Error parsing variant ID {variant_id}: {e}")
        
        # Get all relevant items in one query to minimize DB hits
        unique_item_ids = list(set(v['item_id'] for v in variant_data))
        items_by_id = {}
        
        # Find all items in one query
        items_cursor = items_collection.find(
            {"_id": {"$in": [ObjectId(id) for id in unique_item_ids]}},
            {"_id": 1, "colorVariants.color": 1}
        )
        
        for item in items_cursor:
            items_by_id[str(item['_id'])] = item
        
        # Find existing embeddings in one query
        existing_embeddings = {}
        existing_cursor = embeddings_collection.find(
            {
                "item_id": {"$in": unique_item_ids},
                "color": {"$in": [v['color'] for v in variant_data]}
            },
            {"_id": 1, "item_id": 1, "color": 1}
        )
        
        for emb in existing_cursor:
            key = f"{emb['item_id']}_{emb['color']}"
            existing_embeddings[key] = emb['_id']
        
        # Second pass: create bulk operations
        updates = 0
        inserts = 0
        skipped = 0
        
        for variant in variant_data:
            item_id = variant['item_id']
            color = variant['color']
            embedding = variant['embedding']
            
            # Skip if item not found
            if item_id not in items_by_id:
                logger.warning(f"Item {item_id} not found in database")
                skipped += 1
                continue
                
            item = items_by_id[item_id]
            
            # Check if color variant exists
            color_exists = any(cv.get('color') == color for cv in item.get('colorVariants', []))
            if not color_exists:
                logger.warning(f"Color {color} not found for item {item_id}")
                skipped += 1
                continue
                
            # Prepare embedding document
            now = datetime.now()
            embedding_doc = {
                "item_id": item_id,
                "color": color, 
                "outfitgtn_embedding": embedding.tolist(),
                "updatedAt": now
            }
            
            # Check if entry exists and create appropriate operation
            lookup_key = f"{item_id}_{color}"
            if lookup_key in existing_embeddings:
                # Update operation
                embedding_bulk_operations.append(
                    UpdateOne(
                        {"_id": existing_embeddings[lookup_key]},
                        {"$set": embedding_doc}
                    )
                )
                updates += 1
            else:
                # Insert operation
                embedding_bulk_operations.append(
                    InsertOne(embedding_doc)
                )
                inserts += 1
            
            # Add operation to update the items collection
            items_bulk_operations.append(
                UpdateOne(
                    {"_id": ObjectId(item_id), "colorVariants.color": color},
                    {"$set": {"colorVariants.$.generated_outfitgtn_embedding": True}}
                )
            )
        
        # Execute bulk operations in batches
        if embedding_bulk_operations:
            batch_size = 1000
            for i in range(0, len(embedding_bulk_operations), batch_size):
                batch = embedding_bulk_operations[i:i+batch_size]
                embeddings_collection.bulk_write(batch)
                
        if items_bulk_operations:
            batch_size = 1000
            for i in range(0, len(items_bulk_operations), batch_size):
                batch = items_bulk_operations[i:i+batch_size]
                items_collection.bulk_write(batch)
        
        logger.info(f"Embeddings collection updated: {inserts} inserts, {updates} updates, {skipped} skipped")
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        # Print full stack trace for easier debugging
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

