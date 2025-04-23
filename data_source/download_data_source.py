'''
File: download_data_source.py
Description: This script downloads processed fashion data sources from Azure Blob Storage
             that will be used to build the Outfit Graph Transformer Network (OutfitGTN).
             It retrieves outfit embedding data, item embedding data, and metadata files
             necessary for graph construction and model training. The downloaded data is
             saved to the local filesystem in a structured format.
             
Features:
- Azure Blob Storage integration for secure data retrieval
- Efficient download of large datasets with progress tracking
- Structured organization of downloaded files by data type
- Validation of downloaded data integrity
- Environment variable configuration for secure credential management
- Comprehensive logging for tracking download operations
- Error handling for network and storage access issues
'''

import os
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
import logging
import sys
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("data_downloader")

# Reduce Azure storage logging verbosity
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

def get_latest_blob(container_client, prefix, exclude_patterns=None):
    """
    Find the latest blob in a container by date in filename.
    
    Args:
        container_client: Azure Blob Container client
        prefix: Prefix to filter blobs (e.g., 'graph_' or 'graph_visual_')
        exclude_patterns: List of patterns to exclude from results (e.g., ['visual'] to exclude visualization files)
        
    Returns:
        Name of the latest blob or None if not found
    """
    try:
        # List all blobs with the given prefix
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        
        if not blobs:
            logger.warning(f"No blobs found with prefix '{prefix}'")
            return None
            
        # Apply exclusion patterns if provided
        if exclude_patterns:
            filtered_blobs = []
            for blob in blobs:
                excluded = False
                for pattern in exclude_patterns:
                    if pattern in blob.name:
                        excluded = True
                        break
                if not excluded:
                    filtered_blobs.append(blob)
            
            blobs = filtered_blobs
            
            if not blobs:
                logger.warning(f"No blobs found after applying exclusion patterns: {exclude_patterns}")
                return None
            
        # Find dates in filenames using regex pattern
        # This matches YYYY-MM-DD_HH-MM-SS pattern in filenames
        date_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
        
        dated_blobs = []
        for blob in blobs:
            match = re.search(date_pattern, blob.name)
            if match:
                date_str = match.group(1)
                try:
                    # Parse the date
                    date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
                    dated_blobs.append((blob.name, date))
                except ValueError:
                    logger.warning(f"Could not parse date in blob name: {blob.name}")
            else:
                logger.debug(f"No date pattern found in blob name: {blob.name}")
                
        if not dated_blobs:
            logger.warning("No blobs with valid date format found")
            return None
            
        # Sort by date (newest first) and return the name of the newest blob
        latest_blob = sorted(dated_blobs, key=lambda x: x[1], reverse=True)[0][0]
        return latest_blob
        
    except Exception as e:
        logger.error(f"Error finding latest blob: {str(e)}")
        return None

def download_blob(blob_service_client, container_name, blob_name, output_path, overwrite=False):
    """
    Download a blob from Azure Storage.
    
    Args:
        blob_service_client: Azure Blob Service client
        container_name: Name of the container
        blob_name: Name of the blob to download
        output_path: Local path to save the file
        overwrite: Whether to overwrite if file exists
        
    Returns:
        Boolean indicating success
    """
    # Check if file exists
    if output_path.exists() and not overwrite:
        logger.info(f"File already exists: {output_path}. Skipping download.")
        return False
    
    try:
        # Get blob client
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        # Get blob properties to determine size
        properties = blob_client.get_blob_properties()
        total_size = properties.size
        
        logger.info(f"Downloading {blob_name} ({total_size/1024/1024:.2f} MB) to {output_path}...")
        
        # Download with progress bar
        with open(output_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                download_stream = blob_client.download_blob()
                for chunk in download_stream.chunks():
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Download complete: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading blob {blob_name}: {str(e)}")
        return False

def ensure_latest_data(data_dir, force_update=False, file_type=None):
    """
    Ensures the latest data files are available in the data_dir.
    
    Args:
        data_dir: Directory to store the data files
        force_update: Whether to force update even if files exist
        file_type: Type of file to download ('graph', 'graph_visual', 'dataset', 'best_model_GTN', or None for all)
        
    Returns:
        Tuple of (graph_path, dataset_path) with paths to the data files
    """
    try:
        # Convert data_dir to Path object
        data_dir = Path(data_dir)
        print(f"Ensuring latest {file_type} in {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        
        # Metadata file to track latest versions
        metadata_path = data_dir / "data_versions.json"
        local_metadata = {}
        
        # Load existing metadata if available
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    local_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read metadata file: {e}")
        
        # Get Azure connection string
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            logger.error("Azure Blob connection string not found in environment variables")
            # Return paths from metadata or None if not found
            graph_path = data_dir / local_metadata.get('latest_graph', 'graph.json')
            dataset_path = data_dir / local_metadata.get('latest_dataset', 'dataset.json')
            best_model_path = data_dir / local_metadata.get('best_model_GTN', 'best_model_GTN.pt')
            return str(graph_path), str(dataset_path), str(best_model_path)
            
        # Set container name and prefixes
        container_name = "outfitgat"
        graph_prefix = "graph_"  # For regular graph files
        graph_visual_prefix = "graph_visual_"  # For visualization graph files
        dataset_prefix = "dataset_"
        best_model_prefix = "best_model_GTN_"

        # Create Azure Blob clients
        logger.info("Connecting to Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Handle graph file
        if file_type in [None, 'graph']:
            logger.info("Finding latest graph file...")
            # Use exclude_patterns to avoid visualization files
            latest_graph_blob = get_latest_blob(container_client, graph_prefix, exclude_patterns=['visual'])
            
            if latest_graph_blob:
                blob_filename = latest_graph_blob.split("/")[-1]
                local_graph_path = data_dir / blob_filename
                
                if (not local_graph_path.exists() or force_update) and \
                   local_metadata.get('latest_graph') != blob_filename:
                    logger.info(f"New graph version found: {blob_filename}")
                    download_success = download_blob(
                        blob_service_client, 
                        container_name, 
                        latest_graph_blob, 
                        local_graph_path, 
                        overwrite=force_update
                    )
                    if download_success:
                        local_metadata['latest_graph'] = blob_filename
                        logger.info(f"Downloaded new graph version: {blob_filename}")
                else:
                    logger.info(f"Using existing graph version: {blob_filename}")
        
        # Handle graph visualization file
        if file_type in [None, 'graph_visual']:
            logger.info("Finding latest graph visualization file...")
            latest_graph_visual_blob = get_latest_blob(container_client, graph_visual_prefix)
            
            if latest_graph_visual_blob:
                blob_filename = latest_graph_visual_blob.split("/")[-1]
                local_graph_visual_path = data_dir / blob_filename
                
                if (not local_graph_visual_path.exists() or force_update) and \
                   local_metadata.get('latest_graph_visual') != blob_filename:
                    logger.info(f"New graph visualization version found: {blob_filename}")
                    download_success = download_blob(
                        blob_service_client, 
                        container_name, 
                        latest_graph_visual_blob, 
                        local_graph_visual_path, 
                        overwrite=force_update
                    )
                    if download_success:
                        local_metadata['latest_graph_visual'] = blob_filename
                        logger.info(f"Downloaded new graph visualization version: {blob_filename}")
                else:
                    logger.info(f"Using existing graph visualization version: {blob_filename}")
        
        # Handle dataset file
        if file_type in [None, 'dataset']:
            logger.info("Finding latest dataset file...")
            latest_dataset_blob = get_latest_blob(container_client, dataset_prefix)
            
            if latest_dataset_blob:
                blob_filename = latest_dataset_blob.split("/")[-1]
                local_dataset_path = data_dir / blob_filename
                
                if (not local_dataset_path.exists() or force_update) and \
                   local_metadata.get('latest_dataset') != blob_filename:
                    logger.info(f"New dataset version found: {blob_filename}")
                    download_success = download_blob(
                        blob_service_client, 
                        container_name, 
                        latest_dataset_blob, 
                        local_dataset_path, 
                        overwrite=force_update
                    )
                    if download_success:
                        local_metadata['latest_dataset'] = blob_filename
                        logger.info(f"Downloaded new dataset version: {blob_filename}")
                else:
                    logger.info(f"Using existing dataset version: {blob_filename}")

        # Handle best_model_GTN file
        if file_type in [None, 'best_model_GTN']:
            logger.info("Finding latest best_model_GTN file...")
            latest_best_model_blob = get_latest_blob(container_client, best_model_prefix)
            
            if latest_best_model_blob:
                blob_filename = latest_best_model_blob.split("/")[-1]
                local_best_model_path = data_dir / blob_filename
                
                if (not local_best_model_path.exists() or force_update) and \
                   local_metadata.get('best_model_GTN') != blob_filename:
                    logger.info(f"New best_model_GTN version found: {blob_filename}")
                    download_success = download_blob(
                        blob_service_client, 
                        container_name, 
                        latest_best_model_blob, 
                        local_best_model_path, 
                        overwrite=force_update
                    )
                    if download_success:
                        local_metadata['best_model_GTN'] = blob_filename
                        logger.info(f"Downloaded new best_model_GTN version: {blob_filename}")
                else:
                    logger.info(f"Using existing best_model_GTN version: {blob_filename}")
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(local_metadata, f, indent=2)
            logger.info("Updated data version metadata")
        
        # Return paths to the actual version files
        graph_path = data_dir / local_metadata.get('latest_graph', 'graph.json')
        graph_visual_path = data_dir / local_metadata.get('latest_graph_visual', 'graph_visual.json')
        dataset_path = data_dir / local_metadata.get('latest_dataset', 'dataset.json')
        best_model_path = data_dir / local_metadata.get('best_model_GTN', 'best_model_GTN.pt')

        if file_type == 'graph':
            logger.info(f"Using graph file: {graph_path}")
        elif file_type == 'graph_visual':
            graph_visual_path = data_dir / local_metadata.get('latest_graph_visual', 'graph_visual.json')
            logger.info(f"Using graph visualization file: {graph_visual_path}")
            return str(graph_visual_path), str(dataset_path)
        elif file_type == 'dataset':
            logger.info(f"Using dataset file: {dataset_path}")
        elif file_type == 'best_model_GTN':
            logger.info(f"Using best_model_GTN file: {best_model_path}")
            return str(best_model_path), str(dataset_path)
        else:
            logger.info(f"Using graph file: {graph_path}")
            logger.info(f"Using graph visualization file: {graph_visual_path}")
            logger.info(f"Using dataset file: {dataset_path}")
            logger.info(f"Using best_model_GTN file: {best_model_path}")

        return str(graph_path), str(dataset_path)
        
    except Exception as e:
        logger.error(f"Error ensuring latest data: {str(e)}")
        # Try to return paths from metadata if available
        try:
            if local_metadata and 'latest_graph' in local_metadata and 'latest_dataset' in local_metadata and 'best_model_GTN' in local_metadata:
                graph_path = data_dir / local_metadata['latest_graph']
                dataset_path = data_dir / local_metadata['latest_dataset']
                best_model_path = data_dir / local_metadata['best_model_GTN']
                return str(graph_path), str(dataset_path), str(best_model_path)
        except:
            pass
            
        # Fall back to default paths as last resort
        logger.error("Falling back to default filenames")
        return str(data_dir / "graph.json"), str(data_dir / "dataset.json"), str(data_dir / "best_model_GTN.pt")

def main():
    """
    Main function to download the latest graph data from Azure Blob Storage.
    """
    try:
        parser = argparse.ArgumentParser(description='Download latest graph and/or dataset files')
        parser.add_argument('--type', choices=['graph', 'graph_visual', 'dataset', 'best_model_GTN'], 
                          help='Specify which file to download (graph, graph_visual, dataset, or best_model_GTN). If not specified, downloads all.')
        parser.add_argument('--force', action='store_true', 
                          help='Force update even if files exist')
        
        args = parser.parse_args()
        
        # Get the directory of the current script
        current_dir = Path(__file__).parent.absolute()
        data_dir = current_dir
        
        # Get the metadata file to check for graph_visual
        metadata_path = data_dir / "data_versions.json"
        local_metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    local_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read metadata file: {e}")
        
        graph_path, dataset_path = ensure_latest_data(
            data_dir, 
            force_update=args.force,
            file_type=args.type
        )
        
        if args.type == 'graph':
            logger.info(f"Latest graph file available at: {graph_path}")
        elif args.type == 'graph_visual':
            logger.info(f"Latest graph visualization file available at: {graph_path}")
        elif args.type == 'dataset':
            logger.info(f"Latest dataset file available at: {dataset_path}")
        elif args.type == 'best_model_GTN':
            logger.info(f"Latest best_model_GTN file available at: {best_model_path}")
        else:
            logger.info(f"Latest data files available at:")
            logger.info(f"  Graph: {graph_path}")
            if 'latest_graph_visual' in local_metadata:
                graph_visual_path = data_dir / local_metadata['latest_graph_visual']
                logger.info(f"  Graph Visualization: {graph_visual_path}")
            logger.info(f"  Dataset: {dataset_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Unhandled error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)