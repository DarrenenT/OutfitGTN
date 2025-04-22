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

def get_latest_blob(container_client, prefix):
    """
    Find the latest blob in a container by date in filename.
    
    Args:
        container_client: Azure Blob Container client
        prefix: Prefix to filter blobs (e.g., 'graph_' or 'dataset_')
        
    Returns:
        Name of the latest blob or None if not found
    """
    try:
        # List all blobs with the given prefix
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        
        if not blobs:
            logger.warning(f"No blobs found with prefix '{prefix}'")
            return None
            
        # Filter out 'graph_visual_' when looking for 'graph_' files
        if prefix == "graph_":
            blobs = [blob for blob in blobs if not blob.name.startswith("graph_visual_")]
            if not blobs:
                logger.warning("No graph files found after filtering out visual graphs")
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

def ensure_latest_data(data_dir, force_update=False):
    """
    Ensures the latest data files are available in the data_dir.
    
    Args:
        data_dir: Directory to store the data files
        force_update: Whether to force update even if files exist
        
    Returns:
        Tuple of (graph_path, dataset_path) with paths to the data files
    """
    try:
        # Convert data_dir to Path object
        data_dir = Path(data_dir)
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
            return str(graph_path), str(dataset_path)
            
        # Set container name and prefixes
        container_name = "outfitgat"
        graph_prefix = "graph_"
        dataset_prefix = "dataset_"
        
        # Create Azure Blob clients
        logger.info("Connecting to Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Get latest files from Azure
        logger.info("Finding latest graph file...")
        latest_graph_blob = get_latest_blob(container_client, graph_prefix)
        
        logger.info("Finding latest dataset file...")
        latest_dataset_blob = get_latest_blob(container_client, dataset_prefix)
        
        # Handle graph file
        if latest_graph_blob:
            # Extract just the filename part from the full blob path
            blob_filename = latest_graph_blob.split("/")[-1]
            local_graph_path = data_dir / blob_filename
            
            # Check if we already have this version
            if (not local_graph_path.exists() or force_update) and \
               local_metadata.get('latest_graph') != blob_filename:
                # Download the new version
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
        
        # Handle dataset file
        if latest_dataset_blob:
            # Extract filename from blob path
            blob_filename = latest_dataset_blob.split("/")[-1]
            local_dataset_path = data_dir / blob_filename
            
            # Check if we already have this version
            if (not local_dataset_path.exists() or force_update) and \
               local_metadata.get('latest_dataset') != blob_filename:
                # Download the new version
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
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(local_metadata, f, indent=2)
            logger.info("Updated data version metadata")
        
        # Return paths to the actual version files
        graph_path = data_dir / local_metadata.get('latest_graph', 'graph.json')
        dataset_path = data_dir / local_metadata.get('latest_dataset', 'dataset.json')
        
        logger.info(f"Using graph file: {graph_path}")
        logger.info(f"Using dataset file: {dataset_path}")
        
        return str(graph_path), str(dataset_path)
        
    except Exception as e:
        logger.error(f"Error ensuring latest data: {str(e)}")
        # Try to return paths from metadata if available
        try:
            if local_metadata and 'latest_graph' in local_metadata and 'latest_dataset' in local_metadata:
                graph_path = data_dir / local_metadata['latest_graph']
                dataset_path = data_dir / local_metadata['latest_dataset']
                return str(graph_path), str(dataset_path)
        except:
            pass
            
        # Fall back to default paths as last resort
        logger.error("Falling back to default filenames")
        return str(data_dir / "graph.json"), str(data_dir / "dataset.json")

def main():
    """
    Main function to download the latest graph data from Azure Blob Storage.
    """
    try:
        # Get the directory of the current script
        current_dir = Path(__file__).parent.absolute()
        data_dir = current_dir
        
        # Force update when run as script
        graph_path, dataset_path = ensure_latest_data(data_dir, force_update=True)
        
        logger.info(f"Latest data files available at:")
        logger.info(f"  Graph: {graph_path}")
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