import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import json
import os
from data.fashion_node import FashionNode
from data_source.download_data_source import ensure_latest_data
from concurrent.futures import ThreadPoolExecutor

class FashionDataset(Dataset):
    """Fashion dataset for triplet loss training.
    
    This dataset only includes nodes that have both positive and negative samples,
    as required by the triplet loss function. Nodes missing either type of samples
    are filtered out.
    
    Args:
        nodes: Dictionary of node_id to FashionNode
        positive_samples: Dictionary of node_id to list of positive sample ids
        negative_samples: Dictionary of node_id to list of negative sample ids
        max_pos_samples: Maximum number of positive samples to use per node
        max_neg_samples: Maximum number of negative samples to use per node
        dataset_type: Type of dataset (e.g., "Training", "Validation")
    """
    def __init__(self, nodes, positive_samples, negative_samples, max_pos_samples=2, max_neg_samples=2, dataset_type="Unknown"):
        super().__init__()
        self.nodes = nodes
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.max_pos_samples = max_pos_samples
        self.max_neg_samples = max_neg_samples
        self.dataset_type = dataset_type  # New parameter to track dataset type
        
        # Filter nodes with valid samples
        self.node_list = [
            node_id for node_id in positive_samples.keys() 
            if positive_samples[node_id] and negative_samples[node_id]
        ]
        
        # Enhanced statistics printing
        print(f"\n=== {self.dataset_type} Dataset Statistics ===")
        print(f"Total nodes in graph: {len(nodes)}")
        print(f"Nodes with valid samples (pos+neg): {len(self.node_list)}")
    
    def __len__(self):
        return len(self.node_list)
    
    def __getitem__(self, idx):
        node_id = self.node_list[idx]
        node = self.nodes[node_id]
        
        # Get available samples
        available_pos = self.positive_samples[node_id]
        available_neg = self.negative_samples[node_id]
        
        # Take min(available, max_samples) without replacement
        num_pos = min(len(available_pos), self.max_pos_samples)
        num_neg = min(len(available_neg), self.max_neg_samples)
        
        pos_ids = np.random.choice(available_pos, size=num_pos, replace=False)
        neg_ids = np.random.choice(available_neg, size=num_neg, replace=False)
        
        # Create separate local graphs
        # 1. Query node local graph
        query_local_nodes = set([node_id] + node.neighbors)
        query_id_to_local = {nid: idx for idx, nid in enumerate(query_local_nodes)}
        
        # 2. Positive nodes local graphs (only for actual samples)
        pos_local_nodes = []
        pos_id_to_local = []
        for pid in pos_ids:
            pos_node = self.nodes[int(pid)]
            local_nodes = set([int(pid)] + pos_node.neighbors)
            id_to_local = {nid: idx for idx, nid in enumerate(local_nodes)}
            pos_local_nodes.append(local_nodes)
            pos_id_to_local.append(id_to_local)
        
        # 3. Negative nodes local graphs (only for actual samples)
        neg_local_nodes = []
        neg_id_to_local = []
        for nid in neg_ids:
            neg_node = self.nodes[int(nid)]
            local_nodes = set([int(nid)] + neg_node.neighbors)
            id_to_local = {nid: idx for idx, nid in enumerate(local_nodes)}
            neg_local_nodes.append(local_nodes)
            neg_id_to_local.append(id_to_local)
        
        # Create edge indices and features for each graph
        query_edges = self._create_edges(node_id, query_local_nodes, query_id_to_local)
        query_features = self._create_features(query_local_nodes)
        query_data = Data(
            x=query_features,
            edge_index=query_edges,
            node_type=self._get_node_types(query_local_nodes),
            root_idx=torch.tensor([query_id_to_local[node_id]])
        )
        
        # Create Data objects only for actual samples
        pos_data = [Data(
            x=self._create_features(nodes),
            edge_index=self._create_edges(pid, nodes, id_to_local),
            node_type=self._get_node_types(nodes),
            root_idx=torch.tensor([id_to_local[int(pid)]])
        ) for pid, nodes, id_to_local in zip(pos_ids, pos_local_nodes, pos_id_to_local)]
        
        neg_data = [Data(
            x=self._create_features(nodes),
            edge_index=self._create_edges(nid, nodes, id_to_local),
            node_type=self._get_node_types(nodes),
            root_idx=torch.tensor([id_to_local[int(nid)]])
        ) for nid, nodes, id_to_local in zip(neg_ids, neg_local_nodes, neg_id_to_local)]
        
        return query_data, pos_data, neg_data
    
    def _create_edges(self, root_id, local_nodes, id_to_local):
        edges = []
        for src in local_nodes:
            for dst in self.nodes[src].neighbors:
                if dst in local_nodes:
                    edges.append([id_to_local[src], id_to_local[dst]])
        
        # Ensure edge_index is correctly shaped as [2, num_edges] from the start
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        edge_tensor = torch.tensor(edges, dtype=torch.long)
        return edge_tensor.t().contiguous()  # Shape: [2, num_edges]
    
    def _create_features(self, nodes):
        return torch.stack([
            torch.tensor(self.nodes[nid].embedding, dtype=torch.float)
            for nid in nodes
        ])
    
    def _get_node_types(self, nodes):
        return torch.tensor([
            1 if self.nodes[nid].node_type == "item" else 0
            for nid in nodes
        ], dtype=torch.long)

    def _get_edge_index(self, node_id):
        """Get edge indices for the local subgraph around node_id"""
        edges = []
        node = self.nodes[node_id]
        
        # Create a mapping of large IDs to smaller indices
        # This should only include the current node and its neighbors
        unique_ids = set([int(node.id)] + [int(n) for n in node.neighbors])
        id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
        
        # Use mapped indices instead of raw IDs
        source_idx = id_to_idx[int(node.id)]
        for neighbor_id in node.neighbors:
            target_idx = id_to_idx[int(neighbor_id)]
            # Each neighbor creates 2 edges (bidirectional)
            edges.append([source_idx, target_idx])  # forward edge
            edges.append([target_idx, source_idx])  # backward edge
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

class FashionDataLoader:
    def __init__(self, batch_size, max_pos_samples=2, max_neg_samples=2, auto_download=True, force_update=False):
        self.batch_size = batch_size
        self.max_pos_samples = max_pos_samples
        self.max_neg_samples = max_neg_samples
        self.auto_download = auto_download
        self.force_update = force_update
        self.nodes = {}
        self.positive_samples = {}
        self.negative_samples = {}
        
    def load_data(self, graph_path=None, dataset_path=None, data_dir=None):
        """Load and preprocess the data, optionally downloading latest data first.
        
        Args:
            graph_path: Path to graph file (if not using auto_download)
            dataset_path: Path to dataset file (if not using auto_download)
            data_dir: Directory to store/find data files (used with auto_download)
            
        Returns:
            self (for method chaining)
        """
        # Handle auto-download if enabled
        if self.auto_download:
            try:
                # If paths not specified, use data_dir for both download and loading
                if not (graph_path and dataset_path):
                    if not data_dir:
                        data_dir = "./data"  # Default data directory
                    
                    print(f"Checking for latest data in blob storage...")
                    graph_path,  = ensure_latest_data(data_dir, self.force_update, "graph")
                    dataset_path = ensure_latest_data(data_dir, self.force_update, "dataset")
                    print(f"Using data files:")
                    print(f"  - Graph: {graph_path}")
                    print(f"  - Dataset: {dataset_path}")

                    # Verify files exist
                    if not os.path.exists(graph_path):
                        raise FileNotFoundError(f"Graph file not found: {graph_path}")
                    if not os.path.exists(dataset_path):
                        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
                    
            except ImportError:
                print("Warning: download_data_source module not found. Using specified paths.")
            except Exception as e:
                print(f"Warning: Failed to download latest data: {str(e)}")
                # If auto_download fails but path arguments were provided, fall back to those
                if not (graph_path and dataset_path):
                    raise ValueError("Failed to download data and no manual paths provided")
                
        # Ensure paths are provided
        if not graph_path or not dataset_path:
            raise ValueError("Please provide graph_path and dataset_path, or set auto_download=True with data_dir")
        
        # Parallelize file loading
        def load_json(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
            
        with ThreadPoolExecutor() as executor:
            graph_future = executor.submit(load_json, graph_path)
            dataset_future = executor.submit(load_json, dataset_path)
            
            # Get results (will wait until both are complete)
            graph_data = graph_future.result()
            dataset_data = dataset_future.result()
            
        # Process nodes
        self.nodes = {int(node_data['id']): FashionNode(node_data) 
                     for node_data in graph_data['nodes']}
        
        try:
            # Process samples
            self.positive_samples = {int(k): [int(v) for v in data['positive_samples']] 
                                for k, data in dataset_data.items()}
            self.negative_samples = {int(k): [int(v) for v in data['negative_samples']] 
                                for k, data in dataset_data.items()}
        except Exception as e:
            raise ValueError(f"Failed to process dataset samples: {str(e)}")
        
        return self  # Enable method chaining

    def get_dataloader(self, train=True, shuffle=True):
        """Initialize and return the DataLoader"""
        if not self.nodes:
            raise ValueError("Please call load_data() before getting the dataloader")
            
        # Create dataset instance with appropriate type
        dataset_type = "Training" if train else "Testing"
        dataset = FashionDataset(
            nodes=self.nodes,
            positive_samples=self.positive_samples if train else {},
            negative_samples=self.negative_samples if train else {},
            max_pos_samples=self.max_pos_samples,
            max_neg_samples=self.max_neg_samples,
            dataset_type=dataset_type  # Add dataset type
        )
        
        # Create and return DataLoader with optimal settings
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=3,
            pin_memory=True,
            pin_memory_device='cuda',  # Add this parameter
            prefetch_factor=1  # Add this parameter
        )

    def get_train_val_dataloaders(self, val_ratio=0.2, shuffle=True):
        """Get separate train and validation dataloaders"""
        if not self.nodes:
            raise ValueError("Please call load_data() before getting the dataloaders")
            
        # Get all node IDs
        all_ids = list(self.positive_samples.keys())
        
        # Shuffle if requested
        if shuffle:
            np.random.shuffle(all_ids)
            
        # Split into train and validation
        split_idx = int(len(all_ids) * (1 - val_ratio))
        train_ids = all_ids[:split_idx]
        val_ids = all_ids[split_idx:]
        
        # Create train dataset with dataset_type parameter
        train_dataset = FashionDataset(
            nodes=self.nodes,
            positive_samples={k: self.positive_samples[k] for k in train_ids},
            negative_samples={k: self.negative_samples[k] for k in train_ids},
            max_pos_samples=self.max_pos_samples,
            max_neg_samples=self.max_neg_samples,
            dataset_type="Training"  # Add dataset type
        )
        
        # Create validation dataset with dataset_type parameter
        val_dataset = FashionDataset(
            nodes=self.nodes,
            positive_samples={k: self.positive_samples[k] for k in val_ids},
            negative_samples={k: self.negative_samples[k] for k in val_ids},
            max_pos_samples=self.max_pos_samples,
            max_neg_samples=self.max_neg_samples,
            dataset_type="Validation"  # Add dataset type
        )
        
        # Return both dataloaders
        return (
            DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=3,
                pin_memory=True,
                pin_memory_device='cuda',  # Add this parameter
                prefetch_factor=1  # Add this parameter
            ),
            DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=3,
                pin_memory=True,
                pin_memory_device='cuda',  # Add this parameter 
                prefetch_factor=1  # Add this parameter
            )
        )

    def _get_node_features(self, node_id):
        node = self.nodes[node_id]
        # Convert numpy array to torch tensor
        features = torch.tensor(node.embedding, dtype=torch.float)
        return features

    def __getitem__(self, idx):
        node_id = self.node_ids[idx]
        x = self._get_node_features(node_id)
        
        # Ensure edge_index is properly formatted from the start
        edge_index = self._get_edge_index(node_id)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = edge_index.view(2, -1)  # Ensure [2, num_edges] shape
        
        data = Data(
            x=x,
            edge_index=edge_index,
            node_type=self.node_types[idx]
        )
        return data