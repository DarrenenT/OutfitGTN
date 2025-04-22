import numpy as np

class FashionNode:
    def __init__(self, node_data):
        """
        Initialize a fashion node from graph.json data
        Args:
            node_data (dict): Node data from graph.json containing id, type, embedding, etc.
        """
        self.id = node_data['id'] #in integer
        self.node_type = node_data['type']  # 'item' or 'outfit'
        self.embedding = np.array(node_data['embedding']) # in np array
        self.neighbors = [int(n) for n in node_data.get('neighbors', [])] # in list of integers