# OutfitGTN Inference Guide

This guide explains how to use the three inference scripts for different deployment scenarios of the OutfitGTN model.

## Setup

Before running any of the inference scripts, make sure you have:

1. Trained the OutfitGTN model (using `train_GTN.py`)
2. Have the model checkpoint saved (typically in the `checkpoints` directory)
3. Have the necessary data files (graph data, embeddings, etc.)

## 1. Single-Item Recommendation (Static Embedding)

This approach pre-computes item embeddings for your entire catalog by using synthetic graph construction. It's the most lightweight deployment option, ideal for "similar items" recommendations.

### Step 1: Generate Item Embeddings

```bash
python inference_static_embedding.py --config config/config_GTN.yaml --checkpoint checkpoints/best_model_GTN.pt --graph_path data_source/graph.json --output_path data/embeddings.npz --num_similar 10
```

Parameters:
- `--config`: Path to the model configuration file
- `--checkpoint`: Path to the trained model checkpoint
- `--graph_path`: Path to the graph data file
- `--output_path`: Path to save the computed embeddings
- `--num_similar`: Number of similar items to use for synthetic graph construction
- `--batch_size`: Batch size for processing (default: 32)

This will save the embeddings to the specified output path in `.npz` format.

### Usage in Production

Once you have generated the embeddings, you can use them in a vector database (like MongoDB Atlas, Pinecone, or Qdrant) for efficient similarity search. The script includes a simple example of finding similar items at the end.

## 2. Multi-Item Recommendation (Virtual Outfit Node)

This approach creates a virtual outfit node from multiple selected items to recommend compatible items to complete an outfit.

### Usage

```bash
python inference_virtual_outfit.py --config config/config_GTN.yaml --checkpoint checkpoints/best_model_GTN.pt --embeddings data/embeddings.npz --items "123,456,789" --top_k 5
```

Parameters:
- `--config`: Path to the model configuration file
- `--checkpoint`: Path to the trained model checkpoint
- `--embeddings`: Path to the pre-computed embeddings (from the first script)
- `--items`: Comma-separated list of item IDs that the user has selected
- `--top_k`: Number of recommendations to return (default: 5)

The script will output the top compatible items to complete the outfit.

## 3. Contextual Multi-Item Recommendation

This approach incorporates contextual information alongside items to provide more personalized recommendations based on factors like season, occasion, demographics, etc.

### Step 1: Create Context Embeddings (if needed)

Before using this script, you need context embeddings. You can uncomment the `create_sample_context_embeddings()` function call in the script to generate sample embeddings, or create your own:

```bash
# Uncomment this line in inference_contextual.py before running it:
# create_sample_context_embeddings()
```

This will create a JSON file with random embeddings for various contexts like seasons, occasions, etc.

### Usage

```bash
python inference_contextual.py --config config/config_GTN.yaml --checkpoint checkpoints/best_model_GTN.pt --embeddings data/embeddings.npz --contexts "winter,formal" --items "123,456" --top_k 5
```

Parameters:
- `--config`: Path to the model configuration file
- `--checkpoint`: Path to the trained model checkpoint
- `--embeddings`: Path to the pre-computed embeddings
- `--context_path`: Path to the context embeddings file (default: `data/context_embeddings.json`)
- `--contexts`: Comma-separated list of context tokens (e.g., "winter,formal")
- `--items`: Optional comma-separated list of selected item IDs
- `--top_k`: Number of recommendations to return (default: 5)

You can use this script with or without selected items. If no items are provided, it will recommend items purely based on the context.

## Performance Considerations

1. **Single-Item Recommendation**: 
   - Compute embedding once per item (offline)
   - Sub-second retrieval time using a vector database
   - Very scalable for large catalogs

2. **Multi-Item Recommendation**:
   - Requires real-time inference
   - More computationally expensive
   - Best for interactive outfit builders

3. **Contextual Recommendation**:
   - Most computationally intensive
   - Provides the most personalized experience
   - Consider caching common context+item combinations

## Example Workflow

A typical workflow might include:

1. Use the static embedding approach to pre-compute embeddings for your entire catalog
2. Use these embeddings for standard "similar items" recommendations
3. For users building outfits, use the virtual outfit approach
4. For personalized recommendations, use the contextual approach with user-specific context

## Troubleshooting

- If you get CUDA out-of-memory errors, reduce the batch size
- Ensure your item IDs match those in the original graph data
- For large catalogs, consider sharding the embedding computation 