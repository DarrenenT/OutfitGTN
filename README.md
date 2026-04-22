# OutfitGTN

A **Graph Transformer Network** for learning compatibility-aware outfit and item embeddings. Given a heterogeneous fashion graph (items ↔ outfits) with text-derived node features, OutfitGTN produces a 768-d metric-space embedding where compatible items and outfits lie close together and incompatible ones are pushed apart.

Trained with a custom triplet objective (top-*k* hard-negative mining + margin/penalty terms) under mixed precision on a single H100. Serves embeddings behind a FastAPI endpoint.

## Model

`models/OutfitGTN.py` — a two-layer graph transformer over a heterogeneous item/outfit graph.

### Graph structure

Each training example is a **local subgraph** extracted on the fly: a root node plus its 1-hop neighborhood. Each node carries:

- A **type tag** (`item` or `outfit`) dispatched to one of two MLP encoders
- A precomputed 1536-d text embedding (OpenAI `text-embedding-3-small`) describing the garment / outfit

The root of every subgraph is the node being embedded; the same mechanism is used to build query, positive, and negative graphs per training step.

### Architecture

```
 1536-d text embedding
        │
        ▼
┌────────────────────┐   ┌────────────────────┐
│ item encoder (MLP) │   │ outfit encoder (MLP)│    dispatched by node_type
└────────────────────┘   └────────────────────┘
        │ 1024-d
        ▼
  TransformerConv  (12 heads, learnable β skip, root_weight=True)   →  1-hop
        │
        ▼
  TransformerConv  (12 heads, learnable β skip, root_weight=True)   →  2-hop
        │
        ▼
  α₁·(1-hop) + α₂·(2-hop)     α = softmax over a learnable Parameter
        │
        ▼
  σ-gated self-interaction  (feature gate)
        │
        ▼
  + residual skip from 1-hop
        │
        ▼
  LayerNorm + Linear → 768-d
        │
        ▼
  L2-normalize,  ÷ temperature
```

Key design choices:

- **`TransformerConv` (Shi et al. 2021)** rather than plain GAT — multi-head attention with a **learnable skip-connection coefficient β** per message-passing step.
- **Learnable hop attention** softmaxes a 2-vector `[α₁, α₂]` so the model can decide how much 1-hop vs 2-hop context to trust per training run (initialized to 0.4 / 0.6).
- **Feature gating**: a sigmoid layer over the post-MP representation, element-wise multiplied in — lets the model attenuate uninformative channels before projection.
- **DropEdge regularization** during training (`enable_drop`, `drop_rate=0.1`) — randomly drops graph edges per batch to prevent over-reliance on any single neighbor.
- **Temperature-scaled, L2-normalized** outputs so cosine similarity is the natural distance in the embedding space.
- **On-the-fly batched graph construction** via `Batch.from_data_list` to amortize message-passing cost across all queries, positives, and negatives in a mini-batch.

A GAT baseline (`models/OutfitGAT.py`) with the same training harness is included for ablation.

## Training

`trainers/trainer.py`, driven by `train_GTN.py` + `config/config_GTN.yaml`.

### Loss

A **triplet loss with hard-negative mining and explicit saturation penalties**:

```
loss = triplet_loss + 0.4 · pos_penalty + 0.3 · neg_penalty

   triplet_loss = ReLU( margin − sim(q, p) + mean_topk( sim(q, n) ) )
   pos_penalty  = ReLU( 0.55 − sim(q, p) )         pulls positives past 0.55
   neg_penalty  = ReLU( mean_topk(sim(q,n)) + 0.1 ) pushes hard negatives past −0.1
```

- **Hard-negative mining**: per query, average the top-*k* most similar negatives, with *k* clamped to `[3, 8]` (≈15 % of actual negatives). This matters more than raw ratio because the model quickly learns to separate easy negatives.
- **Dynamic sample count**: queries can have anywhere from 1 to `max_pos_samples` (16) positives and up to `max_neg_samples` (48) negatives — the loss infers the per-query ratio from tensor shapes rather than assuming a fixed one.

### Optimization

- **Adam** (lr `1.5e-5`, weight-decay `3e-4`)
- **Cosine Annealing with Warm Restarts** (`T_0=8`, `T_mult=2`, `η_min=5e-6`) — stronger than plain cosine for this problem, gives the model periodic chances to escape sharp minima.
- **Mixed precision** (`torch.cuda.amp` `GradScaler` + `autocast`) for ~2× throughput on H100.
- **Gradient accumulation** (4 steps by default) so effective batch size stays at 2048 even when a single subgraph batch spikes memory.
- **Checkpoint-on-signal**: `SIGINT` / `SIGTERM` handlers serialize the latest model state before exit, so preempted runs never lose more than one epoch.
- **Early stopping** (patience = 100) on validation loss.

### Logged metrics

Every step writes to TensorBoard:

- `train_loss` / `val_loss`
- `avg_pos_sim` / `avg_neg_sim` (the two numbers you actually care about)
- `pos_sim_std` / `neg_sim_std`
- `embedding_norm` (sanity check that normalization is holding)

## Data pipeline

`data/data_loader.py` + `data/fashion_node.py`.

The dataset is loaded from two JSON artifacts:

- `graph.json` — `{nodes: [{id, type, embedding, neighbors}]}`
- `dataset.json` — `{node_id: {positive_samples: [...], negative_samples: [...]}}`

Positives / negatives are defined externally (not learned from the graph structure) — typically from human-curated outfit labels or from a downstream compatibility signal.

Per training example (in `FashionDataset.__getitem__`):

1. Filter to nodes that have **both** positives and negatives (triplet loss requires both).
2. Sample without replacement: `min(available, max_samples)` positives and negatives.
3. Construct **three separate local subgraphs** (query, each positive, each negative), each with its own `edge_index`, `x`, `node_type`, and `root_idx`.
4. A `Batch.from_data_list` collate in the model fuses them for a single pass of message-passing.

Loading is parallelized with `ThreadPoolExecutor`, and the DataLoader uses `num_workers=3`, `pin_memory=True`, and `prefetch_factor=1` to keep the H100 fed.

## Inference

`inference/api_server.py` — FastAPI server with API-key auth (`X-API-Key` header).

- `inference_outfit.py` — generate an outfit embedding from text (calls Azure OpenAI for the 1536-d text embedding, then runs OutfitGTN).
- `inference_items_batch.py` — batch encode a list of items.
- `calc_avg_outfit_emb.py` — computes the training-set mean outfit embedding, cached in `data_source/average_outfit_embedding.npz` and loaded at startup as a neutral baseline for cold-start queries.

See `inference/README_INFERENCE.md` for the API contract.

## Repository layout

```
models/        OutfitGTN and OutfitGAT architectures
data/          FashionDataset, FashionDataLoader, FashionNode
trainers/      OutfitTrainer (loss, mixed precision, checkpointing)
config/        YAML configs for GTN / GAT training
train_GTN.py   training entry point (GTN)
train_GAT.py   training entry point (GAT ablation)
inference/     embedding generation + FastAPI server
data_source/   data download helpers, cached average outfit embedding
utils/         GPU configuration utilities
```

## Setup

```bash
pip install -r requirements.txt      # torch 2.1.0 + cu121, torch-geometric stack
python train_GTN.py                   # reads config/config_GTN.yaml
uvicorn inference.api_server:app      # serve embeddings
```

`requirements.txt` pins `torch-scatter` / `torch-sparse` / `torch-cluster` against the PyG wheel index for CUDA 12.1 + PyTorch 2.1. Adjust the `--find-links` line if your environment differs.

Required env vars for inference: `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_KEY`, `API_KEY`.

## Status

Research project — no longer under active development. Published as a reference implementation of a graph-transformer approach to outfit compatibility with non-trivial training mechanics.

## License

MIT — see [LICENSE](LICENSE).
