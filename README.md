# OutfitGTN

Graph-based deep learning models for outfit representation and compatibility.

OutfitGTN models an outfit as a heterogeneous graph where nodes are fashion items and edges capture co-occurrence / compatibility. Two architectures are implemented:

- **OutfitGAT** — Graph Attention Network baseline
- **OutfitGTN** — Graph Transformer Network with multi-head attention over typed relations

The trained models produce fixed-size outfit embeddings that can be used for retrieval, recommendation, or downstream compatibility scoring.

## Repository layout

```
models/        OutfitGAT and OutfitGTN architectures (PyTorch + torch_geometric)
data/          dataset loaders and fashion graph construction
trainers/      training loop (mixed-precision, cosine warm-restarts, checkpointing)
config/        YAML configs for GAT / GTN training
inference/     embedding generation, FastAPI serving
utils/         GPU utilities
```

## Setup

Requires Python 3.10+, CUDA 12.1, and PyTorch 2.1.

```bash
pip install -r requirements.txt
```

`requirements.txt` pulls `torch-scatter`, `torch-sparse`, and `torch-cluster` from the PyG wheel index for `torch 2.1.0 + cu121` — if you use a different CUDA/Torch combination, adjust the `--find-links` line accordingly.

## Training

```bash
python train_GTN.py    # uses config/config_GTN.yaml
python train_GAT.py    # uses config/config_GAT.yaml
```

Training writes TensorBoard events and per-run logs to `logs/` (gitignored). Checkpoints go to the path configured in the YAML.

## Inference

See `inference/README_INFERENCE.md` for the embedding pipeline.

```bash
# Serve embeddings via FastAPI
uvicorn inference.api_server:app --host 0.0.0.0 --port 8000
```

Set `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_KEY`, and `API_KEY` in a `.env` file before starting.

## Notes

This was a research project and is no longer under active development. Published as a reference implementation for graph-based outfit modeling.

## License

MIT — see [LICENSE](LICENSE).
