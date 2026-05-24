# Time-MoE-LLF

This repository is a submodule of [LLF](https://github.com/kimzemian/LLF). It runs a KNN ablation on [Time-MoE](https://github.com/Time-MoE/Time-MoE) embeddings for the datasets released in *Lead-Lag Forecasting for Social Platforms: ArXiv and GitHub* (datasets available at [lead-lag-forecasting.github.io](https://lead-lag-forecasting.github.io/)).

## Overview

This repository extracts intermediate and final hidden-state embeddings from a pretrained Time-MoE model, then evaluates them with a K-Nearest Neighbors ablation study. The pipeline supports four benchmark configurations:

| Case | Dataset | Target |
|---|---|---|
| `arxiv` | ArXiv | Citations |
| `arxiv_accesses` | ArXiv | Accesses |
| `github` | GitHub | Forks |
| `github_pushes` | GitHub | Pushes |

## Requirements

- Python 3.8+
- CUDA-capable GPU(s)

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional packages needed at runtime: `faiss-cpu` (or `faiss-gpu`).

## Usage

### 1. Generate embeddings

Edit `run.sh` to set paths for your data, cache, and output directories, then run:

```bash
bash run.sh
```

This will:
1. Run `bench_arxiv.py` / `bench_github.py` to extract embeddings from the Time-MoE model (one GPU per task).
2. Run `concat_embed.py` to consolidate batch-wise outputs into HuggingFace datasets.

### 2. Run KNN evaluation

```bash
python knn2.py --case <case>
```

where `<case>` is one of `arxiv`, `arxiv_accesses`, `github`, or `github_pushes`.

## Project Structure

```
├── run.sh                  # End-to-end pipeline script
├── bench_arxiv.py          # Embedding extraction for ArXiv
├── bench_github.py         # Embedding extraction for GitHub
├── concat_embed.py         # Consolidate embeddings into HF datasets
├── knn2.py                 # KNN ablation evaluation
├── lr_manual.py            # Logistic regression baseline
├── baseline_datasets.py    # ArXiv/GitHub dataset loaders
├── main.py                 # Time-MoE training entry point
├── run_eval.py             # Time-MoE evaluation (MSE/MAE)
├── torch_dist_run.py       # Distributed training launcher
├── requirements.txt
└── time_moe/               # Core Time-MoE package
    ├── models/             # Model architecture & config
    ├── datasets/           # Dataset implementations
    ├── trainer/            # HuggingFace trainer wrapper
    └── utils/              # Logging & distributed utilities
```

## Acknowledgements

Built on [Time-MoE](https://github.com/Time-MoE/Time-MoE) (Maple728/TimeMoE-50M).
