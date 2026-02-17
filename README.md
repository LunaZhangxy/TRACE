# Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction

## Overview

TRACE consists of three core components:

| Component | Symbol | Description |
|---|---|---|
| **Static Intent Estimator** | f_θ(y\|x) | Captures intrinsic conversion intent from pre-click features |
| **Dynamic Trajectory Estimator** | g_ψ(ξ\|x,y) | Scores how the evolving post-click trajectory aligns with each conversion hypothesis |
| **Retrospective Trajectory Completer** | q_φ(y\|x,ξ) | Provides posterior guidance for unrevealed samples using full-lifecycle patterns |

## Quick Start
### 0. Dataset

Replace the data path in criteo_data.py and taobao_data.py
```
_local_path = "/path/to/data.txt"
```

### 1. Pretraining

```bash
# Criteo
python pretrain.py --dataset criteo --device cuda:0

# Taobao
python pretrain.py --dataset taobao --device cuda:0
```

Checkpoints are saved to `./pretrain_model/`.

### 2. Streaming Evaluation

Run the full streaming train-and-predict loop:

```bash
# TRACE on Criteo
python main.py --params_name criteo_trace

# TRACE on Taobao
python main.py --params_name taobao_trace

# Baselines
python main.py --params_name criteo_ce
python main.py --params_name criteo_oracle
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy
- scikit-learn (for metrics)
