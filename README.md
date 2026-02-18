# Follow the TRACE: Exploiting Post-Click Trajectories for Online Delayed Conversion Rate Prediction

## Quick Start
### 0. Dataset

Replace the data path in criteo_data.py and taobao_data.py:
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
- NumPy ≥ 1.21
- scikit-learn ≥ 1.0
- SciPy ≥ 1.7
- pandas ≥ 1.3
