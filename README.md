<div align="center">
  <h2><b><img src="figures/timemoe-logo.png" width=25/>Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts </b></h2>
</div>

## Introduction

Time-MoE consists of a family of decoder-only transformer models with a mixture-of-experts architecture, operating in an auto-regressive manner to support any forecasting horizon and accommodate context lengths of up to 4096.

<p align="center">
    <img src="figures/time_moe_framework.png" alt="" align=center />
</p>

## Usage

### Installation
1. Install Python 3.10+, and then install the dependencies:
```shell
pip install -r requirements.txt
```

2. [Optional] Install flash-attn. (For faster training and inference)
```shell
pip install flash-attn==2.6.3
```

### Download Models

### Forecast

General purpose:
```python
import torch
from transformers import AutoModelForCausalLM

context_length = 12
seqs = torch.randn(2, context_length)  # tensor shape is [batch_size, context_length]

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)

# normalize seqs
mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
normed_seqs = (seqs - mean) / std

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]

# inverse normalize
predictions = normed_predictions * std + mean
```

If the sequences are normalized already:
```python
import torch
from transformers import AutoModelForCausalLM

context_length = 12
normed_seqs = torch.randn(2, context_length)  # tensor shape is [batch_size, context_length]

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]
```

### Evaluation

1. Prepare benchmark datasets.

You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`.

2. Running the follow command to evaluate ETTh1 benchmark.

```shell
python run_eval.py -d dataset/ETT-small/ETTh1.csv -p 96
```

