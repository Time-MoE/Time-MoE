<div align="center">
  <h3><b>Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/Time-MoE/Time-MoE?color=green)
![](https://img.shields.io/github/stars/Time-MoE/Time-MoE?color=yellow)
![](https://img.shields.io/github/forks/Time-MoE/Time-MoE?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">

**[<a href="https://arxiv.org/pdf/2409.16040">Paper Page</a>]**
**[<a href="https://mp.weixin.qq.com/s/LaYn0IJAOlN9Ufp_qus96Q">中文解读</a>]**

</div>

<p align="center">

<img src="./figures/timemoe-logo.png" width="70">

</p>


> 1️⃣ Time-MoE is **the first work to scale time series foundation models up to 2.4 billion parameters**, trained from
> scratch.

> 2️⃣ Time-300B is **the largest open-access time series data collection** comprising over 300 billion time points across more than 9 domains.

## TODO list
- [ ] Enable fine-tuning of TimeMoE for forecasting with dynamic features and support time-series classification.

## Updates/News:

🚩 **News** (Oct 2024): Time-MoE introduction in [Chinese](https://mp.weixin.qq.com/s/LaYn0IJAOlN9Ufp_qus96Q) is now available!

🚩 **News** (Oct 2024): Time-300B dataset is now available 
on [🤗 Hugging Face](https://huggingface.co/datasets/Maple728/Time-300B)!

🚩 **News** (Oct 2024): Time-MoE (large) is now available
on [🤗 Hugging Face](https://huggingface.co/Maple728/TimeMoE-200M)!

🚩 **News** (Sept 2024): Time-MoE (base) is now available
on [🤗 Hugging Face](https://huggingface.co/Maple728/TimeMoE-50M)!

🚩 **News** (Sept 2024): Time-MoE preprint has been made available on [arXiv](https://arxiv.org/pdf/2409.16040)!

## Introduction

Time-MoE comprises a family of decoder-only time series foundation models with a mixture-of-experts architecture,
designed to operate in an auto-regressive manner, enabling universal forecasting with arbitrary prediction horizons and
context lengths of up to 4096.

<p align="center">
    <img src="figures/time_moe_framework.png" alt="" align="center" width="700px" />
</p>


<div align="center">

### 🗂️ Time-MoE Model Card

| Model                                                                | Activated Params. | Total Params. |
|----------------------------------------------------------------------|-------------------|---------------|
| [**Time-MoE (base)**](https://huggingface.co/Maple728/TimeMoE-50M)   | 50M               | 113M          |
| [**Time-MoE (large)**](https://huggingface.co/Maple728/TimeMoE-200M) | 200M              | 453M          |
| **Time-MoE (ultra)**                                                 | 1.1B              | 2.4B          |

</div>

## 📚 Training Data

Time-300B dataset is available on [🤗 Hugging Face](https://huggingface.co/datasets/Maple728/Time-300B)!

Here's an example of how to use this dataset:
```python
import random
from time_moe.datasets.time_moe_dataset import TimeMoEDataset

ds = TimeMoEDataset('Time-300B')
seq_idx = random.randint(0, len(ds) - 1)
seq = ds[seq_idx]
```

This code snippet shows how to load a random data sequence from the Time-300B dataset. First, download the dataset to the local 'Time-300B' folder, import the TimeMoEDataset class from time_moe.datasets, instantiate the class, and finally retrieve a sequence using a random index.

## 🚀 Getting Started

### Installation

1. Install Python 3.10+, and then install the dependencies:

```shell
pip install -r requirements.txt
```

**Note: Time-MoE requires `transformers==4.40.1` .**

2. [Optional but **recommended**] Install [flash-attn](https://github.com/Dao-AILab/flash-attention) for faster training and inference speeds with reduced memory usage.

```shell
pip install flash-attn==2.6.3
```

### Making Forecasts

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

# use it when the flash-attn is available
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)

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

+ If the sequences are normalized already:

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

# use it when the flash-attn is available
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]
```

### Evaluation

+ Prepare the benchmark datasets.

You can access the well pre-processed datasets
from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place
the downloaded contents under `./dataset`.

+ [Example] Running the follow command to evaluate on ETTh1.

```shell
python run_eval.py -d dataset/ETT-small/ETTh1.csv -p 96
```

### 🔥 Fine-tuning Time-MoE

#### Preparing Your Dataset

To start fine-tuning Time-MoE, your dataset should be converted into a `jsonl` format. Each line represents a time-series data as a dictionary object, where the `sequence` field contains a list of time-series observations. For example:

```jsonl
{"sequence": [1.0, 2.0, 3.0, ...]}
{"sequence": [11.0, 22.0, 33.0, ...]}
```

You have the flexibility to save your converted data in `jsonl`, `json`, or `pickle` format. If you are using the [Time-300B](https://huggingface.co/datasets/Maple728/Time-300B) dataset, you can proceed without any additional preprocessing.

#### Training

**Single GPU**

For training with a single GPU, execute the following command and ensure to replace `<data_path>` with the path to your prepared dataset:

```bash
python main.py -d <data_path>
```

**Single Node Multi-GPU**

To leverage multiple GPUs on a single node, use this command:

```bash
python torch_dist_run.py main.py -d <data_path>
```

**Multi-Node Multi-GPU**

For training across multiple nodes, additional environment configurations are necessary to facilitate inter-node communication:

```bash
export MASTER_ADDR=<master_addr>
export MASTER_PORT=<master_port>
export WORLD_SIZE=<world_size>
export RANK=<rank>

python torch_dist_run.py main.py -d <data_path>
```

To explore additional command-line arguments and their usage, invoke the help command:

```bash
python main.py --help
```

## Citation

> 🙋 Please let us know if you find out a mistake or have any suggestions!

> 🌟 If you find the Time-MoE models helpful in your research, please consider to star this repository and cite the
> corresponding [paper](https://arxiv.org/pdf/2409.16040):

```
@misc{shi2024timemoe,
      title={Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts}, 
      author={Xiaoming Shi and Shiyu Wang and Yuqi Nie and Dianqi Li and Zhou Ye and Qingsong Wen and Ming Jin},
      year={2024},
      eprint={2409.16040},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2409.16040}, 
}
```

## Related Resources
* Foundation Models for Time Series Analysis: A Tutorial and Survey, in *KDD*
  2024. [\[paper\]](https://arxiv.org/abs/2403.14735) [\[Tutorial\]](https://wenhaomin.github.io/FM4TS.github.io/)
* What Can Large Language Models Tell Us about Time Series Analysis, in *ICML*
  2024. [\[paper\]](https://arxiv.org/abs/2402.02713)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects, in *TPAMI*
  2024. [\[paper\]](https://arxiv.org/abs/2306.10125) [\[Website\]](https://github.com/qingsongedu/Awesome-SSL4TS)
* A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection, in
  *TPAMI* 2024. [\[paper\]](https://arxiv.org/abs/2307.03759) [\[Website\]](https://github.com/KimMeen/Awesome-GNN4TS)
* Transformers in Time Series: A Survey, in *IJCAI*
  2023. [\[paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* Towards Neural Scaling Laws for Time Series Foundation Models, arXiv 2024. [\[paper\]](https://arxiv.org/pdf/2410.12360)
* TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis, in arXiv 2024. [\[paper\]](https://arxiv.org/abs/2410.16032) [\[GitHub Repo\]](https://github.com/kwuking/TimeMixer)


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- Time-LLM [\[repo\]](https://github.com/KimMeen/Time-LLM)
- TimeMixer [\[repo\]](https://github.com/kwuking/TimeMixer)
- Time-Series-Library [\[repo\]](https://github.com/thuml/Time-Series-Library)
- Large (Language) Models and Foundation Models (LLM, LM, FM) for Time Series and Spatio-Temporal
  Data [\[repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

## License

This project is licensed under the Apache-2.0 License.
