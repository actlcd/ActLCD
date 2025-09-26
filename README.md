# ActLCD: Active Layer-Contrastive Decoding  
**Official implementation for “Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation”**  

[Paper (arXiv)](https://arxiv.org/pdf/2505.23657) · [Project site](https://actlcd.github.io/)


## 📝 Abstract

Modern large language models (LLMs) are susceptible to hallucinations—plausible but factually incorrect outputs. Although existing decoding methods (e.g., contrastive decoding, layer-wise contrast) help reduce this, they often operate uniformly at every step, incurring inefficiencies or overcorrections. We introduce **ActLCD** (Active Layer-Contrastive Decoding), a decoding strategy that dynamically **decides when** to apply layer-wise contrastive adjustments during generation. We frame decoding as a sequential decision problem, using a reinforcement-learned policy to trade off factual precision and fluency. The result: improved factuality across diverse benchmarks with minimal latency overhead.  

## 🚀 Key Features
Novel Decoding Strategy: ActLCD introduces a new method for text generation that actively engages contrastive decoding.

Reinforcement Learning Framework: A reinforcement learning agent learns to decide when to apply the contrastive decoding layers, optimizing for long-term factuality.

State-of-the-Art Performance: ActLCD outperforms existing methods in reducing hallucinations on a wide range of benchmarks.

## 🛠️ Installation
To get started with ActLCD, clone the repository and install the required dependencies:
```
pip install -e transformers-4.46.3
pip install datasets
pip install accelerate
pip install openai # -> only for evaluation
```
## Repository Overview
```
ActLCD/
├── eva_dataset/ # evaluation dataset setup / scripts
├── model/ # model / policy components
├── transformers-4.46.3/ # vendor-pinned transformer utilities
├── dola.py # DoLa & ActLCD implementation
├── gsm8k_eval.py # script for **GSM8K**
├── package_long_truthqa.py # script for **package hallucination & LongFact & TruthfulQA**
├── strqa_eval.py # script for **StrategyQA**
└── README.md
```

## Quick Start
### Baseline
```bash
python gsm8k_eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --output-path ./results/gsm8k_result.json --early-exit-layers -1 --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/TruthfulQA.csv/TruthfulQA.csv --output-path ./results/TruthfulQA_result.json --early-exit-layers -1 --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/longfact_concepts_random.json --output-path ./results/longfact_result.json --early-exit-layers -1 --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/Prompt_Data_Set/Python/LLM_Recent.json --output-path ./results/Python_recent_result.json --early-exit-layers -1 --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/Prompt_Data_Set/JavaScript/JS_LLM_Recent.json --output-path ./results/JS_recent_result.json --early-exit-layers -1 --loaded-model-name -1
python strqa_eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --output-path ./results/strqa_result.json --early-exit-layers -1 --loaded-model-name -1
```

### DoLa
```bash
python gsm8k_eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --output-path ./results/gsm8k_result.json --early-exit-layers low --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/TruthfulQA.csv/TruthfulQA.csv --output-path ./results/TruthfulQA_result.json --early-exit-layers low --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/longfact_concepts_random.json --output-path ./results/longfact_result.json --early-exit-layers low --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/Prompt_Data_Set/Python/LLM_Recent.json --output-path ./results/Python_recent_result.json --early-exit-layers low --loaded-model-name -1
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/Prompt_Data_Set/JavaScript/JS_LLM_Recent.json --output-path ./results/JS_recent_result.json --early-exit-layers low --loaded-model-name -1
python strqa_eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --output-path ./results/strqa_result.json --early-exit-layers low --loaded-model-name -1
```

### ActLCD
```bash
python gsm8k_eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --output-path ./results/gsm8k_result.json --early-exit-layers low --loaded-model-name ./model/llama_bcq_gsm8k-low.pth
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/TruthfulQA.csv/TruthfulQA.csv --output-path ./results/TruthfulQA_result.json --early-exit-layers low --loaded-model-name ./model/llama_bcq_qa.pth
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/longfact_concepts_random.json --output-path ./results/longfact_result.json --early-exit-layers low --loaded-model-name ./model/llama_bcq_long-low.pth
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/Prompt_Data_Set/Python/LLM_Recent.json --output-path ./results/Python_recent_result.json --early-exit-layers low --loaded-model-name ./model/llama_bcq_py_recent.pth
python package_long_truthqa.py --model-name meta-llama/Llama-3.1-8B-Instruct --data-path ./eva_dataset/Prompt_Data_Set/JavaScript/JS_LLM_Recent.json --output-path ./results/JS_recent_result.json --early-exit-layers low --loaded-model-name ./model/llama_bcq_js_recent.pth
python strqa_eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --output-path ./results/strqa_result.json --early-exit-layers low --loaded-model-name ./model/llama_bcq_strqa-low.pth
```

## Reference Repositories
- DoLa: https://github.com/voidism/DoLa
- FastChat: https://github.com/lm-sys/FastChat
- ContrastiveDecoding: https://github.com/XiangLi1999/ContrastiveDecoding
- TruthfulQA: https://github.com/sylinrl/TruthfulQA
- zero_shot_cot: https://github.com/kojima-takeshi188/zero_shot_cot

## 📜 Citation
If you find our work useful, please consider citing our paper:
```
@article{zhang2025active,
  title={Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation},
  author={Zhang, Hongxiang and Chen, Hao and Chen, Muhao and Zhang, Tianyi},
  journal={arXiv preprint arXiv:2505.23657},
  year={2025}
}
```