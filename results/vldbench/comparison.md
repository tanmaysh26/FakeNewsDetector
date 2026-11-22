| Model | Result type | Setting | Evaluation data | Accuracy | Precision | Recall | F1 | Source |
|---|---|---|---|---:|---:|---:|---:|---|
| TF-IDF + Linear SVM (C=1.0) | this experiment | supervised, trained on our 80% split | VLDBench, our 20% test split (n=6268) | 81.45 | 80.42 | 81.45 | 80.03 | results/vldbench/metrics.json |
| Majority class (always Likely) | this experiment (reference) | no learning | VLDBench, our 20% test split (n=6268) | 73.96 | 54.71 | 73.96 | 62.89 | results/vldbench/metrics.json |
| Outlet majority label | this experiment (reference) | outlet name only | VLDBench, our 20% test split (n=6268) | 74.23 | 70.31 | 74.23 | 64.68 | results/vldbench/metrics.json |
| Phi-3-mini-128k-Instruct | published | zero-shot prompting, text only | VLDBench, whole dataset | 57.15 | 55.12 | 58.21 | 56.62 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| Vicuna-7B-v1.5 | published | zero-shot prompting, text only | VLDBench, whole dataset | 55.21 | 56.78 | 52.48 | 54.55 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| Mistral-7B-Instruct-v0.3 | published | zero-shot prompting, text only | VLDBench, whole dataset | 68.58 | 68.10 | 65.09 | 66.56 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| Qwen2-7B-Instruct | published | zero-shot prompting, text only | VLDBench, whole dataset | 69.92 | 68.34 | 69.10 | 68.72 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| InternLM2-7B | published | zero-shot prompting, text only | VLDBench, whole dataset | 51.19 | 49.85 | 51.16 | 50.50 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| DeepSeek-V2-Lite-Chat | published | zero-shot prompting, text only | VLDBench, whole dataset | 51.96 | 52.61 | 53.53 | 51.96 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| GLM-4-9B-Chat | published | zero-shot prompting, text only | VLDBench, whole dataset | 51.14 | 60.19 | 53.28 | 56.52 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| LLaMA-3.1-8B-Instruct | published | zero-shot prompting, text only | VLDBench, whole dataset | 68.21 | 62.13 | 62.10 | 62.11 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| LLaMA-3.2-1B-Instruct | published | zero-shot prompting, text only | VLDBench, whole dataset | 70.29 | 69.78 | 68.92 | 69.35 | Raza et al., VLDBench, arXiv:2502.11361v5, Table 9 |
| Phi-3-mini-128k-Instruct (IFT) | published | instruction fine-tuned, text only | VLDBench, paper's own 15% test split | n/a | n/a | n/a | 67.20 | Raza et al., VLDBench, arXiv:2502.11361v5, Section 5.2 / Figure 9 |
