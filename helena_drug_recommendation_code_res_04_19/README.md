# Health Prediction Project – Drug Recommendation (Multi-label)

## Overview
This project, part of the **Health Prediction Project**, focuses on predicting the set of medications administered/prescribed in that current admission using the **MIMIC-III** database (2001–2012) from PhysioNet. A based line model based on a vanilla Transformer is compared again using instead a RETAIN (reverse-time attention) architecture.

## Hypothesis
> **Clinical notes contain hidden risk signals that structured codes alone miss.**

To test this, an **ablation study** was conducted: (1)Removing the Time Attention Component/treating all past visits with equal temporal weight (2)Removing the Level Attention Component/remove the distinction between specific diagnosis or procedure codes within visits.

## Model Architecture

1. **Baseline Architecture**
   - A vanilla Transformer is used and the drug recommendation is predicted via multi-label.

2. **Improvement using RETAIN**
   - Implement an attention-based sequence model via the reverse-time attention in RETAIN to improve predictive performance.

## Results

> Each of the 4 resulting training run were ran with 300 epochs.

| Model                          | PR‑AUC    | Precision@10   | Recall@10    | Precision@20   | Recall@20    |
|--------------------------------|-----------|----------------|--------------|----------------|--------------|
| **Baseline Transformer**       | 0.6337    | 0.8036         | 0.2551       | 0.6999         | 0.4301       |
| **Improved via RETAIN**        | 0.6610    | 0.8256         | 0.2632       | 0.7215         | 0.4458       |
|                                                                                                            |
| **Ablation 1**                 | 0.6521    | 0.8212         | 0.2616       | 0.7146         | 0.4407       |
|     - Remove Time Attention    |           |                |              |                |              |
| **Ablation 2**                 | 0.6517    | 0.8166         | 0.2595       | 0.7144         | 0.4402       |
|     - Remove Level Attention   |           |                |              |                |              |


**Key takeaways**  
- The improved RETAIN model achieves higher PR-AUC, Precision@10, Recall@10, Precision@20, and Recall@20 than the baseline model.
- Given the 2 ablation runs had lower results, it indicates the two-level attention is not redundant.
- Both "Time" and "Level/Code" attention work together to capture different clinical insights.  When removing either one, ~0.5% is lost in PR-AUC.
- The RETAIN model performs slightly better when it knows which specific diseases/drugs are present, even if it doesn't weight their timing.
- Overall, the ablation study confirms that both level-level(alpha) and visit-level(beta) attention mechanisms contribute to the model's predictive power.
- The slight performance lead of the Alpha-only model over the Beta-only model suggests that feature selection within a visit is slightly more critical for drug recommendation in MIMIC-III than temporal weighting of past visits.

## Repository Structure
```
...drug_recommendation_code_res.../
├── results
│   ├── training_res1___baseline_transformer.txt
│   ├── training_res2___improved_retain.txt
│   ├── training_res3___ablation1.txt
│   └── training_res4___ablation2.txt
├── source
│   ├── script1___drug_recommendation_baseline.py
│   ├── script2___drug_recommendation_improved_retain.py
│   ├── script3___drug_recommendation_ablation1.py
│   └── script4___drug_recommendation_ablation2.py
├── requirements.txt
├── README.md
```

## Getting Started

### Prerequisites
- Python 3.12+
- PyHealth, PyTorch, Transformers, scikit‑learn

### Hardware Requirements
- Machine used for this project on this part was a MacBook Air M5 with 16GB of RAM.
- CPU-only training is possible but significantly slower.
- For the best experience, use a machine with a compatible NVIDIA GPU and CUDA installed if not using a Mac.
- If GPU resources are limited, consider using cloud platforms like Google Colab or AWS SageMaker, which provide access to powerful GPUs.
- For those without access to GPU resources, we recommend running the baseline model first to get familiar with the codebase and then using cloud services for the LLM‑augmented model.

### Installation
```bash
git clone https://github.com/your-username/Health-Prediction-Project.git
cd Health-Prediction-Project
pip install -r requirements.txt

### Running the Model
# Train the baseline model
python src/script1___drug_recommendation_baseline.py

# Train the retain model
python src/script2___drug_recommendation_improved_retain.py

# Train the retain model with ablation 1 (remove time attention)
python src/script3___drug_recommendation_ablation1.py

# Train the retain model with ablation 2 (remove level/code attention)
python src/script3___drug_recommendation_ablation2.py
```

## Citations & Acknowledgements

- MIMIC-III Database: https://physionet.org/content/mimiciii/1.4/
- RETAIN: https://arxiv.org/abs/1608.05745
- PyHealth: https://pyhealth.readthedocs.io/
- ChatGPT/Gemini/DeepSeek, etc. for assistance with debugging and code optimization.
- This project was developed as part of the Health Prediction Project, with contributions from Helena Zhang and collaborators.
