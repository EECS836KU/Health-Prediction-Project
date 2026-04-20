# Health Prediction Project – Readmission Risk Modeling

## Overview
This project, part of the **Health Prediction Project**, focuses on predicting patient readmission using the **MIMIC-III** database (2001–2012) from PhysioNet. A baseline model was compared using only structured clinical codes (diagnoses, procedures, medications) with an LLM‑augmented model that also processes clinical notes via **PubMedBERT**.

## Hypothesis
> **Clinical notes contain hidden risk signals that structured codes alone miss.**

To test this, an **ablation study** was conducted: the LLM component (PubMedBERT + notes) is removed while keeping all other hyperparameters identical.

## Model Architecture

Model performs three main functions:

1. **Feature Fusion**  
   - A **RNN** encodes structured data (the “what” and “when” of diagnoses and medications).  
   - **PubMedBERT** encodes unstructured clinical notes (the “why” and clinical nuance).  
   - The two representations are concatenated and fed into a classifier.

2. **Dimensionality Reduction → Base Risk**  
   - The high‑dimensional combined vector (`RNN_dim + 768`) is projected down to a single scalar – the **base risk**.

3. **Logit Production & Time Decay**  
   - The linear layer outputs a raw logit, which is then modified by a learnable time‑decay factor before a sigmoid function produces the final readmission probability (0–1).

## Results

| Model                  | Best Val PR‑AUC| Test PR‑AUC (tuned)  | Test ROC‑AUC | Test F1 (tuned)| Overfitting gap  |
|------------------------|----------------|----------------------|--------------|----------------|------------------|
| **LLM (notes)**        | 0.286          | **0.278**            | **0.626**    | **0.337**      | 0.008            |
| **Baseline (no notes)**| 0.319          | 0.236                | 0.539        | 0.262          | 0.083            |

**Key takeaways**  
- The LLM‑augmented model achieves **higher test PR‑AUC, ROC‑AUC, and F1** than the baseline.  
- The baseline overfits more (validation performance does not generalise to the test set).  
- Clinical notes provide both predictive signal and a regularising effect.

## Repository Structure

dynamic_readmission_risk_llm/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── train.py              
├── notebooks/                 
├── results/                  
└── config/                    


## Getting Started

### Prerequisites
- Python 3.8+
- PyHealth, PyTorch, Transformers, scikit‑learn

### Hardware Requirements
- GPU with at least 12GB VRAM recommended for training with LLM (notes)
- CPU-only training is possible but significantly slower
- For the best experience, use a machine with a compatible NVIDIA GPU and CUDA installed.
- If GPU resources are limited, consider using cloud platforms like Google Colab or AWS SageMaker, which provide access to powerful GPUs.
- Note: Training the LLM‑augmented model on CPU may take several hours, while the baseline can be trained in under an hour.
- For those without access to GPU resources, we recommend running the baseline model first to get familiar with the codebase and then using cloud services for the LLM‑augmented model.
- If you encounter memory issues when training the LLM‑augmented model, try reducing the batch size or using gradient accumulation to fit the model within your GPU's memory limits.

### Installation
```bash
git clone https://github.com/your-username/Health-Prediction-Project.git
cd Health-Prediction-Project
pip install -r requirements.txt

### Running the Model
# Train with LLM (notes)
python src/train.py --use_llm True --pos_weight 3.0

# Train baseline (no notes)
python src/train.py --use_llm False --pos_weight 3.0


## Citations & Acknowledgements

- MIMIC-III Database: https://physionet.org/content/mimiciii/1.4/
- PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
- PyHealth: https://pyhealth.readthedocs.io/
- DeepSeek for assistance with debugging and code optimization.
- This project was developed as part of the Health Prediction Project, with contributions from Ilia Parish and collaborators.
