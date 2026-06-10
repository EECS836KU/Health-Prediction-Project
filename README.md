# Health Prediction Project

Machine learning system for healthcare clinical prediction using the MIMIC-III dataset and PyHealth. This project implements baseline and advanced deep learning methods for multiple clinical prediction tasks, including drug recommendation, in-hospital mortality prediction, and 30-day hospital readmission risk modeling.

## Overview

This project explores how machine learning models can be used to support clinical prediction from electronic health record data. Using the MIMIC-III database, the team developed and evaluated models for three healthcare tasks:

* Drug recommendation
* In-hospital mortality prediction
* 30-day hospital readmission prediction

The project compares PyHealth baseline models with more advanced architectures, including recurrent neural networks, Transformer-based methods, RETAIN-style attention models, knowledge-informed approaches, and language-model-based methods using clinical notes.

Across tasks, the goal was to evaluate whether richer model architectures and additional clinical context could improve predictive performance over standard baseline approaches.

## Research Motivation

Electronic health records contain both structured clinical data and unstructured clinical notes. Structured data, such as diagnoses, procedures, and medication codes, captures important clinical events, while clinical notes may contain additional context about patient condition, physician reasoning, disease progression, and discharge risk.

This project investigates whether combining structured EHR features with advanced deep learning architectures can improve clinical prediction performance and generalization.

## Dataset

This project uses the MIMIC-III clinical database from PhysioNet.

MIMIC-III contains de-identified health data associated with ICU admissions, including:

* diagnoses
* procedures
* medications
* laboratory events
* clinical notes
* admission and discharge information
* mortality and readmission-related outcomes

Due to data access requirements and privacy restrictions, the MIMIC-III dataset is not included in this repository. Users must obtain access through PhysioNet and follow all required data-use agreements.

## Tasks

### 1. Drug Recommendation

The drug recommendation task focuses on predicting medication recommendations from patient records. The team evaluated deep learning approaches such as Transformer-based models and RETAIN-style architectures.

The RETAIN architecture, which uses attention mechanisms to improve interpretability across visits and clinical features, was compared against plain Transformer approaches. Results suggested that RETAIN-style dual attention can be useful for medication-related prediction tasks.

### 2. In-Hospital Mortality Prediction

The in-hospital mortality task focuses on predicting patient mortality risk during admission. The team explored models designed to capture both short-term and long-term dependencies in clinical sequences.

A hybrid RETAIN-Transformer approach was evaluated for risk stratification and compared with PyHealth baseline models. This task emphasized the importance of temporal modeling, class weighting, and hyperparameter tuning when working with imbalanced clinical outcomes.

### 3. 30-Day Hospital Readmission Prediction

The readmission task focuses on predicting whether a patient will be readmitted within 30 days. This part of the project compared structured-data baselines with language-model-augmented approaches that incorporate clinical notes.

A PubMedBERT-based workflow was used to encode unstructured clinical notes, while structured clinical codes were used to represent diagnoses, procedures, and medication history. The project tested the hypothesis that clinical notes contain additional risk signals that may not be fully captured by structured codes alone.

## Methods

The project used a combination of baseline and advanced methods, including:

* PyHealth baseline models
* recurrent neural networks for structured clinical sequences
* Transformer-based architectures
* RETAIN and hybrid RETAIN-Transformer models
* PubMedBERT for biomedical clinical text representation
* ablation studies comparing structured-only and note-augmented models
* class weighting for imbalanced outcomes
* hyperparameter tuning and model evaluation across validation and test sets

## Overall Findings

Across the three clinical prediction tasks, the project found that advanced architectures often improved performance over simpler baselines when paired with careful preprocessing, hyperparameter tuning, and evaluation.

Key findings included:

* Incorporating clinical notes through PubMedBERT improved readmission prediction and generalization.
* RETAIN-style attention models were effective for drug recommendation tasks.
* Hybrid RETAIN-Transformer models helped capture short- and long-term dependencies for mortality prediction.
* Ablation studies were useful for understanding whether added model components provided meaningful improvements.
* Class weighting and careful evaluation were important due to imbalance in clinical prediction tasks.

## Repository Structure

```text
Health-Prediction-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
├── notebooks/
├── results/
├── src/
│   ├── __init__.py
│   └── train.py
└── task_readmes/
```

Individual team members may also include task-specific READMEs, notebooks, or result summaries for their assigned components.

## Getting Started

### Prerequisites

Recommended environment:

* Python 3.8+
* PyTorch
* PyHealth
* Transformers
* scikit-learn
* pandas
* numpy

### Installation

```bash
git clone https://github.com/your-username/Health-Prediction-Project.git
cd Health-Prediction-Project
pip install -r requirements.txt
```

### Running Models

Example commands may vary by task and implementation.

```bash
# Train a note-augmented readmission model
python src/train.py --use_llm True --pos_weight 3.0

# Train a structured-only baseline
python src/train.py --use_llm False --pos_weight 3.0
```

## Hardware Notes

Training models that use clinical notes and biomedical language models may require GPU resources. A CUDA-compatible GPU is recommended for PubMedBERT-based workflows.

If GPU resources are limited, structured-data baselines can be run first to test the pipeline before moving to note-augmented models. Reducing batch size or using gradient accumulation may help with memory constraints.

## Data Access Notice

MIMIC-III is a restricted-access clinical dataset. It is not included in this repository.

To reproduce this project, users must request access through PhysioNet and complete the required credentialing process.

## Citations and Acknowledgements

* MIMIC-III Database: https://physionet.org/content/mimiciii/1.4/
* PyHealth: https://pyhealth.readthedocs.io/
* PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract

This project was developed as part of a team-based health prediction project at the University of Kansas, with contributions from Ilia and collaborators.

## Project Status

This repository represents a course/research project focused on evaluating machine learning approaches for clinical prediction. Future work may include additional model tuning, external validation, improved documentation, and expanded comparison across clinical prediction tasks.
