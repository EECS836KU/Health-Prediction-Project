"""
Readmission risk prediction using structured EHR data and clinical notes.
Supports ablation: with LLM (PubMedBERT) or without (codes only).
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, average_precision_score

from pyhealth.datasets.utils import split_by_patient, get_dataloader
from pyhealth.tasks import ReadmissionPredictionMIMIC3
from pyhealth.models import RNN
from pyhealth.trainer import Trainer
from pyhealth.models import BaseModel
from transformers import AutoTokenizer, AutoModel

# Reproducibility 
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data alignment
def align_clinical_notes(base_dataset):
    patients_dict = getattr(base_dataset, "patients", {})
    for patient in patients_dict.values():
        for visit in patient.visits:
            notes = [event.attr_dict.get('text', " ") for event in visit.get_event_list('NOTEEVENTS')]
            visit.text_note = " ".join(notes[:2]) if notes else " "
    return base_dataset

# Model definition 
class ClinicalDecayModel(BaseModel):
    def __init__(self, dataset, use_llm: bool = True, pos_weight: float = 3.0, dropout_rate: float = 0.3):
        super(ClinicalDecayModel, self).__init__(dataset=dataset)
        self.use_llm = use_llm
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        self.lm = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )

        self.rnn = RNN(dataset=dataset)
        self.rnn_dim = self.rnn.hidden_dim
        self.llm_dim = 768

        combined_dim = self.rnn_dim + (self.llm_dim if use_llm else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(combined_dim, 1)
        self.decay_rate = nn.Parameter(torch.tensor([0.05]))

    def forward(self, **kwargs):
        y_true = None
        for key in self.label_keys:
            if key in kwargs:
                y_true = kwargs[key]
                break

        text_input = kwargs.pop("text_note", None)
        rnn_output = self.rnn(**kwargs)

        if 'feature' in rnn_output:
            rnn_features = rnn_output['feature']
        else:
            rnn_features = rnn_output['logit']
            actual_batch = rnn_features.shape[0]
            if rnn_features.shape[1] < self.rnn_dim:
                pad_size = self.rnn_dim - rnn_features.shape[1]
                padding = torch.zeros(actual_batch, pad_size, device=rnn_features.device)
                rnn_features = torch.cat([rnn_features, padding], dim=1)

        actual_batch = rnn_features.shape[0]

        if self.use_llm:
            if text_input is None:
                text_input = [" "] * actual_batch
            elif isinstance(text_input, list) and len(text_input) != actual_batch:
                if len(text_input) > actual_batch:
                    text_input = text_input[:actual_batch]
                else:
                    text_input = text_input + [" "] * (actual_batch - len(text_input))
            elif isinstance(text_input, str):
                text_input = [text_input] * actual_batch

            tokens = self.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(rnn_features.device)

            lm_output = self.lm(**tokens).last_hidden_state[:, 0, :]
            combined = torch.cat((rnn_features, lm_output), dim=1)
        else:
            combined = rnn_features

        combined = self.dropout(combined)
        base_risk = self.classifier(combined)

        days = torch.ones((actual_batch, 1), device=rnn_features.device)
        positive_decay = F.softplus(self.decay_rate)
        final_risk = base_risk * torch.exp(-positive_decay * days)

        y_prob = torch.sigmoid(final_risk)
        loss = None
        if y_true is not None:
            if y_true.shape[0] != actual_batch:
                y_true = y_true[:actual_batch]
            loss = self.loss_fn(final_risk.view(-1), y_true.view(-1).float())

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": final_risk
        }

# Evaluation helper
def evaluate_with_threshold(model, dataloader, threshold: float = 0.5, device=None):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(**batch)
            probs = outputs["y_prob"].cpu().numpy().flatten()
            labels = outputs["y_true"].cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    return {
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_prob": all_probs,
        "y_true": all_labels
    }

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Train readmission prediction model.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to MIMIC-III dataset root folder.")
    parser.add_argument("--use_llm", action="store_true", default=False,
                        help="Include clinical notes via PubMedBERT.")
    parser.add_argument("--pos_weight", type=float, default=3.0,
                        help="Positive class weight for BCE loss.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate before final classifier.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 regularization (weight decay).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset
    dataset = MIMIC3Dataset(
        root=args.data_root,
        tables=["DIAGNOSES_ICD", "PRESCRIPTIONS", "NOTEEVENTS", "PROCEDURES_ICD"],
        dev=False
    )
    dataset = align_clinical_notes(dataset)

    task = ReadmissionPredictionMIMIC3(exclude_minors=False)
    sample_dataset = dataset.set_task(task)
    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])

    # Compute theoretical pos_weight (info only)
    label_key = list(sample_dataset.output_schema.keys())[0]
    train_labels = [sample[label_key] for sample in train_ds]
    num_neg = train_labels.count(0)
    num_pos = train_labels.count(1)
    theoretical = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"Training set: neg={num_neg}, pos={num_pos} => theoretical pos_weight = {theoretical:.2f}")
    print(f"Using pos_weight = {args.pos_weight}")

    # Data loaders
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = ClinicalDecayModel(
        dataset=sample_dataset,
        use_llm=args.use_llm,
        pos_weight=args.pos_weight,
        dropout_rate=args.dropout
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Running on device: {device}")

    # Trainer
    trainer = Trainer(model=model, device=device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="pr_auc",
        weight_decay=args.weight_decay
    )

    # Tune threshold on validation set
    print("\n--- Tuning decision threshold on validation set ---")
    val_metrics = evaluate_with_threshold(model, val_loader, threshold=0.5, device=device)
    val_probs = val_metrics["y_prob"]
    val_labels = val_metrics["y_true"]
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    best_f1 = f1_scores[best_idx]
    print(f"Best threshold = {best_threshold:.4f} (F1 = {best_f1:.4f})")

    # Evaluate test set with best threshold
    print("\n--- Final evaluation on test set with custom threshold ---")
    test_metrics = evaluate_with_threshold(model, test_loader, threshold=best_threshold, device=device)
    print(f"Test results (threshold={best_threshold:.4f}):")
    print(f"  PR-AUC = {test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC = {test_metrics['roc_auc']:.4f}")
    print(f"  F1 = {test_metrics['f1']:.4f}")

    # Default Trainer evaluation (threshold=0.5)
    print("\n--- Default Trainer evaluation (threshold=0.5) ---")
    default_eval = trainer.evaluate(test_loader)
    print(default_eval)

    #  Save results to JSON 
    os.makedirs("results", exist_ok=True)
    # Use the default evaluation for baseline metrics (threshold 0.5)
    # For LLM run, we also store the tuned threshold metrics separately
    results = {
        "task": "readmission",
        "use_llm": args.use_llm,
        "hyperparameters": {
            "pos_weight": args.pos_weight,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "seed": args.seed
        },
        "default_threshold_metrics": {
            "pr_auc": default_eval.get("pr_auc", 0.0),
            "roc_auc": default_eval.get("roc_auc", 0.0),
            "f1": default_eval.get("f1", 0.0)
        },
        "tuned_threshold_metrics": {
            "threshold": float(best_threshold),
            "pr_auc": float(test_metrics["pr_auc"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "f1": float(test_metrics["f1"])
        }
    }

    filename = f"results/readmission_results_{'llm' if args.use_llm else 'baseline'}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()
