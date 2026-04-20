"""
Ablation Study: Beta-only RETAIN (Code-Level Attention Neutralized)
Model: RETAIN with removing the Beta Componenet (Time attention)
Track: Category 1 - Advanced Deep Learning
"""

import polars as pl
import torch
import torch.nn as nn
import numpy as np
import types
from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import RETAIN, RETAINLayer
from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed
from pyhealth.metrics import multilabel_metrics_fn

# --- Polars Patch for macOS/Python 3.13 ---
_orig_collect = pl.LazyFrame.collect
def _patched_collect(self, *args, **kwargs):
    if kwargs.get("engine") == "streaming":
        kwargs["engine"] = "cpu"
    return _orig_collect(self, *args, **kwargs)
pl.LazyFrame.collect = _patched_collect

def calculate_manual_at_k(y_true, y_prob, k=10):
    precisions = []
    recalls = []
    for i in range(len(y_true)):
        top_k_idx = np.argsort(y_prob[i])[-k:]
        true_idx = np.where(y_true[i] == 1)[0]
        if len(true_idx) == 0: continue 
        intersect = np.intersect1d(top_k_idx, true_idx)
        precisions.append(len(intersect) / k)
        recalls.append(len(intersect) / len(true_idx))
    return np.mean(precisions), np.mean(recalls)

if __name__ == "__main__":
    set_seed(42)

    # 1. Load data (Using your exact working original setup)
    print(">> Step 1: Loading MIMIC-III")
    base_dataset = MIMIC3Dataset(
        root="./mimic-iii-clinical-database-1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=False, 
    )
    
    # 2. Set task
    print(">> Step 2: Setting Drug Recommendation Task")
    sample_dataset = base_dataset.set_task(DrugRecommendationMIMIC3())
    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
    
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # 3. Define Model
    print(">> Step 3: Initializing RETAIN for Ablation")
    model = RETAIN(
        dataset=sample_dataset,
        embedding_dim=128
    )

    # --- ABLATION PATCH: Force Beta = 1.0 (Alpha-only RETAIN) ---
    def patched_compute_beta(self, rx, lengths):
        # We replace the RNN-generated beta weights with a tensor of ones
        # This makes the "Time Attention" uniform across all visits
        return torch.ones_like(rx)

    # Find the RETAINLayer inside the model and swap the method
    found = False
    for module in model.modules():
        if isinstance(module, RETAINLayer):
            module.compute_beta = types.MethodType(patched_compute_beta, module)
            found = True
    
    if found:
        print(">> Ablation Applied: Time Attention (Beta) has been disabled.")
    else:
        print(">> Warning: Could not find RETAINLayer. Ablation might not be active.")
    # -----------------------------------------------------------

    # 4. Trainer
    trainer = Trainer(
        model=model,
        device="cpu", 
        metrics=["pr_auc_samples"],
    )
    
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=300, # Back to 300 to match your previous baseline runs
        monitor="pr_auc_samples",
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 0.0005},
    )

    # 5. Final Evaluation
    print("\n>> Step 5: Final Evaluation on Test Set")
    y_true, y_prob, _ = trainer.inference(test_dataloader)
    
    standard_results = multilabel_metrics_fn(y_true, y_prob, metrics=["pr_auc_samples"])
    p10, r10 = calculate_manual_at_k(y_true, y_prob, k=10)
    p20, r20 = calculate_manual_at_k(y_true, y_prob, k=20)

    print("-" * 30)
    print(f"ABLATION RESULT: Alpha-Only (No Time Attention)")
    print(f"PR-AUC (Samples):  {standard_results['pr_auc_samples']:.4f}")
    print(f"Precision@10:      {p10:.4f}")
    print(f"Recall@10:         {r10:.4f}")
    print(f"Precision@20:      {p20:.4f}")
    print(f"Recall@20:         {r20:.4f}")
    print("-" * 30)