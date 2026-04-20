"""
Helena - Task 3.1.1 Baselin: Drug Recommendation (Multi-Label)
Model: Standard Transformer
Track: Baseline
"""

# --- ADD THIS PATCH FIRST ---
import polars as pl

_orig_collect = pl.LazyFrame.collect
def _patched_collect(self, *args, **kwargs):
    if kwargs.get("engine") == "streaming":
        kwargs["engine"] = "cpu"
    return _orig_collect(self, *args, **kwargs)
pl.LazyFrame.collect = _patched_collect
# ----------------------------

import torch
import numpy as np
from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed
from pyhealth.metrics import multilabel_metrics_fn

def calculate_manual_at_k(y_true, y_prob, k=10):
    """
    Calculates Precision@k and Recall@k manually as required by Project Section 3.1.1
    """
    precisions = []
    recalls = []
    for i in range(len(y_true)):
        # Get indices of the top k predicted probabilities
        top_k_idx = np.argsort(y_prob[i])[-k:]
        # Get indices of the actual drugs (where true label is 1)
        true_idx = np.where(y_true[i] == 1)[0]
        
        if len(true_idx) == 0: 
            continue 
        
        # Calculate intersection
        intersect = np.intersect1d(top_k_idx, true_idx)
        
        precisions.append(len(intersect) / k)
        recalls.append(len(intersect) / len(true_idx))
    
    return np.mean(precisions), np.mean(recalls)

if __name__ == "__main__":

    """
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">>>>>>>> MPS device found. Moving to GPU <<<<<<<<<<")
    else:
        device = torch.device("cpu")
        print("-------- MPS not available, staying on CPU --------")
    """

    ####################################
    # 0. SET SEED (Required by Section 3.1, page 2)
    ####################################
    set_seed(42)

	####################################
    # STEP 1: load data
	####################################
    print(">> Step 1: Load data")
    root_dir = "./mimic-iii-clinical-database-1.4"
    base_dataset = MIMIC3Dataset(
        root = root_dir,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=False, # Set to True for quick testing, False for final results
    )
    base_dataset.stats()

	####################################
	# Step 2: Set task(s)
	####################################
    print(">> Step 2: Set task(s)")
    sample_dataset = base_dataset.set_task(DrugRecommendationMIMIC3())
    print(f">> Created {len(sample_dataset)} samples.")

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

	####################################
    # STEP 3: define Transformer baseline model
	####################################
    print(">> Step 3: Define model(s)")
    model = Transformer(
        dataset=sample_dataset,
    )

    

	####################################
    # STEP 4: define trainer
	####################################
    print(">> Step 4: Define trainer(s)")
    #device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    trainer = Trainer(
        model=model,
        device=device,
        metrics=["jaccard_samples", "f1_samples", "pr_auc_samples"],
    )

    import torch
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=300,
        monitor="pr_auc_samples",
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 0.0005},
    )

	####################################
    # STEP 5: evaluate
	####################################
    print(">> Step 5: Evaluate model(s)")
    #print(trainer.evaluate(test_dataloader))

    # Inference returns (y_true, y_prob, loss)
    y_true, y_prob, _ = trainer.inference(test_dataloader)
    
    # 1. Standard PyHealth Metrics
    standard_results = multilabel_metrics_fn(
        y_true, 
        y_prob, 
        metrics=["jaccard_samples", "f1_samples", "pr_auc_samples"]
    )
    
    # 2. Manual Top-K Metrics (Required by project documentation)
    p10, r10 = calculate_manual_at_k(y_true, y_prob, k=10)
    p20, r20 = calculate_manual_at_k(y_true, y_prob, k=20)

    print("-" * 30)
    print(f"Jaccard (Samples): {standard_results['jaccard_samples']:.4f}")
    print(f"F1 (Samples):      {standard_results['f1_samples']:.4f}")
    print(f"PR-AUC (Samples):  {standard_results['pr_auc_samples']:.4f}")
    print(f"Precision@10:      {p10:.4f}")
    print(f"Recall@10:         {r10:.4f}")
    print(f"Precision@20:      {p20:.4f}")
    print(f"Recall@20:         {r20:.4f}")
    print("-" * 30)