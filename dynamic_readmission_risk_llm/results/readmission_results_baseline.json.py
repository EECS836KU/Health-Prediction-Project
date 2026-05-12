{
  "task": "readmission",
  "model_type": "baseline",
  "hyperparameters": {
    "pos_weight": 3.0,
    "dropout": 0.3,
    "batch_size": 4,
    "epochs": 10,
    "weight_decay": 1e-5,
    "seed": 42
  },
  "default_threshold_metrics": {
    "pr_auc": 0.2363,
    "roc_auc": 0.5394,
    "f1": 0.2058
  },
  "tuned_threshold_metrics": {
    "threshold": 0.4282,
    "pr_auc": 0.2363,
    "roc_auc": 0.5394,
    "f1": 0.2615
  }
}
