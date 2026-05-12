{
  "task": "readmission",
  "model_type": "llm",
  "hyperparameters": {
    "pos_weight": 3.0,
    "dropout": 0.3,
    "batch_size": 4,
    "epochs": 10,
    "weight_decay": 1e-5,
    "seed": 42
  },
  "default_threshold_metrics": {
    "pr_auc": 0.2783,
    "roc_auc": 0.6259,
    "f1": 0.2264
  },
  "tuned_threshold_metrics": {
    "threshold": 0.4650,
    "pr_auc": 0.2783,
    "roc_auc": 0.6259,
    "f1": 0.3372
  }
}
