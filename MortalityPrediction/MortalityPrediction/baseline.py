from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer
import pyhealth.utils

pyhealth.utils.set_seed(42)
if __name__ == "__main__":
    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="~/Desktop/School/S26/MachineLearning/project/1.4/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "LABEVENTS"],
        cache_dir="./temp",
        dev=False,
    )
    base_dataset.stats()

    # STEP 2: set task
    task = MortalityPredictionMIMIC3()
    sample_dataset = base_dataset.set_task(task)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 3: define model
    model = RNN(dataset=sample_dataset, embedding_dim=128, hidden_dim=128)

    # STEP 4: define trainer
    trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc"])
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=50,
        monitor="pr_auc",
        monitor_criterion="max",
    )

    # STEP 5: evaluate
    results = trainer.evaluate(test_dataloader)
    print(results)
    with open("Baseline2.txt", "w") as f:
        f.write(str(results))
