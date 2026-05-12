from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RETAINLayer, TransformerLayer, EmbeddingModel, BaseModel
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer
from torch.utils.data import DataLoader
import pyhealth.utils
import torch
import torch.nn as nn

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
    print("NUM ENTRIES")
    print(len(sample_dataset))
    print("NUM POSITIVE ENTRIES")
    count = sum(1 for sample in sample_dataset if sample["mortality"] == 1)
    print(count)
    print("PERCENT POSITIVE")
    print(count / len(sample_dataset))

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    def collate_fn_dict(batch):
        return {key: [d[key] for d in batch] for key in batch[0]}

    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 3: define model
    class RetainTransformer(BaseModel):
        def __init__(
            self,
            dataset,
            embedding_dim=128,
            feature_keys=["conditions", "procedures", "drugs"],
        ):
            super().__init__(dataset=dataset)

            self.embedding = EmbeddingModel(dataset, embedding_dim=embedding_dim)

            self.retain = RETAINLayer(feature_size=embedding_dim, dropout=0.5)

            self.transformer = TransformerLayer(
                feature_size=embedding_dim, heads=4, num_layers=2
            )

            self.classifier = nn.Linear(embedding_dim, 1)

        def forward(self, **kwargs):
            inputs, masks = {}, {}
            for key in self.feature_keys:
                feature = kwargs[key]
                if isinstance(feature, torch.Tensor):
                    feature = (feature,)
                schema = self.dataset.input_processors[key].schema()
                inputs[key] = feature[schema.index("value")]

                if "mask" in schema:
                    masks[key] = feature[schema.index("mask")]

            x_embedded = self.embedding(inputs, masks=masks)
            pooled = [x_embedded[k].mean(dim=1) for k in self.feature_keys]

            stack = torch.stack(pooled, dim=0)

            stack = stack.permute(1, 0, 2)

            context = self.retain(stack)

            context = context.unsqueeze(0)

            output, _ = self.transformer(context)

            logits = self.classifier(output)

            logits = logits.permute(2, 1, 0)

            logits = logits.squeeze(2)
            logits = logits.permute(1, 0)

            y_true = kwargs["mortality"].to(self.device)

            return {
                "loss": self.get_loss_function()(logits, y_true),
                "y_prob": self.prepare_y_prob(logits),
                "y_true": y_true,
                "logit": logits,
            }

    model = RetainTransformer(dataset=sample_dataset)
    # STEP 4: define trainer
    trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc"])
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="pr_auc",
        monitor_criterion="max",
    )

    # STEP 5: evaluate
    results = trainer.evaluate(test_dataloader)
    print(results)
    with open("RETAIN_Transformer2.txt", "w") as f:
        f.write(str(results))
