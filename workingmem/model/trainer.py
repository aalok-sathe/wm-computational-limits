import dataclasses

import torch
from transformers import Trainer, TrainingArguments

from workingmem.task.interface import GeneratedCachedDataset


@dataclasses.dataclass
class TrainingConfig(TrainingArguments):
    epochs: int = 50
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0003
    output_dir: str = "output"

    logging_strategy: str = "epoch"
    logging_steps: int = 1  # log every epoch
    save_strategy: str = "epoch"
    save_steps: int = None


class MaskedLossTrainer(Trainer):
    """
    The trainer takes a model and a dataset and trains the model on the dataset.
    """

    def __init__(self, model, train_dataset, eval_dataset, *args, **kwargs):
        self.label_mask = train_dataset.label_mask
        super().__init__(
            *args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ) -> torch.types.Tensor:
        # implement custom logic here
        # b, seq_len, vocab_size
        logits = model(**inputs).logits
        raw_loss = torch.nn.functional.cross_entropy(
            logits, inputs["labels"], reduction="none"
        )

        if self.label_mask:
            batch_label_mask = torch.einops.repeat(
                self.label_mask,
                "seq_len -> (b seq_len)",
                b=logits.shape[0],
                reduction="none",
            )
            masked_loss = raw_loss * batch_label_mask
        else:
            masked_loss = raw_loss  # don't apply a label mask

        return masked_loss.mean()
