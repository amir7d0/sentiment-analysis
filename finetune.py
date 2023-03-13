

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import accuracy_score
from datasets import load_metric
import numpy as np

from src.my_datasets import Dataset
from src.config import config


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    wandb.init(project=config.WANDB_PROJECT, job_type="fine-tune", tags=['staging'], reinit=True)

    ds = Dataset(dataset_path=config.dataset_path, dataset_config=config.dataset_config,
                 config.text_column, config.label_column,
                 num_labels=config.num_labels, tokenizer_checkpoint=config.checkpoint)

    train_dataset, validation_dataset, test_dataset = ds.get_dataset(config.batch_size)
    data_collator = DataCollatorWithPadding(tokenizer=ds.tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(config.checkpoint)
    metric = load_metric("accuracy")

    training_args = TrainingArguments("fine-tune-trainer",
                                      overwrite_output_dir=True,
                                      evaluation_strategy='epoch',
                                      save_strategy="epoch",
                                      lr_scheduler_type='linear',
                                      learning_rate=config.learning_rate,
                                      num_train_epochs=config.epoch,
                                      report_to='wandb')
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=ds.tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    wandb.finish()

