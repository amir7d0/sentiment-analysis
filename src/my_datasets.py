from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import pandas as pd


class Dataset(object):
    def __init__(self, dataset_path, dataset_config, text_column, label_column,
                 num_labels=2, tokenizer_checkpoint="bert-base-uncased"):
        self.dataset_path = dataset_path
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.label_column = label_column
        self.number_of_classes = num_labels
        self.raw_datasets = None
        self.tokenized_datasets = None
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.tokenizer = None

    def load_preprocess_data(self):
        raw_datasets = load_dataset(self.dataset_path, self.dataset_config)
        raw_datasets = raw_datasets.rename_columns({self.text_column: 'text',
                                                    self.label_column: 'labels'})
        drop_columns = list(set(raw_datasets["train"].column_names) - set(['text', 'labels']))
        raw_datasets = raw_datasets.remove_columns(drop_columns)

        # find number of classes and map 1-5 stars to a range of 0 to 4
        assert self.number_of_classes == len(pd.unique(raw_datasets['train']['labels']))
        class_map = dict(zip(pd.unique(raw_datasets['train']['labels']),
                             pd.unique(raw_datasets['train']['labels'])-1))

        self.raw_datasets = raw_datasets.map(lambda example: {'labels': class_map[example['labels']]})

    def tokenize(self, padding=True, truncation=True, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)

        def tokenization(example):
            return self.tokenizer(example['text'], padding=padding, truncation=truncation,
                                  max_length=max_length)

        tokenized_datasets = self.raw_datasets.map(tokenization, batched=True)
        self.tokenized_datasets = tokenized_datasets.remove_columns(['text'])

    def to_tf_dataset(self, batch_size):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        tokenized_datasets = self.tokenized_datasets
        tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            label_cols=["labels"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            label_cols=["labels"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
            columns=["attention_mask", "input_ids", "token_type_ids"],
            label_cols=["labels"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        return tf_train_dataset, tf_validation_dataset, tf_test_dataset

    def get_dataset(self, batch_size=8):
        self.load_preprocess_data()
        self.tokenize()
        train_ds, valid_ds, test_ds = self.to_tf_dataset(batch_size=batch_size)
        return train_ds, valid_ds, test_ds
