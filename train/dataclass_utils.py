import torch

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from nlp.splits import Split

from transformers import DataCollator

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T5DataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["target_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }

@dataclass
class BartDataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["target_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        decoder_input_ids = torch.stack([example["target_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

@dataclass
class Bert2BertDataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example["input_ids"] for example in batch])
        lm_labels = torch.stack([example["target_ids"] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        decoder_input_ids = torch.stack([example["target_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        decoder_attention_mask = torch.stack(
            [example["target_attention_mask"] for example in batch]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lm_labels": lm_labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

@dataclass
class ClassArguments:
    """
    Arguments to init the SummarizationTrainer class.
    """

    input_max_length: int = field(metadata={"help": "Max tokens of the input"})
    target_max_length: int = field(metadata={"help": "Max tokens of the targer"})

    summary_column_name: str = field(metadata={"help": "Name of the summary column"})
    document_column_name: str = field(metadata={"help": "Name of the document column"})

    wandb_project: str = field(metadata={"help": "Name of the wandb project"})
    wandb_run_name: str = field(metadata={"help": "Name of the wandb run"})

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    model_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    title_column_name: Optional[str] = field(default=None, metadata={"help": "Name of the title column"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file_path: Optional[str] = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        metadata={"help": "Max input length for the target text"},
    )


@dataclass
class DatasetArguments:
    """
    Arguments to load the dataset.
    """

    path: str = field(
        metadata={
            "help": "Path to the dataset processing script with the dataset builder"
        },
    )
    ignore_verifications: bool = field(default=None)
    save_infos: bool = field(default=None)
    name: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)