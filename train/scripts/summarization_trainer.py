import torch
import nlp
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import wandb

import numpy as np
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


class SummarizationTrainer(object):
    def __init__(
        self,
        max_len,
        target_max_len,
        summary_column_name,
        document_column_name,
        wandb_project,
    ):
        self.input_max_length = max_len
        self.target_max_length = target_max_len

        self.summary_column_name = summary_column_name
        self.document_column_name = document_column_name

        wandb.init(project=wandb_project)

    def cache_dataset(self, train_file_path, valid_file_path):
        torch.save(self.train_dataset, os.path.abspath(train_file_path))
        torch.save(self.valid_dataset, os.path.abspath(valid_file_path))

    def load_and_process_data(self, name, **load_dataset_kwargs):
        # load train and validation split of Multi-news
        self.train_dataset = nlp.load_dataset(
            name, split="train", **load_dataset_kwargs
        )
        self.valid_dataset = nlp.load_dataset(
            name, split="validation", **load_dataset_kwargs
        )

        # map format_text function to the dataset example wise
        self.train_dataset = self.train_dataset.map(
            self.format_text, load_from_cache_file=False
        )
        # map convert_to_features batch wise
        self.train_dataset = self.train_dataset.map(
            self.convert_to_features, batched=True, load_from_cache_file=False
        )

        self.valid_dataset = self.valid_dataset.map(
            self.format_text, load_from_cache_file=False
        )
        self.valid_dataset = self.valid_dataset.map(
            self.convert_to_features, batched=True, load_from_cache_file=False
        )

        # set the tensor type and the columns which the dataset should return
        columns = ["input_ids", "target_ids", "attention_mask", "target_attention_mask"]
        self.train_dataset.set_format(type="torch", columns=columns)
        self.valid_dataset.set_format(type="torch", columns=columns)

    def format_text(self, example):
        example["input_text"] = example[self.document_column_name]
        example["target_text"] = example[self.summary_column_name]
        return example

    def convert_to_features(self, example_batch):
        # tokenize the examples
        input_encodings = self.tokenizer.batch_encode_plus(
            example_batch["input_text"],
            pad_to_max_length=True,
            max_length=self.input_max_length,
        )
        target_encodings = self.tokenizer.batch_encode_plus(
            example_batch["target_text"],
            pad_to_max_length=True,
            max_length=self.target_max_length,
        )

        encodings = {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "target_ids": target_encodings["input_ids"],
            "target_attention_mask": target_encodings["attention_mask"],
        }

        return encodings

    @classmethod
    def train(cls, args_json_filename):
        parser = HfArgumentParser(
            (ClassArguments, DatasetArguments, DataTrainingArguments, TrainingArguments)
        )
        class_args, dataset_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(args_json_filename)
        )

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info("Start training")
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)

        # Set seed
        set_seed(training_args.seed)

        summarization_trainer = cls(**class_args)

        logger.info("Load and process dataset")
        summarization_trainer.load_and_process_data(**dataset_args)
        summarization_trainer.cache_dataset(
            data_args.train_file_path, data_args.valid_file_path
        )
        train_dataset = torch.load(data_args.train_file_path)
        valid_dataset = torch.load(data_args.valid_file_path)
        logger.info("Dataset ready")

        # Initialize our Trainer
        trainer = Trainer(
            model=summarization_trainer.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=T2TDataCollator(),
            prediction_loss_only=True,
        )

        # Training
        if training_args.do_train:
            trainer.train(
                model_path=class_args.model_name_or_path
                if os.path.isdir(class_args.model_name_or_path)
                else None
            )
            trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_master():
                summarization_trainer.tokenizer.save_pretrained(
                    training_args.output_dir
                )

        # Evaluation
        results = {}
        if training_args.do_eval and training_args.local_rank in [-1, 0]:
            logger.info("*** Evaluate ***")

            eval_output = trainer.evaluate()

            output_eval_file = os.path.join(
                training_args.output_dir, "eval_results.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(eval_output.keys()):
                    logger.info("  %s = %s", key, str(eval_output[key]))
                    writer.write("%s = %s\n" % (key, str(eval_output[key])))

            results.update(eval_output)

        return None


# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator(DataCollator):
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
            "lm_labels": lm_labels,
            "decoder_attention_mask": decoder_attention_mask,
        }


@dataclass
class ClassArguments:
    """
    Arguments to init the SummarizationTrainer class.
    """

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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    max_len: int = field(metadata={"help": "Max tokens of the input"})
    target_max_len: int = field(metadata={"help": "Max tokens of the targer"})

    summary_column_name: str = field(metadata={"help": "Name of the summary column"})

    document_column_name: str = field(metadata={"help": "Name of the document column"})

    wandb_project: str = field(metadata={"help": "Name of the wandb project"})


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
    name: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    data_files: Union[Dict, List] = field(default=None)
    split: Optional[Union[str, Split]] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    download_config: Optional[DownloadConfig] = field(default=None)
    download_mode: Optional[GenerateMode] = field(default=None)
    ignore_verifications: bool = field(default=None)
    save_infos: bool = field(default=None)
