from scripts.t5 import T5SummarizationTrainer
from scripts.t5_with_title import T5WithTitleSummarizationTrainer
from scripts.bart import BartSummarizationTrainer
from scripts.auto_model import AutoModelSummarizationTrainer

import wandb

wandb.login()

T5SummarizationTrainer.train("train/args/t5.json")
T5WithTitleSummarizationTrainer.train("train/args/t5_with_title.json")
BartSummarizationTrainer.train("train/args/bart.json")
BartSummarizationTrainer.train("train/args/bart_cnn.json")
AutoModelSummarizationTrainer.train("train/args/bert.json")
AutoModelSummarizationTrainer.train("train/args/gpt2.json")