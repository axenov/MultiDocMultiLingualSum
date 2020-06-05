from scripts.t5 import T5SummarizationTrainer
from scripts.t5_with_title import T5WithTitleSummarizationTrainer
from scripts.bart import BartSummarizationTrainer
from scripts.bert2bert import Bert2BertSummarizationTrainer

import wandb

wandb.login()
Bert2BertSummarizationTrainer.train("train/args/bert2bert.json")
T5SummarizationTrainer.train("train/args/t5.json")
BartSummarizationTrainer.train("train/args/bart.json")
T5WithTitleSummarizationTrainer.train("train/args/t5_with_title.json")
BartSummarizationTrainer.train("train/args/bart_cnn.json")
