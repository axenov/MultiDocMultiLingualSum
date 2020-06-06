from scripts.t5 import T5SummarizationTrainer
from scripts.t5_with_title import T5WithTitleSummarizationTrainer
from scripts.bart import BartSummarizationTrainer
from scripts.bert2bert import Bert2BertSummarizationTrainer

import wandb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bert2bert", action="store_true")
parser.add_argument("--bart", action="store_true")
parser.add_argument("--bart_cnn", action="store_true")
parser.add_argument("--t5", action="store_true")
parser.add_argument("--t5_with_title", action="store_true")

args = parser.parse_args()


wandb.login()
if args.bert2bert:
    Bert2BertSummarizationTrainer.train("train/args/bert2bert.json")
if args.bart:
    BartSummarizationTrainer.train("train/args/bart_cnn.json")
if args.bart_cnn:
    BartSummarizationTrainer.train("train/args/bart.json")
if args.t5:
    T5SummarizationTrainer.train("train/args/t5.json")
if args.t5_with_title:
    T5WithTitleSummarizationTrainer.train("train/args/t5_with_title.json")
