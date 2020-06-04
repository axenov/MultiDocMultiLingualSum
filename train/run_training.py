from scripts.t5 import T5SummarizationTrainer

import wandb

wandb.login()

T5SummarizationTrainer.train("train/args/t5_args.json")
