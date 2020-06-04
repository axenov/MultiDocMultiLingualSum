import train.scripts.summarization_trainer as st

import wandb

wandb.login()

st.SummarizationTrainer.train("train/args/t5_args.json")
