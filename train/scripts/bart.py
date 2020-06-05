from scripts.summarization_trainer import SummarizationTrainer
from dataclass_utils import BartDataCollator

from transformers import BartForConditionalGeneration, BartTokenizer


class BartSummarizationTrainer(SummarizationTrainer):

    data_collator = BartDataCollator()

    def __init__(
        self,
        model_name_or_path,
        tokenizer_name,
        model_cache_dir,
        input_max_length,
        target_max_length,
        summary_column_name,
        document_column_name,
        wandb_project,
        wandb_run_name,
        **kwargs,
    ):
        super().__init__(
            input_max_length,
            target_max_length,
            summary_column_name,
            document_column_name,
            wandb_project,
            wandb_run_name,
        )
        self.tokenizer = BartTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=model_cache_dir,
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=model_cache_dir,
        )