from scripts.summarization_trainer import SummarizationTrainer
from dataclass_utils import Bert2BertDataCollator

from transformers import EncoderDecoderModel, BertTokenizer


class Bert2BertSummarizationTrainer(SummarizationTrainer):

    data_collator = Bert2BertDataCollator()

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
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=model_cache_dir,
        )
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name_or_path, model_name_or_path, cache_dir=model_cache_dir,
        )
