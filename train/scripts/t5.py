from train.scripts.summarization_trainer import SummarizationTrainer

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction


class T5SummarizationTrainer(SummarizationTrainer):
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name,
        cache_dir,
        input_max_length,
        target_max_length,
        summary_column_name,
        document_column_name,
        wandb_project,
    ):
        super().__init__(
            input_max_length,
            target_max_length,
            summary_column_name,
            document_column_name,
            wandb_project,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_name if tokenizer_name else model_name_or_path,
            cache_dir=cache_dir,
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=cache_dir,
        )

    def format_text(self, example):
        # process the examples in input and target text format and the eos token at the end
        example["input_text"] = (
            "summarize: %s </s>" % example[self.document_column_name]
        )
        example["target_text"] = "%s </s>" % example[self.summary_column_name]
        return example
