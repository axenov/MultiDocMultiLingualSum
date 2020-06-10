from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from tqdm import tqdm

from baselines.baseline import Baseline


class Bert2Bert(Baseline):

    """ Description 
    EncoderDecoder model from HuggingFace with Bert as encoder and decoder
    """

    def __init__(self, name, model_name, input_max_length, device, batch_size):
        super().__init__(name)
        if isinstance(model_name, str):
            model_name = [model_name, model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name[0])
        self.model = AutoModelWithLMHead.from_pretrained(model_name[1])
        self.input_max_length = input_max_length
        self.device = device
        self.batch_size = batch_size

    def get_summaries(
        self, dataset, document_column_name, **kwargs,
    ):
        dataloader = self.prepare_dataset(dataset, document_column_name)
        self.model = self.model.to(self.device)

        hypotheses = []
        for example_batch in tqdm(dataloader):
            batch_hypotheses_toks = self.model.generate(
                input_ids=example_batch["input_ids"].to(self.device),
                attention_mask=example_batch["attention_mask"].to(self.device),
                decoder_start_token_id=self.model.config.decoder.pad_token_id,
                **kwargs,
            )
            batch_hypotheses = [
                self.tokenizer.decode(
                    toks, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for toks in batch_hypotheses_toks
            ]
            hypotheses.extend(batch_hypotheses)

        dataset = self.append_column(dataset, hypotheses, f"{self.name}_hypothesis")
        return dataset

    def prepare_dataset(self, dataset, document_column_name):
        def convert_to_features(
            example_batch,
            input_max_length=self.input_max_length,
            document_column_name=document_column_name,
        ):
            input_encodings = self.tokenizer.batch_encode_plus(
                example_batch[document_column_name],
                pad_to_max_length=True,
                max_length=input_max_length,
            )
            encodings = {
                "input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
            }
            return encodings

        dataset = dataset.map(convert_to_features, batched=True)
        columns = ["input_ids", "attention_mask"]
        dataset.set_format(type="torch", columns=columns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        return dataloader
