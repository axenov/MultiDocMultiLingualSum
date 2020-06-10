from nltk.tokenize import sent_tokenize
import random

from baselines.baseline import Baseline


class Random(Baseline):

    """ Description 
    Give a random score to all sentences
    """

    def rank_sentences(self, dataset, document_column_name, seed=42, **kwargs):
        random.seed(seed)
        all_sentences = list(map(sent_tokenize, dataset[document_column_name]))
        scores = [
            [random.random() for sentence in sentences] for sentences in all_sentences
        ]

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, scores)
        ]
        return Baseline.append_column(dataset, data, self.name)
