import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from baselines.baseline import Baseline


class TFIDF(Baseline):

    """ Description
    Sentences are scored with their similarity with the title of the article.
    """

    """ Implementation
    """

    def rank_sentences(
        self, dataset, document_column_name, title_column_name, **kwargs,
    ):
        all_sentences = []
        all_scores = []
        for title, document in zip(
            dataset[title_column_name], dataset[document_column_name]
        ):
            sentences, scores = self.run_single(title, document)
            all_sentences.append(sentences)
            all_scores.append(scores)

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, all_scores)
        ]
        return Baseline.append_column(dataset, data, self.name)

    def run_single(self, title, document):

        sentences = sent_tokenize(document)

        vectorizer = TfidfVectorizer()
        documents_vector = vectorizer.fit_transform([title] + sentences)
        scores = cosine_similarity(documents_vector[:1], documents_vector[1:])[0]

        return sentences, list(scores)
