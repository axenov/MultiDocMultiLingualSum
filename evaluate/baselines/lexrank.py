import numpy as np
from nltk.tokenize import sent_tokenize
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from baselines.baseline import Baseline


class LexRank(Baseline):

    """ Description from https://arxiv.org/pdf/1905.13164.pdf
    LexRank (Erkan and Radev, 2004) is a widely-used graph-based extractive summarizer; 
    we build a graph with paragraphs as nodes andedges weighted by tf-idf cosine similarity; 
    we run a PageRank-like algorithm on this graph to rank and select paragraphs until 
    the length of the ground-truth summary is reached.
    """

    """ Implementation
    PageRank function from: https://medium.com/analytics-vidhya/sentence-extraction-using-textrank-algorithm-7f5c8fd568cd
    """

    def rank_sentences(self, dataset, document_column_name, **kwargs):
        all_sentences = []
        all_scores = []
        for document in dataset[document_column_name]:
            sentences, scores = self.run_single(document)
            all_sentences.append(sentences)
            all_scores.append(scores)

        data = [
            {"sentences": sentences, "scores": scores}
            for sentences, scores in zip(all_sentences, all_scores)
        ]
        return Baseline.append_column(dataset, data, self.name)

    def run_single(self, document):

        sentences = sent_tokenize(document)

        # Run tf-idf cosine similarity
        vectorizer = TfidfVectorizer()
        documents_vector = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(documents_vector)

        # Run PageRank
        scores = self._run_page_rank(similarity_matrix)

        return sentences, list(scores)

    def _run_page_rank(self, similarity_matrix):
        # constants
        damping = 0.85  # damping coefficient, usually is .85
        min_diff = 1e-5  # convergence threshold
        steps = 100  # iteration steps

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(steps):
            pr_vector = (1 - damping) + damping * np.matmul(
                similarity_matrix, pr_vector
            )
            if abs(previous_pr - sum(pr_vector)) < min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector
