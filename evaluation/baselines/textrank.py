import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from baselines.baseline import Baseline


class TextRank(Baseline):

    """ Description from https://medium.com/analytics-vidhya/sentence-extraction-using-textrank-algorithm-7f5c8fd568cd
    TextRank is an algorithm based on PageRank, which often used in keyword extraction and text summarization.
    """

    """ Implementation
    Code from: https://medium.com/analytics-vidhya/sentence-extraction-using-textrank-algorithm-7f5c8fd568cd
    """

    def rank_sentences(self, dataset, document_column_name, **kwargs):
        all_sentences = []
        all_scores = []
        for document in tqdm(dataset[document_column_name]):
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
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(tokenized_sentences)

        # Run PageRank
        scores = self._run_page_rank(similarity_matrix)

        return sentences, list(scores)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(
                    sentences[idx1], sentences[idx2], stopwords=stopwords
                )

        # Get Symmeric matrix
        sm = self._get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(
            sm, norm, where=norm != 0
        )  # this is to ignore the 0 element in norm

        return sm_norm

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return cosine_similarity([vector1], [vector2])[0][0]

    def _get_symmetric_matrix(self, matrix):
        return matrix + matrix.T - np.diag(matrix.diagonal())

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
