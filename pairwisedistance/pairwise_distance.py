from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict
import fasttext
import numpy as np
import json
import logging


class PairwiseDistances:
    def __init__(self):
        self.vocab = None

    def fit(self, ft_model: fasttext.FastText, vocab: List[str]):
        self.vocab = vocab
        self.vocab_indices = {}
        logging.info("Setting vocab indices...")
        for idx, word in enumerate(vocab):
            self.vocab_indices[word] = idx

        logging.info("Computing pairwise distances")
        self.pairwise_distances = self._compute_pairwise_dists(
            word_vectors=self._get_word_vectors_from_model(ft_model)
        )
        logging.info("Computing mean distances")
        self.mean_distances = self._compute_mean_dists()

    def _get_word_vectors_from_model(self, ft_model) -> np.array:
        """
        Extracts word vectors from model and stacks them
        each word is in the row corresponding to its index from vocab_indices
        """
        return np.stack(
            [ft_model.get_word_vector(word) for word in self.vocab]
        )

    def _compute_pairwise_dists(self, word_vectors) -> np.array:
        return 1 - cosine_similarity(word_vectors)

    def _compute_mean_dists(self) -> np.array:
        # we exclude the elements on the diagonal, which are the distance
        # of a word's vector to itself
        return (
            self.pairwise_distances.sum(axis=1)
            - np.diag(self.pairwise_distances)
        ) / (self.pairwise_distances.shape[1] - 1)

    def get_mean_dist(self, word: str) -> Optional[np.float32]:
        if word in self.vocab:
            idx = self.vocab_indices[word]
            return self.mean_distances[idx]
        else:
            return None

    def get_pairwise_dists(self, word: str) -> Optional[Dict]:
        if word in self.vocab:
            idx = self.vocab_indices[word]
            distances = dict(zip(self.vocab, self.pairwise_distances[idx, :]))

            # remove the distance of the word to itself
            _ = distances.pop(word)

            return distances

        else:
            return None

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "vocab_indices": self.vocab_indices,
                    "mean_distances": self.mean_distances.tolist(),
                    # "pairwise_distances": self.pairwise_distances.tolist(),
                },
                f,
            )

    def load(self, file_path: str):
        with open(file_path, "r") as f:
            loaded = json.load(f)

        self.vocab = loaded["vocab"]
        self.vocab_indices = loaded["vocab_indices"]
        self.mean_distances = loaded["mean_distances"]
        # self.pairwise_distances = np.array(loaded["pairwise_distances"])
