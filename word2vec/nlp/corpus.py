import logging
import os

import numpy as np
from tqdm import tqdm

from word2vec.nlp.tokenizer import naive_tokenizer
from word2vec.nlp.vocabulary import Vocabulary


MAX_NEG_SAMPLE_POOL = 5e7


class Corpus:

    def __init__(
            self,
            data_path,
            preprocessing=None,
            tokenizer=None,
            min_count=1,
            sub_sample_thres=0.001):

        self.data_path = data_path
        self.min_count = min_count
        self.tokenizer = tokenizer or naive_tokenizer
        self.preprocessing = preprocessing or (lambda x: x)
        self.vocab = Vocabulary()
        self.neg_sample_weights = None
        self.sub_sample_weights = None
        self.sub_sample_thres = sub_sample_thres
        self.pool = None

    def profile_data(self):
        raw_word_frequency = {}
        logger = logging.getLogger(__name__)
        logger.info("Processing data ... ")
        with tqdm(total=os.path.getsize(self.data_path)) as pbar:
            with open(self.data_path, "r") as rs:
                for i, line in enumerate(rs):
                    for token in self.tokenizer(self.preprocessing(line)):
                        raw_word_frequency[token] = raw_word_frequency.get(token, 0) + 1
                    pbar.update(len(line))
                    pbar.set_postfix(line=i)
        logger.info("Building vocabulary ... ")
        for key, value in raw_word_frequency.items():
            if value >= self.min_count:
                self.vocab.add_word(key, value)
        self.n_tokens = sum(self.vocab.word_frequency.values())

        if self.vocab.word_frequency:
            wf_vec = np.array(list(self.vocab.word_frequency.values()))
            self.neg_sample_weights = wf_vec ** 0.75
            self.neg_sample_weights /= self.neg_sample_weights.sum()
            wf_vec = wf_vec / wf_vec.sum()
            ratio = wf_vec / self.sub_sample_thres
            self.sub_sample_weights = (ratio**0.5 + 1.0) * (1 / ratio)

        logger.info("Generating negative sample pool ...")
        self.pool_size = min(self.n_tokens, MAX_NEG_SAMPLE_POOL)
        self.pool = np.concatenate([
            np.array(
                [idx]*int(self.pool_size*value),
                dtype=np.int32,
            )
            for idx, value in enumerate(self.neg_sample_weights)
        ])
        np.random.shuffle(self.pool)
        logger.info("Corpus summary")


    def neg_samples(self, size=1):
        return np.random.choice(self.pool, size=size)
