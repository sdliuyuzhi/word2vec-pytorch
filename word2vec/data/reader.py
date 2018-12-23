
import logging
import os

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from word2vec.nlp.corpus import Corpus
from word2vec.nlp.tokenizer import UNK


class DataReader(Dataset):

    def __init__(self, data_path, window_size=5, min_count=1, n_negs=6, batch_size=50, padding_idx=0):

        self.data_path = data_path
        self.window_size = window_size
        self.min_count = min_count
        self.n_negs = n_negs
        self.batch_size = batch_size
        self.padding_idx = padding_idx
        self.corpus = Corpus(
            data_path,
            preprocessing=str.lower,
            min_count=min_count,
        )
        self.corpus.profile_data()
        self.contexts = []
        self.centers = []
        self.sliding_read()

    def sliding_read(self):
        logger = logging.getLogger(__name__)
        tokenizer = self.corpus.tokenizer
        preprocessing = self.corpus.preprocessing
        context = []
        sliding_window = []
        r = self.window_size
        padding_idx = self.padding_idx
        logger.info(f"Sliding window {self.window_size}")
        with tqdm(total=os.path.getsize(self.data_path)) as pbar:
            with open(self.data_path, "r") as rs:
                for line_id, line in enumerate(rs):
                    ids = self.corpus.vocab.encode_idx(
                        tokenizer(preprocessing(line))
                    )
                    len_ids = len(ids)
                    if len_ids:
                        sliding_window = [padding_idx] * r + ids + [padding_idx] * r
                        for i in range(r, r + len_ids):
                            self.contexts.append(
                                np.concatenate([
                                    sliding_window[i-r:i],
                                    sliding_window[i+1:i+1+r]
                                ])
                            )
                            self.centers.append(sliding_window[i])
                    pbar.update(len(line))
                    pbar.set_postfix(line=line_id)
        
        self.size_ = len(self.centers)
        self.id_array_ = np.array(range(self.size_), dtype=np.int32)
        self.unmask = np.full(self.size_, True, dtype=bool)
        self.id_array = self.id_array_[self.unmask]
        logger.info("Training data summary:")
        logger.info(f"Number of training samples: {self.size_}")

    def __len__(self):
        return len(self.id_array)
        
    def __getitem__(self, idx):
        idx_ = self.id_array[idx]
        return self.centers[idx_], self.contexts[idx_], self.corpus.neg_samples(size=self.n_negs)

    def subsamples(self):
        sub_sample_weights = self.corpus.sub_sample_weights
        self.unmask = np.random.uniform(size=self.size_) < np.array([
            sub_sample_weights[idx]
            for idx in self.centers
        ])
        self.id_array = self.id_array_[self.unmask]
