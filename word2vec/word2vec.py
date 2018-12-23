import logging
from os import path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.data.reader import DataReader
from word2vec.model.skipgram import SkipGram


class Word2vec:

    def __init__(
            self,
            data_path,
            min_count=3,
            window_size=5,
            batch_size=50,
            n_negs=6,
            lr=0.005,
            embed_dim=100,
            epochs=2,
            model_dir="/tmp"):
        self.data_path = data_path
        self.min_count = min_count
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_negs = n_negs
        self.lr = lr
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.model_dir = model_dir

    def load_data(self):
        return DataReader(
            data_path=self.data_path,
            window_size=self.window_size,
            min_count=self.min_count,
            n_negs=self.n_negs,
        )

    def train(self):

        dataset = self.load_data()
        model = SkipGram(
            vocab_size=len(dataset.corpus.vocab),
            embed_dim=self.embed_dim,
            padding_idx=0,
        )
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        logger = logging.getLogger(__name__)
        logger.info("Start training ... ")

        for epoch in range(self.epochs):
            dataset.subsamples()
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            pbar = tqdm(dataloader)
            pbar.set_description(f"[Epoch {epoch}")
            for center, contexts, neg_samples in pbar:
                loss = model(center, contexts, neg_samples)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_postfix(loss=loss.item())

        vectors = model.wi.weight.data.cpu().numpy()
        vec_out_path = path.join(self.model_dir, "vector.txt")
        with open(vec_out_path, "w") as ws:
            for vec in vectors:
                ws.write(" ".join(["{:5.6f}".format(x) for x in vec]))
                ws.write("\n")
        logger.info(f"Saved vectors to {vec_out_path}")
        model_out_path = path.join(self.model_dir, "model.state")
        torch.save(model.state_dict(), model_out_path)
        logger.info(f"Saved model to {model_out_path}")
        optim_out_path = path.join(self.model_dir, "optim.state")
        torch.save(optim.state_dict(), optim_out_path)
        logger.info(f"Saved optim params to {optim_out_path}")
