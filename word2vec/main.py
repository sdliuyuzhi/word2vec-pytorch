import logging
import os

import click

from word2vec.word2vec import Word2vec


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@click.group()
def word2vec():
    pass


@click.command()
@click.option("--data-path", default=None, help="Path to the text corpus")
@click.option("--min-count", default=3, help="Drop words with counts lower than this")
@click.option("--window-size", default=5, help="One-size context window size")
@click.option("--batch-size", default=50, help="Trainign batch size")
@click.option("--n-negs", default=6, help="Number of negative samples for each content word")
@click.option("--lr", default=0.005, help="Initial step size of gradient descent")
@click.option("--embed-dim", default=100, help="Embedding dimension")
@click.option("--epochs", default=1, help="Numbrt of cycles through the full training data")
@click.option("--model-dir", default="/tmp/model", help="Fold to save the model")
@click.option("--debug", default=False, help="Whether to output debug messages")
def train(data_path, min_count, window_size, batch_size, n_negs, lr, embed_dim,
        epochs, model_dir, debug):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FORMAT)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    w2v = Word2vec(
        data_path,
        min_count=min_count,
        window_size=window_size,
        batch_size=batch_size,
        n_negs=n_negs,
        lr=lr,
        embed_dim=embed_dim,
        epochs=epochs,
        model_dir=model_dir,
    )
    click.echo(f"Traing model using {data_path}")
    w2v.train()
    click.echo(f"Done")


@click.command()
def evaluate():
    click.echo("Evaluating model ...")

word2vec.add_command(train)
word2vec.add_command(evaluate)
