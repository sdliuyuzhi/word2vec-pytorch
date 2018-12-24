# word2vec-pytorch

Implement word2vec (skipgram) with negative sampling/sub sampling using pytorch.

## Usage

Setup a virtual env locally using `venv` or `virtualenv` and install dependency. For example,
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Prepare corpus (text) in the format of one sentence (or paragraph) per line. Train word2vec using command line
```bash
word2vec train --data-path //path/to/your/corpus.txt --batch-size 300 --lr 0.01
```

For more information of the training parameters, try
```bash
word2vec train --help
```

## Development


# README for travis-lab

[![Build status](https://travis-ci.org/USERNAME/travis-lab.svg?master)](https://travis-ci.org/USERNAME)
