
from word2vec.nlp.tokenizer import UNK


class Vocabulary:

    def __init__(self):
        self.reset()

    def reset(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}
        self.word_frequency = {}
        self.add_word(UNK)
    
    def add_word(self, word, ct=1):
        if not (word in self.word2idx):
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        self.word_frequency[self.word2idx[word]] = self.word_frequency.get(word, 0) + ct

    def encode_idx(self, word_array):
        return [self.word2idx.get(word, 0) for word in word_array]

    def size(self):
        return len(self.word2idx)

    def __len__(self):
        return self.size()
