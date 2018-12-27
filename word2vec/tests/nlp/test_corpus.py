"""
Tests for corpus models.

"""
import pkg_resources

from hamcrest import assert_that, contains, is_

from word2vec.nlp.corpus import Corpus


class TestCorpus:

    def setup(self):
        data_path = pkg_resources.resource_filename(
            "word2vec.tests.fixtures",
            "train.txt",
        )
        print(data_path)
        self.corpus = Corpus(data_path)

    def test_profile_data(self):
        self.corpus.profile_data()
        vocab = self.corpus.vocab
        assert_that(len(vocab.word2idx) == 15)
        two_index = vocab.word2idx["two"]
        assert_that(vocab.word_frequency[two_index] == 2)
