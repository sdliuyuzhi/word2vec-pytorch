
import re


UNK = "<UNK"


def naive_tokenizer(s):
    return re.split(r"\s", s.strip())
