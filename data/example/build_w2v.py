"""Build an np.array from some glove file and some vocab file

You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Guillaume Genthial"

from pathlib import Path

import numpy as np

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"
DATADIR = root_dir + 'example/'

if __name__ == '__main__':
    # Load vocab
    with Path(DATADIR + 'vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 100))

    # Get relevant glove vectors
    found = 0
    print('Reading W2V file (may take a while)')
    with open('/data/kongyy/nlp/word_vectors/typical_opinion_token_100.txt', "r",encoding="utf8", errors="ignore") as f:
        for line_idx, line in enumerate(f):
            if line_idx % 1000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 100 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed(root_dir + 'w2v.npz', embeddings=embeddings)
