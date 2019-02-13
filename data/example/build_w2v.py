"""Build an np.array from some glove file and some vocab file

You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Guillaume Genthial"

from pathlib import Path

import numpy as np

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("usage: python build_w2v.py opinion_id")
        sys.exit(0)
    opinion_id = sys.argv[1]
    DATADIR = root_dir + 'example/{}/'.format(opinion_id)
    # Load vocab
    with Path(DATADIR + 'vocab.words.txt').open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)
    emb_szie = 60
    level = "char"

    # Array of zeros
    embeddings = np.zeros((size_vocab, emb_szie))

    # Get relevant glove vectors
    found = 0
    print('Reading W2V file (may take a while)')
    with open('/data/kongyy/nlp/word_vectors/typical_opinion_{}_{}.txt'.format(level, emb_szie), "r", encoding="utf8",
              errors="ignore") as f:
        for line_idx, line in enumerate(f):
            if line_idx % 1000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != emb_szie + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed(DATADIR + 'w2v_{}.npz'.format(opinion_id), embeddings=embeddings)
