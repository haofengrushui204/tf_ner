"""Script to build words, chars and tags vocab"""

__author__ = "Guillaume Genthial"

from collections import Counter
from pathlib import Path

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"
# DATADIR = root_dir + 'example/'

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("usage: python build_w2v.py opinion_id")
        sys.exit(0)
    opinion_id = sys.argv[1]
    DATADIR = root_dir + 'example/{}/'.format(opinion_id)


    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return DATADIR + '{}.words.txt'.format(name)


    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'test']:
        with Path(words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path(DATADIR + 'vocab.words.txt').open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path(DATADIR + 'vocab.chars.txt').open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))


    # 3. Tags
    # Get all tags from the training set

    def tags(name):
        return DATADIR + '{}.tags.txt'.format(name)


    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    with Path(tags('train')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path(DATADIR + 'vocab.tags.txt').open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))
