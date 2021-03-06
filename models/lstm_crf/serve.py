"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
from tensorflow.contrib import predictor
import numpy as np

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"

# LINE = '探 歌 还 在 坑 里 ， 探 岳 又 要 跳 进 去 吗 ？'
# PTAGS = '1 1 0 0 0 0 0 1 1 0 0 0 0 0 0 0'

LINE = " ".join(list("南北差异有多大？看途岳和探岳就知道了！"))
PTAGS = "0 0 0 0 0 0 0 0 0 1 1 0 1 1 0 0 0 0 0"


def get_words():
    with open(root_dir + "example/xuanjia.words.txt", "r", encoding="utf8") as file_read:
        words = [[w.encode() for w in line.strip().split()] for line in file_read if len(line.strip()) > 0]
    return words


def save_predict_rst(words, predictions, file_write):
    tags = predictions["tags"]
    if b"B-11" in tags[0]:
        file_write.write(b" ".join(words) + b"\n")
        file_write.write(b" ".join(tags[0]) + b"\n")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("usage: python serve.py opinion_id")
        sys.exit(0)
    opinion_id = sys.argv[1]

    export_dir = root_dir + 'saved_model_{}'.format(opinion_id)
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    words = [w.encode() for w in LINE.split()]
    nwords = len(words)
    ptags = [t.encode() for t in PTAGS.split()]
    # words = get_words()
    # nwords = [len(_words) for _words in words]

    if int(opinion_id) == 10000:
        predictions = predict_fn({'words': [words], 'nwords': [nwords], "ptags": [ptags]})
        print(predictions)
    elif int(opinion_id) == 10002:
        predictions = predict_fn({'words': [words], 'nwords': [nwords]})
        print(predictions)
    elif int(opinion_id) == 10001:
        predictions = predict_fn({'words': [words], 'nwords': [nwords]})
        print(predictions)
        tags = [t.decode() for t in predictions["tags"][0].tolist()]
        print(" ".join([t.decode() for t in predictions["tags"][0].tolist()]).replace("B-ENTITY", "1").replace("I-ENTITY", "1"))
    else:
        print("opinion_id must be in (10000,10001,10002)")

    # file_write = open(root_dir + "predict_rst.txt", "wb")
    # for idx, _words in enumerate(words):
    #     predictions = predict_fn({'words': [_words], 'nwords': [nwords[idx]]})
    #     save_predict_rst(_words, predictions, file_write)
