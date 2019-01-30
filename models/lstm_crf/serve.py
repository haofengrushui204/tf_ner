"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
from tensorflow.contrib import predictor
import numpy as np

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"

LINE = '风 噪 , 胎 噪 有 些 大 , 加 速 时 发 动 机 轰 鸣 声 大 （ 对 比 我 家 的 2 . 0 L 3 0 7 三 厢 ）'


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
    export_dir = root_dir + 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    # words = [w.encode() for w in LINE.split()]
    # nwords = len(words)
    words = get_words()
    nwords = [len(_words) for _words in words]

    # predictions = predict_fn({'words': [words], 'nwords': [nwords]})

    file_write = open(root_dir + "predict_rst.txt", "wb")
    for idx, _words in enumerate(words):
        predictions = predict_fn({'words': [_words], 'nwords': [nwords[idx]]})
        save_predict_rst(_words, predictions, file_write)
