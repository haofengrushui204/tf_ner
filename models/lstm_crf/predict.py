"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
from tensorflow.contrib import predictor

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"

LINE = '风 噪 , 胎 噪 有 些 大 , 加 速 时 发 动 机 轰 鸣 声 大 （ 对 比 我 家 的 2 . 0 L 3 0 7 三 厢 ）'


def get_words():
    with open(words_path, "r", encoding="utf8") as file_read:
        words = [[w.encode() for w in line.strip().split()] for line in file_read if len(line.strip()) > 0]
    return words


def get_tags():
    with open(tags_path, "r", encoding="utf8") as file_read:
        tags = [[w.encode() for w in line.strip().split()] for line in file_read if len(line.strip()) > 0]
    return tags


def save_predict_rst(words, predictions, file_write):
    tags = predictions["tags"]
    scores = predictions["scores"]
    file_write.write(b" ".join(words) + b"\n")
    file_write.write(str(scores[0]).encode(encoding="utf8") + b"\t" + b" ".join(tags[0]) + b"\n")


def eval_entity(tags_true, tags_pred):
    """
    flag = same:1, ture in pred:2, pred in ture:3, other:0
    :param tags_true:
    :param tags_pred:
    :return:
    """
    flag = 0
    tags_true = b"".join([tag for tag in tags_true if tag != b"O"])
    tags_pred = b"".join([tag for tag in tags_pred if tag != b"O"])
    if tags_pred == tags_true:
        flag = 1
    elif len(tags_true) > 0 and tags_true in tags_pred:
        flag = 2
    elif len(tags_pred) > 0 and tags_pred in tags_true:
        flag = 3

    return flag


if __name__ == '__main__':
    import sys
    import os

    if len(sys.argv) < 2:
        print("usage: python predict.py data_path")
        sys.exit(0)
    words_path = sys.argv[1]
    if len(sys.argv) > 2:
        tags_path = sys.argv[2]
    else:
        tags_path = ""

    words = get_words()
    if tags_path != "":
        tags_true = get_tags()
    else:
        tags_true = None
    nwords = [len(_words) for _words in words]

    predict_rst_dict = {}
    for opinion_id in range(1, 51):
        export_dir = root_dir + 'saved_model_{}'.format(opinion_id)
        if os.path.exists(export_dir):
            subdirs = [x for x in Path(export_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
            latest = str(sorted(subdirs)[-1])
            predict_fn = predictor.from_saved_model(latest)

            for idx, _words in enumerate(words):
                predictions = predict_fn({'words': [_words], 'nwords': [nwords[idx]]})
                # save_predict_rst(_words, predictions, file_write)

                tags = predictions["tags"]
                scores = predictions["scores"]
                if b" ".join(_words) not in predict_rst_dict:
                    predict_rst_dict[b" ".join(_words)] = {}
                if tags_true is not None:
                    predict_rst_dict[b" ".join(_words)][opinion_id] = [scores[0], tags[0], tags_true[idx]]

    file_write = open(root_dir + "predict_rst_merge.txt", "wb")
    count_dict = {1: 0, 2: 0, 3: 0, 0: 0}
    for sent, opinion_ids in predict_rst_dict.items():
        best_score = 1.0e+10
        best_tags = []
        cur_tags_true = []
        for opinion_id, item in opinion_ids.items():
            if len(best_tags) == 0:
                best_tags = item[1]
            else:
                for idx, tag in enumerate(best_tags):
                    if item[1][idx] != b"O":
                        best_tags[idx] = item[1][idx]
            # if item[0] < best_score:
            #     best_score = item[0]
            #     # merge different position
            #     if len(best_tags) == 0:
            #         best_tags = item[1]
            #     else:
            #         assert len(best_tags) == len(item[1]), "best_tags and curtags lengths don't match " + best_tags + " tag " + item[1]
            #         for idx, tag in enumerate(best_tags):
            #             if item[1][idx] != b"O":
            #                 best_tags[idx] = item[1][idx]
            #
            if tags_true is not None and len(cur_tags_true) == 0:
                cur_tags_true = item[-1]
        cnt = eval_entity(cur_tags_true, best_tags)
        file_write.write(str(cnt).encode() + b"\t" + sent + b"\n")
        file_write.write(b"true\t" + b" ".join(cur_tags_true) + b"\n")
        file_write.write(b"pred\t" + b" ".join(best_tags) + b"\n")

        count_dict[cnt] += 1

    print(count_dict)

    for k, v in count_dict.items():
        if k == 1:
            print("same cnt is {}, {}".format(v, v / sum(count_dict.values())))
        elif k == 2:
            print("ture in pred cnt is {},{}".format(v, v / sum(count_dict.values())))
        elif k == 3:
            print("pred in ture cnt is {},{}".format(v, v / sum(count_dict.values())))
        else:
            print("others ".format(v, v / sum(count_dict.values())))
