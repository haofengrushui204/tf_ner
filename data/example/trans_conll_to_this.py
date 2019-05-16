# -*- coding:utf-8 -*-
"""
@file name: trans_conll_to_this.py
Created on 2019/1/24
@author: kyy_b
@desc: 将 conll 格式的标注文件转为本程序可用的标注文件
"""
import sys
import traceback


def trans(data_path, words_path, tags_path):
    with open(words_path, "w", encoding="utf8") as file_word, \
            open(tags_path, "w", encoding="utf8") as file_tag, \
            open(data_path, "r", encoding="utf8") as file_read:
        wlist = []
        taglist = []
        for line in file_read:
            if len(line.strip()) == 0:
                file_word.write(" ".join(wlist) + "\n")
                file_tag.write(" ".join(taglist) + "\n")
                wlist = []
                taglist = []
            else:
                items = line.rstrip("\n").split(" ")
                if len(items) == 2:
                    wlist.append(items[0])
                else:
                    taglist.append("")
                taglist.append(items[-1])

        if len(wlist) > 0:
            file_word.write(" ".join(wlist) + "\n")
            file_tag.write(" ".join(taglist) + "\n")


def trans_zero_one_label(tags_path, tags01_path):
    with open(tags01_path, "w", encoding="utf8") as file_write:
        with open(tags_path, "r", encoding="utf8") as file_read:
            for line in file_read:
                tags = line.strip().split(" ")
                new_tags = []
                i = 0
                while i < len(tags):
                    if tags[i].startswith("B-"):
                        label = tags[i].split("-")[-1]
                        j = i+1
                        while j < len(tags) and tags[j] == "I-" + label:
                            j += 1
                            new_tags.append("0")
                        if j == i+1:
                            new_tags.append("1")
                        i = j
                    else:
                        new_tags.append("1")
                        i += 1
                new_tags.append("1")
                file_write.write(" ".join(new_tags) + "\n")


def trans_pred(data_path, ptags_path, words_path):
    with open(ptags_path, "w", encoding="utf8") as file_ptag, \
            open(data_path, "r", encoding="utf8") as file_read, \
            open(words_path, "r", encoding="utf8") as file_word:
        words_list = [line.strip() for line in file_word]
        taglist = []
        wlist = []
        words_with_entity_dict = {}
        for line in file_read:
            if len(line.strip()) == 0:
                words_with_entity_dict[" ".join(wlist)] = " ".join(taglist)
                wlist = []
                taglist = []
            else:
                items = line.rstrip("\n").split(" ")
                if len(items) != 3:
                    print(line)
                    traceback.print_exc()
                wlist.append(items[0])
                if items[-1] != "O":
                    taglist.append("1")
                else:
                    taglist.append("0")

        if len(wlist) > 0:
            words_with_entity_dict[" ".join(wlist)] = " ".join(taglist)

        for sent in words_list:
            if sent not in words_with_entity_dict:
                taglist = " ".join(["0"] * len(sent.split(" ")))
            else:
                taglist = words_with_entity_dict[sent]
            file_ptag.write(taglist + "\n")


def trans2entity_label(data_path, words_path, tags_path):
    with open(words_path, "w", encoding="utf8") as file_word, \
            open(tags_path, "w", encoding="utf8") as file_tag, \
            open(data_path, "r", encoding="utf8") as file_read:
        wlist = []
        taglist = []
        for line in file_read:
            if len(line.strip()) == 0:
                if len([tag for tag in taglist if tag != "O"]) > 0:
                    file_word.write(" ".join(wlist) + "\n")
                    file_tag.write(" ".join(taglist) + "\n")
                wlist = []
                taglist = []
            else:
                items = line.rstrip("\n").split(" ")
                if len(items) == 2:
                    wlist.append(items[0])
                else:
                    taglist.append("")
                if items[-1] in ["B-MODEL", "B-BRAND", "B-PRODUCTNAME"]:
                    new_label = "B-ENTITY"
                elif items[-1] in ["I-MODEL", "I-BRAND", "I-PRODUCTNAME"]:
                    new_label = "I-ENTITY"
                else:
                    new_label = "O"
                taglist.append(new_label)

        if len(wlist) > 0:
            file_word.write(" ".join(wlist) + "\n")
            file_tag.write(" ".join(taglist) + "\n")


if __name__ == "__main__":
    stdopinion = 10000
    # data_dir = "E:/nlp_experiment/typical_opinion_extract/sequence_label/"
    data_dir = "E:/nlp_experiment/auto-ner/gpu/"

    trans_zero_one_label("/data/kongyy/nlp/tf_ner_guillaumegenthial/example/10001/train.tags.txt",
                         "D:/workspace/tensorflow/tf_ner/data/example/train.tags01.txt")

    trans_zero_one_label("/data/kongyy/nlp/tf_ner_guillaumegenthial/example/10001/test.tags.txt",
                         "D:/workspace/tensorflow/tf_ner/data/example/test.tags01.txt")

    # for data_type in ["test", "train"]:
    #     if len(str(stdopinion)) == 0:
    #         trans2entity_label(data_dir + "{}.txt".format(data_type),
    #                            data_dir + "{}.words.txt".format(data_type),
    #                            data_dir + "{}.tags.txt".format(data_type),
    #                            )
    #     else:
    #         trans2entity_label(data_dir + "{}/{}.txt".format(stdopinion, data_type),
    #                            data_dir + "{}/{}.words.txt".format(stdopinion, data_type),
    #                            data_dir + "{}/{}.tags.txt".format(stdopinion, data_type),
    #                            )

    # trans_pred(data_dir + "{}/{}.preds.txt".format(stdopinion, data_type),
    #            data_dir + "{}/{}.ptags.txt".format(stdopinion, data_type),
    #            data_dir + "{}/{}.words.txt".format(stdopinion, data_type)
    #            )
