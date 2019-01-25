# -*- coding:utf-8 -*-
"""
@file name: trans_conll_to_this.py
Created on 2019/1/24
@author: kyy_b
@desc: 将 conll 格式的标注文件转为本程序可用的标注文件
"""


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


if __name__ == "__main__":
    data_type = "dev"
    trans("E:/nlp_experiment/typical_opinion_extract/ner/{}.txt".format(data_type),
          "E:/nlp_experiment/typical_opinion_extract/ner/{}.words.txt".format(data_type),
          "E:/nlp_experiment/typical_opinion_extract/ner/{}.tags.txt".format(data_type),
          )
