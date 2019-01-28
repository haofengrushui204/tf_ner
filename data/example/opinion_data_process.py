# -*- coding:utf-8 -*-
"""
file name: ner_data_process.py
Created on 2019/1/16
@author: kyy_b
@desc: 将语料转为 BIO 标注方式
"""
import re
import sys
from collections import Counter
import traceback
from random import shuffle
import random
import jieba
import re
import pandas as pd

max_seq_len = 64

inline_marks_dict = {}
inline_marks = []
# break_marks = {"。", "?", "？", "!", "！", ";", "；"}
break_marks = {"。", "?", "？", "!", "！"}
stop_words = {"的", "了", "呢", "哎"}


def break_long_doc(doc):
    """
    长文本分句
    :param doc: 输入长文档
    :return: 句子list
    """
    num_words = len(doc)
    if num_words == 0:
        return doc

    mark_char = ""
    mark_pos = 0
    sent_list = []
    for i in range(num_words):
        if doc[i] in inline_marks_dict:
            if mark_char == doc[i]:
                sent_list.append(doc[mark_pos:i + 1])
                mark_pos = i + 1
                mark_char = ""
            else:
                if mark_pos != i:
                    sent_list.append(doc[mark_pos:i])
                    mark_pos = i
                mark_char = inline_marks_dict[doc[i]]
        elif mark_char == "":
            if doc[i] in break_marks or i - mark_pos + 1 >= max_seq_len:
                sent_list.append(doc[mark_pos:i + 1])
                mark_pos = i + 1
        elif i - mark_pos + 1 >= max_seq_len:
            sent_list.append(doc[mark_pos:i + 1])
            mark_pos = i + 1
            mark_char = ""
    if mark_pos < num_words:
        sent_list.append(doc[mark_pos:num_words])

    return sent_list


def break_long_sentence(sentence, red_content):
    """
    :param sentence:
    :param red_content: 长段文本中，标红的短文本
    :return:
    """
    if len(red_content) >= max_seq_len:
        return "-1"

    try:
        rst = re.search(re.escape(red_content), sentence)
    except:
        traceback.print_exc()
        print(red_content)
        print(sentence)
        sys.exit(0)
    if rst is not None:
        left, right = rst.span()
        for i in range(right + 1, len(sentence)):  # 跳过最近的标点符号
            if sentence[i] in {"！", "!", "。", "?", "?", ";", "；"}:
                right = i
                break
        if right < max_seq_len:
            return sentence[right + 1]
        for i in range(left - 2, -1, -1):  # 跳过最近的标点符号
            if sentence[i] in {"！", "!", "。", "?", "?", ";", "；"}:
                left = i
                break
        text = sentence[left:right + 1]
        if len(text) < max_seq_len:
            return text
        if len(sentence[left:rst.span()[-1]]) < max_seq_len:
            return sentence[left:rst.span()[-1]]

        return sentence[right - max_seq_len:rst.span()[-1]]
    return "-2"


def get_tag_set(data_path):
    """
    从 抓取的 opinion-content 平行语料中 解析出不同的 opinion
    :param data_path:
    :return:
    """
    with open(data_path, "r", encoding="utf-8") as file_read:
        tag_list = [re.sub("\(\d+\)", "", line.split("\t")[0]) for line in file_read]
        tag_dict = dict(Counter(tag_list))

        for tag, c in tag_dict.items():
            print(tag, "\t", c)


def stats_sentence_len_dist(data_path):
    """
    统计 口碑中包含典型意见的 sentence 的 长度分布
    :param data_path:
    :return:
    """
    from operator import itemgetter
    data_len = []
    tag_not_in_sentence_cnt = 0
    with open(data_path, "r", encoding="utf8") as file_read:
        for line in file_read:
            tag, tagcontent, text = line.strip().replace(" ", "").split("\t")
            if len(text) > max_seq_len:
                sentence = break_long_sentence(text, tagcontent)
                if sentence == "-1":
                    print(tagcontent)
                elif sentence == "-2":
                    tag_not_in_sentence_cnt += 1
                else:
                    data_len.append(len(sentence))
            else:
                data_len.append(len(text))

        print(sorted(dict(Counter(data_len)).items(), key=itemgetter(0), reverse=True))
        print("total = ", len(data_len))
        print(tag_not_in_sentence_cnt)


def org_opinoin(data_path):
    """
    对细粒度的 Opinion 进行归并，归并后的opinion 称为 std_opinion
    :param data_path:
    :return:
    """
    opinion_df = pd.read_excel(data_path, sheet_name="类别展开", names=["opinion", "cnt", "stdopinion"], header=None)
    stdopinion_cnt_df = opinion_df.groupby("stdopinion", as_index=False) \
        .sum() \
        .sort_values(by=["cnt"], ascending=False)

    opinion_raw2std = {}
    opinion_std2id = {}
    stdopinion_id2cnt = {}
    opinion_id2std = {}

    for index, row in opinion_df.iterrows():
        opinion_raw2std[row["opinion"].strip()] = row["stdopinion"].strip()

    index = 1
    for _, row in stdopinion_cnt_df.iterrows():
        opinion_std2id[row["stdopinion"].strip()] = index
        opinion_id2std[index] = row["stdopinion"].strip()
        stdopinion_id2cnt[index] = row["cnt"]
        index += 1

    for k, v in stdopinion_id2cnt.items():
        print(k, opinion_id2std[k], v)

    return stdopinion_id2cnt, opinion_raw2std, opinion_std2id


def get_label_dist(data_path):
    label_list = []
    with open(data_path, "r", encoding="utf8") as file_read:
        for line in file_read:
            if len(line.strip()) > 0:
                items = line.strip().split(" ")
                if items[-1].startswith("B-"):
                    label_list.append(items[-1])

    print("label dist")
    print(dict(Counter(label_list)).values())
    print("total samples")
    print(sum(dict(Counter(label_list)).values()))


def generate_samples(data_path, dst_path, level="char", min_cnt=2000):
    """
    生成样本，每种意见对应的样本量差异较大，需要采样
    :param data_path:
    :param dst_path:
    :param level:
    :return:
    """
    stdopinion_id2cnt, opinion_raw2std, opinion_std2id = org_opinoin(work_dir + "merge-tag-v2.xlsx")
    corpus = []
    tag_dist = {}
    # min_cnt = min(list(stdopinion_id2cnt.values())) * 3

    with open(data_path, "r", encoding='utf8') as file_read:
        for line in file_read:
            opinion, opinion_desc, text = line.strip().replace(" ", "").split("\t")
            opinion = re.sub("\(\d+\)", "", opinion).replace("(", "（").replace(")", "）")

            if len(text) > max_seq_len:
                sentence = break_long_sentence(text, opinion_desc)
                if sentence not in ["-1", "-2"]:
                    text = sentence
                else:
                    text = ""

            if len(text) == 0:
                continue

            if opinion not in opinion_raw2std:
                continue

            opinion_id = opinion_std2id[opinion_raw2std[opinion]]

            if stdopinion_id2cnt[opinion_id] > min_cnt and random.random() > min_cnt / stdopinion_id2cnt[opinion_id]:
                continue

            if stdopinion_id2cnt[opinion_id] < min_cnt:
                continue

            if opinion_id not in tag_dist:
                tag_dist[opinion_id] = 0
            tag_dist[opinion_id] += 1

            try:
                rst = re.search(re.escape(opinion_desc), text)
            except:
                print(text)
                traceback.print_exc()
                continue
            if rst is not None:
                rst = rst.span()
                corpus.append([rst, opinion_id, text])

    corpus_dict = {}
    for item in corpus:
        if item[-1] not in corpus_dict:
            corpus_dict[item[-1]] = []
        corpus_dict[item[-1]].append(item[:2])
    text_list = list(corpus_dict.keys())

    print("opinion cnt desc")
    for k, v in tag_dist.items():
        print(k, v)

    shuffle(text_list)
    total = len(text_list)
    with open(dst_path + "train.txt", "w", encoding="utf8") as file_train, \
            open(dst_path + "test.txt", "w", encoding="utf8") as file_test, \
            open(dst_path + "dev.txt", "w", encoding="utf8") as file_dev:
        idx = 0
        for text in text_list:
            _tag_list = corpus_dict[text]
            idx += 1
            if idx < total * 0.8:
                file_write = file_train
            elif idx < total * 1.0:
                file_write = file_test
            else:
                file_write = file_dev

            if level == "char":
                wlist = list(text)
                tag_list = ["O"] * len(wlist)
                for item in _tag_list:
                    rng, tag_id = item[:]
                    tag_list[rng[0]] = "B-{}".format(tag_id)
                    for i in range(rng[0] + 1, rng[-1]):
                        tag_list[i] = "I-{}".format(tag_id)
                for i in range(len(wlist)):
                    file_write.write(wlist[i] + " " + tag_list[i] + "\n")
            else:
                wlist = jieba.lcut(text)
                tag_list = ["O"] * len(wlist)
                for item in _tag_list:
                    rng, tag_id = item[:]
                    start_idx = 0
                    for i, w in enumerate(wlist):
                        if sum([len(w) for w in wlist[:i + 1]]) >= rng[0]:
                            start_idx = i
                            break
                    end_idx = 0
                    for j in range(start_idx + 1, len(wlist)):
                        if sum([len(w) for w in wlist[:j + 1]]) >= rng[1]:
                            end_idx = j
                            break
                    tag_list[start_idx] = "B-{}".format(tag_id)
                    for i in range(start_idx + 1, end_idx + 1):
                        tag_list[i] = "I-{}".format(tag_id)
                for i in range(len(wlist)):
                    file_write.write(wlist[i] + " " + tag_list[i] + "\n")

            file_write.write("\n\n")


def check_completeness_of_labelling(opinion_path, opinion_kws_path, opinion_completeness_path):
    """
    校验每个样本标注的完整性， 方式是一种粗检测方式：check key words
    存在很多的样本，对某个label，仅仅标注的了部分
    :param opinion_path 包含典型意见的全部文本
    :param opinion_kws_path 典型意见对应的关键词
    :return: 具有完整典型意见标注的样本
    """
    with open(opinion_kws_path, "r", encoding="utf8") as file_read:
        kws = "|".join([line.strip() for line in file_read])

    re_kws = re.compile(kws)

    with open(opinion_completeness_path, "w", encoding="utf8") as file_write:
        with open(opinion_path, "r", encoding='utf8') as file_read:
            for line in file_read:
                tag, tagcontent, text = line.strip().replace(" ", "").split("\t")
                # tag = re.sub("\(\d+\)", "", line.split("\t")[0]).replace("(", "（").replace(")", "）")
                content_residual = re.sub(re.escape(tagcontent), "", text)

                if re_kws.search(content_residual) is not None:
                    continue

                file_write.write(line)


def check_coverage_rate_of_tokens():
    """
    检查采样后的样本中 token or char 的覆盖率
    :return:
    """


def get_raw_tag_dist(typical_opinion_corpus, std_tag_path):
    _, raw_opinion_to_std, _ = org_opinoin(std_tag_path)
    tag_cnt = {}
    with open(typical_opinion_corpus, "r", encoding="utf8") as file_read:
        for line in file_read:
            tag = line.strip().replace(" ", "").split("\t")[0]
            tag = re.sub("\(\d+\)", "", tag).replace("(", "（").replace(")", "）")
            if tag not in raw_opinion_to_std:
                print(tag)
            else:
                std_tag = raw_opinion_to_std[tag]
                if std_tag not in tag_cnt:
                    tag_cnt[std_tag] = 0
                tag_cnt[std_tag] += 1

    print(len(tag_cnt))
    rst = sorted(tag_cnt.items(), key=lambda x: x[1], reverse=True)
    for r in rst:
        print(r[0], "\t", r[1])


def stats_token(data_path, stdopinion):
    """
    给定 stdopinion, 统计该 opinion 对应的 token分布
    :param data_path:
    :param stdopinion:
    :return:
    """
    stdopinion_id2cnt, opinion_raw2std, opinion_std2id = org_opinoin(work_dir + "merge-tag-v2.xlsx")
    opinion_set = set([raw for (raw, std) in opinion_raw2std.items() if std == stdopinion])

    token_freq = {}
    with open(data_path, "r", encoding="utf8") as file_read:
        for line in file_read:
            try:
                opinion, opinion_desc, text = line.strip().split("\t")
                if re.sub("\(\d+\)", "", opinion.replace(" ", "")) in opinion_set:
                    tokens = text.split(" ")
                    for token in tokens:
                        if len(re.sub("([\u4E00-\u9FD5]+)", "", token)) == 0:
                            # if len(token) > 0:
                            if token not in token_freq:
                                token_freq[token] = 0
                            token_freq[token] += 1
            except:
                print(line.strip())

    token_freq_sorted = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    with open(work_dir + stdopinion + "_token_freq.txt", "w", encoding="utf8") as file_write:
        for (token, cnt) in token_freq_sorted:
            file_write.write(token + "\t" + str(cnt) + "\n")


def labelling_by_machine(src_path, dst_path):
    with open(src_path, "r", encoding="utf8") as file_read, open(dst_path, "w", encoding="utf8") as file_write:
        labeled_opinion_desc_set = set()
        sub_text_set = set()
        re_sent_cut = re.compile(",|，|。|？|\?|!|！|;|；")
        for line in file_read:
            try:
                _, opinion_desc, text = line.strip().split("\t")
                labeled_opinion_desc_set.add(opinion_desc)
                sents = re_sent_cut.split(text)
                sub_text_set |= set(sents)
            except:
                print(line.strip())
        sub_text_list = list(sub_text_set)
        labeled_opinion_desc_list = list(labeled_opinion_desc_set)


if __name__ == "__main__":
    work_dir = "E:/work/nlp_experiment/typical_opinion_extract/"
    # get_tag_set(work_dir + "typical_opinion_corpus")

    # stats_sentence_len_dist(work_dir + "typical_opinion_corpus")

    # generate_samples(work_dir + "typical_opinion_corpus", work_dir + "ner/", level="char")

    stats_token(work_dir + "typical_opinion_corpus_seg", "腿部空间")

    # get_label_dist(work_dir + "ner_corpus")

    # check_completeness_of_labelling(work_dir + "typical_opinion_corpus",
    #                                 work_dir + "opinion_kws.txt",
    #                                 work_dir + "opinion_completeness")

    # get_raw_tag_dist(work_dir + "opinion_completeness", work_dir + "merge-tag-v2.xlsx")
