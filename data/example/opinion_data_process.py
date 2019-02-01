# -*- coding:utf-8 -*-
"""
file name: ner_data_process.py
Created on 2019/1/16
@author: kyy_b
@desc: 将语料转为 BIO 标注方式
"""
import re
import sys
import os
from collections import Counter
import traceback
from random import shuffle
import random
import jieba
import re
import pandas as pd
import gensim
from sif_sent2vec import SIFSent2Vec
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

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

    # for k, v in stdopinion_id2cnt.items():
    #     print(k, opinion_id2std[k], v)

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


def get_opinion_desc(data_path, stdopinion="", min_cnt=100):
    """
    选择仅仅包含有 opinion_desc 的文本
    :param data_path:
    :return:
    """
    stdopinion_id2cnt, opinion_raw2std, opinion_std2id = org_opinoin(work_dir + "merge-tag-v2.xlsx")
    corpus = set()
    tag_dist = {}

    with open(data_path, "r", encoding='utf8') as file_read:
        for line in file_read:
            opinion, opinion_desc, text = line.strip().replace(" ", "").replace("　", "").split("\t")
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
            if opinion_raw2std[opinion] == stdopinion:  # 不包含指定的stdopinion
                continue

            opinion_id = opinion_std2id[opinion_raw2std[opinion]]

            if opinion_id > 50:
                continue

            if stdopinion_id2cnt[opinion_id] > min_cnt and random.random() > min_cnt / stdopinion_id2cnt[opinion_id]:
                continue

            if opinion_id not in tag_dist:
                tag_dist[opinion_id] = 0
            tag_dist[opinion_id] += 1

            corpus.add(opinion_desc)

    print("opinion cnt desc")
    for k, v in tag_dist.items():
        print(k, v)

    return corpus


def generate_samples(data_path, dst_path, level="char", min_cnt=3000, is_std=False, stdopinion=""):
    """
    生成样本，每种意见对应的样本量差异较大，需要采样
    :param data_path:
    :param dst_path:
    :param level:
    :param is_std: corpus中的opinion列是不是stdopinion
    :return:
    """
    stdopinion_id2cnt, opinion_raw2std, opinion_std2id = org_opinoin(work_dir + "merge-tag-v2.xlsx")
    corpus = []
    tag_dist = {}
    # min_cnt = min(list(stdopinion_id2cnt.values())) * 3

    with open(data_path, "r", encoding='utf8') as file_read:
        for line in file_read:
            opinion, opinion_desc, text = line.strip().replace(" ", "").replace("　", "").split("\t")
            opinion = re.sub("\(\d+\)", "", opinion).replace("(", "（").replace(")", "）")

            if len(text) > max_seq_len:
                sentence = break_long_sentence(text, opinion_desc)
                if sentence not in ["-1", "-2"]:
                    text = sentence
                else:
                    text = ""

            if len(text) == 0:
                continue

            if not is_std and opinion not in opinion_raw2std:
                continue
            if is_std and opinion not in opinion_std2id:
                continue
            if is_std and opinion != stdopinion:
                continue

            # 筛选给定 stdopinion 对应的语料
            if stdopinion != "" and not is_std and stdopinion != opinion_raw2std[opinion]:
                continue
            if stdopinion != "" and is_std and stdopinion != opinion:
                continue

            if not is_std:
                opinion_id = opinion_std2id[opinion_raw2std[opinion]]
            else:
                opinion_id = opinion_std2id[opinion]

            if opinion_id > 50:
                continue

            if stdopinion_id2cnt[opinion_id] > min_cnt and random.random() > min_cnt / stdopinion_id2cnt[opinion_id]:
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

    if len(stdopinion) > 0:
        dst_path = os.path.join(dst_path, str(opinion_id))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    if not dst_path.endswith("/"):
        dst_path += "/"

    # 非给定 stdopinion 语料
    non_stdopinion_corpus = get_opinion_desc(data_path, stdopinion)

    train_ratio = 0.8
    test_ratio = 0.2
    # val_ratio = 1 - train_ratio - test_ratio

    with open(dst_path + "train.txt", "w", encoding="utf8") as file_train, \
            open(dst_path + "test.txt", "w", encoding="utf8") as file_test, \
            open(dst_path + "dev.txt", "w", encoding="utf8") as file_dev:

        idx = 0
        _total = len(non_stdopinion_corpus)
        for text in non_stdopinion_corpus:
            idx += 1
            if idx <= _total:
                file_write = file_train
            elif idx <= _total * (train_ratio + test_ratio):
                file_write = file_test
            else:
                file_write = file_dev

            if level == "char":
                wlist = list(text)
            else:
                wlist = text.split(" ")
            for w in wlist:
                file_write.write(w + " O\n")
            file_write.write("\n")

        idx = 0
        for text in text_list:
            _tag_list = corpus_dict[text]
            idx += 1
            if idx < total * train_ratio:
                file_write = file_train
            elif idx <= total * (train_ratio + test_ratio):
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

            file_write.write("\n")


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


def load_embeddings(we_path):
    """
    'E:/Work/jobs/data/ner/vectors-segsentence.bin.gz'
    :param we_path:
    :return:
    """
    return gensim.models.KeyedVectors.load_word2vec_format(we_path, binary=False, unicode_errors="ignore")


def get_sif_vec(corpus, we_path, embedding_size=200):
    word_embeddings = load_embeddings(we_path)
    sif_vec = SIFSent2Vec(corpus, word_embeddings, embedding_size, stop_words_set=None)  # list
    return sif_vec.sentence_vecs


def _cosine_similarity(seed_vecs, candidate_vecs):
    X = np.array(seed_vecs)
    # Y = np.array(candidate_vecs)
    # print(X.shape, Y.shape)
    return cosine_similarity(X, candidate_vecs)


def show_result(cosine_matrix, seed_corpus, candidate_corpus, dst_path, topN=10):
    rst = np.argsort(cosine_matrix)
    with open(dst_path, "w", encoding="utf8") as file_write:
        for idx, sentence in enumerate(seed_corpus):
            idx_arr = rst[idx, -topN:]
            score_arr = cosine_matrix[idx, idx_arr]
            file_write.write("".join(sentence) + "\n")
            for i in range(topN - 1, -1, -1):
                file_write.write("\t" + "".join(candidate_corpus[idx_arr[i]]) + "\t" + str(score_arr[i]) + "\n")


def labelling_by_machine(src_path, dst_path, we_path):
    stdopinion_id2cnt, opinion_raw2std, opinion_std2id = org_opinoin(work_dir + "merge-tag-v2.xlsx")

    with open(src_path, "r", encoding="utf8") as file_read, open(dst_path, "w", encoding="utf8") as file_write:
        labeled_opinion_desc_set = set()
        sub_text_set = set()
        opinion_desc_to_stdopinion = {}
        re_sent_cut = re.compile(",|，|。|？|\?|!|！|;|；")
        for line in file_read:
            try:
                opinion, opinion_desc, text = line.strip().split("\t")
                opinion = re.sub("\(\d+\)", "", opinion.replace(" ", ""))
                stdopinion = opinion_raw2std[opinion]
                opinion_id = opinion_std2id[stdopinion]
                if 32 < int(opinion_id) < 50:
                    labeled_opinion_desc_set.add(opinion_desc)
                    opinion_desc_to_stdopinion[opinion_desc] = stdopinion
                    sents = re_sent_cut.split(text)
                    sub_text_set |= set(sents)
            except:
                print(line.strip())
                traceback.print_exc()
        sub_text_list = list(sub_text_set)
        labeled_opinion_desc_list = list(labeled_opinion_desc_set)
        print("sub_text_list len = ", len(sub_text_list))
        print("labeled_opinion_desc_list len = ", len(labeled_opinion_desc_list))

        sif_sent_vec = get_sif_vec(labeled_opinion_desc_list + sub_text_list, we_path, embedding_size=embedding_size)
        assert len(sif_sent_vec) == len(sub_text_list) + len(labeled_opinion_desc_list)

        labeled_opinion_desc_array = np.array(sif_sent_vec[:len(labeled_opinion_desc_list)])
        text_vecs = sif_sent_vec[len(labeled_opinion_desc_list):]

        batch_size = 100
        batch_num = int(len(text_vecs) / batch_size) + 1
        for i in range(batch_num):
            cur_vecs = text_vecs[i * batch_size:(i + 1) * batch_size]
            cosine_matrix = _cosine_similarity(cur_vecs, labeled_opinion_desc_array)
            rst = np.argsort(cosine_matrix)
            cur_texts = sub_text_list[i * batch_size:(i + 1) * batch_size]
            for idx, sentence in enumerate(cur_texts):
                idx_max = rst[idx, -1]
                score = cosine_matrix[idx, idx_max]
                # file_write.write("".join(sentence) + "\n")
                file_write.write(
                    str(opinion_desc_to_stdopinion[labeled_opinion_desc_list[idx_max]]) + "\t" + sentence + "\t" + str(
                        score) + "\n")


def check_model_results(data_path):
    with open(data_path, "r", encoding="utf-8") as file_read, \
            open(data_path[:-3] + "check", "w", encoding="utf8") as file_pred:
        wlist = []
        taglist_true = []
        taglist_pred = []
        for line in file_read:
            if len(line.strip()) == 0:
                if "".join(taglist_pred) != "".join(taglist_true):
                    file_pred.write("\t".join(wlist) + "\n")
                    file_pred.write("\t".join(taglist_true) + "\n")
                    file_pred.write("\t".join(taglist_pred) + "\n")
                wlist = []
                taglist_true = []
                taglist_pred = []
            else:
                items = line.rstrip("\n").split(" ")
                if len(items) == 3:
                    wlist.append(items[0])
                else:
                    wlist.append("")
                taglist_true.append(items[1])
                taglist_pred.append(items[2])


if __name__ == "__main__":
    import platform

    if platform.system() == "Windows":
        work_dir = "E:/nlp_experiment/typical_opinion_extract/"
    else:
        work_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/example/"

    embedding_size = 100

    # get_tag_set(work_dir + "typical_opinion_corpus")

    # stats_sentence_len_dist(work_dir + "typical_opinion_corpus")

    # generate_samples(work_dir + "typical_opinion_corpus",
    #                  work_dir + "sequence_label/",
    #                  level="char", stdopinion="胎噪", min_cnt=3000, is_std=False)

    corpus = get_opinion_desc(work_dir + "typical_opinion_corpus", "胎噪", min_cnt=1000)
    with open(work_dir + "胎噪_extract","w",encoding="utf8") as file_write:
        for text in corpus:
            file_write.write(" ".join(list(text)) + "\n")

    # check_model_results(work_dir + "sequence_label/check/train.preds.txt")

    # labelling_by_machine(work_dir + "typical_opinion_corpus_seg", work_dir + "typical_opinion_corpus_opt",
    #                      we_path="/data/kongyy/nlp/word_vectors/typical_opinion_token_{}.txt".format(embedding_size))

    # stats_token(work_dir + "typical_opinion_corpus_seg", "腿部空间")

    # get_label_dist(work_dir + "ner_corpus")

    # check_completeness_of_labelling(work_dir + "typical_opinion_corpus",
    #                                 work_dir + "opinion_kws.txt",
    #                                 work_dir + "opinion_completeness")

    # get_raw_tag_dist(work_dir + "opinion_completeness", work_dir + "merge-tag-v2.xlsx")
