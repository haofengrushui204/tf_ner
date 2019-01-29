# -*- coding:utf-8 -*-
"""
file name: sif_sent2vec.py
Created on 2019/1/14
@author: kyy_b
@desc:
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List
from collections import Counter

import pickle


class SIFSent2Vec:
    def __init__(self, corpus, word_embeddings, embedding_size, a=1.0e-3, stop_words_set=None):
        """
        分词后的list
        :param corpus:
        """
        self.embedding_size = embedding_size
        self.a = a
        self.corpus = corpus
        self.we = word_embeddings
        self.word_freq = {}
        for sentence in self.corpus:
            if isinstance(sentence, str):
                sentence = sentence.split(" ")
            self.word_freq.update(dict(Counter(sentence)))

        total = sum(self.word_freq.values())
        self.word_freq = {k: v / total for (k, v) in self.word_freq.items()}

        print("sif vocab size = ", len(self.word_freq))

        self.sentence_vecs = self.sentence_to_vec()

    def get_word_frequency(self, word):
        if word in self.word_freq:
            return self.word_freq[word]
        return 0.0001

    def sentence_to_vec(self):
        sentence_set = []
        for sentence in self.corpus:
            if isinstance(sentence, str):
                sentence = sentence.split(" ")
            vs = np.zeros(self.embedding_size)  # add all word2vec values into one vector for the sentence
            sentence_length = len(sentence)
            for word in sentence:
                if word in self.we.wv:
                    a_value = self.a / (self.a + self.get_word_frequency(self.get_word_frequency(word)))
                    vs = np.add(vs, np.multiply(a_value, self.we.wv[word]))  # vs += sif * word_vector

            vs = np.divide(vs, sentence_length)  # weighted average
            sentence_set.append(vs)  # add to our existing re-calculated set of sentences

        assert len(self.corpus) == len(sentence_set)

        # calculate PCA of this sentence set
        pca = PCA()
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        # pad the vector?  (occurs if we have less sentences than embeddings_size)
        if len(u) < self.embedding_size:
            for i in range(self.embedding_size - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs

        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u, vs)
            sentence_vecs.append(np.subtract(vs, sub))

        # return zip([" ".join(sentence) for sentence in self.corpus], sentence_vecs)
        # return dict(zip([" ".join(sentence) for sentence in self.corpus], sentence_vecs))
        return sentence_vecs

    def save(self, path):
        """
        save to pickle
        :return:
        """
        pickle.dump(self.sentence_vecs, open(path, "wb"))

    @staticmethod
    def load(path):
        """
        load the pickle model
        :return:
        """
        return pickle.load(open(path, "rb"))
