import os
import random
import re

import numpy
import torch
from pyfillet import TextEmbedder


class TextReader:
    def __init__(self, input_dir):
        self.text = ''
        files = os.listdir(input_dir)
        for file in files:
            fie_path = os.path.join(input_dir, file)
            if os.path.isfile(fie_path):
                self.text = self.text + ' ' + self.read_text(fie_path)
        self.words = re.findall('\w+', self.text.lower())
        self.embedder = TextEmbedder()
        self.words_emb = self.embedder(' '.join(self.words))
        self.index = 0

    def read_text(self, file_name):
        text = ""
        file_name = file_name
        print("Read text from file '{}'...".format(file_name))
        try:
            text_file = open(file_name, 'r', encoding='windows-1251')
            text = text_file.read()
            text_file.close()
        except Exception:
            print("File '{}' cannot be read".format(file_name))
        return text

    def next_packet(self):
        packet = []
        if self.index <= len(self.words_emb) - 3:
            window = self.words_emb[self.index:self.index + 3]
            print("prefix: {}, {} \tpredition: {}".format(window[0][0], window[1][0], window[-1][0]))
            packet = list(map(lambda word: word[1], window))
            self.index += 1
        return packet

    def word2vector(self, word):
        default_dim = len(self.words_emb[0][1])
        vector = [0.] * default_dim
        word_embs = list(filter(
            lambda w_emb: w_emb[0] == word, self.words_emb))
        if word_embs:
            vector = word_embs[0][1]
        print("for '{}' was found vector: {}".format(word, numpy.mean(vector)))
        return vector

    def nearest_word(self, vector):
        distances = torch.tensor(list(map(
            lambda emb: vector.dist(torch.tensor(emb[1])),
            self.words_emb)))
        idx = torch.argmin(distances)
        word = self.words_emb[idx][0]
        return word

    def random_word(self):
        word_idx = random.randint(0, len(self.words_emb))
        return self.words_emb[word_idx][0]
