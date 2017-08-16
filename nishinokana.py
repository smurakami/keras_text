
'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

import ipdb

import numpy as np
import random
import sys
import os

import tensorflow as tf

import MeCab
mecab = MeCab.Tagger("-Owakati")

model_filepath = "data/nishinokana.dat"

class WDict:
    def __init__(self, path):
        """
        filepathにはスペースで分かち書きされた日本語が入っている。
        """
        vocab = set()
        for line in open(path, encoding='utf-8'):
            line = mecab.parse(line)
            line += '_n'
            words = line.strip().split()
            vocab.update(words)

        w2i = {w: np.int32(i+2) for i, w in enumerate(vocab)}
        w2i['<s>'], w2i['</s>'] = np.int32(0), np.int32(1) # 文の先頭・終端記号
        i2w = {i: w for w, i in w2i.items()}

        self.w2i = w2i
        self.i2w = i2w

    def to_i(self, word):
        return self.w2i[word]

    def to_w(self, index):
        return self.i2w[index]

    def encode(self, sentence):
        """
        sentence: wordの配列
        """
        encoded = []
        for w in sentence:
            encoded.append(self.to_i(w))
        return encoded

    def decode(self, sequence):
        words = [self.to_w(i) for i in sequence]
        return ''.join(words)

    def num_words(self):
        return len(self.i2w.keys())


def to_one_hot(x, depth):
    y = np.zeros((len(x), depth), dtype=np.bool)
    y[np.arange(len(x)), x] = True
    return y


def main():
    # path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    path = 'data/nishinokana.txt'
    wdict = WDict(path)

    data = []

    for line in open(path, encoding='utf-8'):
        line = mecab.parse(line)
        if line.strip() == '':
            continue
        line += '_n'
        s = line.strip().split()
        # s = ['<s>'] + s + ['</s>']
        enc = wdict.encode(s)
        data.append(enc)

    data_flatten = sum(data, [])

    # data = sum(data, [])

    # print('corpus length:', len(text))

    # chars = sorted(list(set(text)))
    # print('total chars:', len(chars))
    # char_indices = dict((c, i) for i, c in enumerate(chars))
    # indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 30
    step = 1
    sentences = []
    next_words = []
    for i in range(0, len(data_flatten) - maxlen, step):
        sentences.append(data_flatten[i: i + maxlen])
        next_words.append(data_flatten[i + maxlen])

    print('nb sequences:', len(sentences))

    print('Vectorization...')

    # X = np.zeros((len(sentences), maxlen, wdict.num_words()), dtype=np.bool)
    # y = np.zeros((len(sentences), wdict.num_words()), dtype=np.bool)

    # for i, sentence in enumerate(sentences):
    #     for t, word in enumerate(sentence):
    #         X[i, t, word] = 1
    #     y[i, next_words[i]] = 1

    X = np.array(sentences, dtype=np.int)
    y = np.array(next_words, dtype=np.int)
    # y = np.zeros((len(sentences), wdict.num_words()), dtype=np.bool)
    # for i, word in enumerate(next_words):
    #     y[i, word] = 1
    # y = tf.one_hot(next_words, wdict.num_words())

    # build the model: a single LSTM
    print('Build model...')


    if os.path.exists(model_filepath):
        print('load model')
        model = keras.models.load_model(model_filepath)
    else:
        model = Sequential()
        model.add(Embedding(wdict.num_words(), 128, input_length=maxlen))
        # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(LSTM(64, input_shape=(maxlen, 128)))
        model.add(Dropout(0.2))
        model.add(Dense(wdict.num_words()))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)


    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # train the model, output generated text after each iteration
    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)


        batch_size = 128
        counter = 0

        for i in range(0, len(X), batch_size):
            y_ = to_one_hot(y[i:i+batch_size], wdict.num_words())
            loss = model.train_on_batch(
                X[i:i+batch_size],
                y_
                )
            if counter % 10 == 0:
                print("batch: {}/{}, loss: {}".format(i, len(X), loss))
            counter += 1

        # model.fit(X, y,
        #           batch_size=128,
        #           epochs=1)

        model.save(model_filepath, overwrite=True)

        start_index = random.randint(0, len(data_flatten) - maxlen - 1)


        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = data_flatten[start_index: start_index + maxlen]
            generated += wdict.decode(sentence)
            print('----- Generating with seed: "' + generated + '"')
            sys.stdout.write(generated)

            for i in range(400):
                # x = np.zeros((1, maxlen, wdict.num_words()))
                x = np.array([sentence], dtype=np.int)
                # for t, word in enumerate(sentence):
                #     x[0, t, word] = 1.

                # ipdb.set_trace()

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = wdict.to_w(next_index)

                generated += next_word
                sentence = sentence[1:] + [next_index]

                if next_word == '_n':
                    print()
                else:
                    sys.stdout.write(next_word)
                    sys.stdout.flush()
            print()

main()