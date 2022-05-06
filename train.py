# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 17:09
# @Author  : Zhao Zijun
# @Email   : uchihazzj@outlook.com
# @File    : train.py
# @Software: PyCharm
import os

import yaml
import sys
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.models import Word2Vec

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml

np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd

sys.setrecursionlimit(1000000)  # 递归的最大深度
# %% set parameters:
vocab_dim = 100
n_iterations = 1  # ideally more..
n_exposures = 10  # 词频数少于10的截断
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 99  # LSTM输入 注意与下长度保持一致
maxlen = 99  # 统一句长
cpu_count = multiprocessing.cpu_count()
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
afterW_folder = os.path.join(BASE_DIR, "data_clean")
good_filepath = os.path.join(afterW_folder, "good.csv")
bad_filepath = os.path.join(afterW_folder, "bad.csv")

# %%加载训练文件
def loadfile():
    neg = pd.read_csv(good_filepath, header=0)
    pos = pd.read_csv(bad_filepath, header=0)

    old_combined = np.concatenate((pos, neg))
    combined = [str(temp_df[0]) for temp_df in old_combined]
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))  # 添加标注

    return combined, y


# %%对句子进行分词，并去掉换行符
def tokenizer(text_tok):
    # for document_temp in text_tok:
    #     print(type(document_temp))
    text_ret = [jieba.lcut(document_temp.replace('\n', '')) for document_temp in text_tok]
    # print(text_ret)
    return text_ret


# %%创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 前方补0 为了进入LSTM的长度统一
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# %%创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(vector_size=vocab_dim,  # 特征向量维度
                     min_count=n_exposures,  # 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
                     window=window_size,  # 窗口大小，表示当前词与预测词在一个句子中的最大距离是多少
                     workers=cpu_count,  # 用于控制训练的并行数
                     epochs=n_iterations)
    model.build_vocab(combined)  # 创建词汇表， 用来将 string token 转成 index
    model.train(combined, total_examples=model.corpus_count, epochs=10)
    model.save(os.path.join(BASE_DIR, "Word2vec_model.pkl"))  # 保存训练好的模型
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined  # word_vectors字典类型{word:vec}


# %%最终的数据准备
def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]

    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


# %%定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever #堆叠
    # 嵌入层将正整数（下标）转换为具有固定大小的向量
    model.add(Embedding(output_dim=vocab_dim,  # 词向量的维度
                        input_dim=n_symbols,  # 字典(词汇表)长度
                        mask_zero=True,  # 确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input
    # Length#当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。

    # 输入数据的形状为188个时间长度（句子长度），每一个时间点下的样本数据特征值维度（词向量长度）是100。
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    # 输出的数据，时间维度仍然是188，每一个时间点下的样本数据特征值维度是50
    model.add(Dropout(0.5))
    model.add(Dense(1))  # 全连接层
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_test, y_test))
    # 22208s 7ms/step - loss: 0.3218 - accuracy: 0.8660 - val_loss: 0.3307 - val_accuracy: 0.8601

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open(os.path.join(BASE_DIR, "lstm.yml"), 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights(os.path.join(BASE_DIR, "lstm.h5"))
    print('Test score:', score)  # Test score: [0.33069735049322874, 0.8600956797599792]


# %%
# 训练模型，并保存
def train():
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    train()
