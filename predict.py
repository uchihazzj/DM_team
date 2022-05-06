# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 23:35
# @Author  : Zhao Zijun
# @Email   : uchihazzj@outlook.com
# @File    : predict.py
# @Software: PyCharm

# %%测试模型
import os

import jieba
import numpy as np
import yaml
from gensim.models import Word2Vec
from keras.engine.saving import model_from_yaml

from train import create_dictionaries

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load(os.path.join(BASE_DIR, "Word2vec_model.pkl"))
    _, _, combined = create_dictionaries(model, words)
    return combined


def lstm_predict(string):
    print('loading model......')
    with open(os.path.join(BASE_DIR, "lstm.yml"), 'r') as f:
        # yaml_string = yaml.load(f)
        yaml_string = yaml.load(f, Loader=yaml.FullLoader)
        # yaml.load(f, Loader=yaml.FullLoader)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights(os.path.join(BASE_DIR, "lstm.h5"))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    # print data
    result = model.predict_classes(data)
    if result[0][0] == 1:
        print(string, ' negative')
    else:
        print(string, ' positive')


# %%测试
if __name__ == '__main__':
    # train()
    # string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    #    string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    #    string='酒店的环境非常好，价格也便宜，值得推荐'
    #    string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    #    string='我傻了'
    #    string='你傻了'
    #    string='屏幕较差，拍照也很粗糙。'
    #  string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    while(True):
        string = input("句子 : ")  # 1
        # train()
        lstm_predict(string)
