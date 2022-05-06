# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 16:23
# @Author  : Zhao Zijun
# @Email   : uchihazzj@outlook.com
# @File    : pltData.py
# @Software: PyCharm
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from matplotlib import font_manager

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# 设置matplotlib绘图时的字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")

# 读取数据
afterW_filename = "comments.csv"
afterW_folder = os.path.join(BASE_DIR, "data_clean")
good_filepath = os.path.join(afterW_folder, "good.csv")
bad_filepath = os.path.join(afterW_folder, "bad.csv")

neg = pd.read_csv(good_filepath, header=0)
pos = pd.read_csv(bad_filepath, header=0)
old_df = np.concatenate((pos, neg))
df = [temp_df[0] for temp_df in old_df]
# %%句子长度分布直方图
Num_len = [len(str(text)) for text in df]
# print(Num_len)
bins_interval = 10  # 区间长度
bins = range(min(Num_len), max(Num_len) + bins_interval - 1, bins_interval)  # 分组
plt.xlim(min(Num_len), max(Num_len))
plt.title("Probability-distribution")
plt.xlabel('Interval')

plt.ylabel('Probability')
# 频率分布normed=True，频次分布normed=False
prob, left, rectangle = plt.hist(x=Num_len, bins=bins, density=True, color=['r'])  # 分布直方图
plt.show()

plt.ylabel('Cumulative distribution')
prob, left, rectangle = plt.hist(x=Num_len, bins=bins, density=True, cumulative=True, histtype='step',
                                 color=['r'])  # 累计分布图
plt.show()

# %%求分位点
import math


def quantile_p(data, p):
    data.sort()
    pos = (len(data) + 1) * p
    # pos = 1 + (len(data)-1)*p
    pos_integer = int(math.modf(pos)[1])
    pos_decimal = pos - pos_integer
    Q = data[pos_integer - 1] + (data[pos_integer] - data[pos_integer - 1]) * pos_decimal
    return Q


quantile = 0.90  # 选取分位数
Q = quantile_p(Num_len, quantile)
print("\n分位点为%s的句子长度:%d." % (quantile, Q))
