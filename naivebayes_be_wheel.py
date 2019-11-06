# -*- coding: utf-8 -*-
# @Created on : 2019/11/6 15:01
# @Author : YYC

"""
    朴素贝叶斯 ，贝叶斯估计  λ=1  K=2， S=3；
    λ=1 拉普拉斯平滑
    1,求p(y)
    2,求p(x|y)
    3,求p(y)*p(x|y)
"""
import csv

import pandas as pd
import numpy as np


class NaiveBayes(object):
    # 初始化贝叶斯估计中的值
    def __init__(self):
        self.a = 1
        self.k = 2
        self.s = 3

    def load_data(self):
        data_set = pd.read_csv('./naivebayes_data.csv')
        data_set_np = np.array(data_set)
        train_set = data_set_np[:, 0:data_set_np.shape[1] - 1]
        labels = data_set_np[:, data_set_np.shape[1] - 1]
        return train_set, labels

    def classify(self, train_set, labels, features):
        labels = list(labels)

        # 求先验概率
        # p(y)
        p_y = {}
        for label in labels:
            p_y[label] = (labels.count(label) + self.a) / float(len(labels) + self.k * self.a)

        # 求条件概率
        # p(x|y)
        p = {}
        for y in p_y.keys():
            y_index = [i for i, label in enumerate(labels) if label == y]  # 找出所有label == y的索引值
            y_count = labels.count(y)  # y 在labels中出现的次数
            for j in range(len(features)):
                x_index = [i for i, x in enumerate(train_set[:,j]) if x == features[j]]  # s所有 x == feature[j] 的索引
                xy_count = len(set(x_index) & set(y_index))
                p_key = str(features[j]) + '|' + str(y)
                p[p_key] = (xy_count + self.a) / float(y_count + self.s * self.a)

        # 预测类别
        # 求p(y) * p(x|y)
        p_type = {}
        for y in p_y.keys():
            p_type[y] = p_y[y]
            for x in features:
                p_type[y] = p_type[y] * p[str(x) + '|' + str(y)]
        result = max(p_type, key=p_type.get)
        # return p_type[result]
        return result, p_type[result]


def main():
    nb = NaiveBayes()
    train_set, labels = nb.load_data()
    features = [2, 'S']
    result, p = nb.classify(train_set, labels, features)
    print(features, '属于', result, "概率为", p)


if __name__ == '__main__':
    main()
