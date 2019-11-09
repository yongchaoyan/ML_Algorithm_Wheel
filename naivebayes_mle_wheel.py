# -*- coding: utf-8 -*-
# @Created on : 2019/11/5 22:05
# @Author : YYC
import numpy as np
import pandas as pd

'''
    朴素贝叶斯算法，极大似然估计
    1,求p(y)
    2,求p(x,y)
    3,求p(x|y) = p(x,y) / p(y)
    4,求p(y)*p(x|y) 找出概率最大的，类别就为y
'''


class NaiveBayes(object):
    def load_data(self):
        data_set = pd.read_csv('./dataset/naivebayes_data.csv')
        data_set_np = np.array(data_set)
        '''numpy中的shape()属性返回一个元组，代表矩阵的维度，
        # b = np.array([[1,2,3],[3,4,5]])
        # print(b.shape)
        # 输出(2,3) 表示数组两行三列
        
        numpy[:,k]表示所有行的第k列        
        numpy[:,m:n]表示所有行的m到n列
        '''
        train_set = data_set_np[:, 0: data_set_np.shape[1] - 1]  # 训练数据 x1,x2
        labels = data_set_np[:, data_set_np.shape[1] - 1]  # 训练数据所对应的类型Y
        return train_set, labels

    def classify(self, train_set, labels, features):
        # 求每个labels中每个label的先验概率
        # p(y)
        labels = list(labels)
        p_y = {}
        for label in labels:
            p_y[label] = labels.count(label) / float(len(labels))  # p = count(y) / count(Y)

        # 求label与feature同时发生的概率
        # p(x,y)
        p_xy = {}
        for y in p_y.keys():
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            y_index = [i for i, label in enumerate(labels) if label == y]  # 去除labels中所有出现y值的数据的下表索引
            for j in range(len(features)):
                x_index = [i for i, feature in enumerate(train_set[:, j]) if feature == features[j]]
                # set()函数创建一个无序不重复元素集 ， & 求两个set的交集
                xy_count = len(set(x_index) & set(y_index))
                p_key = str(features[j]) + '*' + str(y)
                p_xy[p_key] = xy_count / float(len(labels))  # 求 p(x,y)

        # 求条件概率
        # p(x|y)
        p = {}
        for y in p_y.keys():
            for x in features:
                p_key = str(x) + '|' + str(y)
                p[p_key] = p_xy[str(x) + '*' + str(y)] / float(p_y[y])  # p(x|y) = p(x,y)/p(y)

        # 求[2,'s']所属类别
        # p(y) * p(x|y)
        p_type = {}  # 存储各类别的概率
        for y in p_y:
            p_type[y] = p_y[y]
            for x in features:
                p_type[y] = p_type[y] * p[str(x) + '|' + str(y)]
        result = max(p_type, key=p_type.get)
        return result,p_type[result]


def main():
    nb = NaiveBayes()
    # 获取训练数据
    train_set, labels = nb.load_data()
    features = [2, 'S']  # 预测数据
    result,p = nb.classify(train_set, labels, features)
    print(features, '属于', result, "概率为", p)


if __name__ == '__main__':
    main()
