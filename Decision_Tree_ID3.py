# -*- coding: utf-8 -*-
# @Created on : 2019/11/7 9:10
# @Author : YYC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import namedtuple


def load_data():
    # header = None表示原始文件数据没有索引，read_csv会自动加上列索引
    # pandas.read_csv :读取csv文件到DataFrame
    DataFrame = pd.read_csv('./dataset/zoo.data.csv', header=None)
    # DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
    # 删除[0]列，第一列是animal_name
    DataFrame = DataFrame.drop([0], axis=1)
    #
    dataClass = namedtuple('data', ['data', 'target'])
    # pandas中的 iloc() ： 取出指定行，或者指定列
    dataClass.data = DataFrame.iloc[:, :-1].values  # 取除所有行的从0到最后一列
    dataClass.target = DataFrame.iloc[:, -1].values  # 取出所有行的最后一列
    return dataClass


class ID3:
    def __init__(self):
        self.tree = None
        self.dataset = None

    def __entropy(self, feature):
        uni_val, cnt = np.unique(feature, return_counts=True)
        h = np.sum([(-cnt[i] / np.sum(cnt)) * np.log2(cnt[i] / np.sum(cnt)) for i in range(len(uni_val))])
        return h

    # 计算信息增益
    def __InfoGain(self, dataset, f_test_col, Y_col=-1):
        entropy_befor = self.__entropy(dataset.iloc[:, Y_col])  # 经验熵
        uni_val, cnt = np.unique(dataset.iloc[:, f_test_col], return_counts=True)
        entropy_cond = np.sum([(cnt[i] / np.sum(cnt)) * self.__entropy(
            dataset.where(dataset.iloc[:, f_test_col] == uni_val[i]).dropna().iloc[:, Y_col]) for i in
                               range(len(uni_val))])
        return entropy_befor - entropy_cond

    def __gen_tree(self, dataset, org_dataset, f_cols, Y_col=-1, p_node_cls=None):
        """
        dataset:用于分割的数据
        org_dataset:最原始的数据
        f_cols:备选特征，列序号
        """
        # 如果数据中的Y已经纯净,即 Y只有一种类别，则返回Y的取值
        if len(np.unique(dataset.iloc[:, Y_col])) <= 1:
            return np.unique(dataset.iloc[:, Y_col])[0]  # 返回类别Y，此时数组中只有一个数，所以【0】

        #  如果此时特征为空，（对应空叶节点），则返回原始数据中数量较多的CK的值
        elif len(dataset) == 0:
            # return_count = True,返回两个数组，第一个是不重复的元素的数组，第二个是相应的个数
            uni_cls, cnt = np.unique(org_dataset.iloc[:, Y_col], return_counts=True)
            # np.argmax()返回最大值的索引
            return uni_cls[np.argmax(cnt)]

        # 如果没有特征可用于划分，则返回父节点中数量较多的label
        # 由于初始传入的是Index类型，所以这里不能用if not
        elif len(f_cols) == 0:
            return p_node_cls

        # 否则进行分裂
        else:
            # 得到当前节点中数量最多的label,递归时会赋给下层函数的p_node_cls
            cur_uni_cls, cnt = np.unique(dataset.iloc[:, Y_col], return_counts=True)
            cur_node_cls = cur_uni_cls[np.argmax(cnt)]
            # del 删除的是变量，而不是数据
            del cur_uni_cls, cnt

            # 根据信息增益选出最佳分裂特征
            gains = [self.__InfoGain(dataset, f_col, Y_col) for f_col in f_cols]
            best_f = f_cols[np.argmax(gains)]

            # 更新备选特征
            f_cols = [col for col in f_cols if col != best_f]

            # 按最佳特征的不同取值，划分数据集并递归
            tree = {best_f: {}}
            for val in np.unique(dataset.iloc[:, best_f]):
                sub_data = dataset.where(dataset.iloc[:, best_f] == val).dropna()
                sub_tree = self.__gen_tree(sub_data, dataset, f_cols, Y_col, cur_node_cls)
                tree[best_f][val] = sub_tree
            return tree

    def fit(self, x_train, y_train):
        # np_c np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
        # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
        data_set = np.c_[x_train, y_train]
        self.dataset = pd.DataFrame(data_set, columns=list(range(data_set.shape[1])))
        self.tree = self.__gen_tree(self.dataset,self.dataset, list(range(self.dataset.shape[1] - 1)))

    def __predict_one(self, x_test, tree, default=-1):
        for feature in list(x_test.keys()):
            if feature in list(tree.keys()):  # 如果该特征与根节点的划分特征相同
                try:
                    sub_tree = tree[feature][x_test[feature]]  # 根据特征的取值来获取左右分支

                    if isinstance(sub_tree, dict):  # 判断是否还有子树
                        return self.__predict_one(x_test, tree=sub_tree)  # 有则继续查找
                    else:
                        return sub_tree  # 是叶节点则返回结果
                except:  # 没有查到则说明是未见过的情况，只能返回default
                    return default

    def predict(self, x_test):
        x_test = pd.DataFrame(x_test, columns=list(range(x_test.shape[1]))).to_dict(orient='record')
        y_pred = list()
        for item in x_test:
            y_pred.append(self.__predict_one(item, tree=self.tree))
        return y_pred


if __name__ == '__main__':
    data = load_data()
    X = data.data
    Y = data.target
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=33)
    id3_tree = ID3()
    id3_tree.fit(x_train, y_train)

    y_pred = id3_tree.predict(x_test)
    print('acc:{}'.format(np.sum(np.array(y_test) == np.array(y_pred)) / len(y_test)))
