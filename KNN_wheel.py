# -*- coding: utf-8 -*-
# @Created on : 2019/11/5 15:10
# @Author : YYC
import csv
import math
import operator
import random


def load_data(file_name, split, train_set, test_set):
    # rt模式下，python在读取文本时会自动把\r\n转换成\n
    with open(file_name, "rt") as csv_file:
        lines = csv.reader(csv_file)  # 返回一个reader对象，该对象将遍历csv文件中的行。从csv文件中读取的每一行都作为字符串列表返回。
        data_set = list(lines)  # 转化成二维数组
        for x in range(len(data_set) - 1):
            for y in range(4):
                data_set[x][y] = float(data_set[x][y])
            if random.random() < split:  # random.random()生成0-1之间的数，将文本分为训练集和测试集
                train_set.append(data_set[x])
            else:
                test_set.append(data_set[x])


# 计算距离
def cal_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# 返回K个最近邻
def get_neighbors(train_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    # 计算每一个测试用例到训练数据集实例的距离
    for x in range(len(train_set)):
        dist = cal_distance(train_set[x], test_instance, length)
        distances.append((train_set[x], dist))
    # 对所有的距离排序
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])  # 将前K个距离最近的训练集加入
    return neighbors


'''
operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号），下面看例子。

a = [1,2,3] 
>>> b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
>>> b(a) 
2 
>>> b=operator.itemgetter(1,0)   //定义函数b，获取对象的第1个域和第0个的值
>>> b(a) 
(2, 1) 

'''


# 找出包含最多的种类
def get_response(neighbors):
    class_sum = {}
    for x in range(len(neighbors)):
        value = neighbors[x][-1]  # 数据集的最后一个数据是种类
        if value in class_sum:
            class_sum[value] += 1
        else:
            class_sum[value] = 1
    # 排序
    # Python 字典 items() 函数作用：以列表返回可遍历的(键, 值) 元组数组。
    after_sort_class = sorted(class_sum.items(), key=operator.itemgetter(1), reverse=True)
    return after_sort_class[0][0]


# 计算准确率
def ger_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    train_set = []  # 训练数据集
    test_set = []  # 测试数据集
    split = 0.7  # 分割的比例
    load_data("./dataset/iris.txt", split, train_set, test_set)  # 加载数据
    print("Train set :" + repr(len(train_set)))  # repr()的作用是将对象转化为供解释器读取的形式。此处转化成str
    print("Test set :" + str(len(test_set)))

    predictions = []
    k = 10  # 设置k值
    for x in range(len(test_set)):
        neighbors = get_neighbors(train_set, test_set[x], k)
        result = get_response(neighbors)
        predictions.append(result)
        print("Predicted = " + repr(result) + ",actual = " + repr(test_set[x][-1]))
    accuracy = ger_accuracy(test_set, predictions)
    print("Accuracy:" + repr(accuracy) + "%")


if __name__ == "__main__":
    main()
