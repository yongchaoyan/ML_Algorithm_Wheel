# -*- coding: utf-8 -*-
# @Created on : 2019/11/10 14:16
# @Author : YYC
import random

import numpy
from numpy import mat, zeros, inf, power, nonzero, mean, sqrt, square

import matplotlib.pyplot as plt


def load_data(file_name):
    data_set = []
    fp = open(file_name)
    for line in fp.readlines():
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
        cur_line = line.strip('\n').split('\t')
        data_set.append([float(cur_line[0]), float(cur_line[1])])
    return data_set


# 随机选择四个点做出初始点
def init_centroids(data_set, k):
    num_samples, dim = data_set.shape
    centroids = zeros((k, dim))
    for i in range(k):
        # 随机生成（0，num_samples)之间的实数
        index = int(random.uniform(0, num_samples))
        centroids[i, :] = data_set[index, :]
    return centroids


def cal_distance(vector1, vector2):
    return numpy.linalg.norm(vector2 - vector1)


def k_means(data_set, k):
    # 多少个点
    num_samples = data_set.shape[0]

    # 类的存储，第一列是该点的类别，第二列是该点距离中心的距离
    cluster_assment = mat(zeros((num_samples, 2)))
    cluster_changed = True

    # step1:初始化中心点
    centroids = init_centroids(data_set, k)

    # 一旦分组变化了，就继续聚类
    while cluster_changed:
        cluster_changed = False
        for i in range(num_samples):
            min_dist = 100000.0
            min_index = 0.0
            # 找出这个点所属的类别,遍历每个类
            for j in range(k):
                distance = cal_distance(centroids[j, :], data_set[i, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            # 更新这个点的类

            if cluster_assment[i, 0] != min_index:
                cluster_assment[i:, ] = min_index, min_dist

                cluster_changed = True
        # 更新聚类的中心点
        for j in range(k):
            # .A 将矩阵转化为数组
            # nonzero 取出数组中非零元素的索引
            pointsInCluster = data_set[nonzero(cluster_assment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)
    print("congratulations ,cluster complete")
    return centroids, cluster_assment


# 图像展示
def show_cluster(data_set, k, centroids, cluster_assment):
    num_samples, dim = data_set.shape
    if dim != 2:
        print("sorry")
        return 1
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(num_samples):
        mark_index = int(cluster_assment[i, 0])
        plt.plot(data_set[i, 0], data_set[i, 1], mark[mark_index])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


def main():
    print("step1:load data ...")
    data_set = []
    data_set = load_data('./dataset/kmeans_data.txt')

    print("step2:clustering...")
    # list转化为mat
    data_set = mat(data_set)
    k = 4
    # centroids 存放中心点
    # clster_assment 存放每个点所属类的索引和距离
    centroids, cluster_assment = k_means(data_set, k)
    print("step3:show the result")
    show_cluster(data_set, k, centroids, cluster_assment)


if __name__ == '__main__':
    main()
