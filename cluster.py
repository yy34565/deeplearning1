
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris

# 从文件中读取数据，支持txt、xls(x)、csv格式，这里简单以txt格式读取为例
def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过第一行（如果有数据个数那一行的话）
            point = list(map(float, line.strip().split(',')))
            data.append(point)
    return np.array(data)


# 执行K-means聚类算法
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.cluster_centers_, kmeans.labels_


# 可视化聚类结果
def visualize_result(data, cluster_centers, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # 定义一些颜色用于区分不同簇
    for i in range(len(data)):
        ax.scatter(data[i][0], data[i][1], data[i][2], c=colors[labels[i]])
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='*', s=200, c='black')
    plt.show()


if __name__ == "__main__":
    # file_path = "C:/Users/yy/Desktop/3.txt"  # 替换为实际的数据文件路径
    # data = read_data(file_path)
    iris = load_iris()      #导入鸢尾花数据集，该数据集有150个样本，4个特征
    data1 = iris.data
    # print(data1)
    data=data1[:,:3]       #取前3列特征输入
    # print(data)
    n_clusters = 3  # 这里假设聚为3类，可以根据实际需求调整
    cluster_centers, labels = kmeans_clustering(data, n_clusters)
    visualize_result(data, cluster_centers, labels)
