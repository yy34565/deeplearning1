import numpy as np
import pandas as pd
import plotly.express as px
# 计算欧几里得距离
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
# K-Means算法核心实现
class KMeans:
    def __init__(self, n_clusters=3, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels_ = None  # 添加属性用于存储聚类后的标签

    def fit(self, X):
        # 随机初始化质心
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iterations):
            # 分配数据点到最近的质心
            self.labels_ = self.assign_labels(X)  # 在每次迭代分配标签时更新这个属性
            # 更新质心位置
            old_centroids = self.centroids.copy()
            for i in range(self.n_clusters):
                if np.any(self.labels_ == i):
                    self.centroids[i] = np.mean(X[self.labels_ == i], axis=0)
            # 检查质心是否收敛
            if np.all(old_centroids == self.centroids):
                break

    def assign_labels(self, X):
        labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            distances = [euclidean_distance(X[i], centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels


# 从sklearn库里加载鸢尾花数据集（这里假设还是用这个数据集来做示例）
from sklearn.datasets import load_iris
iris = load_iris()
# 获取数据部分的前3列作为特征数据，前3列数据分别作x轴，y轴，z轴坐标
X = iris.data[:, :3]
# 初始化我们自己实现的K-means算法，这里假设聚为3类（鸢尾花通常有3类品种，可以根据实际情况调整）
kmeans = KMeans(n_clusters=3)

# 对数据进行聚类训练
kmeans.fit(X)

# 获取聚类后的标签（每个数据点所属的类别）
labels = kmeans.labels_

# 将特征数据和对应的聚类标签合并为一个DataFrame，方便后续可视化
df = pd.DataFrame(X, columns=['Sepal Length', 'Sepal Width', 'Petal Length'])
df['Cluster'] = labels
# 使用Plotly进行交互式3D可视化展示
fig = px.scatter_3d(df, x='Sepal Length', y='Sepal Width', z='Petal Length',
                    color='Cluster', title='Iris Dataset K-means Clustering',
                    hover_data=['Cluster'])
# 设置图表布局，可根据需要调整视角等参数
fig.update_layout(scene=dict(xaxis_title='Sepal Length',
                             yaxis_title='Sepal Width',
                             zaxis_title='Petal Length'))
# 显示图表
fig.show()