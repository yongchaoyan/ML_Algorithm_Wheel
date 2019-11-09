from sklearn.datasets import load_iris  # 导入鸢尾花数据包
from sklearn.neighbors import KNeighborsClassifier  # 导入sklearn包中的KNN类
from sklearn.model_selection import train_test_split

iris = load_iris()
# 将数据包分出训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=33)
# 取得KNN分类器，并使用内置参数调整KNN三要素,
# 参数具体参考官方说明：https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
knn = KNeighborsClassifier(weights="distance", n_neighbors=5)
# 对训练集训练
knn.fit(X_train, y_train)
# 预测类别
y_predict = knn.predict(X_test)
# 计算准确率
print('The accuracy of K-Nearest Neighbor Classifier is: ', knn.score(X_test, y_test))
