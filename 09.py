from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(0)
x_data=np.random.randn(10000,2)
y=np.where(np.sum(x_data**2,axis=1)<1,0,1)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=42)


clf = svm.SVC(C=0.1,kernel='linear') #惩罚函数设为1，kernel设为线性核
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
float_accuracy=round(accuracy,9)
print("预测准确率:", float_accuracy)
