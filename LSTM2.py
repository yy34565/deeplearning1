#循环核;参数时间共享，循环层提取时间信息
#ht=tanh(xtwxh+ht-1whh+bhh) ht为记忆体内存储的状态信息ht
#3个待训练的参数矩阵wxh whh why
# yt=softmax(htwhy+by) 当前循环体输出的状态特征
# 记忆体的个数被指定，待训练的参数维度也被指定了
# X_train维度[送入样本数，循环核展开步数，每个时间输入特征个数]
# 独热码 数据量大的话过于稀疏，映射是独立的没有相关性
# Embedding（单词编码方法）用低维向量编码，能表示相关性
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Dense ,Dropout,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math

hotel_minst1=pd.read_csv('D:\\datas\\房屋贷款预测数据集\\train.csv')
hotel_minst2=pd.read_csv('D:\\datas\\房屋贷款预测数据集\\test.csv')


training_set=hotel_minst1.iloc[0:120000,1:51].values #取120000名用户数据，第二列至第50列做输入特征
training_label=hotel_minst1.iloc[0:120000,51:].values #取每个用户的是否逾期做训练集标签
test_set=hotel_minst2.iloc[0:30000,1:51].values    #取前30000用户数据做测试
training_set = training_set.astype('float128')
test_set = test_set.astype('float128')

# 归一化
sc = MinMaxScaler()  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化

print(training_set_scaled.shape)
print(123)
print(test_set.shape)

train_x=[]
train_y=[]

test_x=[]
test_y=[]









