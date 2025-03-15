#第一个项目
#回归任务

#导入相应模块
import numpy as np #numpy模块用于数组和矩阵的运算
import pandas as pd #pandas模块用于处理表格数据
import matplotlib.pyplot as plt #用于绘图
import tensorflow as tf # 用于训练模型
import keras    #高级神经网络的API，辅助tensorflow进行模型的训练
import warnings
warnings.filterwarnings('ignore')


file_path="D:\\datas\\01\\china_sites_20200820.csv"  #将下载好的本地数据集导入北京co,co2,No等气体，excel表数据集
features = pd.read_csv(file_path)
print(features.head())  #查看里面数据是啥样


print('数据维度：',features.shape)  #查看数据维度


'''
import os
base_dir='D:\\VMware'
train_dir=os.path.join(base_dir,"文件名")
其他同理



'''

