#对于一个框架pytorch是比较基础的，temsorflow是目前比较主流的框架
#里面有API，tensorflow计算机视觉和自然文本处理
#框架就是工具，帮我们完成实验和任务
#keras这个库提供了许多预训练的神经网络模型
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#oneDNN是一个开源的深度学习性能库，由Intel开发，旨在优化深度学习工作负载在Intel CPU上的性能
import tensorflow as tf
import numpy as np
print(tf.__version__)

# x=[[1,]]
# m=tf.matmul(x,x) #.matmul()函数计算两个矩阵的乘积
# print(m)

# x=tf.constant([[1,9],[3,6]])#。constant()函数用于创建张量tensor,表示数值数据
# print(x)
# #tf.Tensor标准输出格式([值列表], shape=(形状,), dtype=数据类型)
# x=tf.constant([1,2,3])
# y=tf.constant([4,5,6])
# z=tf.add(x,y)#.add()函数其中接收的参数形状必须相同，以上例子比如都是一维*一维
# print(z)#输出新的tensor张量
#
# #格式转换
# m=x.numpy()
# print(m)#将一个tensor张量转化为一个数组
#
# n=tf.cast(x,tf.float32)
# print(n) #实现一个类型转换

u=np.ones([2,2]) #ones() 函数用于创建一个新的数组，维度为2*2，其元素都是 1
i=tf.multiply(u,2)#矩阵的乘法运算，矩阵中每个元素都乘以2
print(i)

import cv2

