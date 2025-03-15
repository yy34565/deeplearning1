#tensor张量可0到n阶
# 创建一个张量
import tensorflow as tf
import numpy as np

# tf.constant(张量，dtype=)
# dtype表示数据类型
a=tf.constant([1,23,4],dtype=tf.int64)
print(a)

# 降numpy数据类型转换为tensorflow数据类型
c=np.arange(0,5)
b=tf.convert_to_tensorflow(c,dtype=tf.int64)
# 转换数据类型
# 创建一个tensor张量
# tf.zeros(维度) 全为0的张量
# tf.ones(维度)   值全为1
# tf.fill(维度，指定值)  创建多维张量
'''
s=1 2 3
v=[1,2,3] 一维向量
m=[[1,2,3],[1,2,3]] 二维矩阵
t=[[[’‘’  几个左方框为几维矩阵
‘’‘]]]
'''


