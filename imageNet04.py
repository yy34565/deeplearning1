# 训练CIFAR10数据集
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np  # numpy模块用于数组和矩阵的运算
import tensorflow as tf  # 用于训练模型
import warnings
warnings.filterwarnings('ignore')  #忽略一些弱警告

model=tf.keras.models.load_model('CIFAR10_CNN_weights.h5')
cifar10 = tf.keras.datasets.cifar10
(train_x,train_y),(test_x,test_y) = cifar10.load_data()

#数据预处理,使数据图像格式相同
train_x,test_x = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
#归一化处理，将图像数据的像素值转化为【0,1】内，并将类型转化为浮点型
train_y,test_y = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)
#归一化处理，并将标签数据转化为整型
# print("test_x:", test_x)
# 这里的数据类型我非常折磨，可视化搞不出来老是报错，后来才发现数据类型对不上加了test_y = tf.squeeze(test_y)才搞出来了
# print("test_y:", test_y)
test_y = tf.squeeze(test_y) #这里我压缩了一个维度，不然的话会报错

#10个类别的标签值，等会评估的时候很直观
class_names=['airplane/top','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

predictions=model.predict(test_x)
print(predictions.shape)#输出它的形状，主要看他是不是二维的
print(predictions[0])#所属于10个类别的概率值
print(np.argmax(predictions[0]))#取它的最大值



def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # 如果 img 是 RGB 图像，不需要 cmap 参数
    plt.imshow(img)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} {:2.0f}%({})'.format(class_names[predicted_label],
                                        100 * np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)
def plot_value_array(i,predictions_array,true_label):
    predictions_array,true_label=predictions_array,true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot=plt.bar(range(10),predictions_array,color='#777777')
    plt.ylim([0,1])
    predicted_label=np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows=5
num_cols=3
num_images=num_rows*num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions[i],test_y,test_x)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i,predictions[i],test_y)
plt.tight_layout()
plt.show()

