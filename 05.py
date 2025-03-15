# 简单的练手fashion数据集
import numpy as np #numpy模块用于数组和矩阵的运算
import pandas as pd #pandas模块用于处理表格数据
import matplotlib.pyplot as plt #用于绘图
import tensorflow as tf # 用于训练模型
# import keras    #高级神经网络的API，辅助tensorflow进行模型的训练
import warnings

from tensorboard.plugins.image.summary import image

warnings.filterwarnings('ignore')

model=tf.keras.models.load_model('fashion_modle.h5')

fashion_minst=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_minst.load_data()


train_images,test_images = tf.cast(train_images/255.0,tf.float32),tf.cast(test_images/255.0,tf.float32)     #归一化
train_labels,test_labels = tf.cast(train_labels,tf.int16),tf.cast(test_labels,tf.int16)


class_names=['T-shit/top','Trouser','puillover','dress','coat','sandal','shirt','sneaker','bag','anklwe']

predictions=model.predict(test_images)


print("predictions:", predictions)
print("test_y:", test_labels)


def plot_image(i,predicttions_arry,ture_label,img):
    predicttions_arry,true_label,img=predicttions_arry,ture_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img,cmap=plt.cm.binary)
    predicted_label=np.argmax(predicttions_arry)
    if predicted_label==true_label:
        color='blue'
    else:
        color='red'
    plt.xlabel('{} {:2.0f}%({})'.format(class_names[predicted_label],
                                        100*np.max(predicttions_arry),
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
    plot_image(i,predictions[i],test_labels,test_images)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i,predictions[i],test_labels)
plt.tight_layout()
plt.show()


