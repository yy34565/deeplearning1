import tensorflow as tf
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import cv2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from pathlib import Path
np.set_printoptions(threshold=np.inf)


#
# x_data=[]
# y_data=[]
#
#
# input_shape=(227,227,3)
#
# # 定义包含所有子文件夹的父文件夹路径
# parent_folder_path = "D:/datas/dataninst/datasminst/Train"
#
# # 用于存储所有图片路径的列表
# all_image_paths = []
#
# # 遍历每个子文件夹
# for root, dirs, files in os.walk(parent_folder_path):
#     for file in files:
#         if file.endswith(('.png')):  # 根据你的图片格式进行调整
#             all_image_paths.append(os.path.join(root, file))
#
# # 设置要抽取的样本数量
# sample_size = 10000 # 这里可以根据需要调整
#
# # 从所有图片路径中随机抽取样本
# sample_paths = random.sample(all_image_paths, sample_size)
#
# for index, path in enumerate(sample_paths):
#     path1 = Path(sample_paths[index]).parent
#     file=os.path.basename(path1)
#     lab=int(file)
#     img=cv2.imread(path,1)
#     img=cv2.resize(img,(input_shape[0],input_shape[1]))
#     x_data.append(img)
#     y_data.append(lab)
#
#
#
# np.random.seed(7)
# np.random.shuffle(x_data)
# np.random.seed(7)
# np.random.shuffle(y_data)
# tf.random.set_seed(7)
#
# X_data=np.array(x_data)
# Y_data=np.array(y_data)


def load_images(parent_folder_path, sample_size, input_shape):
    all_image_paths = []
    x_data = []
    y_data = []
    # 遍历每个子文件夹
    for root, dirs, files in os.walk(parent_folder_path):
        for file in files:
            if file.endswith(('.png')):
                all_image_paths.append(os.path.join(root, file))
    sample_paths = random.sample(all_image_paths, sample_size)
    for index, path in enumerate(sample_paths):
        path1 = Path(sample_paths[index]).parent
        file = os.path.basename(path1)
        lab = int(file)
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (input_shape[0], input_shape[1]))
        x_data.append(img)
        if parent_folder_path_train == "D:/datas/dataninst/datasminst/Train":
            y_data.append(lab)
        else:
            y_data.append(0)
    return np.array(x_data), np.array(y_data)

input_shape = (227, 227, 3)
sample_size = 10000

# 加载训练集数据
parent_folder_path_train = "D:/datas/dataninst/datasminst/Train"
X_data, Y_data = load_images(parent_folder_path_train, sample_size, input_shape)
# x_data为训练集，y_data为训练集标签
# 设置随机种子并打乱数据顺序
np.random.seed(7)
np.random.shuffle(X_data)
np.random.seed(7)
np.random.shuffle(Y_data)
tf.random.set_seed(7)

# 加载测试集数据
parent_folder_path_test = "D:/datas/dataninst/datasminst/Test"
X_data1, Y_data2 = load_images(parent_folder_path_test, sample_size, input_shape)
hotel_minst2=pd.read_csv('D:/datas/gtsrb/gtsrb/GT-final_test.csv')
Y_data1=hotel_minst2.iloc[1:10001,3].values
y_data1=np.array(Y_data1)
# X_data1为测试集数据，Y_data2为测试集标签
# 设置随机种子并打乱测试集数据顺序
np.random.seed(7)
np.random.shuffle(X_data1)
np.random.shuffle(Y_data2)
tf.random.set_seed(7)

# 对训练集进行切割，然后进行训练
x_train,x_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.3)
x_train,x_val = tf.cast(X_data/255.0,tf.float32),tf.cast(x_val/255.0,tf.float32)     #归一化
y_train,y_val = tf.cast(y_train,tf.int16),tf.cast(y_val,tf.int16)
x_test,y_test=tf.cast(X_data1/255.0,tf.float32),tf.cast(y_data1,tf.int16)

print('\n x_train:%s, y_train:%s, x_test:%s, y_test:%s'%(x_train.dtype,y_train.dtype,y_test.dtype,y_test.dtype))

class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        # part1
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('relu')  # 激活层1
        #part2
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('relu')  # 激活层1
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)  # dropout层
        # part3
        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('relu')  # 激活层1
        #part4
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('relu')  # 激活层1
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)  # dropout层
        #part5
        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('relu')  # 激活层1
        #
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('relu')  # 激活层1
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('relu')  # 激活层1
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('relu')  # 激活层1
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('relu')  # 激活层1
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('relu')  # 激活层1
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(43, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y


model = VGG16()

optimizer = tf.keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/VGG16.weights.h5"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train,y_train, batch_size=32, epochs=5, validation_data=(x_val,y_val), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()
test_loss, test_acc = model.evaluate(x_test, y_test)
print('分类准确率:', test_acc)
model.save('VGG16.h5')
###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
