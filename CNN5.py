import tensorflow as tf
import os
from keras.backend import dropout
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

x_data=[]
y_data=[]
input_shape=(227,227,3)

# 定义包含所有子文件夹的父文件夹路径
parent_folder_path = "D:/datas/dataninst/datasminst/Train"

# 用于存储所有图片路径的列表
all_image_paths = []

# 遍历每个子文件夹
for root, dirs, files in os.walk(parent_folder_path):
    for file in files:
        if file.endswith(('.png')):  # 根据你的图片格式进行调整
            all_image_paths.append(os.path.join(root, file))

# 设置要抽取的样本数量
sample_size = 10000 # 这里可以根据需要调整

# 从所有图片路径中随机抽取样本
sample_paths = random.sample(all_image_paths, sample_size)
# print(sample_paths)
for index, path in enumerate(sample_paths):
    path1 = Path(sample_paths[index]).parent
    file=os.path.basename(path1)
    lab=int(file)
    # print(path)
    # path2=os.listdir(path1)
    # for photo_file in path2:
    #     if photo_file[0]=='G':
    #         continue
    #     photo_file_path = os.path.join(path1, photo_file)
    #     # img=cv2.imdecode(np.fromfile(photo_file_path,dtype=np.uint8),1)
    #     # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.imread(path,1)
    img=cv2.resize(img,(input_shape[0],input_shape[1]))
    x_data.append(img)
    y_data.append(lab)

# X_data=np.array(x_data)
# batch_size = 64
# num_batches = len(x_data) // batch_size
# for i in range(num_batches):
#     batch = x_data[i * batch_size:(i + 1) * batch_size]
#     batch = batch.astype(np.float32)/255.0


np.random.seed(7)
np.random.shuffle(x_data)
np.random.seed(7)
np.random.shuffle(y_data)
tf.random.set_seed(7)

X_data=np.array(x_data)
Y_data=np.array(y_data)


print(X_data.shape,Y_data.shape)


# 对训练集进行切割，然后进行训练
x_train,x_test,y_train,y_test = train_test_split(X_data,Y_data,test_size=0.5)
x_train,x_test = tf.cast(x_train/255.0,tf.float32),tf.cast(x_test/255.0,tf.float32)     #归一化
y_train,y_test = tf.cast(y_train,tf.int16),tf.cast(y_test,tf.int16)


print('\n x_train:%s, y_train:%s, x_test:%s, y_test:%s'%(x_train.dtype,y_train.dtype,y_test.dtype,y_test.dtype))
# train_x:(7841, 227, 227, 3), train_y:(7841, 43), test_x:(1961, 43), test_y:(1961, 43)

model = tf.keras.Sequential()
#part 1
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=tf.nn.relu,data_format='channels_last',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
#part 2
model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
#part 3
model.add(tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
#part 4
model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
#part 5
model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))
#part 6
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(43, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = "./checkpoint/CNN5.weights.h5"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train,y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

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


