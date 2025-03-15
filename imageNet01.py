import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os



plt.rcParams['font.sans-serif'] = ['SimHei']
# 加载 CIFAR-10 数据集
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 数据预处理
train_images, test_images = tf.cast(train_images / 255.0, tf.float32), tf.cast(test_images / 255.0, tf.float32)
train_labels, test_labels = tf.cast(train_labels, tf.int16), tf.cast(test_labels, tf.int16)

# 从我本地磁盘选取了一张图片
base_path='C:\\Users\\yy\\Pictures\\联想安卓照片'
train_dir=os.path.join(base_path,'cat.jpg')
image_data = tf.io.read_file(train_dir)
image = tf.image.decode_jpeg(image_data)
# 构建模型  这个模型只用于可视化输出特征图和卷积核
# 正常来说的话，我本来要写一个函数将模型的卷积层和池化层传进去，但在调keras库中tf.keras.model.input()函数老是报错，所以我就放弃了这个方法
# 但条条道路通罗马，我直接新建一个模型只包含卷积层和池化层，我并不用来进行训练，我就是直接传入一张图片然后过一遍卷积层就可以得出卷积后的图像了
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

# 查看模型结构
model.summary()
# example_image = train_images[0] #取训练集中的第一张图片
image=image.numpy()
# # # print('shape',example_image.shape)
image = tf.expand_dims(image, axis=0)
# print('reshaped shape', reshaped_image.shape)
# shape (1099, 1200, 3)
# # 可视化原始图像
# def visualize_train():
#     plt.figure(figsize=(5, 5))
#     plt.imshow(image)
#     plt.title('原始图像')
#     plt.axis('off')
#     plt.show()
# visualize_train()

# def visualize_kernels_tf(model):
#     kernels1 = model.layers[0].weights[0].numpy()
#     kernels2 = model.layers[1].weights[0].numpy()
#
#     fig, axes = plt.subplots(1, kernels1.shape[0], figsize=(15, 15))
#     for i in range(kernels1.shape[0]):
#         ax = axes[i]
#         ax.imshow(kernels1[i, 0, :, :], cmap='gray')
#         ax.axis('off')
#     plt.show()
#
#     fig, axes = plt.subplots(1, kernels2.shape[0], figsize=(15, 15))
#     for i in range(kernels2.shape[0]):
#         ax = axes[i]
#         ax.imshow(kernels2[i, 0, :, :], cmap='gray')
#         ax.axis('off')
#     plt.show()






# 可视化第一层特征图
def visualize_first_layer_feature_maps(module):
    conv_layers = [layer for layer in module.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    # first_layer = conv_layers[3]#这边换成0,1,2,3来改变第几层卷积层的输出
    # 获取当前层的输出张量
    # input_data = train_images[0:1]
    input_data=image
    # print('input_data shape',input_data.shape)
    # feature_maps = first_layer.output
    # print('type',feature_maps.shape)
    feature_maps_pred = model.predict(input_data)
    num_maps = feature_maps_pred.shape[-1]
    num_rows = int(np.ceil(np.sqrt(num_maps)))
    num_cols = int(np.ceil(num_maps / num_rows))
    fig, axes = plt.subplots(num_rows, num_cols)
    for j in range(num_maps):
        row = j // num_cols
        col = j % num_cols
        axes[row, col].imshow(feature_maps_pred[0,:,:,j], cmap='grey')
        axes[row, col].axis('off')
    fig.suptitle('第一层特征图')
    plt.show()
visualize_first_layer_feature_maps(model)

# def visualize_first_conv_kernels(model):
#     conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
#     if len(conv_layers) > 0:
#         first_conv_layer = conv_layers[0]
#         kernels = first_conv_layer.get_weights()[0]
#         num_kernels = kernels.shape[-1]
#         num_rows = int(np.ceil(np.sqrt(num_kernels)))
#         num_cols = int(np.ceil(num_kernels / num_rows))
#         fig, axes = plt.subplots(num_rows, num_cols)
#         for j in range(num_kernels):
#             row = j // num_cols
#             col = j % num_cols
#             if num_kernels > 1:
#                 axes[row, col].imshow(kernels[:, :, :, j], cmap='coolwarm')
#                 axes[row, col].axis('off')
#             else:
#                 axes.imshow(kernels[:, :, :, j], cmap='coolwarm')
#                 axes.axis('off')
#         fig.suptitle('第一层卷积核')
#         plt.show()
#     else:
#         print("No convolutional layers found in the model.")
# visualize_first_conv_kernels(model)