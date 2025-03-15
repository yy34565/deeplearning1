# import tensorflow as tf
# import numpy as np
# import tensorboard.plugins.image.summary as summary_image
# import matplotlib.pyplot as plt
# import os    #导入os模块
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import warnings
# warnings.filterwarnings('ignore')
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 加载 CIFAR-10 数据集
# cifar10 = tf.keras.datasets.cifar10
# (train_x, train_y), (test_x, test_y) = cifar10.load_data()
#
# # 数据预处理
# train_x, test_x = tf.cast(train_x / 255.0, tf.float32), tf.cast(test_x / 255.0, tf.float32)
# train_y, test_y = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)
# # test_y = tf.squeeze(test_y)
#
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, data_format='channels_last', input_shape=train_x.shape[1:]))
# model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
#
# model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
# model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
#
# ##分类识别阶段
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
# print(model.summary())
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#
# history = model.fit(train_x, train_y, batch_size=64, epochs=5 , validation_split=0.2)
#
# model.evaluate(test_x,test_y,verbose=2)
#
# model.save ('imageNet01_module.keras')
#
# weights=model.get_weights()
# model.save_weights('.weights.h5') #保存权重参数
# # model.load_weights('weights.h5')#加载权重参数
#
# example_image = train_x[0]
#
# # # 创建一个辅助模型用于获取中间层的输出
# # layer_outputs = [layer.output for layer in model.layers]
# # activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
# #
# # # 获取各个阶段的输出
# # activations = activation_model.predict(np.expand_dims(example_image, axis=0))
#
#
# def get_layer_outputs(model, input_data):
#     layer_outputs = []
#     input_data = np.clip(input_data, 0, 1)
#     for layer in model.layers:
#         intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
#         output = intermediate_model.predict(input_data[np.newaxis,...])
#         layer_outputs.append(output)
#     return layer_outputs
# activations = get_layer_outputs(model, example_image)
# # 可视化原始图像
# plt.figure(figsize=(10, 10))
# plt.imshow(example_image, interpolation='nearest')
# plt.title('原始图像')
# plt.axis('off')
# plt.show()
# #
# # # 可视化第一次卷积后的图像
# # plt.figure(figsize=(5, 5))
# # plt.imshow(activations[0][0, :, :, :])
# # plt.title('First Convolutional Layer Output')
# # plt.axis('off')
# # plt.show()
# #
# # # 可视化第一次池化后的图像
# # plt.figure(figsize=(5, 5))
# # plt.imshow(activations[1][0, :, :, :])
# # plt.title('First Pooling Layer Output')
# # plt.axis('off')
# # plt.show()
# #
# # # 可视化第二次卷积后的图像
# # plt.figure(figsize=(5, 5))
# # plt.imshow(activations[2][0, :, :, :])
# # plt.title('Second Convolutional Layer Output')
# # plt.axis('off')
# # plt.show()
# #
# # # 可视化第二次池化后的图像
# # plt.figure(figsize=(5, 5))
# # plt.imshow(activations[3][0, :, :, :])
# # plt.title('Second Pooling Layer Output')
# # plt.axis('off')
# # plt.show()
# #
# #
import tensorflow as tf
import numpy as np
import tensorboard.plugins.image.summary as summary_image
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
# 加载 CIFAR-10 数据集
cifar10 = tf.keras.datasets.cifar10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# 数据预处理
train_x, test_x = tf.cast(train_x / 255.0, tf.float32), tf.cast(test_x / 255.0, tf.float32)
train_y, test_y = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, data_format='channels_last', input_shape=train_x.shape[1:]))
model.add(tf.keras.layers.Conv2D(8,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

##分类识别阶段
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
print(model.summary())

dummy_input = np.zeros((1, *train_x.shape[1:]))
layer_output_shapes = []
intermediate_outputs = model(dummy_input)
for i in range(len(model.layers)):
    layer_output_shapes.append(model.layers[i].output.shape)
print("中间层输出形状:", layer_output_shapes)

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# 训练模型
history = model.fit(train_x, train_y, batch_size=64, epochs=5, validation_split=0.2)



# 评估模型
model.evaluate(test_x, test_y, verbose=2)


example_image = train_x[0]
example_image = np.expand_dims(example_image, axis=0)
example_image = tf.convert_to_tensor(example_image)

layer_outputs = []
for layer in model.layers:
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
    output = activation_model.predict(example_image[np.newaxis, ...])
    layer_outputs.append(output)
for i, output in enumerate(layer_outputs):
    if isinstance(model.layers[i], tf.keras.layers.Conv2D):
        plt.figure(figsize=(5, 5))
        plt.imshow(output[0, :, :, 0], cmap='viridis')  # 显示第一个通道的图像
        plt.title(f'Convolution Layer {i} Output')
        plt.axis('off')
        plt.show()
layer_outputs = []
for layer in model.layers:
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)
    output = activation_model.predict(example_image[np.newaxis, ...])
    layer_outputs.append(output)

# # 可视化原始图像
# plt.figure(figsize=(5, 5))
# plt.imshow(example_image)
# plt.title('Original Image')
# plt.axis('off')
# plt.show()

# 可视化每一次卷积后的图像
for i, output in enumerate(layer_outputs):
    if isinstance(model.layers[i], tf.keras.layers.Conv2D):
        plt.figure(figsize=(5, 5))
        plt.imshow(output[0, :, :, 0], cmap='viridis')  # 显示第一个通道的图像
        plt.title(f'Convolution Layer {i} Output')
        plt.axis('off')
        plt.show()

# # 可视化第一次卷积后的图像
# plt.figure(figsize=(5, 5))
# try:
#     plt.imshow(layer_outputs[0][0, :, :, :])
#     plt.title('First Convolutional Layer Output')
#     plt.axis('off')
#     plt.show()
# except IndexError:
#     print("Could not visualize first convolutional layer output.")