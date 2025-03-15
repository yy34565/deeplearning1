import tensorflow as tf     #导入TensorFlow模块 TensorFlow中的keras库中有cifar10的数据集
import numpy as np      #导入numpy模块  用来数据预处理，将数据格式转化为numpy数组格式
import matplotlib.pyplot as plt     #导入matplotlib模块用于可视化处理
import os    #导入os模块
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' #oneDNN是一个开源的深度学习性能库，由Intel开发，旨在优化深度学习工作负载在Intel CPU上的性能
import warnings
warnings.filterwarnings('ignore')  #忽略一些弱警告
#设置matplotlib绘图时为中文黑体
plt.rcParams['font.sans-serif'] = ['SimHei']

#加载CIFAR-10图像数据集
cifar10 = tf.keras.datasets.cifar10
(train_x,train_y),(test_x,test_y) = cifar10.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape))
'''
输出训练集和测试集的形状
结果为train_x:(50000, 32, 32, 3), train_y:(50000, 1), test_x:(10000, 32, 32, 3), test_y:(10000, 1)
这个跟上次MNIST数据集不一样，上次是28*28的灰度图像，很明显这次是32*32*3的RGB图像
'''
#数据预处理,使数据图像格式相同
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
#归一化处理，将图像数据的像素值转化为【0,1】内，并将类型转化为浮点型
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)
#归一化处理，并将标签数据转化为整型
'''
我个人其实很想知道为什么数据为什么不能直接用，为什么一定要进行归一化处理。
除了加快收敛速度，还有什么效果呢
我们都知道加快收敛速度其实就是使梯度变化更为稳定，而梯度又与权重挂钩，
当梯度稳定时，权重不在更新，那么模型基本训练完成
我在网上看了很多博主的解释，其中有一条我觉得解释的很到位
现实数据中可能存在一些异常值，如果不对数据进行归一化处理，
这些异常值可能会对模型产生较大的影响。
归一化可以将数据的范围缩放到一个相对较小的区间内，从而降低异常值对模型的影响程度。
例如，在某些数据集中，可能存在个别数据点的某个特征值非常大或非常小，
如果不进行归一化，这些异常值可能会使模型在训练过程中过度关注这些数据点，导致模型的性能下降。
通过归一化，可以使这些异常值的影响相对减小，使模型更加关注数据的整体分布。
'''
#建立模型
model = tf.keras.Sequential()
#提取特征图
# 第一层 第一次卷积，池化操作
model.add(tf.keras.layers.Conv2D(8,kernel_size=(3,3),padding='same',activation=tf.nn.relu,data_format='channels_last',input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(8,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
'''
下面是我代码运行结果一部分，说明一下
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 32, 32, 8)      │           224 │
卷积层，8个卷积核，大小（3，3），padding='same',表示卷积后得到的图像可以保持原图像大小，relu激活函数，输入形状（32，32，3）
假设我的rgb图像为32*32*3，我的卷积核为3*3*3，那么你扫一次图像的结果本来是30*30*3（32-3+1）的
但加入了paddding操作进行填充，你也可以得到跟输入图像一样的32*32*3的格式
可以看到我第一次卷积后，因为有padding操作得到的还是32*32，如果没有padding,那么应该是(None,30,30,8)
后面那个8是我得到了多少特征图，因为我设置了8个卷积核，后面那个param，参数个数为448
其实我们都知道卷积神经网络，参数是共享的，就是说一个卷积核无论滑到哪个位置参数都是一样的
那么一个卷积核是3*3*3=27个权重参数，有8个卷积核，那么就是27*8=216个权重参数，每个图都会带一个偏置那么就是16个偏置
那么总共的参数就是216+8=224
后面池化操作相当于把图片压缩得到的图片是原来的一半，那么后面进行卷积是在池化操作上进行的得到图片的大小，
'''
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
#池化层，最大值池化，卷积核（2，2） ，最大池化操作，当然也有平均池化操作，但我个人认为一般最大池化是最好的
# 池化操作，也是用卷积核去扫一遍，不过卷积核的大小为2*2，为什么是2*2，我在网络上搜了一下，跟太上老君炼丹一样，经验之谈
#池化操作的目的应该是减少一些参数，很明显用一个2*2的卷积核再去扫一遍卷积的结果，大的值取出来，小的值我不要了，结果肯定是更小的

# 第二层 第二次卷积，池化操作
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# 第三层 图像特征提取完成，构建神经网络
model.add(tf.keras.layers.Flatten())
#改变输入形状
# 卷积操作得到的很明显是一个三维的立体的形状，直接放入神经网络肯定是不行的
# 比如说你卷积得到的是10*10*64的数据图像，你要放入神经网络你得拉成6400*1的形状
# 第五层 隐含层和输出层
model.add(tf.keras.layers.Dense(128,activation='relu'))#全连接网络层，128个神经元，二分类可以用sigmoid,10分类用relu激活函数
model.add(tf.keras.layers.Dense(10,activation='softmax'))#输出层，10个节点，对于多分类问题可以用softmax激活函数

#配置模型训练方法
#adam算法参数采用keras默认的公开参数，损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

#批量训练大小为64，迭代5次，测试集比例0.2（40000条训练集数据，10000条测试集数据）
history = model.fit(X_train,y_train,batch_size=256,epochs=10,validation_split=0.2)

# 评估模型在测试集上的分类正确率
test_loss, test_acc = model.evaluate(X_test, y_test)
print('分类准确率:', test_acc)
print('\n',model.summary()) #简单输出模型的概要

# 用模型预测测试集的正确率
predictions=model.predict(test_x)
# print(predictions.shape)
# print(predictions[0])#所属于10个类别的概率值
# print(np.argmax(predictions[0]))  #概率值最大的那个类别


#保存整个模型
model.save('CIFAR10_CNN_weights.h5')



# 结果可视化的代码跟上次MINST数据集的任务差不多，我直接拿过来用了
#结果可视化
print(history.history) #.history方法: 这是一个在调用 model.fit() 函数时返回的对象，它包含了训练过程中的所有信息，
loss = history.history['loss']
#训练集的损失函数，用于绘图表示
val_loss = history.history['val_loss']
#测试集的损失函数
acc = history.history['sparse_categorical_accuracy']
#训练集的准确率
val_acc = history.history['val_sparse_categorical_accuracy']
#测试集的准确率

plt.figure(figsize=(10,3)) #.figure()函数创建一个图形长为10，高为3
plt.subplot(121)
#.subplot() 函数用于在一个图形（figure）中创建一个或多个子图
#其中数字121每一位都有含义
# 第一位数字（1）表示子图应该被分成几行。
#第二位数字（2）表示子图应该被分成几列。
#第三位数字（1）表示当前创建的子图是这些子图中的哪一个
plt.plot(loss,color='g',label='train')
#.plot()函数绘制一个图形，横坐标为迭代次数,图形中有一条蓝色的线，表示训练集的损失函数
plt.plot(val_loss,color='r',label='test')
#同上，这条红色的线表示测试集的损失函数
plt.ylabel('损失函数') #y轴表示损失函数
plt.legend()   #显示图像

plt.subplot(122)
plt.plot(acc,color='g',label='train')
plt.plot(val_acc,color='r',label='test')
plt.ylabel('准确率')
plt.legend()
#这段代码跟上面那段段代码差不多，只不过是显示的是训练集和测试集的准确率图像

plt.show()
