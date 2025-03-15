import tensorflow as tf

w=tf.Variable(tf.constant(5,dtype=tf.float32))#.variable（）函数表示可训练的权重和偏置 初始权重为5
lr=0.999 #学习率可调,0.2 19轮loss为0  0.001收敛速度很慢   0.999时权重变化在局部震荡，当然loss降的很慢
epoch=40
for epoch in range(epoch):
    with tf.GradientTape() as tape: #自动计算梯度
        loss=tf.square(w+1)
    grads=tape.gradient(loss,w)
    w.assign_sub(lr * grads) #做自减运算 w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f" % (epoch+1, w.numpy(), loss))

# 学习率决定梯度收敛的速度
# 学习率小的话收敛慢，大的话会在最小值震荡
#W(t+1)=w(t)-lr*(loss/w(t)的梯度)

# import tensorflow as tf
#
# w = tf.Variable(tf.constant(5, dtype=tf.float32))
# lr = 0.2
# epoch = 40
#
# for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环40次迭代。
#     with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
#         loss = tf.square(w + 1)
#     grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导
#
#     w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
#     print("After %s epoch,w is %f,loss is %f" % (epoch, w.numpy(), loss))
#
# # lr初始值：0.2   请自改学习率  0.001  0.999 看收敛过程
# # 最终目的：找到 loss 最小 即 w = -1 的最优参数w

