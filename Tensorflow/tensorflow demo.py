#
#可以看到TensorFlow整体流程如下：
#1.输入数据
#2.建立模型
#3.定义损失函数和训练方法
#4.初始化和启动Tensorflow会话
#5.训练
#目标，构造一些满足一元二次方程函数 Y=ax^2 +b 的原始数据。然后构造一个简单的神经网络，仅包含一个输入层，一个隐藏层和一个输出层。
#通过tensorflow的训练，将隐藏层和输出层的weight（权重）和biases（偏置）的值学习出来，并确定损失值是不断减小的
###########第一步骤，输入数据，由于我们是实验 没有数据源，所以自己创造一些
import tensorflow as tf
import  numpy as np
#构造一个满足一元二次方程的函数
x_data = np.linspace(-1,1,300)[:,np.newaxis] #为了使点更密一些，我们构建了300个点，分布在-1和1之间，直接采用np生成等差数列的方法
                                             #并将结果为300个点的一维数组，转换为300*1的二维数组
noise = np.random.normal(0,0.05,x_data.shape) #加入一些噪声点，使它与x_data的维度一致，并且拟合为均值为0，方差为0.05的正态分布
y_data = np.square(x_data)-0.5+noise   #这就是一元二次函数y=x^2 -0.5 +噪声
####输入数据/加载数据
#定义x和y的占位符来作为将要输入神经网络的变量：
xs= tf.placeholder(tf.float32,[None,1])
ys= tf.placeholder(tf.float32,[None,1])

#################第二大步骤是构建一个神经网络模型。输入层，隐藏层，输出层
#1 输入层。作为神经网络的输入层，输入参数有四个：输入数据，输入数据的维度，输出数据的维度，激活函数。每一层经过向量化处理（y=weights*x +biase)
#的处理，并且经过激活函数的非线性化处理后，最终得到输出数据。

def add_layer(inputs,in_size,out_size,activation_function=None):
    #构建权重：何谓权重，权重就是in_size*out_size大小的矩阵
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #构建偏置：1*out_size 的矩阵
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    #矩阵相乘
    wx_plus_b=tf.matmul(inputs,weights)+biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs=activation_function(wx_plus_b)

    return outputs #得到输出数据

#2.构建隐藏层，假设隐藏层有20个神经元
h1 = add_layer(xs,1,20,activation_function=tf.nn.relu)

#3.构建输出层，假设输出层和输入层一样，只有一个神经元
prediction = add_layer(h1,20,1,activation_function=None)
#4.接下来是构建损失函数。何谓损失函数？即：计算输出层的预测值和真实值间的误差，对二者的平方求和再取平均，得到损失函数。
                 #运用梯度下降法，以0.1的学习速率最小化损失：
loss=  tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#训练步数 设置 为0.1
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

######第三大部分训练模型
##训练100次，没50次输出训练的损失值

init=tf.global_variables_initializer()  #初始化所有变量
sess=tf.Session()  #创建对话
sess.run(init)

for i in  range(1000):  #训练1000次
    sess.run(train_step,feed_dict={xs: x_data,ys:y_data})
    if i%50==0:  #每50次打印一次损失值
        print(sess.run(loss,feed_dict={xs: x_data,ys:y_data}))

# 训练拟合，每一步训练队Weights和biases进行更新
#for step in range(201):
 #   sess.run(train_step)
  #  if step % 20 == 0:
   #     print(step, sess.run(weights), sess.run(biases))  ##每20步输出一下W和b
##第四大部分评估模型