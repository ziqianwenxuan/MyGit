from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 隐藏提示警告

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_step = 1000  # 最大迭代次数
learning_rate = 0.001  # 学习率
dropout = 0.9  # dropout时随机保留神经元的比例

data_dir = './tensorboard_input'  # 样本数据存储的路径
log_dir = './tensorboard_output'  # 输出日志保存的路径

mnist = input_data.read_data_sets(data_dir, one_hot=True)

'''
tf.InteractiveSession():它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作
在交互式环境中的人们来说非常便利，比如使用IPython。
tf.Session():需要在启动session之前构建整个计算图，然后启动该计算图。
意思就是在我们使用tf.InteractiveSession()来构建会话的时候，我们可以先构建一个session然后再定义操作（operation），
如果我们使用tf.Session()来构建会话我们需要在会话构建之前定义好全部的操作（operation）然后再构建会话。
'''
sess = tf.InteractiveSession()

'''创建输入数据的占位符，分别创建特征数据x，标签数据y_ '''
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')

'''使用tf.summary.image保存图像信息 '''
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

'''
在构建神经网络模型中，每一层中都需要去初始化参数w,b,为了使代码简介美观，最好将初始化参数的过程封装成方法function。 
创建初始化权重w的方法，生成大小等于传入的shape参数，标准差为0.1,正态分布的随机数，并且将它转换成tensorflow中的variable返回。
'''


# 在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


'''
创建初始换偏执项b的方法，生成大小为传入参数shape的常数0.1，并将其转换成tensorflow的variable并返回
'''


# b = tf.constant(2,shape=[2,2]) = [[2 2],[2 2]]
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
我们知道，在训练的过程在参数是不断地在改变和优化的，我们往往想知道每次迭代后参数都做了哪些变化，可以将参数的信息展现在tenorbord上，
因此我们专门写一个方法来收录每次的参数信息。
'''


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)


'''
创建第一层隐藏层 
创建一个构建隐藏层的方法,输入的参数有： 
input_tensor：特征数据 
input_dim：输入数据的维度大小 
output_dim：输出数据的维度大小(=隐层神经元个数） 
layer_name：命名空间 
act=tf.nn.relu：激活函数（默认是relu)
'''


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 设置命名空间
    with tf.name_scope(layer_name):
        # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        # 执行wx+b的线性计算，并且用直方图记录下来
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('linear', preactivate)
        # 将线性输出经过激励函数，并将输出也用直方图记录下来
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        # 返回激励层的最终输出
        return activations


'''
调用隐层创建函数创建一个隐藏层：输入的维度是特征的维度784，神经元个数是500，也就是输出的维度。
'''
hidden1 = nn_layer(x, 784, 500, 'layer1')
# 创建一个dropout层，,随机关闭掉hidden1的一些神经元，并记录keep_prob
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('droup_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
# 创建一个输出层，输入的维度是上一层的输出:500,输出的维度是分类的类别种类：10，
# 激活函数设置为全等映射identity.（暂且先别使用softmax,会放在之后的损失函数中一起计算）
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

'''
创建损失函数
使用tf.nn.softmax_cross_entropy_with_logits来计算softmax并计算交叉熵损失,并且求均值作为最终的损失值。
'''
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        # 计算所有样本交叉熵损失的均值
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('loss', cross_entropy)

'''
使用AdamOptimizer优化器训练模型，最小化交叉熵损失
'''
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

'''
计算准确率,并用tf.summary.scalar记录准确率
'''
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 分别将预测和真实的标签中取出最大值的索引，若相同则返回1(true),不同则返回0(false)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

'''
将所有的summaries合并，并且将它们写到之前定义的log_dir路径
'''
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
# 运行初始化所有变量
tf.global_variables_initializer().run()

'''
准备训练与测试的两个数据，循环执行整个graph进行训练与评估
如果是train==true，就从mnist.train中获取一个batch样本，并且设置dropout值； 
如果不是train==false,则获取minist.test的测试数据，并且设置keep_prob为1，即保留所有神经元开启。
'''


def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


'''
每隔10步，就进行一次merge, 并打印一次测试数据集的准确率，然后将测试数据集的各种summary信息写进日志中。 
每隔100步，记录原信息 
其他每一步时都记录下训练集的summary信息并写到日志中。
'''

for i in range(max_step):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
        if i % 100 == 99:  # Record execution stats
            # tf.RunOptions定义TensorFlow运行选项，设置trace_lever FULL_TRACE。
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            ##运行时记录运行信息的proto
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()
