import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#看到1,2行代码，不要懵，这个作用是设置日志级别，os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error，等于1是显示所有信息。不加这两行会有个提示（Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2，具体可以看这里）#
#获得数据集  数据读取
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
##########此数据集是本地获取，还有一种是 网站下载的
import tensorflow as tf
#Session(交互方式)  引入会话
sess = tf.InteractiveSession()
#输入图像数据占位符
x = tf.placeholder(tf.float32, [None, 784])

#权值和偏差  ##为模型定义权重W和偏置b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#使用softmax模型  ##拥有一个线性层的softmax回归模型
#实现我们的回归模型了。这只需要一行！我们把向量化后的图片x和权重矩阵W相乘，加上偏置b，然后计算每个分类的softmax概率值
y = tf.nn.softmax(tf.matmul(x, W) + b)
#代价函数占位符
y_ = tf.placeholder(tf.float32, [None, 10])

#交叉熵评估代价
#很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#使用梯度下降算法优化：学习速率为0.5
####训练模型我们已经定义好模型和训练用的损失函数，那么用TensorFlow进行训练就很简单了。
# 因为TensorFlow知道整个计算图，它可以使用自动微分法找到对于各个变量的损失的梯度值。
# TensorFlow有大量内置的优化算法 这个例子中，我们用最速下降法让交叉熵下降，步长为0.01.
#这一行代码实际上是用来往计算图上添加一个新操作，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值。
# 返回的train_step操作对象，在运行时会使用梯度下降来更新参数。因此，整个模型的训练可以通过反复地运行train_step来完成
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#变量需要通过seesion初始化后
# sess.run(tf.initialize_all_variables())

#初始化变量
tf.global_variables_initializer().run()

#训练模型，训练1000次
#每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。
# 注意，在计算图中，你可以用feed_dict来替代任何张量，并不仅限于替换占位符
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#评估模型
#那么我们的模型性能如何呢？首先让我们找出那些预测正确的标签。
# tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，
# 而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
#计算正确率  这两个是评估训练模型，获得准确度的函数，一会儿最后的print 有用到
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
##这里返回一个布尔数组。为了计算我们分类的准确率，
# 我们将布尔值转换为浮点数来代表对、错，然后取平均值。例如：[True, False, True, True]变为[1,0,1,1]，计算出平均值为0.75。

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#最后，我们可以计算出在测试数据上的准确率，大概是91%。
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


###########一次性训练10000次，计算其正确率
#  一次一次 来计算
keep_prob = tf.placeholder("float")
for i in range(2000):
  batch = mnist.train.next_batch(100)
  if i%100 ==0:
    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, train accuracy %g" % (i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))