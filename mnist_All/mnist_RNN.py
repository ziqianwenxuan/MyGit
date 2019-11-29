####第一大步，加载数据导入数据
import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('data', one_hot=True)   #载入数据

##引入会话
sess = tf.InteractiveSession()
#####第二大步，构建模型

#设置训练的超参数
ir = 0.001
training_iters=100000
batch_size = 128

##定义RNN的参数
n_inputs = 28  #输入层的序列长度
n_steps = 28 #输入的步数
n_hidden_units = 128 #隐藏层的神经元的个数
n_classes = 10 #分类的类别，工10个类别

########定义输入的数据和权重
#输入数据占位符
x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

#定义权重
weightes = {'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))}
       #上述公式 实际上就是 weightes = {(28,128),(128,10)}
biases = {'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))}
        #上述公式 实际上就是 biases = {(128,),(10,)}

######定义RNN模型

def RNN(X,weights,biases):
    X=  tf.reshape(X,[-1,n_inputs])
##进入隐藏层
    X_in = tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])
##这里采用基本的LSTM 循环网络单元：basic LSTM Cell

    istm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias = 1.0,state_is_tuple = True)
    init_state = istm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(istm_cell, X_in ,initial_state=init_state,time_major=False)
    results = tf.matmul(final_state[1],weightes['out'])+biases['out']

    return results

pred = RNN(x,weightes,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdadeltaOptimizer(ir).minimize(cost)

#####定义模型预测结果以及准确率计算方法

correct_pred = tf.equal(tf.math.argmax(pred,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

##########第三大部分 训练数据以及评估模型

with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step*batch_size<training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})
        if  step % 20 == 0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step +=1




