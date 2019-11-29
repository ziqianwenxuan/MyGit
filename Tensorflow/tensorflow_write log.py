import tensorflow as tf

#定义一个简单的计算图，实现向量加法的操作
input1=tf.constant([1.0,2.0,3.0],name="input1")
input2=tf.Variable(tf.random_uniform([3]),name="input")
output=tf.add_n([input1,input2],name="add")

#生成一个写日志的writer，并将当前的TensorFlow计算图写入日志。生成的文件在当前文件夹下
writer=tf.summary.FileWriter(".\log",tf.get_default_graph())
writer.close()
init = tf.global_variables_initializer()

tf.Session().run(init)
