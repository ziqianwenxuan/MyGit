import tensorflow as tf

#创建图
a = tf.constant([1.0,2.0])
b = tf.constant([3.0,4.0])
c= a*b

#创建会话
sess = tf.compat.v1.Session()

#计算C
print (sess.run(c))
sess.close()
