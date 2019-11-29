# coding: utf-8
import sys, os
import cv2

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist_train import load_mnist
from PIL import Image
import struct
import shutil
from keras.datasets import mnist


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def setDir():
    if not os.path.exists('./test'):
        os.makedirs('./test')
    else:
        shutil.rmtree('./test')
        os.mkdir('./test')


# 调用系统封装的读入训练数据和测试数据的函数
(x_train, t_train), (x_test, t_test) = mnist.load_data()
# 调用深度学习入门这本书中带有的函数
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)
# 刚开始进行文件夹的处理。
setDir()
# 处理十张手写数字体(mnist数据集带有的)
for i in range(0, 10):
    img = x_train[i]
    label = t_train[i]
    img = img.reshape(28, 28)
    # Image.save('mnistpicture/'+img+'.png')

    fileName = "./test/" + str(i) + "-" + str(t_train[i]) + ".bmp"
    cv2.imwrite(fileName, img)
    print(label)  # 5
# img.show()

# print(img.shape)  # (784,)
# img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
# print(img.shape)  # (28, 28)

# img_show(img)