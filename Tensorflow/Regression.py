#######################第一个机器学习的样例，输入房子面积得到房子面积，输入数据是一维既房子价格，目标数据也是一维，即房子价格。########
#首先导入需要用到的库，很多库都是需要安装的
import numpy as np
import matplotlib.pyplot as plt

#首先定义两个数据的数组，输入数据X代表房子面积和目标数据Y代表房子价格
# Read dataset
x, y = [], []
for sample in open("./prices.txt", "r"):
#在txt里，数据的存储形式使用逗号隔开的，所以要调用python中的split方法将逗号作为参数传入
    xx, yy = sample.split(",")
#将字符串数据转化为浮点数
    x.append(float(xx))
    y.append(float(yy))
##读取完数据后，把提莫转化为Numpy数组以方便进一步的处理
x, y = np.array(x), np.array(y)
# Perform normalization，标准化处理
x = (x - x.mean()) / x.std()
# Scatter dataset，将原始数据以散点图的形式画出
plt.figure()
plt.scatter(x, y, c="g", s=20)
plt.show()
####################以上整个是数据预处理的过程，包括了数据的可视化操作##########################################################

###################接下来是选用相应的学习方法和模型。###############
##通过可视化原始数据，我们发现很有可能通过线性回归中的多项式拟合来得到一个不错的结果#####
#在（-2,4）这个区间上取得100个点作为画图的基础
x0 = np.linspace(-2, 4, 100)
##利用Numpy的函数定义训练并返回多项式回归模型的函数。deg代表着模型参数中的n,即模型多项式的次数
##返回的模型能够根据输入的X（默认是X0），返回相对于的预测y
# Get regression model under LSE criterion with degree 'deg'
def get_model(deg):
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)

###########接下来是评估结果#################################################################
# Get the cost of regression model above under given x, y
#根据参数n,输入的x，y返回相对应的损失
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()
####定义测试参数集并根据它进行各种实验
# Set degrees
test_set = (1, 4, 10)
for d in test_set:
    #输出相应的损失
    print(get_cost(d, x, y))
########然后是可视化结果来查看是否出现了过拟合##############################################
# Visualize results 画出相应的图像
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))
    ####将横轴的范围限制在(-2,4),纵轴的范围限制在(1e5, 8e5)
plt.xlim(-2, 4)
plt.ylim(1e5, 8e5)
#调用legend的方法使曲线对应的label正确显示
plt.legend()
plt.show()
################我们在查看数字结果的时候以为是N=10的时候是最好的，可视化结果之后，才发现n=4的时候才是最好的