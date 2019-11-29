import  matplotlib.pyplot as plt
"""使用matplotlib制作最简单的图表"""
#  金蝶云苍穹，手指的放置方式 可以加快打字速度而已
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.plot(input_values,linewidth = 5)

#设置图表标题，并且给坐标轴加上标签
plt.title("Square Number",fontsize =24)
plt.xlabel ("Value",fontsize = 14)
plt.ylabel("Square of Value",fontsize=24)
#设置刻度标记的大小
plt.tick_params(axis='both',labelsize = 14)

plt.show()
















