import matplotlib.pyplot as plt

# plt.scatter(2,4,s=200)

# x_values = [1,2,3,4,5]
# y_values = [1,4,9,16,25]
x_values = list(range(1,1001))
y_values = [x**2 for x in x_values]

plt.scatter(x_values,y_values,c = y_values,cmap=plt.cm.Blues,edgecolor='none',s= 40)
#设置图标标题，并且给坐标轴加上标签
plt.title ("11Square Numbers",fontsize = 24)
plt.xlabel ("11Value",fontsize = 14)
plt.ylabel("11Square of Value",fontsize=24)

#设置刻度标记的大小
plt.tick_params(axis='both',which = 'major',labelsize =14)
#设置每个坐标轴的取值范围
plt.axis([0,1100,1,1100000])

plt.show()
plt.savefig('hahah.png',bbox_inches = 'tight')




