import csv
from  matplotlib import pyplot as plt
from datetime import datetime

#从文件中获取 最高气温和日期
filename = 'sitka_weather_2014.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    # print(header_row)

    for index,column_header in enumerate(header_row):
        print(index,column_header)

    #已经知道最高温 是在第二个数值的位置。我们已经读过了 next第一行，所以循环会从第二行 数值开始

    dates,highs ,lows =[],[],[]
    for row in reader:
        #此处将  气温由 字符串转换为 数字
        current_date = datetime.strptime(row[0],"%Y-%m-%d")
        dates.append(current_date)

        high = int(row[1])
        highs.append(high)

        low = int(row[3])
        lows.append(low)

    print(highs)
    print(lows)

#根据数据来绘制高温图形
fig = plt.figure(dpi=128,figsize=(10,6))

########居然不能将high 和lows 放在一起 也是醉了
plt.plot(dates,highs,c = 'red')
plt.plot(dates,lows,c = 'blue')
###################################
#设置一下图形的格式
plt.title("Daily high and low temperatures,--- 2014", fontsize = 24)
plt.xlabel('',fontsize =16)
fig.autofmt_xdate()
###填充颜色
plt.plot(dates,highs,c = 'red',alpha = 0.5)
plt.plot(dates,lows,c='blue',alpha =0.5)
plt.fill_between(dates,highs,lows,facecolor = 'blue',alpha = 0.1)
plt.ylabel("Temperature(F)",fontsize = 16)
plt.tick_params(axis='both',which='major',labelsize = 16)
plt.show()

