from die import Die
import  pygal
#类似于C++的创建一个对象   创建一个D6
die = Die()

#掷几次骰子并将结果存储在一个列表中

results = []
for roll_num in range(1000):
    result = die.roll()
    results.append(result)

#分析结果
frequencies = []
for value in range (1,die.num_sides +1):
    frequency = results.count(value)
    frequencies.append(frequency)

#对结果进行可视化处理
hist = pygal.Bar()
hist.title = "Result of rolling one D6 1000 times."
hist.x_labels = ['1','2','3','4','5','6']
hist._x_title = "Result"
hist._y_title = "Frequency of Result"

hist.add('D6',frequencies)
hist.render_to_file('die_visual.svg')




print(results)
print(frequencies)


