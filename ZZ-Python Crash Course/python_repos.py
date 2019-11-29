
import  requests
import pygal
from  pygal.style import LightColorizedStyle as  LCS ,LightenStyle as  LS

#执行API调用并存储响应
url = 'https://api.github.com/search/repositories?q=language:python&sort=stars'    #1.不能有空格 在输入网址的时候2.stars 字母是写对
r = requests.get(url)
print("Status code:",r.status_code)

#将API相应存储在一个变量中
response_dict = r.json()
#处理结果
# print(response_dict.keys())
print("Totle respositorise:" , response_dict['total_count'])
#探索有关仓库的信息

repo_dicts = response_dict['items']
print("Repositories returned:" , len(repo_dicts))

#研究第一个仓库
# repo_dict = repo_dicts[0]


# print("\nKeys:" , len(repo_dict))
# for key in sorted(repo_dict.keys()):
#     print(key)
################################打印仓库的描述   ,可以遍历一下字典

print("\n Selected information about each repository:")

names,stars = [],[]

for repo_dict in repo_dicts:

    names.append(repo_dict['name'])
    stars.append(repo_dict['stargazers_count'])

    print("---------------------我是调皮的分界线-----------------")
    print('Name:', repo_dict['name'])
    print('Owner:', repo_dict['owner'] ['login'] )
    print('Stars:',repo_dict['stargazers_count'])
    print('Repository:',repo_dict['html_url'])
    print('Created:',repo_dict['created_at'])
    print('Updated:',repo_dict['updated_at'])
    print('Description:',repo_dict['description'])


#可视化处理

my_style = LS('#333366',base_style=LCS)
# chart = pygal.Bar(style = my_style,x_label_rotation = 45,show_legend = False)
###########改动一下图标
my_config = pygal.Config()
my_config.x_label_rotation = 45
my_config.show_legend = False
my_config.title_font_size = 24
my_config.label_font_size = 14
my_config.major_label_font_size = 18
my_config.truncate_label = 15
my_config.show_y_guides = False
my_config.width = 1000
chart = pygal.Bar(my_config,style = my_style)

chart.title = 'Most-Starred Python Projects on GitHub'
chart.x_labels=names

chart.add('',stars)
chart.render_to_file('python_repos.2.svg')






