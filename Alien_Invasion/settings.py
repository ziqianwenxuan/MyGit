class Settings():
#     """存数外星人入侵的所有设置的类"""


    def __init__(self):
#         """初始化游戏的静态设置"""
#         #屏幕设置
        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color = (230,230,230)
        self.ship_speed_factor = 1.5
        #设置一些子弹的参数
        self.bullet_speed_factor = 1
        self.bullet_width = 3
        self.bullet_height = 15
        self.bullet_color = 60,60,60
        self.bullets_allowed = 3
        self.alien_speed_factor = 1

        self.fleet_drop_speed = 10
        # self.fleet_direction = 1
        self.ship_limit = 3           #每个玩家有三搜飞船


        #以什么样的速度加快游戏节奏
        self.speedup_scale = 1.1
        self.score_scale = 1.5
        self.initialize_dynamic_settings()



    def initialize_dynamic_settings(self):

        """初始化随游戏而变化的设置"""
        self.ship_speed_factor = 1.5
        self.bullet_speed_factor = 3
        self.alien_speed_factor = 1

        #fleet_direction 为1表示向右，-1 表示像左

        self.fleet_direction = 1
        #记分

        self.alien_points = 50


    def increase_speed(self):
        """提高速度设置和外星人点数"""
        self.ship_speed_factor *= self.speedup_scale
        self.bullet_speed_factor *= self.speedup_scale
        self.alien_speed_factor *= self.speedup_scale
        self.alien_points = int(self.alien_points*self.score_scale)
        print(self.alien_points)