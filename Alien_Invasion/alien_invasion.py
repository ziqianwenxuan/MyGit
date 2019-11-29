import sys
import pygame
from settings import Settings
from ship import  Ship
import  game_functions  as gf      #game_function 是一个模块并不是个类。它没有建立单独的类，而是简单地存放函数
from  pygame.sprite import Group
from  alien import Alien
from  game_stats import GameStats
from  button import Button
from scoreboard import Scoreboard

def run_game():
    pygame.init()   # 初始化游戏

    # screen = pygame.display.set_mode((1200,800))
    ai_settings = Settings()
    screen = pygame.display.set_mode((ai_settings.screen_width,ai_settings.screen_height))

    pygame.display.set_caption("Alien Invasion")
    #创建play按钮
    play_button = Button(ai_settings,screen,"Play")

    #创建一个用于存储游戏信息的实例,并且创建记分牌
    stats = GameStats(ai_settings)
    sb = Scoreboard(ai_settings,screen,stats)


    # bg_corlor = (230,230,230)  #设置背景色
    # 创建一艘飞船
    ship = Ship(ai_settings,screen)
    #创建一个用于存储子弹的编组
    bullets = Group()
    #创建一个外星人编组
    aliens =Group()
    #创建外形人群
    gf.create_fleet(ai_settings,screen,ship,aliens)
    # #创建一个外星人
    # alien = Alien(ai_settings,screen)
    #开始游戏的主循环
    while True:

        gf.check_events(ai_settings,screen,stats,sb,play_button,ship,aliens,bullets)
        if stats.game_active:
            ship.update()                               #飞船位置在检测到鼠标键盘事件后 和更新屏幕前更新
            gf.update_bullets(ai_settings,screen,stats,sb,ship,aliens,bullets)
            # print(len(bullets))   #显示当前还有多少颗子弹
            gf.update_aliens(ai_settings,screen,stats,sb,ship,aliens,bullets)
        gf.update_screen( ai_settings,screen,stats,sb,ship,aliens,bullets,play_button)

run_game()





