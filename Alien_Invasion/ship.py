import  pygame
# from settings import Settings            我们不去调用setting 里面的ship_speed_factor，而是借助一个行参和变量 来调用
from pygame.sprite import  Sprite
class Ship():
    def __init__(self,ai_settings,screen):
        super(Ship, self).__init__()
        self.screen =  screen
        self.ai_settings = ai_settings
        self.image = pygame.image.load('images/ship.bmp')
        self.rect = self.image.get_rect()
        self.screen_rect =  screen.get_rect()
        self.rect.centerx = self.screen_rect.centerx
        self.rect.bottom = self.screen_rect.bottom
        self.center = float (self.rect.centerx)

        #移动标识
        self.moving_right = False
        self.moving_left = False
    def update(self):
        """根据移动标识调整飞船位置"""
        if self.moving_right and self.rect.right < self.screen_rect.right:
            self.center += self.ai_settings.ship_speed_factor
        if self.moving_left and self.rect.left > 0:
            self.center -= self.ai_settings.ship_speed_factor

        self.rect.centerx = self.center
    def blitme(self):
        """在指定位置绘制飞船"""
        self.screen.blit(self.image,self.rect)

    def center_ship(self):
        self.center = self.screen_rect.centerx

