import pygame
import torch
import math
import csv
import time
from random import random
from PIL import Image
from ModuleClass import TextBox, Button
from ScreenFunction import WelcomeScreen, MainScreen, ConfigurationScreen, MazeGeneratorScreen, CollectScreen, IntroScreen, MapScreen
from MazeClass import Cell
from network_train import NeuroNet

pygame.init()

# 颜色定义

WHITE = (255,255,255)
GREY = (20,20,20)
BLACK = (0,0,0)
PURPLE = (100,0,100)
RED = (255,0,0)
GREEN = (0, 200, 0)
BLUE = (0, 204, 255)
YELLOW = (255, 255, 0)

# 窗口尺寸大小

window_size = 1001
chop_size = 200
compensate_size = 100

win = pygame.display.set_mode((window_size, window_size))  # 画布窗口的大小
pygame.display.set_caption("Yiyang Simulator")  # 窗口标题
clock = pygame.time.Clock() # 激活计时器
robot = pygame.image.load("robot.png") # 载入机器人贴图
arrow = pygame.image.load("arrow.png") # 载入箭头指示贴图
photo = pygame.image.load("photo.jpg") # 载入开发者照片
cover = pygame.image.load("cover.png") # 载入圆形遮挡贴图

# 调整尺寸

robot_scale = pygame.transform.scale(robot, (int(window_size/10), int(window_size/10)))
arrow_scale = pygame.transform.scale(arrow, (int(window_size/4), int(window_size/10)))
photo_scale = pygame.transform.scale(photo, (int(window_size/4), int(window_size/3)))
cover_scale = pygame.transform.scale(cover, (int(window_size), int(window_size)))

# 加载背景图

background = pygame.image.load("maze.png") # 载入背景图
##width, height = background.get_size()
x, y, z = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size, 360*random()  # 机器人的起点坐标及朝向
des_x, des_y = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size  # 机器人的终点坐标

# 加载默认参数

moving_speed = 5  # 前进速度
rotating_speed = 15  # 转向速度
UserName = 'Yiyang' # 用户名
screen = 'welcome' # 初始界面
mazewidth = 100 # 迷宫通道尺寸
cols = int(window_size / mazewidth)
rows = int(window_size / mazewidth)

# 加载雷达测距模块

DetectionDistance = int(window_size/2) - 1
BeamNumber = 17
DectectionList = []

for i in range(0,BeamNumber):

    DectectionList.append([])

    for j in range(0,DetectionDistance):

        DectectionList[i].append([int(j*math.cos(i*math.pi/(BeamNumber-1))),int(j*math.sin(i*math.pi/(BeamNumber-1)))])


    # 前向传播

    def forward(self, x):

        output = self.fc(x)

        return output


def name_on_enter(id, final):
    print('enter pressed, username is "{}"'.format(final))

def pass_on_enter(id, final):
    print('enter pressed, password is "{}"'.format(final))

username_settings = {
    "command" : name_on_enter,
    "inactive_on_enter" : False,
}
password_settings = {
    "command" : pass_on_enter,
    "inactive_on_enter" : False,
}
user_name_settings = {
    "command" : pass_on_enter,
    "inactive_on_enter" : False,
}
btn_settings = {
    "clicked_font_color" : (0,0,0),
    "hover_font_color"   : (205,195, 100),
    'font'               : pygame.font.Font(None,16),
    'font_color'         : (255,255,255),
    'border_color'       : (0,0,0),
}

moving_entry = TextBox(rect=(425,350,150,30), **username_settings)
rotating_entry = TextBox(rect=(425,450,150,30), **password_settings)
user_name_entry = TextBox(rect=(425,550,150,30), **user_name_settings)
tbs = [moving_entry, rotating_entry, user_name_entry]


def add_margin(pil_img, top, right, bottom, left, color):

    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def SaveFile(DataLog, StoreFile):

    with open(StoreFile, 'w', newline='') as f:

        csv_writer = csv.writer(f)

        # write object state

        for x1 in range(0, len(DataLog)):

            csv_writer.writerow(DataLog[x1])


def get_textboxes(SpeedOne, SpeedTwo, Name):

    global moving_speed, rotating_speed, UserName
##    print('button pressed, running speed is "{}"'.format(SpeedOne))
##    print('button pressed, turning speed is "{}"'.format(SpeedTwo))

    if SpeedOne != '' and  SpeedTwo != '':

        moving_speed = float(SpeedOne)  # 前进速度
        rotating_speed = float(SpeedTwo)  # 转向速度
        UserName = str(Name)



def get_start():

    global screen
    screen = 'main'


def get_gen():

    global screen, grid, current_cell, next_cell, stack, Save, DataLog
    screen = 'generator'
    Save = True

    # 重置迷宫贴图背景

    grid = []

    for y in range(rows):

        grid.append([])

        for x in range(cols):

            grid[y].append(Cell(x,y,mazewidth))

    current_cell = grid[0][0]
    next_cell = 0
    stack = []
    DataLog = []


def get_configuration():

    global screen
    screen = 'configuration'

def get_intro():

    global screen
    screen = 'intro'

def get_collect():

    global screen
    screen = 'collect'

def get_save():

    global DataLog

    TimeMark = time.strftime("%Y%m%d%H%M%S")
    SaveFile(DataLog, 'DataFile/'+UserName+'_tratining_data_'+TimeMark+'.csv')
    DataLog = []

def get_welcome():

    global screen
    screen = 'welcome'

def get_load():

    global screen,background,x,y,z,des_x,des_y,CollisionWarning

    # 重置背景图

    im = Image.open('maze.png')
    im_new = add_margin(im, int(0.7*chop_size), int(0.7*chop_size), int(0.7*chop_size), int(0.7*chop_size), (0, 0, 0))
    im_new.save('maze.png')

    background = pygame.image.load("maze.png") # 载入背景图
##    background_scale = pygame.transform.scale(background, (window_size, window_size))
    x, y, z = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size, 360*random()  # 机器人的起点坐标及朝向
    des_x, des_y = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size  # 机器人的终点坐标
    CollisionWarning = False
    screen = 'welcome'


btn = Button(rect=(450,650,105,25), command=lambda:get_textboxes(moving_entry.final, rotating_entry.final, user_name_entry.final), text='OK', **btn_settings)
Startbtn = Button(rect=(450,350,105,25), command=lambda:get_start(), text='Start', **btn_settings)
Genbtn = Button(rect=(450,450,105,25), command=lambda:get_gen(), text='Maze Generator', **btn_settings)
Confbtn = Button(rect=(450,550,105,25), command=lambda:get_configuration(), text='Configurations', **btn_settings)
Savebtn = Button(rect=(450,650,105,25), command=lambda:get_save(), text='Save Data', **btn_settings)
Collectbtn = Button(rect=(450,750,105,25), command=lambda:get_collect(), text='Collect Data', **btn_settings)
Introbtn = Button(rect=(450,850,105,25), command=lambda:get_intro(), text='Developer', **btn_settings)
Welcomebtn = Button(rect=(450,750,105,25), command=lambda:get_welcome(), text='Back', **btn_settings)
Loadbtn = Button(rect=(450,550,105,25), command=lambda:get_load(), text='Load Map', **btn_settings)

# 字体大小定义

font = pygame.font.Font('freesansbold.ttf', 32)
font1 = pygame.font.SysFont('didot.ttc', 36)
font2 = pygame.font.SysFont('didot.ttc', 72)
font3 = pygame.font.SysFont('didot.ttc', 54)

# 文字贴图

MovingSpeedText = font1.render('Moving Speed', True, BLACK)
RotatingSpeedText = font1.render('Rotating Speed', True, BLACK)
UserNameText = font1.render('User Name', True, BLACK)
UserConfigurationText = font2.render('User Configuration', True, BLACK)
WelcomeText = font2.render('Robotic Navigation Simulator', True, BLACK)
DeveloperTextOne = font3.render('This simulator is developed by Yiyang Chen', True, BLACK)
DeveloperTextTwo = font3.render('- All RIGHTS RESERVED', True, BLACK)
TaskCompleteText = font3.render('Task Complete', True, GREEN)
CollisionWarningText = font3.render('Collision Warning', True, RED)

# 迷宫贴图背景

grid = []

for i in range(rows):

    grid.append([])

    for j in range(cols):

        grid[i].append(Cell(j,i,mazewidth))

current_cell = grid[0][0]
next_cell = 0
stack = []
Save = True
run = True
LargeScreen = False
CollisionWarning = False
CompleteSign = True
ManualControl = False
AlreadySave = False
LastMotion = 'AHEAD'
DataLog = []

# 构建模型
model = NeuroNet()

# 载入训练好的模型
model = torch.load('model_net.pkl')

# 部署GPU
device = torch.device('cuda:0')
model = model.to(device)

model.eval()

# 开始

while run:

    if screen == 'intro':

        run = IntroScreen(win,Welcomebtn,DeveloperTextOne,DeveloperTextTwo,photo_scale)

    if screen == 'main':

        if LargeScreen:

            win = pygame.display.set_mode((window_size, window_size))
            LargeScreen = False
        
        x, y, z, run, screen, CollisionWarning, step, LastMotion, AlreadySave = MainScreen(x,y,z,des_x,des_y,window_size,chop_size,compensate_size,font,win,background,robot_scale,arrow_scale,cover_scale,moving_speed,rotating_speed,DectectionList,TaskCompleteText,CollisionWarningText,CollisionWarning,ManualControl,model,LastMotion,AlreadySave)

        if step != []:

            DataLog.append(step)

    if screen == 'collect':

        if LargeScreen:

            win = pygame.display.set_mode((window_size, window_size))
            LargeScreen = False

        x, y, z, run, screen, CollisionWarning, CompleteSign, step = CollectScreen(x,y,z,des_x,des_y,window_size,chop_size,compensate_size,font,win,background,robot_scale,arrow_scale,cover_scale,DectectionList,CollisionWarning,CompleteSign)

        if step != []:

            DataLog.append(step)

    if screen == 'configuration':

        run = ConfigurationScreen(win,tbs,btn,Welcomebtn,MovingSpeedText,RotatingSpeedText,UserNameText,UserConfigurationText)

    if screen == 'generator':

        current_cell,run, Save = MazeGeneratorScreen(win,current_cell,Loadbtn,grid,stack,mazewidth,rows,cols,Save)

    if screen == 'map':

        if not LargeScreen:

            win = pygame.display.set_mode((window_size + int(1.4*chop_size), window_size + int(1.4*chop_size)))
            LargeScreen = True

        run, screen, AlreadySave = MapScreen(x,y,des_x,des_y,window_size,chop_size,win,background,DataLog,AlreadySave)


pygame.quit()
