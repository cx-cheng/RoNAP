import pygame
import math
from MazeClass import removeWalls
from random import random
from random import choices
import numpy as np
import torch



WHITE = (255,255,255)
GREY = (20,20,20)
BLACK = (0,0,0)
PURPLE = (100,0,100)
RED = (255,0,0)
GREEN = (0, 200, 0)
BLUE = (0, 204, 255)
YELLOW = (255, 255, 0)

def MainScreen(x,y,z,des_x,des_y,window_size,chop_size,compensate_size,font,win,background,robot_scale,arrow_scale,cover_scale,moving_speed,rotating_speed,DectectionList,TaskCompleteText,CollisionWarningText,CollisionWarning,ManualControl,model,LastMotion,AlreadySave):

    screen = 'main'
    run = True
    step = []

    # distance calculation

    distance = round(math.sqrt(math.pow(des_x - x, 2) + math.pow(des_y - y, 2)),2)

    # angle calculation

    if des_x - x > 0:

        angle = math.atan((des_y - y)/(des_x - x))*180/math.pi

    elif des_x - x < 0:

        angle = math.atan((des_y - y)/(des_x - x))*180/math.pi + 180

    else:

        if des_y - y > 90:

            angle = 90

        else:

            angle = 270

    direction = (angle % 360 - z) % 360

    # precess screen information

    width, height = background.get_size()
    subsurface = background.subsurface((x-chop_size/2, height-y-chop_size/2, chop_size, chop_size)) # chop nearby environment
    rotated_image = pygame.transform.rotate(subsurface, 90-z) # rotate map
    rotated_arrow = pygame.transform.rotate(arrow_scale, 90+direction) # rotate arrow
    w, h = rotated_image.get_size()
    rotated_image_subsurface = rotated_image.subsurface((w-compensate_size)/2, (h-compensate_size)/2, compensate_size, compensate_size) # cut outside part
    rotated_image_subsurface = pygame.transform.scale(rotated_image_subsurface, (window_size, window_size)) # rescale size

    # draw background

    win.blit(rotated_image_subsurface, (0, 0))

    # draw cover

    win.blit(cover_scale, (0, 0))

    # radar distance detection

    DetectionData = []

    for i in range(0, len(DectectionList)):

        j = 0

        while rotated_image_subsurface.get_at((int(window_size/2+DectectionList[i][j][0]), int(window_size/2-DectectionList[i][j][1]))) == (255,255,255) and rotated_image_subsurface.get_at((int(window_size/2+DectectionList[i][j][0]+1), int(window_size/2-DectectionList[i][j][1]))) == (255,255,255) and j < len(DectectionList[i])-1:

            j = j + 1

        DetectionData.append(j)

        # color choice

        if j < 0.3*len(DectectionList[i]):

            Color_line = RED

        elif j >= 0.3*len(DectectionList[i]) and j < 0.6*len(DectectionList[i]):

            Color_line = YELLOW

        else:

            Color_line = GREEN

        pygame.draw.line(win, Color_line, (window_size/2, window_size/2), (window_size/2+DectectionList[i][j][0], window_size/2-DectectionList[i][j][1]))

    # raise collision warning

    if min(DetectionData) < 0.1*len(DectectionList[i]):

        CollisionWarning = True


    # prevent CPU over hot

    pygame.time.delay(100)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            run = False

    keys = pygame.key.get_pressed()


    if ManualControl:

        # direction key response

        if not CollisionWarning:

            if keys[pygame.K_LEFT]:
                z += rotating_speed
                z = z % 360
                step = [DetectionData, distance, direction, 'LEFT']

            if keys[pygame.K_RIGHT]:
                z -= rotating_speed
                z = z % 360
                step = [DetectionData, distance, direction, 'RIGHT']

            if keys[pygame.K_UP]:
                x += moving_speed*math.cos(z*math.pi/180)
                y += moving_speed*math.sin(z*math.pi/180)
                step = [DetectionData, distance, direction, 'AHEAD']

        else:

            if keys[pygame.K_DOWN]:
                x -= moving_speed*math.cos(z*math.pi/180)
                y -= moving_speed*math.sin(z*math.pi/180)
                CollisionWarning = False
                step = [DetectionData, distance, direction, 'WARNING']

    else:

        Input = []

        for i in range(0, len(DetectionData)):

            Input.append(DetectionData[i]/500)

        Input.append(direction/360)

        InputTensor = torch.FloatTensor(Input)

        output = model.forward(InputTensor.cuda())
        output = output.cpu()

        population = [0, 1, 2]
        weights = output.tolist()
        pred = choices(population, weights)[0]

##        pred = np.argmax(output.data.numpy())

        if not CollisionWarning:

            if pred == 1 and LastMotion != 'RIGHT':
                z += rotating_speed
                z = z % 360
                step = [x, y, z, 'LEFT']
                LastMotion = 'LEFT'

            if pred == 2 and LastMotion != 'LEFT':
                z -= rotating_speed
                z = z % 360
                step = [x, y, z, 'RIGHT']
                LastMotion = 'RIGHT'

            if pred == 1 and LastMotion == 'RIGHT':
                z -= rotating_speed
                z = z % 360
                step = [x, y, z, 'RIGHT']
                LastMotion = 'RIGHT'

            if pred == 2 and LastMotion == 'LEFT':
                z += rotating_speed
                z = z % 360
                step = [x, y, z, 'LEFT']
                LastMotion = 'LEFT'

            if pred == 0:
                x += moving_speed*math.cos(z*math.pi/180)
                y += moving_speed*math.sin(z*math.pi/180)
                step = [x, y, z, 'AHEAD']
                LastMotion = 'AHEAD'

        else:

            x -= moving_speed*math.cos(z*math.pi/180)
            y -= moving_speed*math.sin(z*math.pi/180)
            CollisionWarning = False
            step = [x, y, z, 'WARNING']





    # map key response

    if keys[pygame.K_ESCAPE]:

        screen = 'welcome'

    if keys[pygame.K_m]:

        AlreadySave = False
        screen = 'map'


##    # boundary constraint
##
##    if x > width - chop_size/2:
##        x = width - chop_size/2
##
##    if x < chop_size/2:
##        x = chop_size/2
##
##    if y > height - chop_size/2:
##        y = height - chop_size/2
##
##    if y < chop_size/2:
##        y = chop_size/2



    # process text information

    text = font.render('Distance = '+str(distance), True, GREEN, GREY)
    textRect = text.get_rect()
    textRect.center = (window_size-window_size/5, window_size/5)

    # draw arrow

    new_rect = rotated_arrow.get_rect(center = arrow_scale.get_rect(topleft = (window_size/2-window_size/10 - 25, window_size/2-window_size/20)).center)
    win.blit(rotated_arrow, new_rect.topleft)

    # draw robot

    win.blit(robot_scale, (window_size/2-window_size/20, window_size/2-window_size/20))

    # draw distance

    win.blit(text, textRect)

    # draw task complete text

    if distance < 10:

       win.blit(TaskCompleteText, (150, 100))

    # draw collision warning text

    if CollisionWarning:

       win.blit(CollisionWarningText, (150, 200))

    return x, y, z, run, screen, CollisionWarning, step, LastMotion, AlreadySave



def CollectScreen(x,y,z,des_x,des_y,window_size,chop_size,compensate_size,font,win,background,robot_scale,arrow_scale,cover_scale,DectectionList,CollisionWarning,CompleteSign):

    screen = 'collect'
    run = True
    step = []
    mu, sigma = 0, 0.03

    # distance calculation

    distance = round(math.sqrt(math.pow(des_x - x, 2) + math.pow(des_y - y, 2)),2)

    # angle calculation

    if des_x - x > 0:

        angle = math.atan((des_y - y)/(des_x - x))*180/math.pi

    elif des_x - x < 0:

        angle = math.atan((des_y - y)/(des_x - x))*180/math.pi + 180

    else:

        if des_y - y > 90:

            angle = 90

        else:

            angle = 270

    direction = (angle % 360 - z) % 360

    # precess screen information

    width, height = background.get_size()
    subsurface = background.subsurface((x-chop_size/2, height-y-chop_size/2, chop_size, chop_size)) # chop nearby environment
    rotated_image = pygame.transform.rotate(subsurface, 90-z) # rotate map
    rotated_arrow = pygame.transform.rotate(arrow_scale, 90+direction) # rotate arrow
    w, h = rotated_image.get_size()
    rotated_image_subsurface = rotated_image.subsurface((w-compensate_size)/2, (h-compensate_size)/2, compensate_size, compensate_size) # cut outside part
    rotated_image_subsurface = pygame.transform.scale(rotated_image_subsurface, (window_size, window_size)) # rescale size

    # draw background

    win.blit(rotated_image_subsurface, (0, 0))

    # draw cover

    win.blit(cover_scale, (0, 0))

    # radar distance detection

    DetectionData = []

    for i in range(0, len(DectectionList)):

        j = 0

        while rotated_image_subsurface.get_at((int(window_size/2+DectectionList[i][j][0]), int(window_size/2-DectectionList[i][j][1]))) == (255,255,255) and rotated_image_subsurface.get_at((int(window_size/2+DectectionList[i][j][0]+1), int(window_size/2-DectectionList[i][j][1]))) == (255,255,255) and j < len(DectectionList[i])-1:

            j = j + 1

        DetectionData.append(j)

        # color choice

        if j < 0.3*len(DectectionList[i]):

            Color_line = RED

        elif j >= 0.3*len(DectectionList[i]) and j < 0.6*len(DectectionList[i]):

            Color_line = YELLOW

        else:

            Color_line = GREEN

        pygame.draw.line(win, Color_line, (window_size/2, window_size/2), (window_size/2+DectectionList[i][j][0], window_size/2-DectectionList[i][j][1]))

    # raise collision warning

    if min(DetectionData) < 0.1*len(DectectionList[i]):

        CollisionWarning = True


    # prevent CPU over hot

    pygame.time.delay(20)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            run = False

    keys = pygame.key.get_pressed()

    # direction key response

    if not CollisionWarning and CompleteSign:

        if keys[pygame.K_LEFT]:
            x, y, z = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size, angle % 360 + 360*float(np.random.normal(mu, sigma, 1))
            step = [DetectionData, distance, direction, 'LEFT']
            CompleteSign = False

        if keys[pygame.K_RIGHT]:
            x, y, z = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size, angle % 360 + 360*float(np.random.normal(mu, sigma, 1))
            step = [DetectionData, distance, direction, 'RIGHT']
            CompleteSign = False

        if keys[pygame.K_UP]:
            x, y, z = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size, angle % 360 + 360*float(np.random.normal(mu, sigma, 1))
            step = [DetectionData, distance, direction, 'AHEAD']
            CompleteSign = False

    if CollisionWarning:

        x, y, z = window_size*random()+0.7*chop_size, window_size*random() + 0.7*chop_size, angle % 360 + 360*float(np.random.normal(mu, sigma, 1))
        CollisionWarning = False

    if not CompleteSign:

        if keys[pygame.K_DOWN]:
            CompleteSign = True






    # map key response

    if keys[pygame.K_ESCAPE]:

        screen = 'welcome'

    # process text information

    text = font.render('Distance = '+str(distance), True, GREEN, GREY)
    textRect = text.get_rect()
    textRect.center = (window_size-window_size/5, window_size/5)

    # draw arrow

    new_rect = rotated_arrow.get_rect(center = arrow_scale.get_rect(topleft = (window_size/2-window_size/10 - 25, window_size/2-window_size/20)).center)
    win.blit(rotated_arrow, new_rect.topleft)

    # draw robot

    win.blit(robot_scale, (window_size/2-window_size/20, window_size/2-window_size/20))

    # draw distance

    win.blit(text, textRect)

    return x, y, z, run, screen, CollisionWarning, CompleteSign, step






def MapScreen(x,y,des_x,des_y,window_size,chop_size,win,background,DataLog,AlreadySave):

    screen = 'map'
    run = True

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            run = False

    keys = pygame.key.get_pressed()

    # direction key response

    if keys[pygame.K_m]:

        screen = 'main'

    path = []

    for i in range(0, len(DataLog)):

        path.append((int(DataLog[i][0]), int(window_size+1.4*chop_size-DataLog[i][1])))


    # draw background

    win.blit(background, (0, 0))

    # draw robot

##    pygame.draw.circle(win, RED, (int(x), int(window_size+1.4*chop_size-y)), 10)

    # draw destination

    pygame.draw.circle(win, GREEN, (int(des_x), int(window_size+1.4*chop_size-des_y)), 10)

    # draw path

    pygame.draw.lines(win, BLUE, False, path, 4)

    # save path

    if not AlreadySave:

        pygame.image.save(win, "path.png")
        AlreadySave = True


    return run, screen, AlreadySave



def ConfigurationScreen(win,tbs,btn,Welcomebtn,MovingSpeedText,RotatingSpeedText,UserNameText,UserConfigurationText):

    run = True

    win.fill((230,230,230))

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            run = False

        for tb in tbs:

            tb.get_event(event)

        btn.get_event(event)

        Welcomebtn.get_event(event)

    Welcomebtn.draw(win)

    for tb in tbs:

        tb.update()
        tb.draw(win)

    btn.draw(win)
    win.blit(MovingSpeedText, (425, 300))
    win.blit(RotatingSpeedText, (425, 400))
    win.blit(UserNameText, (425, 500))
    win.blit(UserConfigurationText, (300, 150))


    return run






def WelcomeScreen(win,Startbtn,Genbtn,Confbtn,Savebtn,Introbtn,Collectbtn,WelcomeText):

    run = True

    win.fill((230,230,230))

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            run = False

        Startbtn.get_event(event)
        Genbtn.get_event(event)
        Confbtn.get_event(event)
        Savebtn.get_event(event)
        Introbtn.get_event(event)
        Collectbtn.get_event(event)



    Startbtn.draw(win)
    Genbtn.draw(win)
    Confbtn.draw(win)
    Savebtn.draw(win)
    Collectbtn.draw(win)
    Introbtn.draw(win)
    win.blit(WelcomeText, (200, 150))


    return run





def IntroScreen(win,Welcomebtn,DeveloperTextOne,DeveloperTextTwo,photo_scale):

    run = True

    win.fill((230,230,230))

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            run = False

        Welcomebtn.get_event(event)


    Welcomebtn.draw(win)

    win.blit(DeveloperTextOne, (150, 100))
    win.blit(DeveloperTextTwo, (300, 200))
    win.blit(photo_scale, (385, 250))
    return run




def MazeGeneratorScreen(win,current_cell,Loadbtn,grid,stack,width,rows,cols, Save):

    run = True

    TaskComplete = False

    # --- Main event loop

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            run = False

        Loadbtn.get_event(event)

    win.fill(GREY)

    current_cell.visited = True
    current_cell.current = True

    for y in range(rows):

        for x in range(cols):

            grid[y][x].draw(width,win)



    next_cell = current_cell.checkNeighbors(width,cols,rows,grid)
    adjoint_cell = current_cell.checkNeighbors(width,cols,rows,grid)

    if next_cell != False:

        current_cell.neighbors = []

        stack.append(current_cell)

        removeWalls(current_cell,next_cell,width)
        removeWalls(current_cell,adjoint_cell,width)

        current_cell.current = False

        current_cell = next_cell

    elif len(stack) > 0:

        current_cell.current = False
        current_cell = stack.pop()

    elif len(stack) == 0:

        pygame.draw.rect(win,WHITE,(1,1,width,width))
        pygame.draw.line(win,BLACK,(0,0),((0 + width),0),1) # top
        pygame.draw.line(win,BLACK,(0,(0 + width)),(0,0),1) # left

        if Save:
            pygame.image.save(win, "maze.png") # save maze
            Save = False

        TaskComplete = True





    if TaskComplete:

        Loadbtn.draw(win)

    return current_cell, run, Save












