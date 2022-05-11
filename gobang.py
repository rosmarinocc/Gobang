import pygame
import os
import time
import sys
import tkinter.messagebox
import numpy as np
import random as rand

# 参数设置
WIDTH = 800
HEIGHT = 800
SIZE = 15  # 棋盘大小为15*15
SPACE = WIDTH // (SIZE + 1)  # 网格大小
FPS = 60  # 帧率
INF = 999999  # 定义无穷值
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# init_pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GoBang-五子棋")
clock = pygame.time.Clock()
bg_img = pygame.image.load(os.path.join("images", "background.png"))
background = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))
back_rect = background.get_rect()

# 全局变量
flag_win = 0  # 白子获胜:-1 黑子获胜:1
flag_color = 1  # ai执白子
flag_gg = True
flag_running = True
flag_start = True
# 步数记录
step = 0
# 棋盘矩阵
matrix = np.zeros((SIZE + 2, SIZE + 2), dtype=int)
# 搜索范围
min_x, min_y, = 0, 0
max_x, max_y = 0, 0
# 步骤记录
movements = []


# 绘制网格线
def draw_background(surf):
    screen.blit(background, back_rect)

    rect_lines = [((SPACE, SPACE), (SPACE, HEIGHT - SPACE)),
                  ((SPACE, SPACE), (WIDTH - SPACE, SPACE)),
                  ((SPACE, HEIGHT - SPACE), (WIDTH - SPACE, HEIGHT - SPACE)),
                  ((WIDTH - SPACE, SPACE), (WIDTH - SPACE, HEIGHT - SPACE))]
    # 边框线
    for line in rect_lines:
        pygame.draw.line(surf, COLOR_BLACK, line[0], line[1], 3)
    # 网格线
    for i in range(17):
        pygame.draw.line(surf, COLOR_BLACK, (SPACE * (2 + i), SPACE),
                         (SPACE * (2 + i), HEIGHT - SPACE))
        pygame.draw.line(surf, COLOR_BLACK,
                         (SPACE, SPACE * (2 + i)),
                         (HEIGHT - SPACE, SPACE * (2 + i)))
    # 画棋盘上的黑色标记
    dots = [(SPACE * 4, SPACE * 4),
            (SPACE * 8, SPACE * 4),
            (SPACE * 12, SPACE * 4),
            (SPACE * 4, SPACE * 8),
            (SPACE * 8, SPACE * 8),
            (SPACE * 12, SPACE * 8),
            (SPACE * 4, SPACE * 12),
            (SPACE * 8, SPACE * 12),
            (SPACE * 12, SPACE * 12)]

    for dot in dots:
        pygame.draw.circle(surf, COLOR_BLACK, dot, 8)

# 刷新棋盘已占有棋子的外切矩形范围
def update_range(x, y):
    global min_x, min_y, max_x, max_y
    if step == 0:
        min_x, min_y, max_x, max_y = x, y, x, y
    else:
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y

# 棋型评估
model_score = {
    # 一子:两端开
    (0, 1, 0): 2,
    # 死二
    (-1,1,1, -1): -5,
    # 死三
    (-1,1,1, 1, -1): -5,
    # 死四
    (-1,1,1,1, 1, -1): -5,
    # 二子:一端死
    (0, 1, 1, -1): 5,
    (-1, 1, 1, 0): 5,
    # 二子:两端开
    (0, 1, 1, 0): 20,
    # 三子:一端死
    (-1, 1, 1, 1, 0): 20,
    (0, 1, 1, 1, -1): 20,
    (-1, 1, 1, 1, 0,0): 20,
    (0,0, 1, 1, 1, -1): 20,
    (0, 1, 0,1, 1, -1): 20,
    (-1, 1, 1, 0, 1, 0): 20,
    (0, 1, 1,0, 1, -1): 20,
    (-1, 1, 0, 1, 1,0): 20,
    (1, 0, 0, 1, 1): 20,
    (1, 1, 0, 0, 1): 20,
    (1, 0, 1, 0 ,1): 20,
    (-1, 0, 1, 1, 1, 0 ,-1): 20,
    # 三子:两端开
    (0, 1, 1, 1, 0): 40,
    (0, 1, 0, 1, 1, 0): 40,
    (0, 1, 1, 0, 1, 0): 40,
    # 四子:一端死
    (-1, 1, 1, 1, 1, 0): 80,
    (0, 1, 1, 1, 1, -1): 80,
    (0,1,0,1,1,1,0): 80,
    (0,1,1,1,0,1,0): 80,
    (0,1,1,0,1,1,0): 80,
    # 四子:两端开
    (0, 1, 1, 1, 1, 0): 160,
    # 五子
    (0, 1, 1, 1, 1, 1, 0): 320,
    (0, 1, 1, 1, 1, 1, -1): 320,
    (-1, 1, 1, 1, 1, 1, 0): 320,
    (-1, 1, 1, 1, 1, 1, -1): 320
}


# 评估一个节点分值AI为正数 玩家为负数(调用的时候确定符号)
def evaluation(list_h, list_v, list_s, list_b):
    score_h = model_score.get(tuple(list_h), 0)
    score_v = model_score.get(tuple(list_v), 0)
    score_s = model_score.get(tuple(list_s), 0)
    score_b = model_score.get(tuple(list_b), 0)
    rank = [score_h, score_v, score_s, score_b]
    return sum(rank)

# 获得该结点在水平、竖直、左斜、右斜方向上构成的同色的棋子
def get_list(mx, my, color):
    global matrix
    # list_h:水平
    # 向右
    list1 = []
    tx, ty = mx, my
    while matrix[tx][ty] == color:
        list1.append(1)  # 1表示是己方棋子，-1是敌方棋子
        tx = tx + 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list1.append(-1)
    else:
        list1.append(0)

    # 删除拼接交叉点
    list1.pop(0)

    # 向左
    list2 = []
    tx = mx
    ty = my
    while matrix[tx][ty] == color:
        list2.insert(0, 1)
        tx = tx - 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list2.insert(0, -1)
    else:
        list2.insert(0, 0)
    list_h = list2 + list1

    # list_v:垂直方向
    list1 = []
    tx = mx
    ty = my
    while matrix[tx][ty] == color:
        list1.append(1)
        ty = ty + 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list1.append(-1)
    else:
        list1.append(0)
    list1.pop(0)
    list2 = []
    tx = mx
    ty = my
    while matrix[tx][ty] == color:
        list2.insert(0, 1)
        ty = ty - 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list2.insert(0, -1)
    else:
        list2.insert(0, 0)
    list_v = list2 + list1

    # list_l:向左斜
    list1 = []
    tx = mx
    ty = my
    while matrix[tx][ty] == color:
        list1.append(1)
        tx = tx - 1
        ty = ty + 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list1.append(-1)
    else:
        list1.append(0)
    list1.pop(0)
    list2 = []
    tx = mx
    ty = my
    while matrix[tx][ty] == color:
        list2.insert(0, 1)
        tx = tx + 1
        ty = ty - 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list2.insert(0, -1)
    else:
        list2.insert(0, 0)
    list_l = list2 + list1

    # list_r:向右斜
    list1 = []
    tx = mx
    ty = my
    while matrix[tx][ty] == color:
        list1.append(1)
        tx = tx + 1
        ty = ty + 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list1.append(-1)
    else:
        list1.append(0)
    list1.pop(0)
    list2 = []
    tx = mx
    ty = my
    while matrix[tx][ty] == color:
        list2.insert(0, 1)
        tx = tx - 1
        ty = ty - 1
    if matrix[tx][ty] == -color or tx == 0 or ty == 0 or tx > SIZE or ty > SIZE:
        list2.insert(0, -1)
    else:
        list2.insert(0, 0)
    list_r = list2 + list1

    return [list_h, list_v, list_l, list_r]

# 判断搜索范围是否超出边界，返回合法的搜索范围
def legal_range(min_x, min_y, max_x, max_y):
    change = 1
    if min_x - change < 1:
        min_tx = 1
    else:
        min_tx = min_x - change

    if min_y - change < 1:
        min_ty = 1
    else:
        min_ty = min_y - change

    if max_x + change > SIZE:
        max_tx = SIZE
    else:
        max_tx = max_x + change

    if max_y + change > SIZE:
        max_ty = SIZE
    else:
        max_ty = max_y + change
    return [min_tx, min_ty, max_tx, max_ty]

# alpha-beta剪枝搜索
def round_ai():
    global min_x, max_x, min_y, max_y, flag_color, matrix
    time_s = time.time()

    if step != 0:  # step=0 步骤为玩家(对手)行棋
        if step == 1:
            # 第一步抢占中心,若中心被占则随机选择周围四点
            if matrix[(SIZE + 1) // 2][(SIZE + 1) // 2] == 0:
                rx, ry = (SIZE + 1) // 2, (SIZE + 1) // 2
            else:
                case = rand.randint(1, 4)
                if case == 1:
                    dx, dy = 1, 1
                elif case == 2:
                    dx, dy = 1, -1
                elif case == 3:
                    dx, dy = -1, 1
                else:
                    dx, dy = -1, -1
                rx, ry = (SIZE + 1) // 2 + dx, (SIZE + 1) // 2 + dy
        else:

            min_tx1, min_ty1, max_tx1, max_ty1 = legal_range(min_x, min_y, max_x, max_y)

            eva_matrix1 = np.zeros((SIZE + 2, SIZE + 2), dtype=int)  # 第一层的估值矩阵
            value_max = -INF
            rx, ry = 0, 0

            for i in range(min_tx1, max_tx1 + 1):
                for j in range(min_ty1, max_ty1 + 1):
                    # 是否剪枝
                    flag_cut = False
                    eva_matrix2 = np.zeros((SIZE + 2, SIZE + 2), dtype=int)

                    if matrix[i][j] == 0:
                        matrix[i][j] = flag_color
                        min_tx2, min_ty2, max_tx2, max_ty2 = legal_range(min_tx1, min_ty1, max_tx1, max_ty1)
                        [list_h, list_v, list_l, list_r] = get_list(i, j, flag_color)
                        eva1 = evaluation(list_h, list_v, list_l, list_r)

                        for _i in range(min_tx2, max_tx2 + 1):
                            for _j in range(min_ty2, max_ty2 + 1):

                                if matrix[_i][_j] == 0:
                                    matrix[_i][_j] = -flag_color
                                    [list_h, list_v, list_s, list_b] = get_list(_i, _j, -flag_color)
                                    eva2 = -evaluation(list_h, list_v, list_s, list_b)

                                    eva_matrix2[_i][_j] = eva2 + eva1
                                    matrix[_i][_j] = 0
                                    # 剪枝
                                    if eva_matrix2[_i][_j] < value_max:
                                        eva_matrix1[i][j] = eva_matrix2[_i][_j]
                                        flag_cut = 1
                                        break
                            if flag_cut:
                                break

                        if not flag_cut:
                            value_min = INF
                            for _i in range(min_tx2, max_tx2 + 1):
                                for _j in range(min_ty2, max_ty2 + 1):
                                    if eva_matrix2[_i][_j] < value_min and matrix[_i][_j] == 0:
                                        value_min = eva_matrix2[_i][_j]

                            eva_matrix1[i][j] = value_min

                            if value_max < value_min:
                                value_max = value_min
                                rx, ry = i, j

                        matrix[i][j] = 0

        time_e = time.time()
        print("Time cost:", round(time_e - time_s, 4), "s")
        add_chess(rx, ry, flag_color)


def round_ai2():
    global min_x, max_x, min_y, max_y, flag_color, matrix
    time_s = time.time()

    if step != 0:  # step=0 步骤为玩家(对手)行棋
        if step == 1:
            # 第一步抢占中心,若中心被占则随机选择周围四点
            if matrix[(SIZE + 1) // 2][(SIZE + 1) // 2] == 0:
                rx, ry = (SIZE + 1) // 2, (SIZE + 1) // 2
            else:
                case = rand.randint(1, 4)
                if case == 1:
                    dx, dy = 1, 1
                elif case == 2:
                    dx, dy = 1, -1
                elif case == 3:
                    dx, dy = -1, 1
                else:
                    dx, dy = -1, -1
                rx, ry = (SIZE + 1) // 2 + dx, (SIZE + 1) // 2 + dy
        else:
            min_tx1, min_ty1, max_tx1, max_ty1 = legal_range(min_x, min_y, max_x, max_y)

            eva_matrix1 = np.zeros((SIZE + 2, SIZE + 2), dtype=int)  # 第一层的估值矩阵
            value_max = -INF
            rx, ry = 0, 0
            #第一层搜索
            for i in range(min_tx1, max_tx1 + 1):
                for j in range(min_ty1, max_ty1 + 1):
                    # 是否剪枝
                    flag_cut = False
                    eva_matrix2 = np.zeros((SIZE + 2, SIZE + 2), dtype=int)

                    if matrix[i][j] == 0:
                        matrix[i][j] = flag_color
                        min_tx2, min_ty2, max_tx2, max_ty2 = legal_range(min_tx1, min_ty1, max_tx1, max_ty1)
                        [list_h, list_v, list_l, list_r] = get_list(i, j, flag_color)
                        eva1 = evaluation(list_h, list_v, list_l, list_r)
                        #第二层搜索
                        for _i in range(min_tx2, max_tx2 + 1):
                            for _j in range(min_ty2, max_ty2 + 1):

                                if matrix[_i][_j] == 0:
                                    matrix[_i][_j] = -flag_color
                                    [list_h, list_v, list_s, list_b] = get_list(_i, _j, -flag_color)
                                    eva2 = -evaluation(list_h, list_v, list_s, list_b)

                                    eva_matrix2[_i][_j] = eva2 + eva1
                                    matrix[_i][_j] = 0
                                    # 剪枝
                                    if eva_matrix2[_i][_j] < value_max:
                                        eva_matrix1[i][j] = eva_matrix2[_i][_j]
                                        flag_cut = 1
                                        break
                            if flag_cut:
                                break

                        if not flag_cut:
                            value_min = INF
                            for _i in range(min_tx2, max_tx2 + 1):
                                for _j in range(min_ty2, max_ty2 + 1):
                                    if eva_matrix2[_i][_j] < value_min and matrix[_i][_j] == 0:
                                        value_min = eva_matrix2[_i][_j]

                            eva_matrix1[i][j] = value_min

                            if value_max < value_min:
                                value_max = value_min
                                rx, ry = i, j

                        matrix[i][j] = 0

        time_e = time.time()
        print("Time cost:", round(time_e - time_s, 4), "s")
        add_chess(rx, ry, flag_color)

# 玩家行棋
def round_player(pos):
    x = round(pos[0] / SPACE)
    y = round(pos[1] / SPACE)
    if 1 <= x <= SIZE and 1 <= y <= SIZE and matrix[x][y] == 0:
        add_chess(x, y, -flag_color)
        return True

# 添加棋子
def add_chess(x, y, color):
    global step, matrix
    step = step + 1
    movements.append((x, y, color, step))
    matrix[x][y] = color
    update_range(x, y)
    is_gg()

# 撤销棋子
def withdraw_chess():
    global step, matrix
    i = 0
    while i != 2 and len(movements):
        step = step - 1
        x = movements[-1][0]
        y = movements[-1][1]
        del movements[-1]
        matrix[x][y] = 0
        i = i + 1

# 绘制文本
def draw_text(surf, text, size, x, y):
    font = pygame.font.SysFont("华文仿宋", size)
    text_surface = font.render(text, True, COLOR_BLACK)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    surf.blit(text_surface, text_rect)

# 绘制棋子
def draw_chess(surf):
    for move in movements:
        if move[2] == flag_color:
            pygame.draw.circle(surf, COLOR_WHITE, (move[0] * SPACE, move[1] * SPACE), 16)
        else:
            pygame.draw.circle(surf, COLOR_BLACK, (move[0] * SPACE, move[1] * SPACE), 16)

# 判断游戏是否结束
def is_gg():
    global flag_win, flag_gg, flag_start
    x = movements[-1][0]
    y = movements[-1][1]
    color = movements[-1][2]
    [list_h, list_v, list_l, list_r] = get_list(x, y, color)
    if sum(list_h[1:-1]) == 5 or sum(list_v[1:-1]) == 5 or sum(list_l[1:-1]) == 5 or sum(list_r[1:-1]) == 5:
        flag_win = color
        flag_gg = True
        flag_start = True

# 开始界面显示
def init_ui(surf):
    global flag_win, movements, step, matrix, min_x, min_y, max_x, max_y, flag_gg, flag_start
    if flag_start:
        if flag_win != 0:
            root = tkinter.Tk()
            root.withdraw()
            if flag_win == 1:
                tkinter.messagebox.showinfo("游戏结束", "你输了!")
            else:
                tkinter.messagebox.showinfo("游戏结束", "你赢了!")
        else:
            screen.blit(background, back_rect)
        draw_text(surf, "Enter键以开始游戏", 36, WIDTH / 2, HEIGHT / 2)

    pygame.display.flip()
    flag_win = 0
    movements = []
    step = 0
    matrix = [[0 for i in range(SIZE + 2)] for j in range(SIZE + 2)]
    min_x, min_y, max_x, max_y = 0, 0, 0, 0
    flag_gg = False
    flag_waiting = True
    while flag_waiting and flag_start:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN:
                    round_ai()
                    flag_waiting = False
    flag_start = False

# 主循环
while flag_running:
    if flag_gg:
        init_ui(screen)
    clock.tick(FPS)
    if step % 2 == 1:  # ai行奇数步
        round_ai()
    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag_running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                round_player(event.pos)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    flag_gg = True
                elif event.key == pygame.K_w:
                    withdraw_chess()
                elif event.key == pygame.K_q:
                    flag_running = False

    draw_background(screen)
    draw_chess(screen)
    draw_text(screen, "R:重开 -- W:悔棋 -- Q:退出", 28, WIDTH // 2, SPACE//2)
    pygame.display.flip()

pygame.quit()
sys.exit()
