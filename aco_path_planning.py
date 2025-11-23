import pygame
import random
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Game settings
GRID_SIZE = 30
CELL_SIZE = 30
MAP_WIDTH = GRID_SIZE * CELL_SIZE

# GUI Settings
GUI_WIDTH = 250
WIDTH = 1440
HEIGHT = 900

OBSTACLE_DENSITY = 0.25
FPS = 30
MOVE_TICK = 15
DETECTION_RANGE = 7

# ACO Settings
NUM_ANTS = 20
NUM_ITERATIONS = 50
ALPHA = 1.0
BETA = 2.0
RHO = 0.1
Q = 100

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 150, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
PATH_COLOR = (0, 255, 255)
FLOOR_COLOR = (200, 200, 200)
GUI_BG_COLOR = (50, 50, 50)
BUTTON_COLOR = (0, 100, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)
TEXT_COLOR = WHITE
INPUT_BG_COLOR = (100, 100, 100)
INPUT_TEXT_COLOR = WHITE
ACTIVE_INPUT_COLOR = (150, 150, 150)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("ACO dynamic path planning.")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
font_big = pygame.font.Font(None, 30)
formula_font = pygame.font.SysFont("consolas", 26, bold=True)
formula_small = pygame.font.SysFont("consolas", 22)

# -------------------------------
# Map & Game State
def generate_map(size, density):
    game_map = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            if random.random() < density:
                game_map[row][col] = 1
    game_map[0][0] = 0
    game_map[size - 1][size - 1] = 0
    return game_map

def reset_game_state(new_map=False):
    global game_map, pheromones, best_path, is_calculating
    global is_moving_automatically, path_step_index, robot_pos, current_start_pos

    if new_map:
        game_map = generate_map(GRID_SIZE, OBSTACLE_DENSITY)

    current_start_pos = initial_start_pos
    robot_pos[:] = current_start_pos
    pheromones = initialize_pheromones(GRID_SIZE, game_map)
    best_path = None
    is_calculating = False
    is_moving_automatically = False
    path_step_index = 0
    print("Game reset.")

def initialize_pheromones(size, game_map):
    pheromones = [[1.0 for _ in range(size)] for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if game_map[r][c] == 1:
                pheromones[r][c] = 0.0
    return pheromones

def get_neighbors(col, row):
    neighbors = []
    directions = [(0,1),(0,-1),(1,0),(-1,0)]
    for dc, dr in directions:
        n_col, n_row = col+dc, row+dr
        if 0 <= n_row < GRID_SIZE and 0 <= n_col < GRID_SIZE and game_map[n_row][n_col]==0:
            neighbors.append((n_col,n_row))
    return neighbors

def distance(pos1,pos2):
    c1,r1=pos1
    c2,r2=pos2
    return math.sqrt((r1-r2)**2 + (c1-c2)**2) or 0.0001

def ant_move(current_pos, visited, target_pos, pheromones_map):
    c,r=current_pos
    neighbors=get_neighbors(c,r)
    unvisited=[n for n in neighbors if n not in visited]
    if not unvisited: return None
    probabilities=[]
    total=0.0
    for nc,nr in unvisited:
        pheromone = pheromones_map[nr][nc] ** ALPHA
        heuristic = (1.0/distance((nc,nr),target_pos)) ** BETA
        attr = pheromone*heuristic
        probabilities.append(((nc,nr),attr))
        total += attr
    if total==0.0:
        return random.choice(unvisited)
    choices, weights = zip(*[(pos,a/total) for pos,a in probabilities])
    return random.choices(choices, weights=weights, k=1)[0]

def run_aco_iteration(start_pos, record_stats=False):
    global pheromones, best_path
    all_paths=[]
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if game_map[r][c]==0:
                pheromones[r][c]*=(1.0-RHO)
    for _ in range(NUM_ANTS):
        path=[start_pos]
        cur=start_pos
        while cur!=target_pos:
            nxt=ant_move(cur,set(path),target_pos,pheromones)
            if nxt is None: break
            path.append(nxt)
            cur=nxt
        if path[-1]==target_pos:
            all_paths.append(path)
    for path in all_paths:
        L=len(path)-1
        if L>0:
            pheromone_deposit = Q/L
            for c,r in path:
                pheromones[r][c]+=pheromone_deposit
    success=[p for p in all_paths if p[-1]==target_pos]
    if success:
        success.sort(key=len)
        best=success[0]
        if best_path is None or len(best)<len(best_path):
            best_path=best
    if record_stats:
        best_len = len(best_path)-1 if best_path else 0
        pher_sum = sum(sum(row) for row in pheromones)
        return best_len, pher_sum
    return len(success)>0

def run_aco_iteration_local(start_pos, local_pheromones, local_best_path):
    all_paths=[]
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if game_map[r][c]==0:
                local_pheromones[r][c]*=(1.0-RHO)
    for _ in range(NUM_ANTS):
        path=[start_pos]
        cur=start_pos
        while cur!=target_pos:
            nxt=ant_move(cur,set(path),target_pos,local_pheromones)
            if nxt is None: break
            path.append(nxt)
            cur=nxt
        if path[-1]==target_pos:
            all_paths.append(path)
    for path in all_paths:
        L=len(path)-1
        if L>0:
            pheromone_deposit = Q/L
            for c,r in path:
                local_pheromones[r][c]+=pheromone_deposit
    success=[p for p in all_paths if p[-1]==target_pos]
    if success:
        success.sort(key=len)
        best=success[0]
        if local_best_path is None or len(best)<len(local_best_path):
            local_best_path=best
    best_len = len(local_best_path)-1 if local_best_path else 0
    pher_sum = sum(sum(row) for row in local_pheromones)
    return best_len, pher_sum, local_best_path

# -------------------------------
# GUI Draw Functions
def draw_map(screen, game_map, cell_size, current_start):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = pygame.Rect(GUI_WIDTH + col*cell_size, row*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, FLOOR_COLOR, rect)
            if game_map[row][col]==1:
                pygame.draw.rect(screen, RED, rect)
            if (col,row)==current_start and (col,row)!=target_pos:
                pygame.draw.rect(screen, GREEN, rect,3)
            elif (col,row)==target_pos:
                pygame.draw.rect(screen, BLUE, rect)
            pygame.draw.rect(screen, BLACK, rect,1)

def draw_path(screen, path, cell_size):
    if path and len(path)>1:
        points=[]
        for c,r in path:
            cx=GUI_WIDTH+c*cell_size+cell_size//2
            cy=r*cell_size+cell_size//2
            points.append((cx,cy))
            pygame.draw.circle(screen, PATH_COLOR,(cx,cy),cell_size//6)
        pygame.draw.lines(screen, PATH_COLOR, False, points,3)

def draw_robot(screen,pos,cell_size):
    col,row=pos
    cx=GUI_WIDTH+col*cell_size+cell_size//2
    cy=row*cell_size+cell_size//2
    robot_rect=pygame.Rect(cx-cell_size//4, cy-cell_size//4, cell_size//2, cell_size//2)
    pygame.draw.rect(screen, GREEN, robot_rect)

def draw_text_line(screen, text, x, y, color):
    txt=font.render(text, True, color)
    screen.blit(txt,(x,y))

# InputBox
class InputBox:
    def __init__(self,x,y,w,h,text='',label='',is_float=True):
        self.rect=pygame.Rect(x,y,w,h)
        self.color=INPUT_BG_COLOR
        self.text=text
        self.label=label
        self.txt_surface=font.render(text, True, INPUT_TEXT_COLOR)
        self.active=False
        self.is_float=is_float
    def handle_event(self,event):
        if event.type==pygame.MOUSEBUTTONDOWN:
            self.active=self.rect.collidepoint(event.pos)
            self.color=ACTIVE_INPUT_COLOR if self.active else INPUT_BG_COLOR
        if event.type==pygame.KEYDOWN and self.active:
            if event.key==pygame.K_RETURN:
                self.active=False
                self.color=INPUT_BG_COLOR
            elif event.key==pygame.K_BACKSPACE:
                self.text=self.text[:-1]
            else:
                c=event.unicode
                if c.isdigit():
                    self.text+=c
                elif self.is_float and c=='.' and '.' not in self.text:
                    self.text+=c
            self.txt_surface=font.render(self.text, True, INPUT_TEXT_COLOR)
    def draw(self,screen):
        label_surface=font.render(self.label, True, TEXT_COLOR)
        screen.blit(label_surface,(self.rect.x,self.rect.y-18))
        pygame.draw.rect(screen, self.color, self.rect, border_radius=3)
        screen.blit(self.txt_surface,(self.rect.x+5, self.rect.y+(self.rect.height-self.txt_surface.get_height())//2))
        if self.active:
            pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=3)

# Buttons
BUTTONS = {
    "start_aco": pygame.Rect(15, 50, GUI_WIDTH-30,40),
    "reset_map": pygame.Rect(15,100,GUI_WIDTH-30,40),
    "reset_agent": pygame.Rect(15,150,GUI_WIDTH-30,40),
    "save_aco_params": pygame.Rect(15,720,GUI_WIDTH-30,40),
    "save_stats": pygame.Rect(15,780,GUI_WIDTH-30,40)
}

def draw_gui(screen, mouse_pos, input_boxes):
    gui_rect=pygame.Rect(0,0,GUI_WIDTH,HEIGHT)
    pygame.draw.rect(screen, GUI_BG_COLOR, gui_rect)
    title=font_big.render("ACO CONTROL", True, TEXT_COLOR)
    screen.blit(title,(15,15))
    for key,rect in BUTTONS.items():
        is_hover=rect.collidepoint(mouse_pos)
        color=BUTTON_HOVER_COLOR if is_hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=5)
        if key=="start_aco":
            text="Start / Resume ACO"
        elif key=="reset_map":
            text="Reset Map"
        elif key=="reset_agent":
            text="Reset Agent"
        elif key=="save_aco_params":
            text="Save ACO Params"
        else:
            text="Save Stats"
        txt_surf=font.render(text, True, TEXT_COLOR)
        screen.blit(txt_surf,(rect.x+(rect.width-txt_surf.get_width())//2, rect.y+(rect.height-txt_surf.get_height())//2))

    y_offset=180
    if is_calculating:
        status_text=f"ACO: Calculating ({current_iteration}/{NUM_ITERATIONS})"
        status_color=PATH_COLOR
    elif is_moving_automatically:
        length=len(best_path)-1 if best_path else 0
        status_text=f"MOVING: {path_step_index-1}/{length} steps"
        status_color=GREEN
    elif best_path:
        status_text=f"READY! Path length: {len(best_path)-1}"
        status_color=WHITE
    else:
        status_text="ACO Not Started"
        status_color=RED
    draw_text_line(screen,"Status:",15,y_offset,TEXT_COLOR)
    draw_text_line(screen,status_text,15,y_offset+20,status_color)

    y_offset+=60
    draw_text_line(screen,"Robot parameters:",15,y_offset,TEXT_COLOR)
    draw_text_line(screen,f"Automatic detection: Within {DETECTION_RANGE} cells",15,y_offset+20,WHITE)
    draw_text_line(screen,f"Grid Size: {GRID_SIZE}x{GRID_SIZE}",15,y_offset+40,WHITE)

    y_offset+=100
    draw_text_line(screen,"Map Control:",15,y_offset,TEXT_COLOR)
    draw_text_line(screen,"LMB: Add Obstacle",15,y_offset+20,WHITE)
    draw_text_line(screen,"RMB: Remove Obstacle",15,y_offset+40,WHITE)

    y_offset+=80
    draw_text_line(screen,"ACO Parameters:",15,y_offset,PATH_COLOR)
    for box in input_boxes.values():
        box.draw(screen)

# -------------------------------
# Paths
initial_start_pos=(0,0)
target_pos=(GRID_SIZE-1,GRID_SIZE-1)
game_map=generate_map(GRID_SIZE, OBSTACLE_DENSITY)
pheromones=initialize_pheromones(GRID_SIZE, game_map)
best_path=None
is_calculating=False
is_moving_automatically=False
path_step_index=0
move_delay=0
current_start_pos=initial_start_pos
robot_pos=list(initial_start_pos)

# Input boxes
input_box_h=25
input_box_w=70
input_box_x=15 + (GUI_WIDTH-30-input_box_w)//2
current_y=480
input_boxes={
    "NUM_ANTS": InputBox(input_box_x,current_y,input_box_w,input_box_h,text=str(NUM_ANTS),label="Num. of ants",is_float=False),
    "NUM_ITERATIONS": InputBox(input_box_x,current_y+40,input_box_w,input_box_h,text=str(NUM_ITERATIONS),label="Iterations",is_float=False),
    "ALPHA": InputBox(input_box_x,current_y+80,input_box_w,input_box_h,text=str(ALPHA),label="Alpha"),
    "BETA": InputBox(input_box_x,current_y+120,input_box_w,input_box_h,text=str(BETA),label="Beta"),
    "RHO": InputBox(input_box_x,current_y+160,input_box_w,input_box_h,text=str(RHO),label="Rho"),
    "Q": InputBox(input_box_x,current_y+200,input_box_w,input_box_h,text=str(Q),label="Q",is_float=False),
}

# Statistics
def save_stats_experiments():
    folder="stats_output"
    os.makedirs(folder, exist_ok=True)

    best_lengths = []
    pher_sums = []
    local_pheromones = initialize_pheromones(GRID_SIZE, game_map)
    local_best_path = None

    for i in range(NUM_ITERATIONS):
        best_len, pher_sum, local_best_path = run_aco_iteration_local(initial_start_pos, local_pheromones, local_best_path)
        best_lengths.append(best_len)
        pher_sums.append(pher_sum)

    plt.figure()
    plt.plot(range(1,NUM_ITERATIONS+1), best_lengths, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Best Path Length")
    plt.title("Best Path Length per Iteration")
    plt.grid(True)
    plt.savefig(os.path.join(folder,"best_path_length.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1,NUM_ITERATIONS+1), pher_sums, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Sum of Pheromones")
    plt.title("Sum of Pheromones per Iteration")
    plt.grid(True)
    plt.savefig(os.path.join(folder,"pheromone_sum.png"))
    plt.close()

    plt.figure(figsize=(8,8))
    data = np.array(pheromones)
    plt.imshow(data, cmap='Blues', origin='upper')
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if game_map[r][c] == 1:
                plt.scatter(c,r,color='red',s=50)
    plt.title("Pheromone Heatmap")
    plt.colorbar(label="Pheromone Level")
    plt.savefig(os.path.join(folder,"pheromone_heatmap.png"))
    plt.close()

    with open(os.path.join(folder,"parameters.txt"),"w") as f:
        f.write(f"NUM_ANTS={NUM_ANTS}\nNUM_ITERATIONS={NUM_ITERATIONS}\n")
        f.write(f"ALPHA={ALPHA}\nBETA={BETA}\nRHO={RHO}\nQ={Q}\n")

    print("Statistics saved in folder:", folder)

# -------------------------------
# Main loop
running=True
current_iteration=0

while running:
    mouse_pos=pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
        for box in input_boxes.values():
            box.handle_event(event)
        if event.type==pygame.MOUSEBUTTONDOWN:
            mx,my=event.pos
            if mx<GUI_WIDTH:
                if BUTTONS["start_aco"].collidepoint(mx,my):
                    print("Starting ACO...")
                    is_calculating=True
                    current_iteration=0
                    best_path=None
                    is_moving_automatically=False
                    current_start_pos=tuple(robot_pos)
                    pheromones=initialize_pheromones(GRID_SIZE, game_map)
                elif BUTTONS["reset_map"].collidepoint(mx,my):
                    reset_game_state(new_map=True)
                elif BUTTONS["reset_agent"].collidepoint(mx,my):
                    robot_pos[:] = initial_start_pos
                    current_start_pos=initial_start_pos
                    is_moving_automatically=False
                    is_calculating=False
                    path_step_index=0
                    print("Agent reset to start position!")
                elif BUTTONS["save_aco_params"].collidepoint(mx,my):
                    try:
                        NUM_ANTS=int(input_boxes["NUM_ANTS"].text)
                        NUM_ITERATIONS=int(input_boxes["NUM_ITERATIONS"].text)
                        ALPHA=float(input_boxes["ALPHA"].text)
                        BETA=float(input_boxes["BETA"].text)
                        RHO=float(input_boxes["RHO"].text)
                        Q=int(input_boxes["Q"].text)
                        if NUM_ANTS<=0 or NUM_ITERATIONS<=0 or ALPHA<0 or BETA<0 or RHO<=0 or Q<=0:
                            raise ValueError("Parameters must be positive.")
                        print("ACO parameters updated successfully!")
                        pheromones=initialize_pheromones(GRID_SIZE, game_map)
                        best_path=None
                        is_calculating=False
                        is_moving_automatically=False
                        current_iteration=0
                    except ValueError as e:
                        print("ERROR:", e)
                elif BUTTONS["save_stats"].collidepoint(mx,my):
                    save_stats_experiments()
            else:
                col=(mx-GUI_WIDTH)//CELL_SIZE
                row=my//CELL_SIZE
                if 0<=row<GRID_SIZE and 0<=col<GRID_SIZE:
                    if (col,row) not in [initial_start_pos,target_pos]:
                        change=False
                        if event.button==1 and game_map[row][col]!=1:
                            game_map[row][col]=1
                            change=True
                        elif event.button==3 and game_map[row][col]!=0:
                            game_map[row][col]=0
                            change=True
                        if change:
                            robot_c,robot_r=robot_pos
                            dist=abs(col-robot_c)+abs(row-robot_r)
                            if dist<=DETECTION_RANGE and (is_moving_automatically or is_calculating):
                                is_moving_automatically=False
                                is_calculating=True
                                current_start_pos=tuple(robot_pos)
                                current_iteration=0
                                pheromones=initialize_pheromones(GRID_SIZE, game_map)
                                best_path=None
                                print("Map changed within range! Recalculating...")
                            else:
                                pheromones=initialize_pheromones(GRID_SIZE, game_map)
        if not is_calculating and not is_moving_automatically:
            if event.type==pygame.KEYDOWN:
                new_c,new_r=robot_pos
                if event.key==pygame.K_LEFT:
                    new_c-=1
                elif event.key==pygame.K_RIGHT:
                    new_c+=1
                elif event.key==pygame.K_UP:
                    new_r-=1
                elif event.key==pygame.K_DOWN:
                    new_r+=1
                if 0<=new_c<GRID_SIZE and 0<=new_r<GRID_SIZE and game_map[new_r][new_c]==0:
                    robot_pos=[new_c,new_r]
                    current_start_pos=tuple(robot_pos)

    # ACO iteration
    if is_calculating:
        if current_iteration<NUM_ITERATIONS:
            run_aco_iteration(current_start_pos)
            current_iteration+=1
        else:
            is_calculating=False
            if best_path:
                print("ACO finished. Path length:",len(best_path)-1)
                is_moving_automatically=True
                path_step_index=1
            else:
                print("ACO finished. No path found.")

    # Move robot automatically
    if is_moving_automatically:
        if path_step_index<len(best_path):
            next_c,next_r=best_path[path_step_index]
            if game_map[next_r][next_c]==1:
                robot_c,robot_r=robot_pos
                dist=abs(next_c-robot_c)+abs(next_r-robot_r)
                if dist<=DETECTION_RANGE:
                    is_moving_automatically=False
                    is_calculating=True
                    current_start_pos=tuple(robot_pos)
                    current_iteration=0
                    pheromones=initialize_pheromones(GRID_SIZE, game_map)
                    best_path=None
                    print("Obstacle on route! Recalculating...")
                    continue
                else:
                    print("Obstacle out of range! Stopping.")
                    is_moving_automatically=False
                    continue
        move_delay+=1
        if move_delay>=MOVE_TICK:
            move_delay=0
            if path_step_index<len(best_path):
                robot_pos=list(best_path[path_step_index])
                current_start_pos=tuple(robot_pos)
                path_step_index+=1
                if robot_pos==list(target_pos):
                    is_moving_automatically=False
                    print("Robot reached target!")
            else:
                is_moving_automatically=False

    # Draw everything
    screen.fill(WHITE)
    draw_gui(screen, mouse_pos, input_boxes)
    draw_map(screen, game_map, CELL_SIZE, current_start_pos)
    draw_path(screen, best_path, CELL_SIZE)
    draw_robot(screen, robot_pos, CELL_SIZE)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
