import math

import numpy as np
from random import shuffle
import  time

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5

def setup(self):
    np.random.seed()

def act(self, game_state: dict):
    self.logger.info('Pick action at random')
    # Gather information about the game state
    step = game_state['step']

    _, _, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = game_state['others']
    bomb_map = game_state['explosion_map']
    #self.logger.debug(f'bombmap {bomb_map}')
    #for xy,b in bombs:
    #    self.logger.debug(f'bomb L {xy}')
    #    self.logger.debug(f'bomb count {b}')
    #self.logger.debug(f'bombmap {bomb_map}')

    # exclude wall and crates
    free_space = arena == 0
    # exclude lethals bomb areas(including bombs) and explosions
    lethal_area = get_lethal_area(arena,bombs,bomb_map)
    for la in lethal_area:
        free_space[la[0],la[1]] = False
    # exclude others
    for o in [xy for (n, s, b, xy) in others]:
        free_space[o] = False

    if is_lethal(lethal_area,x,y,arena,bombs,bomb_map):
        escape = escape_dir(lethal_area, x, y, arena, bombs, others, self.logger,bomb_map)
        return ACTIONS[escape]
    else:
        if others:
            otherdir, _ = others_dir_upgrade(lethal_area,free_space,x, y, arena, bombs,others, bombs_left, self.logger, bomb_map)
            return ACTIONS[otherdir]

        if coins:
            coindir= coins_dir_updgrade(lethal_area,free_space,x, y, coins, arena,bombs, others, self.logger, bomb_map)
            return ACTIONS[coindir]

        if 1 in arena:
            cratedir, _ = crates_dir_upgrade(lethal_area, free_space,x, y, arena, bombs, others, bombs_left, self.logger, bomb_map)
            return ACTIONS[cratedir]

'''
    else:
        #if others:
        #    otherdir, _ = others_dir_upgrade(x, y, arena, bombs, others, bombs_left, self.logger, bomb_map)
        #    return ACTIONS[otherdir]
        if coins:
            coindir,_ = coins_dir_updgrade(x,y,coins,arena,bombs,others,self.logger, bomb_map)
            # return ACTIONS[crates_dir_upgrade(x, y, arena, bombs, others, bombs_left)]
            return ACTIONS[coindir]

        if 1 in arena:
            cratedir, _ = crates_dir_upgrade(x, y, arena, bombs, others, bombs_left, self.logger,bomb_map)
            return ACTIONS[cratedir]
'''

def get_lethal_area(arena,bombs,bomb_map):
    # record lethals: bombs area and explosion
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    lethal_area = []
    if bombs:
        for (bx, by), _ in bombs:
            lethal_area.append((bx,by))
            for direction in directions:
                ix, iy = bx, by
                ix, iy = increment_position(ix, iy, direction)
                while (not has_object(ix, iy, arena, 'wall') and
                       abs(ix - bx) <= 3 and abs(iy - by) <= 3):
                    lethal_area.append((ix,iy))
                    ix, iy = increment_position(ix, iy, direction)
    for i in range(np.shape(bomb_map)[0]):
        for j in range(np.shape(bomb_map)[1]):
            if bomb_map[i,j] == 1:
                lethal_area.append((i,j))

    return np.array(lethal_area)

def is_lethal(lethal_area,x: int, y: int, arena: np.array, bombs: list,bombsmap) -> (int):
    for i in lethal_area:
        if i[0] == x and i[1] == y:
            return 1
    return 0

def has_object(x: int, y: int, arena: np.array, object: str) -> bool:
    """
    Check if tile at position (x,y) is of the specified type.
    """
    if object == 'crate':
        return arena[x,y] == 1
    elif object == 'free':
        return arena[x,y] == 0
    elif object == 'wall':
        return arena[x,y] == -1
    else:
        raise ValueError(f"Invalid object {object}")

def increment_position(x: int, y: int, direction: str) -> (int, int):
    """
    Standing at position (x,y), take a step in the specified direction.
    """
    if direction == 'UP':
        y -= 1
    elif direction == 'RIGHT':
        x += 1
    elif direction == 'DOWN':
        y += 1
    elif direction == 'LEFT':
        x -= 1
    else:
        raise ValueError(f"Invalid direction {direction}")
    return x, y

def look_for_targets(free_space, start, targets):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.

    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def escape_dir(lethal_area,x: int, y: int, arena: np.array, bombs: list, others: list, logger,bomb_map) -> int:
    """
    Given agent's position at (x,y) find the direction to the closest non-lethal
    tile. Returns a normalized vector indicating the direction. Returns the zero
    vector if the bombs cannot be escaped or if there are no active bombs.
    """

    # Take a step towards the most immediately interesting target
    # exclude walls and crates
    free_space = arena == 0
    # exclude bombs and others
    for o in [xy for (n, s, b, xy) in others]:
        free_space[o] = False
    for b in [xy for (xy, c) in bombs]:
        free_space[b] = False
    #for c in [co for (xy,co) in bombs]:
    #    logger.debug(f'count {c}')
    # get targets
    targets = []

    for i in range(np.shape(arena)[0]):
        for j in range(np.shape(arena)[1]):
            # if not wall
            if arena[i,j] != -1:
                lethal = is_lethal(lethal_area,i, j, arena, bombs,bomb_map)
                #logger.debug(f'lethal {lethal}')
                if lethal == 0:
                    targets.append((i,j))
    #logger.debug(f'Target {targets}')

    #logger.debug(f'targets {targets}')
    #logger.debug(f'targets {targets}')
    d = look_for_targets(free_space, (x,y), targets)
    if d == (x, y - 1):
        #logger.debug(f'1UP {WAIT}')
        return UP
    elif d == (x, y + 1):
        #logger.debug(f'1DOWN {WAIT}')
        return DOWN
    elif d == (x - 1, y):
        #logger.debug(f'1LEFT {WAIT}')
        return LEFT
    elif d == (x + 1, y):
        #logger.debug(f'1RIGHT {WAIT}')
        return RIGHT
    elif d == (x,y):
        lethal = is_lethal(lethal_area,x, y, arena, bombs,bomb_map)
        if lethal == 0:
            logger.debug(f'WAIT {WAIT}')
            return WAIT
        else:
            return BOMB
        #logger.debug(f'1WAIT {WAIT}')
    else:
        #logger.debug(f'1NOESCAPE {WAIT}')
        return WAIT

def crates_dir_upgrade(lethal_area,free_space,x: int, y: int, arena: np.array, bombs: list, others: list, bombs_left: bool,logger,bomb_map) -> (int,bool):
    """
    version 2:
    return different actions of looking for crates, the actions are noted as different value
    No consideration of escaping lethals
    """
    # Take a step towards the most immediately interesting target

    # get crates
    targets = []
    for i in range(np.shape(arena)[0]):
        for j in range(np.shape(arena)[1]):
            # if id crate
            if arena[i][j] == 1 :
                lethal = is_lethal(lethal_area,i, j, arena, bombs,bomb_map)
                if lethal == 0:
                    targets.append((i,j))
    d = look_for_targets(free_space, (x,y), targets)
    reach_crase = False
    if d == (x, y - 1): return UP, reach_crase
    elif d == (x, y + 1): return DOWN,reach_crase
    elif d == (x - 1, y): return LEFT,reach_crase
    elif d == (x + 1, y): return RIGHT,reach_crase
    elif d == (x,y):
        reach_crase = True
        if(bombs_left):
            logger.debug(f'ReachCrate {reach_crase}')
            return BOMB,reach_crase
        else: return WAIT,reach_crase
    else: return WAIT,reach_crase

def coins_dir_updgrade(lethal_area,free_space,x: int, y: int, coins: list, arena: np.array, bombs: list, others: list,logger,bomb_map) -> int:
    """
        version 2:
        return different actions of looking for coins, the actions are noted as different value
        or return the status of Get_Coin
        No consideration of escaping lethals
    """
    # Take a step towards the most immediately interesting target
    # coins xy
    targets = []

    for c in coins:
        lethal = is_lethal(lethal_area,c[0],c[1],arena,bombs,bomb_map)
        if lethal == 0:
            targets.append(c)
    d = look_for_targets(free_space, (x, y), targets)
    '''
    ecounter_cion = False
    for c1 in coins:
    if c1 == d:
        ecounter_cion = True
        logger.debug(f'ReachCoin {ecounter_cion}')
    '''
    if d == (x, y - 1):
        return UP #, ecounter_cion
    elif d == (x, y + 1):
        return DOWN# , ecounter_cion
    elif d == (x - 1, y):
        return LEFT #, ecounter_cion
    elif d == (x + 1, y):
        return RIGHT #, ecounter_cion
    else:
        if arena[x-1][y] == 1 or arena[x+1][y] == 1 or arena[x][y-1] == 1 or arena[x][y+1] == 1:
            return BOMB #, ecounter_cion
        else: return WAIT

def others_dir_upgrade(lethal_area,free_space, x: int, y: int, arena: np.array, bombs: list, others: list, bombs_left: bool, logger,bomb_map)-> (int,bool):
    """
    version 2:
    return different actions of looking for opponents, the actions are noted as different value
    No consideration of escaping lethals
    """
    # Take a step towards the most immediately interesting target
    # opponents xy
    targets = []

    for _, _, b, xy in others:
        lethal = is_lethal(lethal_area,xy[0], xy[1], arena, bombs,bomb_map)
        # get opponents without aggressive power
        if lethal == 0:
            targets.append(xy)

    d = look_for_targets(free_space, (x,y), targets)
    reach_opponent = False
    if d == (x, y - 1):
        return UP,reach_opponent
    elif d == (x, y + 1):
        return DOWN,reach_opponent
    elif d == (x - 1, y):
        return LEFT,reach_opponent
    elif d == (x + 1, y):
        return RIGHT,reach_opponent
    elif d == (x,y):
        for o in [xy for (n, s, b, xy) in others]:
            if math.sqrt(((o[0]-d[0])**2)+((o[1]-d[1])**2) ) == 1:
                reach_opponent = True
        if arena[x-1][y] == 1 or arena[x+1][y] == 1 or arena[x][y-1] == 1 or arena[x][y+1] == 1 or reach_opponent:
            if bombs_left:
                #logger.debug(f'ReachOPP {reach_opponent}')
                return BOMB, reach_opponent

    return WAIT,reach_opponent

