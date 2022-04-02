import math
import os
import pickle
import random
from queue import Queue
from random import shuffle
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDRegressor
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# different features' value of
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5

# ---------------- Parameters ----------------
FILENAME = "Yi_Zhang-Changjing_Hu_agent_old"  # Base filename of model (excl. extensions).
ACT_STRATEGY = 'eps-greedy'  # Options: 'softmax', 'eps-greedy'
ONLY_USE_VALID_ACTIONS = False  # Enable/disable filtering of invalid actions.
# --------------------------------------------

fname = f"{FILENAME}.pt"  # Adding the file extension.


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Save the actions.
    self.actions = ACTIONS

    # Assign the decision strategy.
    self.act_strategy = ACT_STRATEGY

    # Incremental PCA for dimensionality reduction of game state.
    n_comp = 100
    self.dr_override = True  # if True: Use only manual feature extraction.

    self.model_is_fitted = False

    # Setting up the full model.
    if os.path.isfile(fname) and not self.train:
        self.logger.info("Loading model from saved state.")
        with open(fname, "rb") as file:
            self.model, self.dr_model = pickle.load(file)
        self.model_is_fitted = True
        if self.dr_model is not None:
            self.dr_model_is_fitted = True
        else:
            self.dr_model_is_fitted = False

    elif self.train:
        self.logger.info("Setting up model from scratch.")
        self.model = CustomRegressor(SGDRegressor(alpha=0.0001, warm_start=True))
        if not self.dr_override:
            self.dr_model = IncrementalPCA(n_components=n_comp)
        else:
            self.dr_model = None
        self.model_is_fitted = False
        self.dr_model_is_fitted = False
    else:
        raise ValueError(f"Could not locate saved model {fname}")

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # --------- (1) Optionally, only allow valid actions: -----------------
    # Switch to enable/disable filter of valid actions.
    if ONLY_USE_VALID_ACTIONS:
        mask, valid_actions = get_valid_actions(game_state, filter_level='full')
    else:
        mask, valid_actions = np.ones(len(ACTIONS)) == 1, ACTIONS

    # --------- (2a) Softmax decision strategy: ---------------
    if self.act_strategy == 'softmax':
        # Softmax temperature. During training, we anneal the temperature. In
        # game mode, we use a predefined (optimal) temperature. Limiting cases:
        # tau -> 0 : a = argmax Q(s,a) | tau -> +inf : uniform prob dist P(a).
        if self.train:
            tau = self.tau
        else:
            tau = 0.1
        if self.model_is_fitted:
            self.logger.debug("Choosing action from softmax distribution.")
            # Q-values for the current state.
            q_values = self.model.predict(transform(self, game_state))[0][mask]
            # Normalization for numerical stability.
            qtau = q_values / tau - np.max(q_values / tau)
            # Probabilities from Softmax function.
            p = np.exp(qtau) / np.sum(np.exp(qtau))
        else:
            # Uniformly random action when Q not yet initialized.
            self.logger.debug("Choosing action uniformly at random.")
            p = np.ones(len(valid_actions)) / len(valid_actions)
        # Pick choice from valid actions with the given probabilities.
        return np.random.choice(valid_actions, p=p)

    # --------- (2b) Epsilon-Greedy decision strategy: --------
    elif self.act_strategy == 'eps-greedy':
        if self.train:
            random_prob = self.epsilon
        else:
            random_prob = 0.01

        if random.random() < random_prob or not self.model_is_fitted:
            self.logger.debug("Choosing action uniformly at random.")
            execute_action = np.random.choice(valid_actions)
        else:
            self.logger.debug("Choosing action with highest q_value.")
            q_values = self.model.predict(transform(self, game_state))[0][mask]
            execute_action = valid_actions[np.argmax(q_values)]
        return execute_action
    else:
        raise ValueError(f"Unknown act_strategy {self.act_strategy}")


def transform(self, game_state: dict) -> np.array:
    """
    Feature extraction from the game state dictionary. Wrapper that toggles
    between automatic and manual feature extraction.
    """
    # This is the dict before the game begins and after it ends.
    if game_state is None:
        return None
    if self.dr_model_is_fitted and not self.dr_override:
        # Automatic dimensionality reduction.
        return self.dr_model.transform(state_to_features(game_state))
    else:
        # Hand crafted feature extraction function.
        return state_to_features(game_state)

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state dictionary to a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # ---- INFORMATION EXTRACTION ----
    # Getting all useful information from the game state dictionary.
    _, _, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = game_state['others']
    explosion_map = game_state['explosion_map']

    # exclude wall and crates
    free_space = arena == 0
    # exclude lethals bomb areas(including bombs) and explosions
    lethal_area = get_lethal_area(arena, bombs, explosion_map)
    for la in lethal_area:
        free_space[la[0], la[1]] = False
    # exclude others
    for o in [xy for (n, s, b, xy) in others]:
        free_space[o] = False

    # --------------------------------------------------------------------------
    # ---- LETHAL LEVEL ----
    # Int value indicator telling the level of lethal situation, 0 meas no danger, 1-4 means different levels
    lethal_status = False
    lethal = is_lethal(lethal_area, x, y)
    if lethal != 0:
        lethal_status = True

    # ---- ESCAPE ----
    # Direction towards the closest escape from imminent danger.
    escape_direction = escape_act_upgrade(lethal_area,x, y, arena, bombs, others, explosion_map)

    # ---- OTHERS ----
    # Direction towards the best offensive tile against other agents.
    others_direction, others_reached = others_act_upgrade(lethal_area,free_space, x, y, arena, bombs, others, bombs_left,explosion_map)
    opponent_acquired = 0
    if others_reached and not lethal_status:
        opponent_acquired = 1

    # ---- COINS ----
    # Direction towards the closest reachable coin.
    coin_direction = coins_act_updgrade(lethal_area,free_space, x, y, coins, arena, bombs, explosion_map)

    # ---- CRATES ----
    # Direction towards the best offensive tile for destroying crates.
    crates_direction, crates_reached = crates_act_upgrade(lethal_area,free_space, x, y, arena, bombs, bombs_left,
                                                          explosion_map)
    crates_acquired = int(crates_reached and not lethal_status)

    # ---- TARGETS ----
    # if have targets: coins, crates, others
    have_target = 1
    if len(coins) == 0 and len(others) == 0:
        if 1 not in arena:
            have_target = 0

    # target_acquired = int((others_reached or (crates_reached and all(others_direction == (0,0))))
    #                      and bombs_left and not lethal_status)
    # ------------------------------------------------------------------------------------------------------------------------
    # [LETHAL LEVEL, ESCAPE        ,  OTHERS                       ,  COINS                    ,CRATES                           ,  HAVE TARGETS]
    # [lethal level, escape_action, other_action, opponent_acquired , coin_action, crates_direction, crates_acquired,  have_target]
    # [     0      ,       1      ,      2      ,        3          ,      4     ,      5      ,         6       ,       7        ]

    features = np.concatenate((lethal,
                               escape_direction,
                               others_direction,
                               coin_direction,
                               crates_direction,
                               have_target), axis=None)
    return features.reshape(1, -1)

def has_object(x: int, y: int, arena: np.array, object: str) -> bool:
    """
    Check if tile at position (x,y) is of the specified type.
    """
    if object == 'crate':
        return arena[x, y] == 1
    elif object == 'free':
        return arena[x, y] == 0
    elif object == 'wall':
        return arena[x, y] == -1
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

def get_lethal_area(arena, bombs, bomb_map):
    # record lethals: bombs area and explosion
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    lethal_area = []
    if bombs:
        for (bx, by), _ in bombs:
            lethal_area.append((bx, by))
            for direction in directions:
                ix, iy = bx, by
                ix, iy = increment_position(ix, iy, direction)
                while (not has_object(ix, iy, arena, 'wall') and
                       abs(ix - bx) <= 3 and abs(iy - by) <= 3):
                    lethal_area.append((ix, iy))
                    ix, iy = increment_position(ix, iy, direction)
    for i in range(np.shape(bomb_map)[0]):
        for j in range(np.shape(bomb_map)[1]):
            if bomb_map[i, j] == 1:
                lethal_area.append((i, j))

    return np.array(lethal_area)

def is_lethal(lethal_area, x: int, y: int) ->int:

    for i in lethal_area:
        if i[0] == x and i[1] == y:
            return 1
    return 0

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

def escape_act_upgrade(lethal_area,x: int, y: int, arena: np.array, bombs: list, others: list,
                       explosion_map) -> int:
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
    # get targets
    targets = []

    for i in range(np.shape(arena)[0]):
        for j in range(np.shape(arena)[1]):
            # if not wall
            if arena[i, j] != -1:
                lethal = is_lethal(lethal_area, i, j)
                if lethal == 0:
                    targets.append((i, j))

    d = look_for_targets(free_space, (x, y), targets)
    if d == (x, y - 1):
        return UP
    elif d == (x, y + 1):
        return DOWN
    elif d == (x - 1, y):
        return LEFT
    elif d == (x + 1, y):
        return RIGHT
    elif d == (x, y):
        lethal = is_lethal(lethal_area,x, y)
        if lethal == 0:
            return WAIT
        else:
            return BOMB
    else:
        return WAIT

def crates_act_upgrade(lethal_area,free_space, x: int, y: int, arena: np.array, bombs: list,
                       bombs_left: bool, bomb_map) -> (int, bool):
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
            if arena[i][j] == 1:
                lethal = is_lethal(lethal_area,i, j)
                if lethal == 0:
                    targets.append((i, j))
    d = look_for_targets(free_space, (x, y), targets)
    reach_crase = False
    if d == (x, y - 1):
        return UP, reach_crase
    elif d == (x, y + 1):
        return DOWN, reach_crase
    elif d == (x - 1, y):
        return LEFT, reach_crase
    elif d == (x + 1, y):
        return RIGHT, reach_crase
    else:
        if arena[x - 1][y] == 1 or arena[x + 1][y] == 1 or \
                arena[x][y - 1] == 1 or arena[x][y + 1] == 1:
            if bombs_left:
                reach_crase = True
                return BOMB, reach_crase

        return WAIT, reach_crase

def coins_act_updgrade(lethal_area,free_space, x: int, y: int, coins: list, arena: np.array,
                       bombs: list, bomb_map) -> int:
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
        lethal = is_lethal(lethal_area, c[0], c[1])
        if lethal == 0:
            targets.append(c)
    d = look_for_targets(free_space, (x, y), targets)
    if d == (x, y - 1):
        return UP
    elif d == (x, y + 1):
        return DOWN
    elif d == (x - 1, y):
        return LEFT
    elif d == (x + 1, y):
        return RIGHT
    else: return WAIT


def others_act_upgrade(lethal_area,free_space, x: int, y: int, arena: np.array, bombs: list,
                       others: list, bombs_left: bool, bomb_map) -> (int, bool):
    """
    version 2:
    return different actions of looking for opponents, the actions are noted as different value
    No consideration of escaping lethals
    """
    # Take a step towards the most immediately interesting target
    # opponents xy
    targets = []

    for _, _, b, xy in others:
        lethal = is_lethal(lethal_area, xy[0], xy[1])
        # get opponents without aggressive power
        if lethal == 0:
            targets.append(xy)

    d = look_for_targets(free_space, (x, y), targets)
    near_opponent = False
    reach_opponent = False
    if d == (x, y - 1):
        return UP, reach_opponent
    elif d == (x, y + 1):
        return DOWN, reach_opponent
    elif d == (x - 1, y):
        return LEFT, reach_opponent
    elif d == (x + 1, y):
        return RIGHT, reach_opponent
    elif d == (x,y):
        for o in [xy for (n, s, b, xy) in others]:
            if math.sqrt(((o[0] - d[0]) ** 2) + ((o[1] - d[1]) ** 2)) == 1:
                near_opponent = True
                break
        if bombs_left and near_opponent:
            reach_opponent = True
            # logger.debug(f'ReachOPP {reach_opponent}')
            return BOMB, reach_opponent

    return WAIT, reach_opponent

'''
        if arena[x - 1][y] == 1 or arena[x + 1][y] == 1 or arena[x][y - 1] == 1 or arena[x][y + 1] == 1:
            if bombs_left:
                # logger.debug(f'ReachOPP {reach_opponent}')
                return BOMB, reach_opponent
            else: return WAIT, reach_opponent
'''

def get_valid_actions(lethal_area,game_state: dict, filter_level: str = 'basic'):
    """
    Given the gamestate, check which actions are valid. Has two filtering levels,
    'basic' where only purely invalid moves are disallowed and 'full' where also
    bad moves (bombing at inescapable tile, moving into lethal region) are 
    forbidden.
    Avoiding Suicide
        Not going to positions that are on-explosion on the next step.
        Not going to doomed positions, i.e., positions where if the agent were
            to go there the agent would have no way to escape.
        For any bomb, doomed positions can be computed by referring to its blast range,
            and life, together with the local terrain.
    Bomb Placement
        Not place bombs when the agent’s position is covered by the blast of
            any previously placed bomb.
    :param game_state:  A dictionary describing the current game board.
    :param filter_level: Either 'basic' or 'full'
    :return: mask which ACTIONS are executable
             list of VALID_ACTIONS
    """
    # aggressive_play = True # Allow agent to drop bombs.

    # Gather information about the game state
    step = game_state['step']
    _, _, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
    bombs = [xy for (xy, t) in game_state['bombs']]
    others = [xy for (n, s, b, xy) in game_state['others']]
    bomb_map = game_state['explosion_map']

    # Check for valid actions.
    #            [    'UP',  'RIGHT',   'DOWN',   'LEFT', 'WAIT']
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]

    # Initialization.
    valid_actions = []
    mask = np.zeros(len(ACTIONS))
    allow_bombing = True
    if (step == 1):
        allow_bombing = False
    lethal_lev = np.zeros(len(directions))

    # Check the filtering level.
    if filter_level == 'full':
        # Check lethal level in all directions.
        for i, (ix, iy) in enumerate(directions):
            if not has_object(ix, iy, arena, 'wall'):
                lethal_lev[i] = int(is_lethal(lethal_area, ix, iy))
            else:
                lethal_lev[i] = -1
        # Verify that there is at least one non-lethal tile in the surrounding.
        if not any(lethal_lev == 0):
            # No non-lethal tile detected, we can only disallow waiting.
            lethal_lev = np.zeros(len(directions))
            lethal_lev[-1] = 1

        # Check escape status on the current tile.
        # if not is_escapable(x, y, arena):
        #    allow_bombing = False

    elif filter_level == 'basic':
        # Could to other things here.
        pass
    else:
        raise ValueError(f"Invalid option filter_level={filter_level}.")

    # Movement:
    for i, d in enumerate(directions):
        if (arena[d] == 0 and  # Is a free tile
                bomb_map[d] < 1 and  # No ongoing explosion
                not d in others and  # Not occupied by other player
                not d in bombs and  # No bomb placed
                lethal_lev[i] == 0):  # Is non-lethal.
            if (bomb_map[d] == 0):
                valid_actions.append(ACTIONS[i])
                mask[i] = 1  # Append the valid action.
            else:
                if (i != 4):
                    valid_actions.append(ACTIONS[i])  # Append the valid action.
                    mask[i] = 1  # Binary mask

    # Bombing:
    if bombs_left and allow_bombing:
        valid_actions.append(ACTIONS[-1])
        mask[-1] = 1

    mask = (mask == 1)  # Convert binary mask to boolean mask.
    valid_actions = np.array(valid_actions)  # Convert list to numpy array

    if len(valid_actions) == 0:
        # The list is empty, there are no valid actions. Return all actions as
        # to not break the code by returning an empty list.
        return np.ones(len(ACTIONS)) == 1, ACTIONS
    else:
        return mask, valid_actions

class CustomRegressor:
    def __init__(self, estimator):
        # Create one regressor for each action separately.
        self.reg_model = [clone(estimator) for i in range(len(ACTIONS))]

    def partial_fit(self, X, y):
        '''
        Fit each regressor individually on its set of data.

        Parameters:
        -----------
        X: list
            List of length len(ACTIONS), where each entry is a 2d array of
            shape=(n_samples, n_features) with feature data corresponding to the
            given regressor. While n_features must be the same for all arrays,
            n_samples can optionally be different in every array.
        y: list
            List of length len(ACTIONS), where each entry is an 1d array of
            shape=(n_samples,) corresponding to each regressor. Since each
            regressor is fully independent, n_samples need not be equal for
            every array in y, but must however match in size to the
            corresponding array in X mentioned above.
        
        Returns:
        --------
        Nothing.
        '''
        # For every action:
        for i in range(len(ACTIONS)):
            # Verify that we have data.
            if X[i] and y[i]:
                # Perform one epoch of SGD.
                self.reg_model[i].partial_fit(X[i], y[i])

    def predict(self, X, action_idx=None):
        '''
        Get predictions from all regressors on a set of samples. Can also return
        predictions by a single regressor.

        Parameters:
        -----------
        X: np.array shape=(n_samples, n_features)
            Feature matrix for the n_samples each with n_features as the number
            of dimensions.
        action_idx: int
            (Optional) if action_idx is specified, only get predictions from
            the chosen regressor.
        
        Returns:
        --------
        y_predict: np.array
            If action_idx is unspecified, return the predictions by all regressors
            for all samples in an array of shape=(n_samples, len(ACTIONS)). Else
            return predictions for the single specified regressor, in an array
            of shape=(n_samples,).
        '''
        if action_idx is None:
            y_predict = [self.reg_model[i].predict(X) for i in range(len(ACTIONS))]
            return np.vstack(y_predict).T  # shape=(n_samples, len(ACTIONS))
        else:
            return self.reg_model[action_idx].predict(X)  # shape=(n_samples,)
