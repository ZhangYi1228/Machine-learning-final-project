import math
import os
import pickle
import random
from queue import Queue
from random import shuffle
import numpy as np

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import rmsprop_v2

from sklearn.base import clone

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
FILENAME = "Yi_Zhang-Changjing_Hu_agent_DQN"  # Base filename of model (excl. extensions).
ACT_STRATEGY = 'DQN'  # Options: 'DQN', 'DDQN'
# --------------------------------------------
FEATURE_NUM = 8

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

    self.model_is_fitted = False

    # Setting up the full model.
    if os.path.isfile(fname) and not self.train:
        self.logger.info("Loading model from saved state.")
        with open(fname, "rb") as file:
            self.model = pickle.load(file)
        self.model_is_fitted = True

    elif self.train:
        self.logger.info("Setting up model from scratch.")
        eval_model = Eval_Model(num_actions=len(self.actions))
        target_model = Target_Model(num_actions=len(self.actions))
        self.model = DeepQNetwork(len(self.actions), FEATURE_NUM, eval_model, target_model)
        self.model_is_fitted = False
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

    # --------- (1) DQN decision strategy: ---------------
    if self.act_strategy == 'DQN':
            self.logger.debug("Choosing action from max q value.")
            # Q-values for the current state.
            action = self.model.choose_action(state_to_features(game_state))
            return action
    else:
        raise ValueError(f"Unknown act_strategy {self.act_strategy}")

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
                               others_direction, opponent_acquired,
                               coin_direction,
                               crates_direction, crates_acquired,
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
    elif d == (x, y):
        reach_crase = True
        if (bombs_left):
            return BOMB, reach_crase
        else:
            return WAIT, reach_crase
    else:
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
    else:
        if arena[x - 1][y] == 1 or arena[x + 1][y] == 1 or arena[x][y - 1] == 1 or arena[x][y + 1] == 1:
            return BOMB  # , ecounter_cion
        else:
            return WAIT


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
    elif d == (x, y):
        for o in [xy for (n, s, b, xy) in others]:
            if math.sqrt(((o[0] - d[0]) ** 2) + ((o[1] - d[1]) ** 2)) == 1:
                reach_opponent = True
        if arena[x - 1][y] == 1 or arena[x + 1][y] == 1 or arena[x][y - 1] == 1 or arena[x][
            y + 1] == 1 or reach_opponent:
            if bombs_left:
                return BOMB, reach_opponent

    return WAIT, reach_opponent


class Eval_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network')
        self.layer1 = layers.Dense(10, activation='relu')
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits

class Target_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network_1')
        self.layer1 = layers.Dense(10, trainable=False, activation='relu')
        self.logits = layers.Dense(num_actions, trainable=False, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits

class DeepQNetwork:
    def __init__(self, n_actions, n_features, eval_model, target_model):

        self.params = {
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': 0.01,
            'reward_decay': 0.9,
            'e_greedy': 0.8,
            'replace_target_iter': 300,
            'memory_size': 1000,
            'batch_size': 500,
            'e_greedy_increment': None
        }

        # total learning step

        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.epsilon = 0 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
        self.memory = np.zeros((self.params['memory_size'], self.params['n_features'] * 2 + 2))

        self.eval_model = eval_model
        self.target_model = target_model

        self.eval_model.compile(
            optimizer = rmsprop_v2.RMSprop(lr=self.params['learning_rate']),
            loss='mse'
        )
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.params['memory_size']
        self.memory[index, :] = transition

        self.memory_counter += 1

        return self.memory_counter

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_model.predict(observation)
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.params['n_actions'])
        return action

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.params['memory_size']:
            sample_index = np.random.choice(self.params['memory_size'], size=self.params['batch_size'])
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.params['batch_size'])

        batch_memory = self.memory[sample_index, :]

        q_next = self.target_model.predict(batch_memory[:, -self.params['n_features']:])
        q_eval = self.eval_model.predict(batch_memory[:, :self.params['n_features']])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.params['batch_size'], dtype=np.int32)
        eval_act_index = batch_memory[:, self.params['n_features']].astype(int)
        reward = batch_memory[:, self.params['n_features'] + 1]

        q_target[batch_index, eval_act_index] = reward + self.params['reward_decay'] * np.max(q_next, axis=1)

        # check to replace target parameters
        if self.learn_step_counter % self.params['replace_target_iter'] == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('\ntarget_params_replaced\n')

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network

        self.cost = self.eval_model.train_on_batch(batch_memory[:, :self.params['n_features']], q_target)

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.params['e_greedy_increment'] if self.epsilon < self.params['e_greedy'] \
            else self.params['e_greedy']
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



