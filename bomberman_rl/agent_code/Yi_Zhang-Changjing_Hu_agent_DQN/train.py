import os
import pickle
import random
from collections import namedtuple, deque
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-GUI backend, needed for plotting in non-main thread.
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1.5

from sklearn.base import clone
import settings as s
import events as e
from .callbacks import (state_to_features,
                        fname,
                        FILENAME)

# Transition tuple. (s, a, s', r)
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

# Feature nr.
LETHAL_LEV = 0
ESC_ACT = 1
OTHER_ACT = 2
OTHER_ACQ = 3
COIN_ACT = 4
CRATES_ACT = 5
CRATES_ACQ = 6
HAVE_TARGET = 7
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
#Actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
WAIT = 4
BOMB = 5

# ------------------------ HYPER-PARAMETERS -----------------------------------
# General hyper-parameters:
TRANSITION_HISTORY_SIZE = 1000  # Keep only ... last transitions.
BATCH_SIZE              = 500   # Size of batch in TD-learning.
TRAIN_FREQ              = 1     # Train model every ... game.

# N-step TD Q-learning:
GAMMA   = 0.8  # Discount factor.
N_STEPS = 4    # Number of steps to consider real, observed rewards.

# Prioritized experience replay:
PRIO_EXP_REPLAY   = True    # Toggle on/off.
PRIO_EXP_FRACTION = 0.25    # Fraction of BATCH_SIZE to keep.

# Dimensionality reduction from learning experience.
DR_FREQ           = 1000    # Play ... games before we fit DR.
DR_EPOCHS         = 30      # Nr. of epochs in mini-batch learning.
DR_MINIBATCH_SIZE = 10000   # Nr. of states in each mini-batch.
DR_HISTORY_SIZE   = 50000   # Keep the ... last states for DR learning.

# Epsilon-Greedy: (0 < epsilon < 1)
EXPLORATION_INIT  = 1.0
EXPLORATION_MIN   = 0.005
EXPLORATION_DECAY = 0.99995

# Softmax: (0 < tau < infty)
TAU_INIT  = 15
TAU_MIN   = 0.5
TAU_DECAY = 0.9999

# Auxiliary:
PLOT_FREQ = 25
# -----------------------------------------------------------------------------

# File name of historical training record used for plotting.
FNAME_DATA = f"{FILENAME}_data.pt"

# Custom events:
CLOSER_TO_ESCAPE = "CLOSER_TO_ESCAPE"
FURTHER_FROM_ESCAPE = "FURTHER_FROM_ESCAPE"

WAITED_NECESSARILY = "WAITED_NECESSARILY"
WAITED_UNNECESSARILY = "WAITED_UNNECESSARILY"

CLOSER_TO_OTHERS = "CLOSER_TO_OTHERS"
FURTHER_FROM_OTHERS = "FURTHER_FROM_OTHERS"
BOMBED_OTHERS = "BOMBED_OTHERS"
MISSED_OTHERS = "MISSED_OTHERS"
WAITED_OTHERS = "WAITED_OTHERS"
UNWAITED_OTHERS = "UNWAITED_OTHERS"

BOMBED_OTHERS_OBSTACLES = "BOMBED_OTHERS_OBSTACLES"
MISSED_OTHERS_OBSTACLES = "MISSED_OTHERS_OBSTACLES"
BOMBED_COIN_OBSTACLES = "BOMBED_COIN_OBSTACLES"
MISSED_COIN_OBSTACLES = "MISSED_COIN_OBSTACLES"
BOMBED_CRATE_OBSTACLES = "BOMBED_CRATE_OBSTACLES"
MISSED_CRATE_OBSTACLES = "MISSED_CRATE_OBSTACLES"

BAD_BOMB = "BAD_BOMB"

CLOSER_TO_COIN = "CLOSER_TO_COIN"
FURTHER_FROM_COIN = "FURTHER_FROM_COIN"

CLOSER_TO_CRATE = "CLOSER_TO_CRATE"
FURTHER_FROM_CRATE = "FURTHER_FROM_CRATE"
BOMBED_CRATE = "BOMBED_CRATE"
MISSED_CRATE = "MISSED_CRATE"
BOMBED_GOAL = "BOMBED_GOAL"
MISSED_GOAL = "MISSED_GOAL"
UNWAITED_CRATE = "UNWAITED_CRATE"
WAITED_CRATE = "WAITED_CRATE"


SURVIVED_STEP = "SURVIVED_STEP"
PASS_STEP = "PASS_STEP"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Ques to store the transition tuples and coordinate history of agent.
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # For evaluation of the training progress:
    # Check if historic data file exists, load it in or start from scratch.
    if os.path.isfile(FNAME_DATA):
        # Load historical training data.
        with open(FNAME_DATA, "rb") as file:
            self.historic_data = pickle.load(file)
        self.game_nr = max(self.historic_data['games']) + 1
    else:
        # Start a new historic record.
        self.historic_data = {
            'score'  : [],       # subplot 1
            'coins'  : [],       # subplot 2
            'crates' : [],       # subplot 3
            'enemies': [],       # subplot 4
            'games'  : []        # subplot 1,2,3,4,5 x-axis
        }
        self.game_nr = 1

    self.transition_length = 0
    # Initialization
    self.score_in_round    = 0
    self.collected_coins   = 0
    self.destroyed_crates  = 0
    self.killed_enemies    = 0
    self.bomb_loop_penalty = 0
    self.perform_export    = False

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    # ---------- (1) Add own events to hand out rewards: ----------
    # If the old state is not before the beginning of the game:
    if old_game_state:
        # Extract feature vector:
        state_old = state_to_features(old_game_state)
        #self.logger.debug(f'state_old:{state_old}')
        # Extract the lethal indicator from the old state.
        islethal_old = True
        if(state_old[0][LETHAL_LEV] == 0):
            islethal_old = False

        self.logger.debug(f'islethal_old:{islethal_old}')

        # different action for is lethal
        if islethal_old:
            # ---- WHEN IN LETHAL ----
            # When in lethal danger, we only care about escaping. Following the
            # escape direction is rewarded, anything else is penalized.
            escape_act_old = state_old[0][ESC_ACT]
            #self.logger.debug(f'IN DANGER! Recommand:{ACTIONS[escape_act_old]}')
            #self.logger.debug(f'Self action:{self_action}')
            #self.logger.debug(f'check :{check_action(escape_act_old, self_action)}')
            if check_action(escape_act_old, self_action):
                events.append(CLOSER_TO_ESCAPE)
            else:
                events.append(FURTHER_FROM_ESCAPE)
        else:
            # ---- WHEN IN NON-LETHAL ----
            # When not in lethal danger, we are less stressed to make the right
            # decision. Our order of priority is: others > coins > crates.
            #self.logger.debug(f'state_old {state_old}')
            # Extracting information from the old game state.
            others_act_old = state_old[0][OTHER_ACT]
            others_acq_old = state_old[0][OTHER_ACQ]
            self.logger.debug(f'Recommand others ACTION {ACTIONS[others_act_old]}')

            coins_act_old  = state_old[0][COIN_ACT]
            self.logger.debug(f'Recommand coin ACTION {ACTIONS[coins_act_old]}')

            crates_act_old = state_old[0][CRATES_ACT]
            crates_acq_old = state_old[0][CRATES_ACQ]
            self.logger.debug(f'Recommand crates ACTION {ACTIONS[crates_act_old]}')

            self.logger.debug(f'SELF ACTION {self_action}')

            have_targets = state_old[0][HAVE_TARGET]

            # others > coins > crates
            # bomb > move > wait

            # If we chose to bomb in the previous state:
            # check self action with actions for others
            if check_action(others_act_old,self_action):
                # action for others
                # Reward if we successfully bomb-laying the others, else penalize.
                if self_action == 'BOMB':
                    if others_acq_old == 1:
                        events.append(BOMBED_OTHERS)
                    else:
                        events.append(BOMBED_OTHERS_OBSTACLES)
                elif self_action == 'WAIT':
                    if others_acq_old == 1:
                        events.append(WAITED_OTHERS)
                    else:
                        events.append(CLOSER_TO_OTHERS)
                else:
                    events.append(CLOSER_TO_OTHERS)
            else:
                if self_action == 'BOMB':
                    events.append(BAD_BOMB)
                    if others_act_old == WAIT:
                        if others_acq_old == 1:
                            events.append(UNWAITED_OTHERS)
                        else:
                            events.append(FURTHER_FROM_OTHERS)
                    else:
                        events.append(FURTHER_FROM_OTHERS)
                else:
                    if others_act_old == BOMB:
                        if others_acq_old == 1:
                            events.append(MISSED_OTHERS)
                        else:
                            events.append(MISSED_OTHERS_OBSTACLES)
                    elif others_act_old == WAIT:
                        if others_acq_old == 1:
                            events.append(UNWAITED_OTHERS)
                        else:
                            events.append(FURTHER_FROM_OTHERS)
                    else:
                        events.append(FURTHER_FROM_OTHERS)

                # check self action with actions for coins
                if check_action(coins_act_old, self_action):
                    # action for coins
                    # reward if close to coin
                    if coins_act_old == BOMB:
                        events.append(BOMBED_COIN_OBSTACLES)
                    else:
                        events.append(CLOSER_TO_COIN)
                else:
                    if self_action == 'BOMB':
                        events.append(BAD_BOMB)
                        events.append(FURTHER_FROM_COIN)
                    else:
                        if coins_act_old == BOMB:
                            events.append(MISSED_COIN_OBSTACLES)

                    # check self action with actions for crates
                    if check_action(crates_act_old,self_action):
                        # action for crates
                        # reward if close to crate or lay bombs
                        if self_action == 'BOMB':
                            if crates_acq_old == 1:
                                events.append(BOMBED_CRATE)
                            else:
                                events.append(BOMBED_CRATE_OBSTACLES)
                        elif self_action == 'WAIT':
                            if crates_acq_old == 1:
                                events.append(WAITED_CRATE)
                            else:
                                events.append(CLOSER_TO_CRATE)
                        else:
                            events.append(CLOSER_TO_CRATE)
                    else:
                        if self_action == 'BOMB':
                            events.append(BAD_BOMB)
                            if crates_act_old == WAIT:
                                if crates_acq_old == 1:
                                    events.append(UNWAITED_CRATE)
                                else:
                                    events.append(FURTHER_FROM_CRATE)
                            else:
                                events.append(FURTHER_FROM_CRATE)
                        else:
                            if crates_act_old == BOMB:
                                if crates_acq_old == 1:
                                    events.append(MISSED_CRATE)
                                else:
                                    events.append(MISSED_CRATE_OBSTACLES)
                            elif crates_act_old == WAIT:
                                if crates_acq_old == 1:
                                    events.append(UNWAITED_CRATE)
                                else:
                                    events.append(FURTHER_FROM_CRATE)
                            else:
                                events.append(FURTHER_FROM_CRATE)

            if have_targets == 0:
                if self_action == 'WAIT':
                    events.append(WAITED_NECESSARILY)
                else: events.append(WAITED_UNNECESSARILY)

    # Reward for surviving (effectively a passive reward).
    if not 'GOT_KILLED' in events:
        events.append(SURVIVED_STEP)

    if old_game_state:
        self.transition_length = self.model.store_transition(state_to_features(old_game_state)[0], self_action, reward_from_events(self,events), state_to_features(new_game_state)[0])
    # RL take action and get next observation and reward

    # ---------- (4) For evaluation purposes: ----------
    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    if 'CRATE_DESTROYED' in events:
        self.destroyed_crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.killed_enemies += events.count('KILLED_OPPONENT')
    self.score_in_round += reward_from_events(self, events)


    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.transition_length = self.model.store_transition(state_to_features(last_game_state)[0], last_action,
                                                         reward_from_events(self, events),
                                                         state_to_features(None)[0])
    # RL take action and get next observation and reward
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.logger.debug(f'End round:{True}')

    # Logging
    # ---------- (6) Model export: ----------
    # Check if a full model export has been requested.
    if self.perform_export:
        export = self.model, self.dr_model
        with open(fname, "wb") as file:
            pickle.dump(export, file)
        self.perform_export = False # Reset export flag

    # ---------- (7) Performance evaluation: ----------
    # Get the numbers from the last round.
    if 'COIN_COLLECTED' in events:
        self.collected_coins += 1
    if 'CRATE_DESTROYED' in events:
        self.destroyed_crates += events.count('CRATE_DESTROYED')
    if 'KILLED_OPPONENT' in events:
        self.killed_enemies += events.count('KILLED_OPPONENT')
    self.score_in_round += reward_from_events(self, events)

    # Total score in this game.
    score = np.sum(self.score_in_round)
   
    # Append results to each specific list.
    self.historic_data['score'].append(score)
    self.historic_data['coins'].append(self.collected_coins)
    self.historic_data['crates'].append(self.destroyed_crates)
    self.historic_data['enemies'].append(self.killed_enemies)
    self.historic_data['games'].append(self.game_nr)
    #self.historic_data['exploration'].append(self.epsilon)


    # Store the historic record.
    with open(FNAME_DATA, "wb") as file:
        pickle.dump(self.historic_data, file)
    
    # Reset game score, coins collected and one up the game count.
    self.score_in_round   = 0
    self.collected_coins  = 0
    self.destroyed_crates = 0
    self.killed_enemies   = 0
    self.game_nr += 1
    
    # Plot training progress every n:th game.
    if self.game_nr % PLOT_FREQ == 0:
        # Incorporate the full training history.
        games_list = self.historic_data['games']
        score_list = self.historic_data['score']
        coins_list = self.historic_data['coins']
        crate_list = self.historic_data['crates']
        other_list = self.historic_data['enemies']
        #explr_list = self.historic_data['exploration']

        # Plotting
        fig, ax = plt.subplots(5, figsize=(7.2, 5.4), sharex=True)

        # Total score per game.
        ax[0].plot(games_list, score_list)
        ax[0].set_title('Total score per game')
        ax[0].set_ylabel('Score')

        # Collected coins per game.
        ax[1].plot(games_list, coins_list)
        ax[1].set_title('Collected coins per game')
        ax[1].set_ylabel('Coins')

        # Destroyed crates per game.
        ax[2].plot(games_list, crate_list)
        ax[2].set_title('Destroyed crates per game')
        ax[2].set_ylabel('Crates')

        # Eliminiated opponents per game
        ax[3].plot(games_list, other_list)
        ax[3].set_title('Eliminated opponents per game')
        ax[3].set_ylabel('Kills')
        ax[3].set_xlabel('Game Round')

        # Export the figure.
        fig.tight_layout()
        plt.savefig(f'TrainEval_{FILENAME}.pdf')
        plt.close('all')

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    # escape > kill > coin > crate
    
    # Base rewards:
    kill  = s.REWARD_KILL
    coin  = s.REWARD_COIN
    # crates density is 0.75, so has 132 crates, and there are 9 coins, so  9/132 is the reward
    crate = 0.1 * coin
    
    escape_movement    = 0.1 * kill
    bombing_others     = 0.2  * kill
    bombing_crates     = 0.1 * coin
    bad_bomb           = -4 * escape_movement
    bombed_goal        = 0.1 * kill

    waiting            = 0.1  * kill
    wait_others        = 0.5 * bombing_others
    wait_crates        = 0.5 * bombing_crates

    offensive_movement = 0.05 * kill
    coin_movement      = 0.1  * coin
    crate_movement     = 0.5 * crate

    others_obstacles   = 0.5  * bombing_others
    coins_obstacles    = 0.5 * coin
    crates_obstacles   = 0.5 * crate

    passive = 0
    surive = 0

    # Game reward dictionary:
    game_rewards = {
        # ---- CUSTOM EVENTS ----
        # escape movement
        CLOSER_TO_ESCAPE    : escape_movement,
        FURTHER_FROM_ESCAPE : -escape_movement,

        # bombing
        BOMBED_OTHERS : bombing_others,
        MISSED_OTHERS : -bombing_others, #-4*escape_movement,# Needed to prevent self-bomb-laying loops.
        BOMBED_CRATE :  bombing_crates,
        MISSED_CRATE : -bombing_crates,
        BAD_BOMB:       bad_bomb,
        BOMBED_GOAL:   bombed_goal,


        # waiting
        WAITED_NECESSARILY  : waiting,
        WAITED_UNNECESSARILY: -waiting,
        WAITED_OTHERS : wait_others,
        UNWAITED_OTHERS : -wait_others,
        WAITED_CRATE : wait_crates,
        UNWAITED_CRATE: -wait_crates,

        #obstacles
        BOMBED_OTHERS_OBSTACLES: others_obstacles,
        MISSED_OTHERS_OBSTACLES: -others_obstacles,
        BOMBED_COIN_OBSTACLES: coins_obstacles,
        MISSED_COIN_OBSTACLES: -coins_obstacles,
        BOMBED_CRATE_OBSTACLES: crates_obstacles,
        MISSED_CRATE_OBSTACLES: -crates_obstacles,

        # offensive movement
        CLOSER_TO_OTHERS    : offensive_movement,
        FURTHER_FROM_OTHERS : -offensive_movement,

        # coin movement
        CLOSER_TO_COIN      : coin_movement,
        FURTHER_FROM_COIN   : -coin_movement,
        
        # crate movement
        CLOSER_TO_CRATE     : crate_movement,
        FURTHER_FROM_CRATE  : -crate_movement,
        
        # passive
        SURVIVED_STEP       : surive,
        PASS_STEP           : passive,

        # ---- DEFAULT EVENTS ----
        # movement
        e.MOVED_LEFT         :  0,
        e.MOVED_RIGHT        :  0,
        e.MOVED_UP           :  0,
        e.MOVED_DOWN         :  0,
        e.WAITED             :  0,
        e.INVALID_ACTION     : -1,
        
        # bombing
        e.BOMB_DROPPED       : 0,
        e.BOMB_EXPLODED      : 0,

        # crates, coins
        e.CRATE_DESTROYED    : crate,
        e.COIN_FOUND         : 0,
        e.COIN_COLLECTED     : coin,

        # kills
        e.KILLED_OPPONENT    : kill,
        e.KILLED_SELF        : -kill,
        e.GOT_KILLED         : -kill,
        e.OPPONENT_ELIMINATED: 0,

        # passive
        e.SURVIVED_ROUND     : passive,
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def check_action(action: int, self_action: str) -> bool:
    """
    Check if a taken action was the wanted one.
    
    Parameters:
    -----------
    action: int
        action enum indicating the optimal direction to take.
    
    Returns:
    --------
    is_correct_action: bool
        True if the action taken was the correct one given the direction vector.
    """
    return ((action == DOWN and self_action == 'DOWN') or
            (action == RIGHT and self_action == 'RIGHT') or
            (action == UP and self_action == 'UP') or
            (action == LEFT and self_action == 'LEFT')or
            (action == WAIT and self_action == 'WAIT')or
            (action == BOMB and self_action == 'BOMB'))

def check_movement(action: int):
    if action != BOMB and action != WAIT:
        return True
    return False

