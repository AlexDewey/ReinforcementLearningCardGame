import math
import random

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from Environments.BaseEnv.board import Board


def game_view(board):
    print("Needs to be rewritten!")


class KerduGym(py_environment.PyEnvironment):

    def __init__(self, bot_agency):
        # pass (1), attack(5), defend(20) = 26 -> 25
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=25, name='action')
        # boards (2), hand(65), opponent_num_cards(5) = 590
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(590,), dtype=np.int32, minimum=0, maximum=1, name='observation')

        self._episode_ended = False

        # Board for game, P1 is NN and P2 is ENV
        self.board = Board()

        # State needs to be our observation of shape=(590,)
        self.board.fill_hand(1)
        self.board.fill_hand(2)
        self._state = self.transcribe_state()

        self.players = ["P1", "P2"]
        self.playerPass = [True, True]
        self.card_in_play = False
        self.playerNum = 1

        # How aggressive the bot is (4 mostly passive to 0 aggressive)
        self.bot_agency = bot_agency

        # If games are being presented to screen, default false
        self.view = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        # Initializes the board and gives players cards
        self.board = Board()
        self.card_in_play = False
        self.playerNum = 1

        self.correct_actions = list()
        exercise = random.randrange(0, 4)
        if exercise == 0:
            # Optimal first row defence
            while len(self.board.p1_rows[0]) == 0:
                for group in range(0, 5):
                    # Create Board
                    if random.randrange(0, 2) == 1:
                        if group == 0:
                            card_value = random.randrange(0, 2)
                        elif group == 1:
                            card_value = random.randrange(2, 5)
                        elif group == 2:
                            card_value = random.randrange(5, 8)
                        elif group == 3:
                            card_value = random.randrange(8, 11)
                        else:
                            card_value = random.randrange(11, 13)
                        self.board.p1_rows[0].append(card_value)

            random.shuffle(self.board.p1_rows[0])

            # Create optimal hand
            for card in self.board.p1_rows[0]:
                self.board.p1_hand.append(card)

            # Add card action to correct actions "defend, card_used_idx, row, column"
            for index, card in enumerate(self.board.p1_rows[0]):
                self.correct_actions.append(["defend", self.board.p1_hand.index(card), 0, index])
        elif exercise == 1:
            # todo
            # Perfect Defence
            num_of_defences = random.randrange(0, 5)

        elif exercise == 2:
            # todo
            # Pass
        else:
            # todo
            # Assassinate
            num_of_combo = random.randrange(0, 5)

        self._state = self.transcribe_state()
        return ts.restart(self._state)

    def post_action_logic(self, action_used):
        if action_used[0] != "pass":
            self.playerPass[self.playerNum - 1] = False
        else:
            self.playerPass[self.playerNum - 1] = True

        if action_used[0] == "attack":
            if self.playerNum == 1:
                try:
                    self.board.attack_card(2, self.board.p1_hand[action_used[1]])
                except:
                    self.playerPass[self.playerNum - 1] = True
            else:
                self.board.attack_card(1, self.board.p2_hand[action_used[1]])

        if action_used[0] == "defend":
            if self.playerNum == 1:
                if len(self.board.p1_rows[action_used[2]]) > action_used[3]:
                    self.board.defend_card(1, action_used[1], action_used[2], action_used[3])
            else:
                if len(self.board.p2_rows[action_used[2]]) > action_used[3]:
                    self.board.defend_card(2, action_used[1], action_used[2], action_used[3])

        if self.playerNum == 2:
            self.playerNum = 1
        else:
            self.playerNum = self.playerNum + 1

    def interpret_action(self, action):
        action_used = ["pass"]
        if 0 <= action <= 19:
            defend_idx = math.floor((action + 4) / 4) - 1
            row = abs(defend_idx * 4 - action)

            if len(self.board.p1_rows[row]) == 0:
                return None
            elif defend_idx < len(self.board.p1_hand):
                # Finds best fit defence for the row and defending card selected
                # card value, idx
                best_fit = [-1, -1]
                defending_card = self.board.p1_hand[defend_idx]
                for index, card in enumerate(self.board.p1_rows[row]):
                    if defending_card >= card > best_fit[0]:
                        best_fit = [card, index]
                if best_fit[0] == -1:
                    return None
                else:
                    action_used = ["defend", defend_idx, row, best_fit[1]]
        elif 20 <= action <= 24:
            if action - 20 < len(self.board.p1_hand) and len(self.board.p1_hand) > 0:
                action_used = ["attack", action - 100]
        else:
            if not self.card_in_play and len(self.board.p1_hand) > 0:
                action_used = ["attack", 0]
            else:
                action_used = ["pass"]

        return action_used

    def transcribe_state(self):
        p1_board = np.zeros(260).reshape(-1, 1)
        for row_index, row in enumerate(self.board.p1_rows):
            for column_index, card_value in enumerate(row):
                p1_board[row_index + (4 * column_index) + (20 * card_value)] = 1
        p2_board = np.zeros(260).reshape(-1, 1)
        for row_index, row in enumerate(self.board.p2_rows):
            for column_index, card_value in enumerate(row):
                p2_board[row_index + (4 * column_index) + (20 * card_value)] = 1
        hand = np.zeros(65).reshape(-1, 1)
        for hand_index, card_value in enumerate(self.board.p1_hand):
            hand[hand_index + (5 * card_value)] = 1
        opponent_num_cards = np.zeros(5).reshape(-1, 1)
        opponent_num_cards[len(self.board.p2_hand) - 1] = 1

        _state = np.concatenate((p1_board, np.concatenate((p2_board, np.concatenate((hand, opponent_num_cards))))))
        _state = _state.astype('int32')
        _state = _state.reshape(590, )

        return _state

    def _view(self, view_bool):
        if view_bool:
            self.view = True
        else:
            self.view = False

    def _step(self, action):
        # todo

        if self._episode_ended:
            return self.reset()

        action_used = self.interpret_action(action)

        if action_used not in self.correct_actions:
            reward = -100
            return ts.transition(self._state, reward=reward, discount=1.0)
        else:
            # Enact the action that's used onto the board and return if the gym training is complete
            exercise_complete = self.enact_action(action_used)
            reward = 100
            if exercise_complete:
                return ts.termination(self._state, reward=reward)
            else:
                return ts.transition(self._state, reward=reward, discount=1.0)
