import math

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from BaseEnv.board import Board


class KerduGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        # pass (1), attack(5), defend(100) = 106
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(106,), dtype=np.int32, minimum=0, maximum=1, name='action')
        # boards (2), hand(65), opponent_num_cards(5) = 590
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(590,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

        # Board for game, P1 is NN and P2 is ENV
        self.board = Board()
        self.players = ["NN", "ENV"]
        self.playerPass = [True, True]
        self.card_in_play = False
        self.playerNum = 1

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        # Initializes the board and gives players cards
        self.board = Board()
        self.playerPass = [True, True]
        self.card_in_play = False
        self.playerNum = 1
        self.pre_action_logic()
        return ts.restart(np.array([self._state], dtype=np.int32))

    def post_action_logic(self, action_used):
        if action_used != "pass":
            self.playerPass[self.playerNum - 1] = False
        else:
            self.playerPass[self.playerNum - 1] = True

        if action_used[0] == "attack":
            if self.playerNum == 1:
                self.board.attack_card(2, self.board.p1_hand[action_used[1]])
            else:
                self.board.attack_card(1, self.board.p2_hand[action_used[1]])

        if action_used[0] == "defend":
            if self.playerNum == 1:
                self.board.defend_card(1, action_used[1], action_used[2], action_used[3])
            else:
                self.board.defend_card(2, action_used[1], action_used[2], action_used[3])

        if self.playerNum == 2:
            self.playerNum = 1
        else:
            self.playerNum = self.playerNum + 1

    def pre_action_logic(self):
        # If both players passed, draw cards. Automatically the case at the start of the game
        if False not in self.playerPass:
            # End game if cards in first row
            if len(self.board.p1_rows[0]) != 0 or len(self.board.p2_rows[0]) != 0:
                self.board.gameOver = True
                self._episode_ended = True
            # Move all cards up a row
            for index in range(1, 4):
                self.board.p1_rows[index - 1] = self.board.p1_rows[index]
                self.board.p2_rows[index - 1] = self.board.p2_rows[index]
            self.board.p1_rows[3] = []
            self.board.p2_rows[3] = []
            # Refill hands
            for index in range(0, len(self.players)):
                self.board.fill_hand(index + 1)
                self.playerPass[index] = False

        # If there's a card on the board, the player can pass, otherwise no
        self.card_in_play = False
        for row in self.board.p1_rows:
            if len(row) != 0:
                self.card_in_play = True
        for row in self.board.p2_rows:
            if len(row) != 0:
                self.card_in_play = True


    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        if self.card_in_play:
            action_used = ["pass"]
        else:
            action_used = ["attack", 0]

        if 0 <= action <= 99:
            # defend
            hand = math.ceil((action + 1) / 20)
            board_pos = (hand * 20 - action)
            row = math.ceil(board_pos / 5) - 1
            column = abs(row * 5 - board_pos) - 1

            if column > len(self.board.p1_rows[row]):
                action_used = action_used
            elif (hand - 1) < len(self.board.p1_hand):
                if column < len(self.board.p1_rows[row]):
                    if self.board.p1_hand[hand - 1] <= self.board.p1_rows[row][column]:
                        action_used = action_used
                    else:
                        action_used = ["defend", hand, row, column]
        elif 100 <= action <= 105:
            if action - 99 > len(self.board.p1_hand):
                action_used = action_used
            else:
                action_used = ["attack", action - 100]
        elif action == 106:
            if not self.card_in_play:
                action_used = ["attack", 0]
            else:
                action_used = ["pass"]

        self.post_action_logic(action_used)

        self.pre_action_logic()

        if self.card_in_play:
            action_used = ["pass"]
        else:
            action_used = ["attack", 0]

        # Environment action being found
        defence_found = False
        action_used = ["pass"]

        # If immediate threat, defeat
        if len(self.board.p2_rows[0]) > 0:
            for column_index, attacking_card in enumerate(self.board.p2_rows[0]):
                if defence_found:
                    break
                for card_index, card_in_hand in enumerate(self.board.p2_hand):
                    if card_in_hand > attacking_card:
                        action_used = ["defend", card_index, 0, column_index]
                        defence_found = True
                        break
        elif len(self.board.p2_hand) > 0:
            min_index = [0, self.board.p2_hand[0]]
            for hand_index, card in enumerate(self.board.p2_hand):
                if card < min_index[1]:
                    min_index = [hand_index, card]
            action_used = ["attack", min_index[0]]
        else:
            action_used = ["pass"]

        self.post_action_logic(action_used)

        self.pre_action_logic()

        # if action == 1:
        #     self._episode_ended = True
        # elif action == 0:
        #     new_card = np.random.randint(1, 11)
        #     self._state += new_card
        # else:
        #     raise ValueError('action should be 0 or 1')
        #
        # if self._episode_ended or self._state >= 21:
        #     reward = self._state - 21 if self._state <= 21 else -21
        #     return ts.termination(np.array([self._state], dtype=np.int32), reward)
        # else:
        #     return ts.transition(np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
