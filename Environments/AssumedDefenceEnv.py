import math

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from Environments.BaseEnv.board import Board


def game_view(board):
    print("==NN Hand===")
    for card in board.p1_hand:
        print(card + 2, end=" ")
    print(" ")
    print("==NN Board==")
    for row in board.p1_rows:
        print("-", end=" ")
        for card in row:
            print(card + 2, end=" ")
        print(" ")
    print("============")
    print("==ENV Hand===")
    for card in board.p2_hand:
        print(card + 2, end=" ")
    print(" ")
    print("==ENV Board==")
    for row in board.p2_rows:
        print("-", end=" ")
        for card in row:
            print(card + 2, end=" ")
        print(" ")
    print("============")
    print("\n\n\n")


class KerduGameEnv(py_environment.PyEnvironment):

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
        self.playerPass = [True, True]
        self.card_in_play = False
        self.playerNum = 1
        self.pre_action_logic()
        self.board.fill_hand(1)
        self.board.fill_hand(2)
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

    def pre_action_logic(self):
        # If both players passed, draw cards. Automatically the case at the start of the game
        if False not in self.playerPass:
            # End game if cards in first row
            if len(self.board.p1_rows[0]) != 0 or len(self.board.p2_rows[0]) != 0:
                self.board.gameOver = True
                self._episode_ended = True
            else:
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
                break
        for row in self.board.p2_rows:
            if len(row) != 0:
                self.card_in_play = True
                break

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
                action_used = ["attack", action - 20]
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

        if self._episode_ended:
            return self.reset()

        # Completing action
        action_used = self.interpret_action(action)

        if action_used is None:
            if self.card_in_play:
                action_used = ["pass"]
            else:
                action_used = ["attack", 0]

        if self.view:
            print("NN Action used: " + str(action_used))
            game_view(self.board)

        # Changing board based on action
        self.post_action_logic(action_used)

        self.pre_action_logic()

        if self.card_in_play:
            action_used = ["pass"]
        else:
            action_used = ["attack", 0]

        # If immediate threat, defeat
        defence_found = False
        if len(self.board.p2_rows[0]) > 0:
            for column_index, attacking_card in enumerate(self.board.p2_rows[0]):
                if defence_found:
                    break
                for card_index, card_in_hand in enumerate(self.board.p2_hand):
                    if card_in_hand >= attacking_card:
                        action_used = ["defend", card_index, 0, column_index]
                        defence_found = True
                        break
        elif len(self.board.p2_hand) > self.bot_agency:
            min_index = [0, self.board.p2_hand[0]]
            for hand_index, card in enumerate(self.board.p2_hand):
                if card < min_index[1]:
                    min_index = [hand_index, card]
            action_used = ["attack", min_index[0]]
        else:  # ... otherwise pass
            action_used = ["pass"]

        if self.view:
            print("Computer Action:" + str(action_used))
            game_view(self.board)

        self.post_action_logic(action_used)

        self.pre_action_logic()

        self._state = self.transcribe_state()

        if self._episode_ended is False:
            reward = 1

            return ts.transition(self._state, reward=reward, discount=1.0)
        else:
            if len(self.board.p1_rows[0]) != 0 and len(self.board.p2_rows[0]) != 0:
                reward = 10
            elif len(self.board.p2_rows[0]) != 0:
                reward = 100
            else:
                reward = -100

            return ts.termination(self._state, reward=reward)
