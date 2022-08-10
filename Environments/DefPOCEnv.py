import math
import random

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from Environments.BaseEnv.board import Board


def game_view(board):
    print("Needs to be rewritten!")


class DefencePOCGym(py_environment.PyEnvironment):

    def __init__(self):
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

        self.exercise = random.randrange(0, 3)

        if self.exercise == 0:
            # Optimal first row defence
            for num_possible_cards in range(5):
                # Create Board
                card_value = random.randrange(0, 13)
                while card_value in self.board.p1_rows[0]:
                    card_value = random.randrange(0, 13)
                self.board.p1_rows[0].append(card_value)

            random.shuffle(self.board.p1_rows[0])

            # Create hand that may be suboptimal for defence, but a clear answer is given
            for card in self.board.p1_rows[0]:
                max_value = card
                while random.randrange(0, 4) != 0:
                    if (max_value + 1) not in self.board.p1_rows[0]:
                        if max_value == 12 or (max_value + 1) in self.board.p1_rows[0]:
                            continue
                        else:
                            max_value += 1
                    else:
                        continue
                self.board.p1_hand.append(max_value)

            random.shuffle(self.board.p1_hand)
        elif self.exercise == 1:
            # Perfect Defence
            # While the board is empty
            for num_possible_cards in range(5):
                card_value = random.randrange(0, 13)
                if card_value <= 1:
                    self.board.p1_rows[0].append(card_value)
                elif 2 <= card_value <= 8:
                    self.board.p1_rows[1].append(card_value)
                elif 9 <= card_value <= 11:
                    self.board.p1_rows[2].append(card_value)
                elif card_value == 12:
                    self.board.p1_rows[3].append(card_value)
                self.board.p1_hand.append(card_value)

            random.shuffle(self.board.p1_rows[0])
            random.shuffle(self.board.p1_rows[1])
            random.shuffle(self.board.p1_rows[2])
            random.shuffle(self.board.p1_rows[3])
            random.shuffle(self.board.p1_hand)
        elif self.exercise == 2:
            # Pass
            for num_possible_cards in range(5):
                card_value = random.randrange(0, 13)
                self.board.p1_hand.append(card_value)

            while len(self.board.p2_rows[0]) == 0:
                for num_possible_card in range(5):
                    # Create Board
                    if random.randrange(0, 2) == 1:
                        card_value = random.randrange(0, 13)
                        self.board.p2_rows[0].append(card_value)

            self.card_in_play = True

        self._state = self.transcribe_state()
        return ts.restart(self._state)

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
                    action_used = ["defend", defend_idx, row.item(), best_fit[1]]
        elif 20 <= action <= 24:
            if action - 20 < len(self.board.p1_hand) and len(self.board.p1_hand) > 0:
                action_used = ["attack", action - 20]
        else:
            if not self.card_in_play and len(self.board.p1_hand) > 0:
                action_used = ["attack", 0]
            else:
                action_used = "pass"

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
            if hand_index + (5 * card_value) >= 65:
                print("wat")
                hand[99] = 1
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

    def enact_action(self, action_used):
        if self.exercise == 0:
            # Optimal First Defence
            # self.correct_actions.append(["defend", self.board.p1_hand.index(card), 0, index])
            del self.board.p1_hand[action_used[1]]
            del self.board.p1_rows[0][action_used[3]]
            if len(self.board.p1_hand) == 0:
                return True
            else:
                return False
        elif self.exercise == 1:
            # Perfect Defence
            # "defend, card_used_idx, row, column"
            del self.board.p1_hand[action_used[1]]
            del self.board.p1_rows[action_used[2]][action_used[3]]
            if len(self.board.p1_hand) == 0:
                return True
            else:
                return False
        elif self.exercise == 2:
            return True

    def _step(self, action):

        if self._episode_ended or (len(self.board.p1_hand) == 0 and self.exercise != 2):
            return self.reset()

        self.correct_actions = list()

        if self.exercise == 0:
            if len(self.board.p1_rows[0]) == 1:
                self.correct_actions.append(["defend", 0, 0, 0])
            else:
                sorted_indicies_row = np.argsort(self.board.p1_rows[0]).tolist()
                sorted_indicies_hand = np.argsort(self.board.p1_hand).tolist()
                for index in range(len(sorted_indicies_row)):
                    self.correct_actions.append(["defend", self.board.p1_hand.index(self.board.p1_hand[sorted_indicies_hand[index]]), 0, sorted_indicies_row[index]])
        elif self.exercise == 1:
            for card in self.board.p1_hand:
                for row_idx, row in enumerate(self.board.p1_rows):
                    if len(row) != 0:
                        for column_idx, column in enumerate(row):
                            if column == card:
                                self.correct_actions.append(["defend", self.board.p1_hand.index(card), row_idx, column_idx])
        elif self.exercise == 2:
            self.correct_actions = ["pass"]

        action_used = self.interpret_action(action)

        if action_used not in self.correct_actions:
            reward = -100
            # print("Exercise: " + str(self.exercise))
            # print("Hand: " + str(self.board.p1_hand))
            # print("Board: " + str(self.board.p1_rows))
            # print("Correct Actions: " + str(self.correct_actions))
            # print("Action used: " + str(action_used))
            # print("======================")
            return ts.transition(self._state, reward=reward, discount=1.0)
        else:
            # Enact the action that's used onto the board and return if the gym training is complete
            exercise_complete = self.enact_action(action_used)
            # Update state for new return
            self._state = self.transcribe_state()
            reward = 100
            if exercise_complete:
                return ts.termination(self._state, reward=reward)
            else:
                return ts.transition(self._state, reward=reward, discount=1.0)
