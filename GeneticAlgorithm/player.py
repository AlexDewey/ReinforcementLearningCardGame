import math
import numpy as np


class Player:

    def __init__(self, position, model):
        self.position = position
        self.model = model

    # player_can_pass is passed in as variable card_in_play, as if there's a card in play, a player can pass
    def get_player_action(self, board, playerNum, model, player_can_pass):
        # possible action datatypes: ["pass", ["attack", cardHandIndex], ["defend", row, cardTargeted]]

        # 4 rows [0-3], 5 columns [0-4], 13 values [0-12], 4 x 5 x 13 = 260
        self_board = np.zeros(260).reshape(-1, 1)
        if playerNum == 1:
            for row_index, row in enumerate(board.p1_rows):
                for column_index, card_value in enumerate(row):
                    self_board[row_index + (4 * column_index) + (20 * card_value)] = 1
        else:
            for row_index, row in enumerate(board.p2_rows):
                for column_index, card_value in enumerate(row):
                    self_board[row_index + (4 * column_index) + (20 * card_value)] = 1

        # 4 rows [0-3], 5 columns [0-4], 13 values [0-12], 4 x 5 x 13 = 260
        opponenet_board = np.zeros(260).reshape(-1, 1)
        if playerNum == 1:
            for row_index, row in enumerate(board.p2_rows):
                for column_index, card_value in enumerate(row):
                    opponenet_board[row_index + (4 * column_index) + (20 * card_value)] = 1
        else:
            for row_index, row in enumerate(board.p1_rows):
                for column_index, card_value in enumerate(row):
                    opponenet_board[row_index + (4 * column_index) + (20 * card_value)] = 1

        # 5 cards held [0-4], 13 values [0-12], 5 x 13 = 65
        hand = np.zeros(65).reshape(-1, 1)
        if playerNum == 1:
            for hand_index, card_value in enumerate(board.p1_hand):
                hand[hand_index + (5 * card_value)] = 1
        else:
            for hand_index, card_value in enumerate(board.p2_hand):
                hand[hand_index + (5 * card_value)] = 1

        # 5 cards held by opponent = 5 cards counted [0-4] = 5
        opponent_counted = np.zeros(5).reshape(-1, 1)
        if playerNum == 1:
            opponent_counted[len(board.p2_hand) - 1] = 1
        else:
            opponent_counted[len(board.p1_hand) - 1] = 1

        # 590 Total Length
        neural_input = np.concatenate((np.concatenate((np.concatenate((self_board, opponenet_board)), hand)), opponent_counted))
        neural_input = np.atleast_2d(neural_input)
        neural_input = neural_input.reshape(1, -1)

        output_prob = model.predict(neural_input)

        action_prio = np.argsort(output_prob[0])[::-1]

        if playerNum == 1:
            for action in action_prio:
                if 0 <= action <= 99:
                    hand = math.ceil((action + 1) / 20)
                    board_pos = (hand * 20 - action)
                    row = math.ceil(board_pos / 5) - 1
                    column = abs(row * 5 - board_pos) - 1

                    if column > len(board.p1_rows[row]):
                        continue
                    elif (hand - 1) < len(board.p1_hand):
                        if column < len(board.p1_rows[row]):
                            if board.p1_hand[hand - 1] <= board.p1_rows[row][column]:
                                continue
                            else:
                                return ["defend", row, column, hand - 1]
                        else:
                            continue
                    else:
                        continue
                if 100 <= action <= 104:
                    if action - 99 > len(board.p1_hand):
                        continue
                    else:
                        return ["attack", action - 100]
                if action == 105:
                    if not player_can_pass:
                        continue
                    else:
                        return "pass"

        if playerNum == 2:
            for action in action_prio:
                if 0 <= action <= 99:
                    hand = math.ceil((action + 1) / 20)
                    board_pos = (hand * 20 - action)
                    row = math.ceil(board_pos / 5) - 1
                    column = abs(row * 5 - board_pos) - 1

                    if column > len(board.p2_rows[row]):
                        continue
                    elif (hand - 1) < len(board.p2_hand):
                        if column < len(board.p2_rows[row]):
                            if board.p2_hand[hand - 1] <= board.p2_rows[row][column]:
                                continue
                            else:
                                return ["defend", row, column, hand - 1]
                        else:
                            continue
                    else:
                        continue
                if 100 <= action <= 104:
                    if action - 99 > len(board.p2_hand):
                        continue
                    else:
                        return ["attack", action - 100]
                if action == 105:
                    if not player_can_pass:
                        continue
                    else:
                        return "pass"

        return "pass"
