import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


class Player:

    def __init__(self, position, model):
        self.position = position
        self.model = model

    def get_player_action(self, board, playerNum, model):
        # actions: ["pass", ["attack", cardHandIndex], ["defend", row, cardTargeted]]

        if playerNum == 1:
            if len(board.p1_hand) == 5:
                canPass = False
            else:
                canPass = True
        else:
            if len(board.p2_hand) == 5:
                canPass = False
            else:
                canPass = True

        # 4 rows [0-3], 5 columns [0-4], 13 values [0-12], 4 x 5 x 13 = 260
        self_board = np.zeros(260).reshape(-1, 1)
        if playerNum == 1:
            for row_index, row in enumerate(board.p1_rows):
                for column_index, card_value in enumerate(row):
                    self_board[row_index + (4 * column_index) + (20 * card_value) ] = 1
        else:
            for row_index, row in enumerate(board.p2_rows):
                for column_index, card_value in enumerate(row):
                    self_board[row_index + (4 * column_index) + (20 * card_value)] = 1

        # 4 rows [0-3], 5 columns [0-4], 13 values [0-12], 4 x 5 x 13 = 260
        opponenet_board = np.zeros(260).reshape(-1, 1)
        if playerNum == 1:
            for row_index, row in enumerate(board.p2_rows):
                for column_index, card_value in enumerate(row):
                    opponenet_board[row_index + (4 * column_index) + (20 * card_value) ] = 1
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
            opponent_counted[len(board.p2_hand)] = 1
        else:
            opponent_counted[len(board.p1_hand)] = 1

        # 590 Total Length
        neural_input = np.concatenate((np.concatenate((np.concatenate((self_board, opponenet_board)), hand)), opponent_counted))
        neural_input = np.atleast_2d(neural_input)
        neural_input = neural_input.reshape(1, -1)

        output_prob = model.predict(neural_input)

        print(output_prob[0])

        print(output_prob[0][0:3])
        action = np.argmax(output_prob[0][0:3])

        print(output_prob[0][3:8])
        hand_action = np.argmax(output_prob[0][3:8])

        print(output_prob[0][8:])
        defend_action = np.argmax(output_prob[0][8:])

        print("test")

        # # either pass, attack or defend
        # action = np.zeros(3).reshape(-1, 1)
        #
        # # which card to use
        # hand_action = np.zeros(5).reshape(-1, 1)
        #
        # # which row to defend
        # defend_action = np.zeros(20).reshape(-1, 1)
        #
        # output_layer = np.concatenate((np.concatenate((action, hand_action)), defend_action))



