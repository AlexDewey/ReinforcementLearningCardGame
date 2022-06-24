import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


class Player:

    def __init__(self, position, model):
        self.position = position
        self.model = model

    def get_player_action(self, board, playerNum, model, cant_skip):
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

        print(output_prob[0][0:5])
        hand_action_prio = np.argsort(output_prob[0][0:5])[::-1]

        print(output_prob[0][5:])
        defend_action_prio = np.argsort(output_prob[0][5:])[::-1]

        action_prio = np.argsort(output_prob[0])[::-1]

        print("test")

        for action in action_prio:
            if 0 <= action <= 4:
                # check if card exists in hand
                if len(board.p1_hand) > action:
                    return ["attack", action]
                else:
                    continue
            elif 5 <= action <= 28:
                # check if card exists in
                if:

            else:
                if cant_skip:
                    continue
                else:
                    return "pass"


        # Validate available action
        # for action in action_prio:
        #     # If trying to pass
        #     if action == 0:
        #         if cant_skip:
        #             continue
        #         else:
        #             return "pass"
        #     # If trying to attack
        #     elif action == 1:
        #         # If the card exists to attack with and is in the player's hand
        #         if len(board.p1_hand) > hand_action_prio[0]:
        #             return ["attack", hand_action_prio[0]]
        #         else:
        #
        #     # If trying to defend
        #     elif action == 2:
        #         if len(board.pe)





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



