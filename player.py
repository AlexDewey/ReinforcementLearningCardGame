import numpy as np


class Player:

    def __init__(self, position):
        self.position = position

    def getPlayerAction(self, board, playerNum, passed):
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

        # 12 values, 4 rows, 5 columns = binary representation: [0-11], [0-3], [0,4] = 12x4x5 = 240
        self_board = np.zeros(1040).reshape(-1, 1)
        if playerNum == 1:
            for row_index, row in enumerate(board.p1_rows):
                print(row_index)
                for column_index, card in enumerate(row):
                    print(column_index)
                    print(str(card))

        # 12 values, 4 rows, 5 columns = binary representation: [0-11], [0-3], [0,4] = 12x4x5 = 240
        opponenet_board = np.zeros(1040).reshape(-1, 1)

        # 5 cards held x 12 values = binary representation: [0-4], [0-11] = 5x12 = 60
        hand = np.zeros(260).reshape(-1, 1)

        # 5 cards held by player = 5 cards counted [0-4] = 5
        self_counted = np.zeros(5).reshape(-1, 1)

        # 5 cards held by opponent = 5 cards counted [0-4] = 5
        opponent_counted = np.zeros(5).reshape(-1, 1)

        # In total 550