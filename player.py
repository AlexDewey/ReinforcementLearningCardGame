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

        # 4 chances of a card, 13 values, 4 rows, 5 columns = 1040 self board
        self_board = np.zeros(1040).reshape(-1, 1)
        if playerNum == 1:
            for rows in board.p1_rows:
                for index, card in enumerate(rows):
                    


        # 4 chances of a card, 13 values, 4 rows, 5 columns = 1040 opponent board
        opponenet_board = np.zeros(1040).reshape(-1, 1)

        # 5 cards held x 4 suits x 13 values = 260 hand
        hand = np.zeros(260).reshape(-1, 1)

        # 5 cards held by each player = 10 cards counted
        cards_counted = np.zeros(10).reshape(-1, 1)