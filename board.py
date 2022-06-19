from deck import *


class Board:

    def __init__(self):
        # 0 is closest, 3 furthest away
        self.p1_rows = [[], [], [], []]
        self.p2_rows = [[], [], [], []]

        # Hands of both players
        self.p1_hand = list()
        self.p2_hand = list()

        # Base Facts
        self.turn = 1
        self.gameOver = False

        # Create Deck
        self.deck = Deck()

    def attackCard(self, boardAttacked, cardValue):

        row_pos = 3

        if cardValue <= 1:
            row_pos = 0
        if 2 <= cardValue <= 8:
            row_pos = 1
        if 9 <= cardValue <= 11:
            row_pos = 2
        if cardValue == 12:
            row_pos = 3

        if boardAttacked == 1:
            self.p1_rows[row_pos].append(cardValue)
        else:
            self.p2_rows[row_pos].append(cardValue)

    def defendCard(self, boardDefended, row, cardValueTarget):

        if boardDefended == 1:
            self.p1_rows[row].remove(cardValueTarget)
        else:
            self.p2_rows[row].remove(cardValueTarget)

        self.deck.discardCard(cardValueTarget)
