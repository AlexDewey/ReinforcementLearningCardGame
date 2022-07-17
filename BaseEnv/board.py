from BaseEnv.deck import *


class Board:

    def __init__(self):
        # 0 is closest, 3 furthest away
        self.p1_rows = [[], [], [], []]
        self.p2_rows = [[], [], [], []]

        # Hands of both players
        self.p1_hand = []
        self.p2_hand = []

        # Base Facts
        self.turn = 1
        self.gameOver = False

        # Create Deck
        self.deck = Deck()

    def fill_hand(self, playerNum):
        if playerNum == 1:
            while len(self.p1_hand) < 5:
                self.p1_hand.append(self.deck.draw_card())
        else:
            while len(self.p2_hand) < 5:
                self.p2_hand.append(self.deck.draw_card())

    def attack_card(self, boardAttacked, cardValue):

        row_pos = 3

        if cardValue <= 1:
            row_pos = 0
        if 2 <= cardValue <= 8:
            row_pos = 1
        if 9 <= cardValue <= 11:
            row_pos = 2
        if cardValue == 12:
            row_pos = 3

        # We need to remove the card from hand and place it on the board
        if boardAttacked == 1:
            self.p1_rows[row_pos].append(cardValue)
            self.p2_hand.remove(cardValue)
        else:
            self.p2_rows[row_pos].append(cardValue)
            self.p1_hand.remove(cardValue)

    def defend_card(self, boardDefended, row, column, card_used_idx):

        print("P" + str(boardDefended) + " Rows: " + str(self.p1_rows) + " row: " + str(row) + " column: " + str(column))

        if boardDefended == 1:
            # Discard and remove both the hand card and row card
            self.deck.discard_card(self.p1_rows[row][column])
            del self.p1_rows[row][column]

            self.deck.discard_card(self.p1_hand[card_used_idx])
            del self.p1_hand[card_used_idx]
        else:
            self.deck.discard_card(self.p2_rows[row][column])
            del self.p2_rows[row][column]

            self.deck.discard_card(self.p2_hand[card_used_idx])
            del self.p2_hand[card_used_idx]
