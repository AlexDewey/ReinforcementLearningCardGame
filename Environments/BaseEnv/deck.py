import random


def shuffle_deck(deck):
    random.shuffle(deck)
    return deck


def build_deck(discardPile):
    # Shuffling the discard pile into
    if len(discardPile) > 1:
        deck = discardPile
    else:
        deck = list()
    for value in range(0, 13):
        for instance in range(0, 4):
            deck.append(value)
    return deck


class Deck:

    def __init__(self):
        deck = build_deck([])
        self.deck = shuffle_deck(deck)

        self.discardPile = list()

    def draw_card(self):
        # If empty deck first replace
        if len(self.deck) == 0:
            # Basic deck
            deck = build_deck(self.discardPile)
            self.deck = shuffle_deck(deck)

            # Take out the cards in hand
        return self.deck.pop()

    def discard_card(self, card):
        self.discardPile.append(card)
