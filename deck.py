import random


def shuffleDeck(deck):
    deck = random.shuffle(deck)
    return deck


def buildDeck(discardPile):
    # Shuffling the discard pile into
    if len(discardPile) > 1:
        deck = discardPile
    else:
        deck = list()
    for value in range(2, 15):
        for instance in range(0, 4):
            deck.append(value)
    return deck


class Deck:

    def __init__(self):
        deck = buildDeck([])
        self.deck = shuffleDeck(deck)

        self.discardPile = list()

    def drawCard(self):
        # If empty deck first replace
        if len(self.deck) == 0:
            # Basic deck
            deck = buildDeck(self.discardPile)
            self.deck = shuffleDeck(deck)

            # Take out the cards in hand
        return self.deck.pop()

    def discardCard(self, card):
        self.discardPile.append(card)
