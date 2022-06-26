import math
from time import sleep

from board import *
from gameView import *
from player import *


def create_model():

    model = Sequential()
    model.add(Dense(590, input_shape=(590,)))
    model.add(Activation('relu'))
    model.add(Dense(590 * 2, input_shape=(590,)))
    model.add(Activation('relu'))
    # 100 for placement and card in defence, 5 for attack and 1 for pass. = 106
    model.add(Dense(106, input_shape=(590,)))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='adam')

    return model


class KerduGame:

    def __init__(self):
        print("Generating initial population")
        CREATE_NEW_POOL = True
        TOTAL_MODELS = 50
        GENERATIONS = 10
        current_pool = list()

        for i in range(TOTAL_MODELS):
            model = create_model()
            current_pool.append(model)

        if not CREATE_NEW_POOL:
            for i in range(TOTAL_MODELS):
                current_pool[i].load_weights("SavedModels/model_new"+str(i)+".keras")

        for generation in range(GENERATIONS):
            print("Computing fitness")

            random.shuffle(current_pool)
            p1_pool = current_pool[0:math.floor(len(current_pool)/2)]
            p2_pool = current_pool[math.floor(len(current_pool)/2):]

            for player in range(len(p1_pool)):
                board = Board()
                self.play_game(board, p1_pool[player], p2_pool[player])

            print("Selection")

            print("Crossover")

            print("Mutation")

        print("Stop after n generations")

    def play_game(self, board, p1_model, p2_model):

        p1 = Player(1, p1_model)
        p2 = Player(2, p2_model)

        players = [p1, p2]

        playerPass = [True, True]

        playerNum = 1

        while board.gameOver is False:
            # If both players passed, draw cards. Automatically the case at the start of the game
            if False not in playerPass:
                for index in range(0, len(players)):
                    board.fill_hand(index + 1)
                    playerPass[index] = False

            # If there's a card on the board, the player can pass, otherwise no
            card_in_play = False
            for row in board.p1_rows:
                if len(row) != 0:
                    card_in_play = True
            for row in board.p2_rows:
                if len(row) != 0:
                    card_in_play = True

            # actions: ["pass", ["attack", cardHandIndex], ["defend", cardTargetedRow, cardTargetedColumn, cardUsedIdx]]
            action = players[playerNum].get_player_action(board, playerNum, players[playerNum - 1].model, card_in_play)

            if action != "pass":
                playerPass[playerNum - 1] = False
            else:
                playerPass[playerNum - 1] = True

            if action[0] == "attack":
                if playerNum == 1:
                    board.attack_card(playerNum, board.p2_hand[action[1]])
                else:
                    board.attack_card(playerNum, board.p1_hand[action[1]])

            if action[0] == "defend":
                if playerNum == 1:
                    board.defend_card(2, action[1], action[2])

            if playerNum == 2:
                playerNum = 1
            else:
                playerNum = playerNum + 1


KerduGame()
