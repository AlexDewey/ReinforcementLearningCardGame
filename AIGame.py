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
    model.add(Dense(28, input_shape=(28,)))
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

        playerPass = [False, False]

        playerNum = 1

        while board.gameOver is False:
            action = players[playerNum].get_player_action(board, playerNum, playerPass[playerNum - 1])

            if action != "pass":
                playerPass[playerNum - 1] = False
            else:
                playerPass[playerNum - 1] = True
                continue

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
