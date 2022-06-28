import math
from time import sleep

from board import *
from gameView import *
from player import *


def game_view(board):
    print("==P1 Hand===")
    for card in board.p1_hand:
        print(card + 2, end=" ")
    print(" ")
    print("==P1 Board==")
    for row in board.p1_rows:
        print("-", end=" ")
        for card in row:
            print(card + 2, end=" ")
        print(" ")
    print("============")
    print("==P2 Hand===")
    for card in board.p2_hand:
        print(card + 2, end=" ")
    print(" ")
    print("==P2 Board==")
    for row in board.p2_rows:
        print("-", end=" ")
        for card in row:
            print(card + 2, end=" ")
        print(" ")
    print("============")
    print("\n\n\n\n")


def model_mutate(weights):
    # mutate each models weights
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0, 1) > .85:
                change = random.uniform(-.5,.5)
                weights[i][j] += change
    return weights


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
        TOTAL_MODELS = 50 # Even number please!
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

            winnerIdx = list()

            for player in range(len(p1_pool)):
                p1_win = False
                p2_win = False

                while (p1_win and p2_win) or (not p1_win and not p2_win):
                    board = Board()
                    [p1_win, p2_win] = self.play_game(board, p1_pool[player], p2_pool[player])

                if p1_win:
                    winnerIdx.append(1)
                    print("1")
                else:
                    winnerIdx.append(2)
                    print("2")

            print("Selection")
            # P1 Pool are the pools that survive
            for idx in winnerIdx:
                if idx == 2:
                    p1_pool[idx] = p2_pool[idx]

            print("Crossover")
            # Crossover self and another random indexed model
            for parent1 in p1_pool:
                rand_idx = math.floor(random.random() * len(p1_pool))
                parent2 = p1_pool[rand_idx]

                weight1 = parent1.get_weights()
                weight2 = parent2.get_weights()

                new_weight1 = weight1
                new_weight2 = weight2

                gene = math.floor(random.random() * len(new_weight1))

                new_weight1[gene] = weight2[gene]
                new_weight2[gene] = weight1[gene]

                cross_over_weights = np.asarray([new_weight1, new_weight2])

                mutated1 = model_mutate(cross_over_weights[0])
                mutated2 = model_mutate(cross_over_weights[0])

                new_weights = list()
                new_weights.append(mutated1)
                new_weights.append(mutated2)

            print("Mutation")

        print("Stop after n generations")

    def play_game(self, board, p1_model, p2_model):

        p1 = Player(1, p1_model)
        p2 = Player(2, p2_model)

        players = [p1, p2]

        playerPass = [True, True]

        playerNum = 1

        while board.gameOver is False:
            # Update game board
            # game_view(board)

            # If both players passed, draw cards. Automatically the case at the start of the game
            if False not in playerPass:
                # End game if cards in first row
                if len(board.p1_rows[0]) != 0 or len(board.p2_rows[0]) != 0:
                    board.gameOver = True
                    break
                # Move all cards up a row
                for index in range(1, 4):
                    board.p1_rows[index - 1] = board.p1_rows[index]
                    board.p2_rows[index - 1] = board.p2_rows[index]
                board.p1_rows[3] = []
                board.p2_rows[3] = []
                # Refill hands
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
            action = players[playerNum - 1].get_player_action(board, playerNum, players[playerNum - 1].model, card_in_play)

            if action != "pass":
                playerPass[playerNum - 1] = False
            else:
                playerPass[playerNum - 1] = True

            if action[0] == "attack":
                if playerNum == 1:
                    board.attack_card(2, board.p1_hand[action[1]])
                else:
                    board.attack_card(1, board.p2_hand[action[1]])

            if action[0] == "defend":
                if playerNum == 1:
                    board.defend_card(1, action[1], action[2], action[3])
                else:
                    board.defend_card(2, action[1], action[2], action[3])

            if playerNum == 2:
                playerNum = 1
            else:
                playerNum = playerNum + 1

        win_return = [False, False]

        if len(board.p2_rows[0]) != 0:
            win_return[0] = True
        if len(board.p1_rows[0]) != 0:
            win_return[1] = True

        return  win_return


KerduGame()
