from tensorflow import keras

from board import *
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


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(590, input_shape=(590,)),
        keras.layers.Dense((590 * 2), activation="relu"),
        keras.layers.Dense(106, activation="relu")
    ])

    model.compile(loss='mse', optimizer='adam')

    return model


class KerduGame:

    def __init__(self, mutation_rate, model_nums, generations):
        self.mutation_rate = mutation_rate
        CREATE_NEW_POOL = True
        TOTAL_MODELS = model_nums  # Even number please!
        GENERATIONS = generations
        base_pool = list()

        for i in range(TOTAL_MODELS):
            model = create_model()
            model.save_weights("SavedModels/model_base_" + str(i) + ".keras")
            base_pool.append(model)

        if not CREATE_NEW_POOL:
            for i in range(TOTAL_MODELS):
                base_pool[i].load_weights("SavedModels/model_new" + str(i) + ".keras")


        for generation in range(GENERATIONS):
            print("Generation " + str(generation) + "/" + str(generations) + ", mutation rate " + str(self.mutation_rate))

            # todo: rework this code so no break :3
            random.shuffle(base_pool)
            p1_pool = base_pool[0:math.floor(len(base_pool) / 2)]
            p2_pool = base_pool[math.floor(len(base_pool) / 2):]

            winnerIdx = list()

            for player_index in range(len(p1_pool)):
                p1_win = False
                p2_win = False

                # rematches = 0

                while (p1_win and p2_win) or (not p1_win and not p2_win):
                    board = Board()
                    [p1_win, p2_win] = self.play_game(board, p1_pool[player_index], p2_pool[player_index])
                    # rematches += 1

                if p1_win:
                    winnerIdx.append(1)
                    # print("1    " + str(rematches))
                else:
                    winnerIdx.append(2)
                    # print("2    " + str(rematches))

            # print("Selection")
            # P1 Pool are the pools that survive
            for index, winner in enumerate(winnerIdx):
                if winner == 2:
                    p1_pool[index] = p2_pool[index]

            # Crossover self and another random indexed model
            for parent_index, parent1 in enumerate(p1_pool):
                rand_idx = parent_index
                while rand_idx == parent_index:
                    rand_idx = math.floor(random.random() * len(p1_pool))
                parent2 = p1_pool[rand_idx]

                new_weight1 = parent1.get_weights()
                new_weight2 = parent2.get_weights()

                # Crossover
                for layer_index, layer in enumerate(new_weight1):
                    for gene_index in range(len(layer)):
                        if random.random() > 0.85:
                            layer[gene_index] = new_weight2[layer_index][gene_index]
                for layer_index, layer in enumerate(new_weight2):
                    for gene_index in range(len(layer)):
                        if random.random() > 0.85:
                            layer[gene_index] = new_weight2[layer_index][gene_index]

                # Mutation
                for layer in new_weight1:
                    print(layer)
                    for gene_index in range(len(layer)):
                        print(gene_index)
                        print(len(layer[gene_index]) - 1)
                        for connection_index in range(len(layer[gene_index]) - 1):
                            if random.random() > 1 - self.mutation_rate:
                                layer[gene_index][connection_index] += random.uniform(-0.5, 0.5)
                for layer in new_weight2:
                    for gene_index in range(len(layer)):
                        for connection_index in range(len(layer[gene_index]) - 1):
                            if random.random() > 1 - self.mutation_rate:
                                layer[gene_index][connection_index] += random.uniform(-0.5, 0.5)


                # Set values to location
                parent1.set_weights(new_weight1)
                p2_pool[parent_index] = create_model().set_weights(new_weight2)

            # todo: rework this code for no break :3
            base_pool = p1_pool + p2_pool

        original_pool = list()
        for i in range(TOTAL_MODELS):
            model = create_model()
            original_pool.append(model.load_weights("SavedModels/model_base_"+str(i)+".keras"))

        trained_pool = base_pool

        for pool_index in range(len(trained_pool)):
            p_trained_win = False
            p_base_win = False

            rematches = 0

            while (p_trained_win and p_base_win) or (not p_trained_win and not p_base_win):
                board = Board()
                [p_trained_win, p_base_win] = self.play_game(board, trained_pool[pool_index], p_base_win[pool_index])
                rematches += 1

            if p_trained_win:
                print("1    " + str(rematches))
            else:
                print("2    " + str(rematches))



    @staticmethod
    def play_game(board, p1_model, p2_model):

        p1 = Player(1, p1_model)
        p2 = Player(2, p2_model)

        players = [p1, p2]

        playerPass = [True, True]

        playerNum = 1

        while board.gameOver is False:
            # Update game board

            # game view
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
            action = players[playerNum - 1].get_player_action(board, playerNum, players[playerNum - 1].model,
                                                              card_in_play)

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

        return win_return


KerduGame(0.01, 10, 100)
print("Done!")
