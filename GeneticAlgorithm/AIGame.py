from tensorflow import keras

from BaseEnv.board import *
from GeneticAlgorithm.player import *


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
        keras.layers.Dense((590 * 2), input_shape=(590,), activation="relu"),
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

        # CREATING MODELS AND SAVING THEIR BASE SELVES

        for i in range(TOTAL_MODELS):
            model = create_model()
            model.save_weights("SavedModels/model_base_" + str(i) + ".keras")
            base_pool.append(model)

        if not CREATE_NEW_POOL:
            for i in range(TOTAL_MODELS):
                base_pool[i].load_weights("SavedModels/model_new" + str(i) + ".keras")


        # TRAINING MODELS

        for generation in range(GENERATIONS):
            print("Generation " + str(generation) + "/" + str(generations) + ", mutation rate " + str(self.mutation_rate))

            if generation % 5 == 0:
                original_pool = list()
                for i in range(TOTAL_MODELS):
                    model = create_model()
                    model.load_weights("SavedModels/model_base_" + str(i) + ".keras")
                    original_pool.append(model)

                trained_pool = base_pool

                trained_wins = 0
                trained_losses = 0

                for pool_index in range(len(trained_pool)):
                    p_trained_win = False
                    p_base_win = False

                    rematches = 0

                    while (p_trained_win and p_base_win) or (not p_trained_win and not p_base_win):
                        board = Board()
                        [p_trained_win, p_base_win] = self.play_game(board, trained_pool[pool_index],
                                                                     original_pool[pool_index])
                        rematches += 1

                    if p_trained_win:
                        trained_wins += 1
                    else:
                        trained_losses += 1

                print("Generation: " + str(generation) + " " + str(trained_wins / (trained_wins + trained_losses)))

            random.shuffle(base_pool)
            p1_pool = base_pool[0:math.floor(len(base_pool) / 2)]
            p2_pool = base_pool[math.floor(len(base_pool) / 2):]

            winnerIdx = list()

            for player_index in range(len(p1_pool)):
                p1_win = False
                p2_win = False

                # HAVING BOTS PLAY EACH OTHER

                while (p1_win and p2_win) or (not p1_win and not p2_win):
                    board = Board()
                    [p1_win, p2_win] = self.play_game(board, p1_pool[player_index], p2_pool[player_index])

                if p1_win:
                    winnerIdx.append(1)
                else:
                    winnerIdx.append(2)

            # P1 POOL IS USED FOR ALL WINNERS

            for index, winner in enumerate(winnerIdx):
                if winner == 2:
                    p1_pool[index] = p2_pool[index]

            # CREATING NEW POOL BASED OFF OF WINNERS

            for parent_index, parent1 in enumerate(p1_pool):
                rand_idx = parent_index
                while rand_idx == parent_index:
                    rand_idx = math.floor(random.random() * len(p1_pool))
                parent2 = p1_pool[rand_idx]

                new_weight1 = parent1.get_weights()
                new_weight2 = parent2.get_weights()

                # CROSSOVER

                for layer_index, layer in enumerate(new_weight1):
                    for gene_index in range(len(layer)):
                        if random.random() > 0.85:
                            layer[gene_index] = new_weight2[layer_index][gene_index]
                for layer_index, layer in enumerate(new_weight2):
                    for gene_index in range(len(layer)):
                        if random.random() > 0.85:
                            layer[gene_index] = new_weight2[layer_index][gene_index]

                # MUTATION

                for layer in new_weight1:
                    if layer.ndim == 2:
                        for gene_index in range(len(layer)):
                            for connection_index in range(len(layer[gene_index]) - 1):
                                if random.random() > 1 - self.mutation_rate:
                                    layer[gene_index][connection_index] += random.uniform(-0.5, 0.5)
                for layer in new_weight2:
                    if layer.ndim == 2:
                        for gene_index in range(len(layer)):
                            for connection_index in range(len(layer[gene_index]) - 1):
                                if random.random() > 1 - self.mutation_rate:
                                    layer[gene_index][connection_index] += random.uniform(-0.5, 0.5)


                # SAVING CHANGES

                parent1.set_weights(new_weight1)
                new_model = create_model()
                new_model.set_weights(new_weight2)
                p2_pool[parent_index] = new_model

            # CONSTRUCTING NEW POOL

            base_pool = p1_pool + p2_pool

        # GRAB ORIGINAL AI

        original_pool = list()
        for i in range(TOTAL_MODELS):
            model = create_model()
            model.load_weights("SavedModels/model_base_"+str(i)+".keras")
            original_pool.append(model)

        trained_pool = base_pool

        trained_wins = 0
        trained_losses = 0

        for pool_index in range(len(trained_pool)):
            p_trained_win = False
            p_base_win = False

            rematches = 0

            while (p_trained_win and p_base_win) or (not p_trained_win and not p_base_win):
                board = Board()
                [p_trained_win, p_base_win] = self.play_game(board, trained_pool[pool_index], original_pool[pool_index])
                rematches += 1

            if p_trained_win:
                trained_wins += 1
            else:
                trained_losses += 1

        self.trained_wins = trained_wins
        self.trained_losses = trained_losses

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

    def grabWinsLosses(self):
        return [self.trained_wins, self.trained_losses]
