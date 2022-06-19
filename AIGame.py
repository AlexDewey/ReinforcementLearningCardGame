from time import sleep

from board import *
from gameView import *
from player import *


class KerduGame:

    def __init__(self):
        print("Initializing Kerdu game with only Ai Players.")
        board = Board()
        self.play_game(board)


    def play_game(self, board):

        p1 = Player(1)
        p2 = Player(2)

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
