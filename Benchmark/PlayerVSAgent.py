import os

from BaseEnv.board import Board
from C51.TFEnvironment import KerduGameEnv

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.environments import wrappers


def post_action_logic(self, action_used):
    if action_used[0] != "pass":
        self.playerPass[self.playerNum - 1] = False
    else:
        self.playerPass[self.playerNum - 1] = True

    if action_used[0] == "attack":
        if self.playerNum == 1:
            try:
                self.board.attack_card(2, self.board.p1_hand[action_used[1]])
            except:
                self.playerPass[self.playerNum - 1] = True
        else:
            self.board.attack_card(1, self.board.p2_hand[action_used[1]])

    if action_used[0] == "defend":
        if self.playerNum == 1:
            if len(self.board.p1_rows[action_used[2]]) > action_used[3]:
                self.board.defend_card(1, action_used[1], action_used[2], action_used[3])
        else:
            if len(self.board.p2_rows[action_used[2]]) > action_used[3]:
                self.board.defend_card(2, action_used[1], action_used[2], action_used[3])

    if self.playerNum == 2:
        self.playerNum = 1
    else:
        self.playerNum = self.playerNum + 1


if __name__ == "__main__":
    # Loading AI
    saved_policy = tf.saved_model.load('../SavedModels/policy')

    board = Board()
    board.fill_hand(1)
    board.fill_hand(2)
    players = ["PL", "NN"]
    playerPass = [True, True]
    card_in_play = False
    playerNum = 1

    print("test")
