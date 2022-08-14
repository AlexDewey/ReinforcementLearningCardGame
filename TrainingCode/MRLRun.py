import numpy as np
import random

from tf_agents.environments import wrappers
from tf_agents.environments import tf_py_environment


class MRL:

    def __init__(self, seed):

        random.seed(seed)


    def train(self, env, model_name):

        train_py_env = wrappers.TimeLimit(env, duration=100)
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)

        ts = train_py_env.reset()

        player = 1

        while not ts.is_last():
            action = {
                'position': np.asarray(random.randint(0, 8)),
                'vlaue': player
            }
            ts = train_py_env.step(action)
            print('Player:', player, 'Action:', action['position'], 'Reward:', ts.reward)
            player = 1 + player % 2