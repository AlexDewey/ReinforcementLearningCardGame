import os

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

saved_policy = tf.saved_model.load('../SavedModels/policy')

print("test")
