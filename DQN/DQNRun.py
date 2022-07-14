import numpy as np
import tensorflow as tf

from DQN.TFEnvironment import KerduGameEnv

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils

# get_new_card_action = np.array(0, dtype=np.int32)
# end_round_action = np.array(1, dtype=np.int)
#
# env = KerduGameEnv()
# tf_env = tf_py_environment.TFPyEnvironment(env)
# time_step = tf_env.reset()
# print(time_step)
# cumulative_reward = time_step.reward
#
# for _ in range(3):
#     time_step = tf_env.step(get_new_card_action)
#     print(time_step)
#     cumulative_reward += time_step.reward
#
# time_step = tf_env.step(end_round_action)
# print(time_step)
# cumulative_reward += time_step.reward
# print("Final Reward = ", cumulative_reward)
# tf_env.close()



# Hyperparameters

num_iterations = 20000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 0.001
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000

# Environment

env = KerduGameEnv()

env.reset()

# Obs Spec

# Reward Spec

# Action Spec




# time_step = env.reset()
#
# action = np.array(1, dtype=np.int32)
#
# next_time_step = env.step(action)
#
# train_py_env = suite_gym.load(env_name)
# eval_py_env = suite_gym.load(env_name)
#
# train_env = tf_py_environment.TFPyEnvironment(train_py_env)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

















