import numpy as np
import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

from DQN.TFEnvironment import KerduGameEnv

from tf_agents.environments import tf_py_environment

from tf_agents.environments import wrappers
from tf_agents.networks import q_network
from tf_agents.agents import DqnAgent

from tf_agents.replay_buffers import TFUniformReplayBuffer

import matplotlib.pyplot as plt


def train():
    train_py_env = wrappers.TimeLimit(KerduGameEnv(), duration=100)
    eval_py_env = wrappers.TimeLimit(KerduGameEnv(), duration=100)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (1000,)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter)

    tf_agent.initialize()

    replay_buffer = TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size)

    replay_observer = [replay_buffer.add_batch]

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        tf_agent.collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=1)

    episode_len = []

    final_time_step, policy_state = driver.run()

    for i in range(num_iterations):
        final_time_step, _ = driver.run(final_time_step, policy_state)

        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience=experience)
        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            episode_len.append(train_metrics[3].result().numpy())
            print('Average episode length: {}'.format(train_metrics[3].result().numpy()))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
    plt.plot(episode_len)
    plt.show()


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

# env.reset()



train()


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

















