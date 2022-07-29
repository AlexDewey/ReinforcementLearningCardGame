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


# Finding the average return over 100 episodes
def compute_avg_return(environment, policy, num_episodes=100):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def train():
    # Number of iterations during training, 5000000 was used with the original paper to show results
    num_iterations = 3000

    # Initial collection for batch
    initial_collect_steps = 1000
    collect_steps_per_iteration = 1
    # How many elements can be stored in the replay buffer
    replay_buffer_capacity = 100000

    fc_layer_params = (800,)

    # C51 Learning hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    gamma = 0.99
    log_interval = 200

    # Number of atoms to approximate probability distributions, more the better
    num_atoms = 51
    # Values should be set to the min and max step rewards
    min_q_value = -100
    max_q_value = 100
    # Computing error between current time step and next time step using 2 steps
    n_step_update = 2

    num_eval_episodes = 10
    eval_interval = 1000

    # Environments are wrapped in a time limit and then a tf environment wrapper.
    train_py_env = wrappers.TimeLimit(KerduGameEnv(), duration=100)
    eval_py_env = wrappers.TimeLimit(KerduGameEnv(), duration=100)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # A categorical q network is required for our categorical C51
    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params)

    # Adam optimizer used for optimization
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    # Track how many times the network was updated
    train_step_counter = tf.Variable(0)

    # The main difference between this and a vainilla C51 is the min/max q difference
    agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        train_step_counter=train_step_counter)

    # Initialize C51
    agent.initialize()

    # Establishing random policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    # Saves observations and action pairs for training. As to not bias the network to specific situations these are fed
    # in somewhat randomly
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

    # Observes the environment, gets an action, gets the resulting time step, then saves the result to the buffer
    def collect_step(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)

    # Undergoing initial steps to add to replay buffer before official training as it cannot start empty
    for _ in range(initial_collect_steps):
        collect_step(train_env, random_policy)

    # The dataset takes our replay buffer and creates trajectories
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=n_step_update + 1, single_deterministic_pass=False).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Actual training begins
    for _ in range(num_iterations):

        # Collect a few steps using default agent greedy policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns.append(avg_return)

    policy_dir = os.path.join('../SavedModels', 'policy')
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)




    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=110)

    plt.show()


train()
