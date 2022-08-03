import os

from Environments.BasicEnv import KerduGameEnv

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
def compute_avg_return(environment, policy, num_episodes=300):
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


class C51:

    # train_time: (int) number of iterations
    # fc_layer_params: (int) size of fully connected layer
    # min_q: (int) minimum value of q that can be rewarded
    # max_q: (int) maximum value of q that can be rewarded
    # steps_considered: (int) how far into the future are we counting reward values
    def __init__(self, train_time, fc_layer_params, min_q, max_q, steps_considered):
        # Number of iterations during training, 5000000 was used with the original paper to show results
        self.num_iterations = train_time  # 100000

        # Initial collection for batch
        self.initial_collect_steps = 1000
        self.collect_steps_per_iteration = 1
        # How many elements can be stored in the replay buffer
        self.replay_buffer_capacity = 100000

        #
        self.fc_layer_params = (fc_layer_params,)  # 242

        # C51 Learning hyperparameters
        self.batch_size = 64
        self.learning_rate = 0.01
        self.gamma = 0.99
        self.log_interval = 200

        # Number of atoms to approximate probability distributions, more the better
        self.num_atoms = 51
        # Values should be set to the min and max step rewards
        self.min_q_value = min_q  # -100
        self.max_q_value = max_q  # 100
        # Computing error between current time step and next time step using 4 steps
        self.n_step_update = steps_considered

        self.num_eval_episodes = 10
        self.eval_interval = 1000

        # For training
        self.avg_returns = list()


    # env: (PyEnvironment) Environment the model will be training on
    # checkpoint_name: (str) name that the checkpoint is saved under
    def train(self, env, checkpoint_name):

        # Environment setup
        train_py_env = wrappers.TimeLimit(env, duration=100)  # KerduGameEnv()
        eval_py_env = wrappers.TimeLimit(env, duration=100)  # KerduGameEnv()

        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


        # C51 setup
        categorical_q_net = categorical_q_network.CategoricalQNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            num_atoms=self.num_atoms,
            fc_layer_params=self.fc_layer_params)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step_counter =  tf.compat.v1.train.get_or_create_global_step() # tf.Variable(0)
        agent = categorical_dqn_agent.CategoricalDqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            epsilon_greedy=0.1,
            categorical_q_network=categorical_q_net,
            optimizer=optimizer,
            min_q_value=self.min_q_value,
            max_q_value=self.max_q_value,
            n_step_update=self.n_step_update,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=self.gamma,
            train_step_counter=train_step_counter)

        agent.initialize()


        # Creating replay buffer and filling before training. Other setup as well
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                        train_env.action_spec())

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=self.replay_buffer_capacity)

        def collect_step(environment, policy):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)

        for _ in range(self.initial_collect_steps):
            collect_step(train_env, random_policy)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size,
            num_steps=self.n_step_update + 1, single_deterministic_pass=False).prefetch(3)

        iterator = iter(dataset)
        agent.train = common.function(agent.train)
        agent.train_step_counter.assign(0)
        avg_return = compute_avg_return(eval_env, agent.policy, self.num_eval_episodes)
        self.returns = [avg_return]


        # Checkpointer
        checkpoint_dir = os.path.join('SavedModels/Checkpoints', str(checkpoint_name))
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=train_step_counter
        )


        # Training starts
        for _ in range(self.num_iterations):

            # Collect a few steps using default agent greedy policy800 and save to the replay buffer.
            for _ in range(self.collect_steps_per_iteration):
                collect_step(train_env, agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience)

            step = agent.train_step_counter.numpy()

            # if step % log_interval == 0:
            #     print('step = {0}: loss = {1}'.format(step, train_loss.loss))

            if step % self.eval_interval == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
                self.returns.append(avg_return)


        # Save
        train_checkpointer.save(train_step_counter)


    def viewPlot(self):
        steps = range(0, self.num_iterations + 1, self.eval_interval)
        plt.plot(steps, self.returns)
        plt.ylabel('Average Return')
        plt.xlabel('Step')
        plt.ylim(top=110)

        plt.show()
