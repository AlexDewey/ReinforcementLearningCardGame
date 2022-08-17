import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import categorical_q_network, sequential
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import wrappers
from tf_agents.specs import tensor_spec


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

# env: (PyEnvironment) Environment the model will be training on
    # model_name: (str) name that the checkpoint is saved under
    # num_games: (int) number of games you want to watch
def watch(env, model_name, num_games):
    env._view(True)
    saved_policy = tf.saved_model.load('SavedModels/Policies/' + str(model_name))

    eval_py_env = wrappers.TimeLimit(env, duration=1000)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    for _ in range(num_games):
        time_step = eval_env.reset()
        while not time_step.is_last():
            action_step = saved_policy.action(time_step)
            time_step = eval_env.step(action_step.action)
    env._view(False)


class DQN:

    # train_time: (int) number of iterations
    # fc_layer_params: (int) size of fully connected layer
    # min_q: (int) minimum value of q that can be rewarded
    # max_q: (int) maximum value of q that can be rewarded
    # steps_considered: (int) how far into the future are we counting reward values
    def __init__(self, train_time):
        # Number of iterations during training, 5000000 was used with the original paper to show results
        self.num_iterations = train_time  # 100000

        # Initial collection for batch
        self.initial_collect_steps = 1000
        self.collect_steps_per_iteration = 1
        # How many elements can be stored in the replay buffer
        self.replay_buffer_capacity = 100000
        self.learning_rate = 0.001
        self.batch_size = 64
        self.n_step_update = 1

        self.fc_layer_params = (100, 50)  # 242

        self.num_eval_episodes = 10
        self.eval_interval = 10000

        # Empty variables until assigned
        self.avg_returns = list()
        self.model_name = ""

    # env: (PyEnvironment) Environment the model will be training on
    # model_name: (str) name that the checkpoint is saved under
    def train(self, env, model_name):
        self.model_name = str(model_name)

        # Environment setup
        train_py_env = wrappers.TimeLimit(env, duration=100)  # KerduGameEnv()
        eval_py_env = wrappers.TimeLimit(env, duration=100)  # KerduGameEnv()

        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        action_tensor_spec = tensor_spec.from_spec(env.action_spec())

        def dense_layer(num_units):
            return tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode='fan_in', distribution='truncated_normal'))

        # DQN Setup
        dense_layers = [dense_layer(num_units) for num_units in self.fc_layer_params]
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_step_counter = tf.compat.v1.train.get_or_create_global_step()  # tf.Variable(0)
        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
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
        checkpoint_dir = os.path.join('SavedModels/Checkpoints', str(model_name))
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

        # Save Checkpoint for training and policy for deployment
        train_checkpointer.save(train_step_counter)
        policy_dir = os.path.join('SavedModels/Policies', str(model_name))
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        tf_policy_saver.save(policy_dir)

    def viewPlot(self):
        steps = range(0, self.num_iterations + 1, self.eval_interval)
        plt.plot(steps, self.returns)
        plt.ylabel('Average Return')
        plt.xlabel('Step')
        plt.ylim(top=130)

        plt.show()
