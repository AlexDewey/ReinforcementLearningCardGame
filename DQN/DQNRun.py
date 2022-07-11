# Actor and critic network
from DQN import Environment

actor_net = ActorDistributionNetwork(action_spec)
critic_net = CriticNetwork((observation_spec, action_spec), ...)

# Create a Soft-Actor-Critic Agent
tf_agent = sac_agent.SacAgent(
    critic_network=critic_network,
    actor_network=actor_netwrok,
    actor_optimizer=AdamOptimizer(learning_rate=0.001),
    critic_optimizer=AdamOptimizer(learning_rate=0.001))

# Get experience and train
dataset = replay_buffer.as_dataset(num_steps=2).prefetch(3)
for batched_experience in dataset:
    tf_agent.train(batched_experience)



# ---------------------

# Single

# # Defining the Environment
# env = Environment.BreakoutEnv()
# # Define a Policy (How the agent plays)
# policy = MyPolicy()
#
# time_step = env.reset()
# episode_return = 0
#
# while not time_step.is_last():
#     policy_step = policy.action(time_step)
#     time_step = env.step(policy_step.action)
#     episode_return += time_step.reward



# Parallel

# Running the game 4 times
parallel_env = ParallelPyEnvironment([BreakoutEnv() for _ in range(4)])
tf_env = TFPyEnvironment(parallel_env)

time_step = tf_env.reset()
episode_return = tf.zeros([4])
for _ in range(num_steps):
    policy_step = policy.action(time_step)
    time_step = tf_env.step(policy_step.aciton)
    episode_return += time_step.reward




# -----------------------


# Creating an agent
agent = DqnAgent(q_network=..., optimizer=...)

# Policy used to explore & collect training data
collect_policy = agent.collect_policy

# Train from a batch of experience
loss_info = agent.train(batched_experience=trajectories)

# The deployment policy
deployment_policy = agent.policy




# --------------------------

# Training neural network

# Create a Network
q_net = q_network.QNetwork(observation_spec, action_spec, ...)

# Create the Agent
agent = dqn_agent.DqnAgent(q_network=q_net,
                           optimizer=AdamOptimizer(learning_rate=0.001))

# Get experience and train the Agent
dataset = replay_buffer.as_dataset(num_steps=2).prefetch(3)

for batched_experience in dataset:
    agent_train(batched_experience)
























