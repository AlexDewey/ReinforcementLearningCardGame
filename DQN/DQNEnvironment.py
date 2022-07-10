import numpy as np

class DQNGame:

    def __init__(self):
        print("Hello World")

        # create the environment
        env = gym.make("Kerdu-v0")

        # parameters
        action_space_size = env.action_space.n
        state_space_size = env.observation_space.n

        # brain
        q_table = np.zeros(every possible combination)

        # how many training games
        num_episodes = 10000
        # how many steps before termination
        max_steps_per_episode = 100

        # alpha in text, should be 1 and approach 0 as time infinitely grows
        learning_rate = 0.1
        # discount rate is gamma, it measures future insight into actions, low gamma means greedy model
        discount_rate = 0.99

        # how often to chose to explore or be greedy
        exploration_rate = 1
        max_exploration_rate = 1
        min_exploration_rate = 0.01
        exploration_decay_rate = 0.001

        rewards_all_episodes = []
        for episode in range(num_episodes):
            state = env.reset()

            done = False
            rewards_current_episode = 0

            for step in range(max_exploration_rate):

                # Exploration-exploitation trade-off
                # If passes the threshold, take greedy algorithm, otherwise take random action
                exploration_rate_threshold = random.uniform(0, 1)
                if exploration_rate_threshold > exploration_rate:
                    action = np.argmax(q_table[state,:])
                else:
                    action = env.action_space.sample()

                new_state, reward, done, info = env.step(action)

                # Update Q-table for Q(s,a)


                state = new_state
                rewards_current_episode += reward

            exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) \
                               * np.exp(-exploration_decay_rate * episode)

            rewards_all_episodes.append(rewards_current_episode)

    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
    count = 1000
    print("***Average reward per thousand episodes***\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000
