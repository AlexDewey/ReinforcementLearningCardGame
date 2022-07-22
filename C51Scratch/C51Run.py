from C51Scratch import EnvironmentScratch


class C51Agent:

    def __init__(self, state_size, action_size, num_atoms):
        self.state_size = state_size
        self.action_size = action_size

        # Discount rate high
        self.gamma = 0.99
        # Small learning rate
        self.learning_rate = 0.0001
        self.batch_size = 64
        # Exploration data
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.observe = 2000
        self.explore = 50000

        self.num_atoms = num_atoms
        self.q_max = 100
        self.q_min = -100


if __name__ == "__main__":

    env = EnvironmentScratch.KerduScratch()




