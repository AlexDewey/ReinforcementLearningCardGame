# Define a Network using Keras layers
class MyQNetwork(tf_agents.networks.Network):

    def __init__(self, number_of_actions):

        self.forward = tf.keras.Sequential([
            ...
            ...
            tf.keras.layers.Dense(number_of_actions)
        ])

    def call(self, observations, state=()):
        logits = self._forward(observations)
        return logits, state