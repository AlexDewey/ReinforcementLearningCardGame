from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


class Networks(object):

    @staticmethod
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        # Base network structure, already flattened
        state_input = Input(shape=input_shape)
        deep_net = Dense(800, activation='relu')

        # Creating distributions for every output
        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(deep_net))

        # Creating model with adam optimizer
        model = Model(input=state_input, output=distribution_list)
        adam = Adam(lr=learning_rate)
        model.compile(loss='cateogrical_crossentropy', optimizer=adam)

        return model
