#################
# Agent Creators
#################

# NN Size
# Saved Name
# C51_Base = C51Run(242, 'policy242')

#################
# Agent Enhancers
#################



#################
# Environments
#################

# Environment difficulty 1 (aggressive / hard) - 4 (easy)
# BasicEnv = BasicEnv(4)

# Player vs. Agent
# PVA = PVA()

# if __name__ == "__main__":
#     saved_policy = tf.saved_model.load('../SavedModels/policy800')
#
#     eval_py_env = wrappers.TimeLimit(KerduGamePVN(), duration=1000)
#     eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#
#     num_games = 10
#     for _ in range(num_games):
#         time_step = eval_env.reset()
#         while not time_step.is_last():
#             action_step = saved_policy.action(time_step)
#             time_step = eval_env.step(action_step.action)

# Environment difficulty 1 (aggressive / hard) - 4 (easy)
# WatchEnv = WatchEnv(4)
from Agents.C51Run import C51, watch
from Environments.BasicEnv import KerduGameEnv as NormalEnv
from Environments.AssumedDefenceEnv import KerduGameEnv as AssistedEnv
from Environments.PVAEnv import KerduGamePVN as PlayerVsAgent
from Environments.GymEnv import KerduGym as Gym

game_env = Gym()
print("12 ========================================================")
twelve_net = C51(100000, 52, -100, 100, 1)
twelve_net.train(game_env, "Gym52")
twelve_net.viewPlot()
