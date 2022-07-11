import numpy as np

class BreakoutEnv(tf_agents.py_environment.PyEnvironment):
    def observation_specs(self): "Defines the Observations"
    def action_spects(self): "Defines the Actions"

    def _reset(self):

    def _step(self, action):
        "Apply the action and return the next time_step(reward, observation)"
        if is_final(self.state):
            return self.reset()
        observation, reward = self._apply_action(action)
        return TimeStep(observation, reward)