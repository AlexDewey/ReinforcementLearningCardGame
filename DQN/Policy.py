class MyPolicy(tf_policy.Base):

    def __init__(self, network):
        # Takes 1 or more networks like
        # QPolicy, PPOPolicy, ActorPolicy, RandomPolicy, GreedyPolicy

    def _distribution(self, time_step, policy_state):
        logits, next_state = self._network(
            time_step.observation, policy_state
        )
        return PolicyStep(
            tfp.distribution.Categorical(logits),
            next_state,
            info={"logits": logits}
        )