from Sources.RL.Models.Model import Model


class Focus(Model):
    def __init__(self, cfg, target_net=None):
        super(Focus, self).__init__(cfg)
        self.construct_model()

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def construct_policy_net(self):
        raise NotImplementedError("Focus does not have a policy network")
    
    def construct_value_net(self):
        raise NotImplementedError("Focus does not have a value network")