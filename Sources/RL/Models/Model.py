import torch
import torch.nn as nn
from collections import namedtuple


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.n = cfg.n_agents
        self.hidden_dim = cfg.hidden_dim
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim

        self.Transition = namedtuple(
            'Transition', ('state', 'action', 'reward', 'next_state', 'done')
        )

        self.batchnorm = nn.BatchNorm1d(self.n)
        self.cost_batchnorm = nn.BatchNorm1d(self.n)

    def reload_params_to_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.parameters()):
            target_param.data.copy_(param.data)

class ATLA(Model):
    def __init__(self, args, target_net=None):
        super(ATLA, self).__init__(args)
        self.construct_model()

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

