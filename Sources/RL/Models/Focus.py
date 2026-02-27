import torch
import torch.nn as nn

from Sources.RL.Models.Model import Model
from Sources.RL.Models.AttentionCritic import AttentionCritic

class Focus(Model):
    def __init__(self, cfg, target_net=None):
        super(Focus, self).__init__(cfg)
        self.construct_model()

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    
    def construct_value_net(self):
        self.value_dicts = nn.ModuleList([
            AttentionCritic(self.cfg.state_dim, self.cfg.action_dim, self.cfg)
        ])

    def construct_policy_net(self):
        if self.cfg.agent_ids:
            input_shape = self.state_dim + self.n
        else:
            input_shape = self.state_dim

        if self.cfg.agent_type == 'mlp':
            from agents.mlp_agent_gaussian import MLPAgent
            Agent = MLPAgent
        else:
            NotImplementedError()

        if self.cfg.shared_params:
            self.policy_dicts = nn.ModuleList([Agent(input_shape, self.cfg)])
        else:
            self.policy_dicts = nn.ModuleList([Agent(input_shape, self.cfg) for _ in range(self.n)])