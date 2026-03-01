import torch
import torch.nn as nn

from Sources.RL.Models.Model import Model
from Sources.RL.Models.AttentionCritic import AttentionCritic

class Focus(Model):
    def __init__(self, cfg, target_net=None):
        super(Focus, self).__init__(cfg)
        self.construct_model()
        
        self.apply(self.init_weights)

        if target_net is not None:
            self.target_net = target_net
            self.reload_params_to_target()
            
        self.batchnorm = nn.BatchNorm1d(self.n)

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

        print(f"Constructed Focus model with state_dim={self.state_dim} and action_dim={self.action_dim}")

    def construct_value_net(self):
        print(f"Constructing value network with state_dim={self.state_dim} and action_dim={self.action_dim}")
        self.value_dicts = nn.ModuleList([
            AttentionCritic(self.cfg.state_dim, self.cfg.action_dim, self.cfg)
        ])

    def construct_policy_net(self):
        print(f"Constructing policy network with state_dim={self.state_dim}")
        input_shape = self.state_dim
        self.policy_dicts = nn.ModuleList([MLPAgent(input_shape, self.cfg)])

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.cfg.init_type == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            elif self.cfg.init_type == "orthogonal":
                nn.init.orthogonal_(
                    m.weight, gain=nn.init.calculate_gain(self.cfg.hid_activation)
                )

    def get_action(
        self, 
        state, 
        status, 
        exploration, 
        actions_avail, 
        target=False, 
        last_hid=None
    ):
        target_policy = self.target_net.policy if self.cfg.target else self.policy
        
        return target_policy(state, schedule=None, last_act=None, last_hid=last_hid, info=None, stat=None)


class MLPAgent(nn.Module):
    def __init__(self, input_shape, cfg):
        super(MLPAgent, self).__init__()
        self.cfg = cfg

        # Easiest to reuse hid_size variable
        self.fc1 = nn.Linear(input_shape, self.cfg.hidden_size)

        if self.cfg.layernorm:
            self.layernorm = nn.LayerNorm(self.cfg.hidden_size)

        self.fc2 = nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size)
        self.mean = nn.Linear(self.cfg.hidden_size, self.cfg.action_dim)
        self.log_std = nn.Linear(self.cfg.hidden_size, self.cfg.action_dim)

        if self.cfg.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.cfg.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.cfg.hidden_size).zero_()

    def forward(self, inputs, hidden_state):
        x = self.fc1(inputs)
        if self.cfg.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h = self.hid_activation(self.fc2(x))
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.tanh(log_std)
        log_std = self.cfg.LOG_STD_MIN + 0.5 * (self.cfg.LOG_STD_MAX - self.cfg.LOG_STD_MIN) * (log_std + 1)