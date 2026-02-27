import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionCritic(nn.Module):
    def __init__(self, state_size, action_dim, cfg):
        super(AttentionCritic, self).__init__()

        self.hidden_dim = cfg.hidden_dim
        self.attend_heads = cfg.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0

        self.sa_sizes = (state_size, action_dim)
        self.nagents = cfg.n_agents
        self.continuous = cfg.continuous
        
        sdim, adim = self.sa_sizes
        idim = sdim + adim
        if self.continuous:
            odim = 1
        else:
            odim = adim
        
        # s+a encoder
        encoder = nn.Sequential()
        if cfg.norm_in:
            encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
        encoder.add_module('enc_fc1', nn.Linear(idim, self.hidden_dim))
        encoder.add_module('enc_nl', nn.LeakyReLU())
        self.critic_encoder = encoder

        # critic
        critic = nn.Sequential()
        critic.add_module('critic_fc1', nn.Linear(2 * self.hidden_dim, self.hidden_dim))
        critic.add_module('critic_nl', nn.LeakyReLU())
        critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
        self.critic = critic

        # bias
        bias = nn.Sequential()
        bias.add_module('bias_fc1', nn.Linear(self.hidden_dim, self.hidden_dim))
        bias.add_module('bias_nl', nn.LeakyReLU())
        bias.add_module('bias_fc2', nn.Linear(self.hidden_dim, 1))
        self.bias = bias

        # s encoder
        state_encoder = nn.Sequential()
        if cfg.norm_in:
            state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(sdim, affine=False))
        state_encoder.add_module('s_enc_fc1', nn.Linear(sdim, self.hidden_dim))
        state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
        self.state_encoder = state_encoder

        attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList() 
        
        for _ in range(self.attend_heads):
            key_extractor = nn.Linear(self.hidden_dim, attend_dim, bias=False)
            selector_extractor = nn.Linear(self.hidden_dim, attend_dim, bias=False)
            value_extractor = nn.Sequential(
                nn.Linear(self.hidden_dim, attend_dim), nn.LeakyReLU()
            )

            self.key_extractors.append(key_extractor)
            self.selector_extractors.append(selector_extractor)
            self.value_extractors.append(value_extractor)

        