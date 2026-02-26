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

    def update_target(self):
        self.reload_params_to_target()

    def transition_update(self, transition):
        if self.cfg.replay:
            self.memory.push(*transition)
        else:
            self.memory = transition
        
    def episode_update(self, episode, stat=None):
        if self.cfg.replay:
            if len(self.memory) < self.cfg.batch_size:
                return
            transitions = self.memory.sample(self.cfg.batch_size)
            batch = self.Transition(*zip(*transitions))
            self.update(batch, episode, stat)
        else:
            self.update(self.memory, episode, stat)
    
    def construct_model(self):
        raise NotImplementedError
    
    def policy(self, state, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        raise NotImplementedError
    
    def value(self, state, act, last_act=None, last_hid=None):
        raise NotImplementedError
    
    def construct_policy_net(self):
        if self.cfg.agent_id:
            input_shape = self.state_dim + self.n
        else:
            input_shape = self.state_dim
            
        if self.cfg.agent_type == "mlp":
            if self.cfg.gaussian_policy:
                from agents.mlp_agent_gaussian import MLPAgent
            else:
                from agents.mlp_agent import MLPAgent
                
            self.policy_net = MLPAgent(input_shape, self.cfg)
        else:
            raise NotImplementedError
        
        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([Agent(input_shape, self.cfg)])
        else:
            self.policy_dicts = nn.ModuleList(
                [Agent(input_shape, self.cfg) for _ in range(self.n)]
            )
        
    def construct_value_net(self):
        raise NotImplementedError
    
    def init_weights(self, m):
        if not isinstance(m, nn.Linear):
            return
        if self.cfg.init_type == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif self.cfg.init_type == "orthogonal":
            nn.init.orthogonal_(
                m.weight, gain=nn.init.calculate_gain(self.cfg.hid_activation)
            )
        
    def get_actions(self, state, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        raise NotImplementedError
    
    def det_loss(self, state, act, last_act=None, last_hid=None):
        raise NotImplementedError
    
    def train_process(self, stat, episode_i, trainer, hyperparams={}):
        stat_train = {
            'mean_train_reward': 0, 
            'mean_train_solver_infeasible': 0,
            'mean_train_solver_interventions': 0
        }
    
    
class ATLA(Model):
    def __init__(self, args, target_net=None):
        super(ATLA, self).__init__(args)
        self.construct_model()

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

