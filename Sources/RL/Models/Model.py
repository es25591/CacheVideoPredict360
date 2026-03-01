import torch
import torch.nn as nn
from collections import namedtuple

from RL.TrainerCore import EpisodeRunner


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.n = cfg.n_agents
        self.hidden_dim = cfg.hidden_dim
        self.state_dim = cfg.state_dim_meta
        self.action_dim = cfg.action_dim_meta

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
    
    def construct_policy_net(self):
        raise NotImplementedError
        
    def construct_value_net(self):
        raise NotImplementedError
    
    def policy(self, state, schedule=None, last_act=None, last_hid=None, info=None, stat=None):
        raise NotImplementedError
    
    def value(self, state, act, last_act=None, last_hid=None):
        raise NotImplementedError

    def forward(self, state, act=None, last_act=None, last_hid=None, info=None, stat=None):
        if act is None:
            return self.policy(state, schedule=None, last_act=last_act, last_hid=last_hid, info=info, stat=stat)
        else:
            return self.value(state, act, last_act=last_act, last_hid=last_hid)

    def init_weights(self, m):
        if not isinstance(m, nn.Linear):
            return
        if self.cfg.init_type == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
        elif self.cfg.init_type == "orthogonal":
            nn.init.orthogonal_(
                m.weight, gain=nn.init.calculate_gain(self.cfg.hid_activation)
            )
        
    def get_actions(self, state, schedule=None, last_act=None, last_hid=None, info=None, stat=None):
        raise NotImplementedError
    
    def det_loss(self, state, act, last_act=None, last_hid=None):
        raise NotImplementedError
    
    def train_process(self, stat, episode_i, trainer, hyperparams=None):
        if hyperparams is None:
            hyperparams = {}

        stat_train = {
            'mean_train_reward': 0, 
            'mean_train_solver_infeasible': 0,
            'mean_train_solver_interventions': 0
        }
        
        print(f"Episode {episode_i} training...")

        episode = []

        device = getattr(self.cfg, 'device', next(self.parameters()).device if any(True for _ in self.parameters()) else 'cpu')
        init_last_hid = None
        if getattr(self, 'hidden_dim', None):
            init_last_hid = torch.zeros(self.n, self.hidden_dim).to(device)

        runner = EpisodeRunner(
            env=trainer.env,
            max_steps=int(hyperparams.get('max_steps', self.cfg.max_steps)),
        )

        def action_selector(state, schedule=None, last_act=None, last_hid=None, info=None, stat=None):
            return self.get_actions(
                state,
                schedule=schedule,
                last_act=last_act,
                last_hid=last_hid,
                info=info or {},
                stat=stat or {},
            )

        def transition_handler(transition, episode_idx, _step_idx):
            if getattr(self.cfg, 'episodic', False):
                episode.append(transition)
                return

            self.transition_update(transition)
            self.episode_update(episode_idx, stat)

        result = runner.run(
            episode_i=episode_i,
            stat=stat,
            schedule=getattr(trainer, 'schedule', None),
            action_selector=action_selector,
            transition_handler=transition_handler,
            last_hidden=init_last_hid,
            last_action=None,
        )

        stat_train.update(result.stats)

        if getattr(self.cfg, 'episodic', False):
            self.memory = episode
            self.episode_update(episode_i, stat)

        return stat_train