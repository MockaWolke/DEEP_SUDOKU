import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        
        assert torch.is_tensor(logits), "Logits Must be provided ans as tensor"
        
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            
            self.device = logits.device
      
            
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)


class AgentBarebone(nn.Module):
    def __init__(self, mask_actions):
        super(AgentBarebone, self).__init__()

        self.mask_actions = mask_actions

    def get_value(self, obs):
        
        raise NotImplemented()
        
        # return self.critic(self.network(x))
    
    def get_value_and_logits(self, obs):
        
        raise NotImplemented()
    
    def get_action_mask(self, observations):
        
        observations = observations.reshape(-1, 81)
        
        mask = (observations == 0).repeat_interleave(9, dim = 1)
        
        
        return mask
            

    def get_action_and_value(self, obs, action=None):

        value, logits = self.get_value_and_logits(obs)

        if self.mask_actions:

            action_mask = self.get_action_mask(obs)

            probs = CategoricalMasked(logits=logits, masks=action_mask) 
            
            
        else:
            
            probs = Categorical(logits=logits)
            
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value
    
    
    def get_greedy_action(self, obs):
        
        value, logits = self.get_value_and_logits(obs)


        action_mask = self.get_action_mask(obs).to(logits.device)
        
        logits = torch.where(action_mask, logits, torch.tensor(-1e8).to(logits.device))
        
        return torch.argmax(logits, axis = -1)

    def get_action_probs(self, obs):

        value, logits = self.get_value_and_logits(obs)


        action_mask = self.get_action_mask(obs).to(logits.device)
        
        logits = torch.where(action_mask, logits, torch.tensor(-1e8).to(logits.device))
        
        return logits

        
class SeperateOnlyConv(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(SeperateOnlyConv, self).__init__(mask_actions)
    
        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(10, 16, kernel_size=3, padding = 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=9, padding = 4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 9, kernel_size=9, padding = 4), std=0.01),)
        
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(810, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )


    def get_value(self, obs):

        obs = F.one_hot(obs.to(torch.int64), 10)
        obs = obs.float()

        return self.critic(obs)
        
    def get_value_and_logits(self, obs : torch.Tensor):
        
        obs = F.one_hot(obs.to(torch.int64), 10)
        obs = obs.float()

        values = self.critic(obs)
        
        logits = self.actor(obs.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).reshape(-1, 9**3)
        
        
        return values, logits
        
        
class MLP(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(MLP, self).__init__(mask_actions)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(810, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(810, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 9**3), std=0.01),
        )
        
    def get_value(self, obs):

        obs = obs.to(torch.int64).reshape(-1,81)
        obs = F.one_hot(obs,10).float()

        return self.critic(obs.reshape(-1, 810))
        
    def get_value_and_logits(self, obs):

        obs = obs.to(torch.int64).reshape(-1,81)
        obs = F.one_hot(obs, 10).reshape(-1, 810).float()

        values = self.critic(obs.reshape(-1, 810))
        
        logits = self.actor(obs)
        
        return values, logits

class ConvwithFCHead(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(ConvwithFCHead, self).__init__(mask_actions)
    
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(10, 16, kernel_size=3, padding = 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=9, padding = 4)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 9 * 9, 128)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(128, 9 ** 3), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)


    def get_value(self, obs):

        obs = F.one_hot(obs.to(torch.int64), 10)
        obs = obs.float().permute(0, 3, 2, 1)

        return self.critic(self.network(obs))
        
    def get_value_and_logits(self, obs):
        
        obs = F.one_hot(obs.to(torch.int64), 10)
        obs = obs.float().permute(0, 3, 2, 1)

        hidden = self.network(obs)

        values = self.critic(hidden)
        
        logits = self.actor(hidden)
        
        return values, logits
    
    
        
class SharedOnlyConv(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(SharedOnlyConv, self).__init__(mask_actions)
    
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(10, 16, kernel_size=3, padding = 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=9, padding = 4)),
            nn.ReLU(),

        )
        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(32, 9, kernel_size=9, padding = 4), std=0.01),)
        
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(32 * 9 * 9, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1))


    def get_value(self, obs):

        obs = F.one_hot(obs.to(torch.int64), 10)
        obs = obs.float().permute(0, 3, 2, 1)

        return self.critic(self.network(obs))
        
    def get_value_and_logits(self, obs):
        
        obs = F.one_hot(obs.to(torch.int64), 10)
        obs = obs.float().permute(0, 3, 2, 1)

        hidden = self.network(obs)

        values = self.critic(hidden)
        
        logits = self.actor(hidden).permute(0, 3, 2, 1).reshape(-1, 9**3)
        
        
        return values, logits
    
    
    
class SudokuTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super(SudokuTransformer, self).__init__()
        
        # Create the positional embeddings
        xs = torch.arange(9).repeat(9)
        ys = torch.arange(9).repeat_interleave(9)

        blocks = torch.arange(3).repeat_interleave(3).repeat(3)
        blocks = torch.cat((blocks, blocks + 3, blocks + 6))
        
        self.positional_embedding = torch.cat((
            F.one_hot(xs, 9),
            F.one_hot(ys, 9),
            F.one_hot(blocks, 9),
        ), axis=1)
        
        # Transformation Layers
        self.to_embed_dim = nn.Linear(37, embed_dim)
        self.layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.bigger = nn.Linear(embed_dim, embed_dim * 2)
        self.smaller = nn.Linear(embed_dim * 2, embed_dim)
        self.output = nn.Linear(embed_dim, 9)
        self.relu = nn.ReLU()

    def forward(self, obs):

        to_batch_size = self.positional_embedding[None, :].repeat(obs.size(0),1,1).to(obs.device)
        obs = torch.cat((obs, to_batch_size), -1).float()

        # Transformer Layers
        x = self.to_embed_dim(obs)
        x = self.relu(x)
        x, _ = self.layer(x, x, x)
        x = self.relu(self.smaller(self.relu(self.bigger(x))))
        x, _ = self.layer1(x, x, x)
        x = self.output(x)
        
        return x
    

class TransformerAgent(AgentBarebone):
    
    def __init__(self,  mask_actions):
        
        super(TransformerAgent, self).__init__(mask_actions)
            
        self.critic = nn.Sequential(
            layer_init(nn.Linear(810, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
    
        self.actor = SudokuTransformer(128, 8)
    
    def get_value(self, obs):

        obs = obs.reshape(-1, 81)
        obs = F.one_hot(obs.to(torch.int64), 10)
        
        return self.critic(obs.float().reshape(-1, 810))
        
    def get_value_and_logits(self, obs):
        
        obs = obs.reshape(-1, 81)
        obs = F.one_hot(obs.to(torch.int64), 10)
        

        values = self.critic(obs.float().reshape(-1, 810))
        
        logits = self.actor(obs).reshape(-1, 9 ** 3)
        
        
        return values, logits
    