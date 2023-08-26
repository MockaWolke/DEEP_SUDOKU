import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

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
        
        # row is not full
        y_mask = (observations == 0).any(2)
        # column is not full
        x_mask = (observations == 0).any(1)
        
        return y_mask, x_mask, []
            

    def get_action_and_value(self, obs, action=None):

        value, logits = self.get_value_and_logits(obs)

        if self.mask_actions:

            action_masks = self.get_action_mask(obs)

            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(logits, action_masks)
            ]
            
        else:
            
            multi_categoricals = [Categorical(logits=logits) for logits in logits]
            
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        
        return action.T, logprob.sum(0), entropy.sum(0), value


class SplitMLP(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(SplitMLP, self,).__init__(mask_actions)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(81, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(81, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 9 * 3), std=0.01),
        )
        
    def get_value(self, obs):

        obs = obs.reshape(-1, 81) - 4.5 / 4.5
        obs = obs.float()

        return self.critic(obs)
        
    def get_value_and_logits(self, obs):
        
        obs = obs.reshape(-1, 81) - 4.5 / 4.5
        obs = obs.float()

        values = self.critic(obs)
        
        logits = torch.split(self.actor(obs), 9, dim=1)
        
        return values, logits
        

class SplitMLPOnehot(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(SplitMLPOnehot, self).__init__(mask_actions)
        
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
            layer_init(nn.Linear(256, 9 * 3), std=0.01),
        )
        
    def get_value(self, obs):

        obs = obs.to(torch.int64).reshape(-1,81)
        obs = torch.nn.functional.one_hot(obs,10).reshape(-1, 810)
        obs = obs.float()

        return self.critic(obs)
        
    def get_value_and_logits(self, obs):

        obs = obs.to(torch.int64).reshape(-1,81)
        obs = torch.nn.functional.one_hot(obs, 10).reshape(-1, 810)
        obs = obs.float()

        values = self.critic(obs)
        
        logits = torch.split(self.actor(obs), 9, dim=1)
        
        return values, logits
        
    
class ConvActor(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(ConvActor, self,).__init__(mask_actions)
    
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 16, kernel_size=3, padding = 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=9, padding = 4)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 9 * 9, 128)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(128, 9 * 3), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_value(self, obs):

        obs = obs[:,:,:,None] - 4.5 / 4.5
        obs = obs.float().permute(0, 3, 2, 1)
        
        print(obs.shape)
        return self.critic(self.network(obs))
        
    def get_value_and_logits(self, obs):
        
        obs = obs[:,:,:,None] - 4.5 / 4.5
        obs = obs.float().permute(0, 3, 2, 1)

        hidden = self.network(obs)

        values = self.critic(hidden)
        
        logits = torch.split(self.actor(hidden), 9, dim=1)
        
        return values, logits
        
        
class ConvActorOnehot(AgentBarebone):
    
    def __init__(self, mask_actions):
        
        super(ConvActorOnehot, self).__init__(mask_actions)
    
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(10, 16, kernel_size=3, padding = 1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=9, padding = 4)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 9 * 9, 128)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(128, 9 * 3), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)


    def get_value(self, obs):

        obs = torch.nn.functional.one_hot(obs.to(torch.int64), 10)
        obs = obs.float().permute(0, 3, 2, 1)

        return self.critic(self.network(obs))
        
    def get_value_and_logits(self, obs):
        
        obs = torch.nn.functional.one_hot(obs.to(torch.int64), 10)
        obs = obs.float().permute(0, 3, 2, 1)

        hidden = self.network(obs)

        values = self.critic(hidden)
        
        logits = torch.split(self.actor(hidden), 9, dim=1)
        
        return values, logits
        