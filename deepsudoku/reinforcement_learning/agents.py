"""Hold the code for the agents"""
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
import numpy as np
from deepsudoku import PACKAGE_PATH


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """PPO layer init"""
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
    """Wrapper around networks for simplicity."""
    
    def __init__(self, mask_actions, device):
        super(AgentBarebone, self).__init__()

        self.mask_actions = mask_actions
        self.device = device

    def get_value(self, obs):
        
        raise NotImplemented()
        
        # return self.critic(self.network(x))
    
    def get_value_and_logits(self, obs):
        
        raise NotImplemented()
    
    def get_action_mask(self, observations):
        """Get the action mask given sudoku field"""
        
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
        """Get action with highest probability"""
        
        value, logits = self.get_value_and_logits(obs)


        action_mask = self.get_action_mask(obs).to(logits.device)
        
        logits = torch.where(action_mask, logits, torch.tensor(-1e8).to(logits.device))
        
        return torch.argmax(logits, axis = -1)

    def get_action_probs(self, obs):
        """Get probability of  actions"""
        
        value, logits = self.get_value_and_logits(obs)


        action_mask = self.get_action_mask(obs).to(logits.device)
        
        logits = torch.where(action_mask, logits, torch.tensor(-1e8).to(logits.device))
        
        return logits

        
class BestConvModel(AgentBarebone):
    """Our best (OnlyConv) Agent"""
    def __init__(self, mask_actions, device = "cuda"):
        
        super(BestConvModel, self).__init__(mask_actions, device)
        # the actor and critic networks
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


    def forward(self, obs, get_action = None):
        """Bif forward function that returns actions in different formats given and obs"""
                
        assert get_action in [None, "raveled", "unraveled"]
        
        if not torch.is_tensor(obs):
            
            obs = torch.tensor(obs)
        
        if len(obs.shape) == 2:
            
            obs = obs.unsqueeze(0)
            
        obs = obs.to(self.device)
            
        assert obs.shape[1:] == (9,9), f"Wrong shape: {obs.shape}"

        action_mask = self.get_action_mask(obs).to(self.device)

        obs = torch.nn.functional.one_hot(obs.to(torch.int64), 10).float()

        logits = self.actor(obs.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).reshape(-1, 9**3)
        
        logits = torch.where(action_mask, logits, torch.tensor(-1e8).to(logits.device))

        if get_action is None:
            # get the probilies
            return logits.cpu()
        
        elif get_action == "raveled":
            
            
            # get the raveled actions
            raveled = torch.argmax(logits, -1).cpu().numpy().squeeze()
            
            return raveled
            
        
        else:
            
            raveled = torch.argmax(logits, -1).cpu().numpy().squeeze()
            
            # get unraveled
            return np.unravel_index(raveled, (9,9,9))
            

    def get_value(self, obs):

        obs = torch.nn.functional.one_hot(obs.to(torch.int64), 10)
        obs = obs.float()

        return self.critic(obs)
        
    def get_value_and_logits(self, obs : torch.Tensor):
        
        obs = torch.nn.functional.one_hot(obs.to(torch.int64), 10)
        obs = obs.float()

        values = self.critic(obs)
        
        logits = self.actor(obs.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).reshape(-1, 9**3)
        
        
        return values, logits
        
def get_sudoku_agent(device : str = "cuda") -> torch.nn.Module: 
    """Load our best agent"""        
     
    agent = BestConvModel(True, device)
    agent.load_state_dict(torch.load(PACKAGE_PATH /"reinforcement_learning/best_agent.pth")["model_state_dict"])
    
    if device == "cuda":
        agent = agent.cuda()
        
    return agent