"""Code to visualize integradetd gradients"""
from captum.attr import IntegratedGradients
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class SoftMaxWrapper(torch.nn.Module):
    
    def __init__(self, agent):
        super().__init__()
        
        self.softmax = nn.Softmax(dim=1)
        self.agent = agent
        
            
    def forward(self, obs, original_image = None):
        
        log_probs = self.agent.actor(obs.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).reshape(-1, 9 ** 3)
        
        if original_image:     
            
            
            self.action_mask = self.agent.get_action_mask(torch.tensor(original_image)[None,:]).cuda()
            log_probs = torch.where(self.action_mask, log_probs, torch.tensor(-1e8).to(log_probs.device)).reshape(-1, 81, 9)
        
        log_probs = log_probs.reshape(-1, 81, 9)
        
        return self.softmax(log_probs).reshape(-1,  9 **3)
    
    
def get_valid_oberservation_action_pair(env, agent):
    
    obs, action, raveled, original_image = None, None, None, None

    while True:
        original_image, _ = env.reset()

        obs = torch.tensor(original_image).to("cuda")[None,:]

        raveled = agent.get_greedy_action(obs)[0].cpu().numpy()

        action = np.unravel_index(raveled, (9,9,9))

        reward = env.step(action)[1]

        if reward == 1:
            break
    
    return obs, action, raveled, original_image, 

def calculate_integrated_gradients(agent, obs, raveled):

    if not torch.is_tensor(obs):
        
        obs = torch.tensor(obs)
        
    if obs.shape[0] != 1:
        
        obs = torch.unsqueeze(obs, 0)
    
    assert obs.shape == (1,9,9)

    wrapper = SoftMaxWrapper(agent).cuda().float()

    obs = torch.nn.functional.one_hot(obs.to(torch.int64), 10)
    obs = obs.float().cuda()

    ig = IntegratedGradients(wrapper)
    attributions, approximation_error = ig.attribute(obs, target = torch.tensor([int(raveled)]),
                                        return_convergence_delta=True, 
                                        )

    atribution = attributions.cpu().detach().numpy().squeeze().sum(-1)

    atribution = np.maximum(atribution, 0)
    return atribution


def farm_and_viz_integrated_gradients(env,agent):

    obs, action, raveled, original_image = get_valid_oberservation_action_pair(env, agent)
        
    atribution = calculate_integrated_gradients(agent, obs, raveled)

    fig = plt.figure(figsize=(6,6 ), dpi = 100)

    field = original_image

    plt.imshow(atribution, "Blues")

    plt.axis("off")
    # Draw grid lines
    for i in range(10):
        if i % 3 == 0:
            plt.vlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=2)
            plt.hlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=2)
        else:
            plt.vlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=1)
            plt.hlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=1)

    # now plot the numbers of field
    for i in range(9):
        for j in range(9):
            if field[i, j] != 0:
                plt.text(
                    j,
                    i,
                    str(field[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                )

            if (i,j) == action[:2]:
                
                plt.text(
                    j,
                    i,
                    str(action[-1] + 1),
                    ha="center",
                    va="center",
                    color="maroon",
                    fontsize=16,
                )

    return fig

def get_winning_playthrough(agent, env):


    while True:

        obs, _ = env.reset()
        frames = [obs.copy()]
        actions = []

        terminated = False

        while not terminated:
            
            action = agent(obs, "unraveled")
            
            obs, _, terminated, _,_ = env.step(action)
            
            frames.append(obs.copy())
            actions.append(action)
            
            
            
        if (obs==0).any():
            continue
        
        return frames, actions

def get_animation_of_full_playthrough(agent, env):
    
    frames, actions = get_winning_playthrough(agent, env)

    raveled = [np.ravel_multi_index(action, (9,9,9)) for action in actions]

    atributions = [calculate_integrated_gradients(agent, obs, action) for obs, action in zip(frames[:-1], raveled)]

    fig = plt.figure(figsize=(6,6 ), )

    plt.axis("off")
    # Draw grid lines
    for i in range(10):
        if i % 3 == 0:
            plt.vlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=2)
            plt.hlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=2)
        else:
            plt.vlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=1)
            plt.hlines(i - 0.5, -0.5, 8.5, colors="black", linewidth=1)

    field = frames[-1]

    texts = []

    max_val = np.max(atributions)

    atributions = [atr / atr.max() * max_val for atr in atributions]


    img = plt.imshow(np.zeros_like(field), cmap="Blues", vmin=0, vmax=max_val)

    # now plot the numbers of field
    for i in range(9):
        for j in range(9):
            if frames[0][i, j] != 0:
                text = plt.text(
                    j,
                    i,
                    str(field[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                )
                
            else: 
                
                text = plt.text(
                    j,
                    i,
                    str(field[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                    visible = False
                )

            texts.append(text)
            
    def update(data):
        
        if data is None:
            return tuple()
        
        atribution, action, last_action = data
        img.set_data(atribution)
        
        if last_action:
            
            y,x = last_action[:-1]
            
            digit = texts[y*9+x]
            digit.set_color("black")
            
        y,x = action[:-1]
        
        digit = texts[y*9+x]
        digit.set_color("red")
        digit.set_visible(True)
        
        return img, tuple(texts)
        
    last_actions = [None] + actions[:-1]
        
    animation_frames = [a for a in zip(atributions, actions, last_actions)]
    animation_frames = [None, None] + animation_frames

    ani = FuncAnimation(fig, update, frames=animation_frames,)
    
    return ani