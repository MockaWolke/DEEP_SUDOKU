"""Simple code to evaluate the agent"""
import torch
import deepsudoku.reinforcement_learning
from deepsudoku.reinforcement_learning.ensemble import *
import gymnasium as gym
from typing import Tuple
from deepsudoku.reinforcement_learning.agents import AgentBarebone
import numpy as np
import tqdm

def eval_one_round(env, agent : AgentBarebone) -> Tuple[int, int, bool]:
    
    obs, _ = env.reset()
    
    terminated = False
    episodic_reward = 0
    episode_length = 1
    
    while not terminated:
        
        action = agent(obs, get_action = "unraveled")
        
        obs, reward, terminated , _, _ = env.step(action)

        episodic_reward += reward

        if terminated:
            
            is_win = (obs == 0).sum() == 0
            
            return episodic_reward, episode_length, is_win
            
        episode_length += 1



def calculate_winrate(n_steps, env, agent):
    
    results = [eval_one_round(env, agent)[-1] for _ in tqdm.tqdm(range(n_steps))]
    
    return np.mean(results)
    