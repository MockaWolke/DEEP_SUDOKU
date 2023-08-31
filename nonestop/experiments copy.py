import deepsudoku.reinforcement_learning
from agents import Bigger, SeperateOnlyConv, CriticAsConv
import gymnasium as gym
import torch

Single_Action_Agents = {
    "SeperateOnlyConv" :SeperateOnlyConv,
    "OnlyConvSeperateValueBigger" :Bigger,
    "CriticAsConv" : CriticAsConv
}
    

if __name__ == "__main__":




    env = gym.make_vec('Sudoku-v0', 5, render_mode = "human", upper_bound_missing_digist = 5)
    obs,_ = env.reset()


    obs = torch.tensor(obs).to("cuda")

    for name, agent in Single_Action_Agents.items():
        print("Testing:", name)

        agent = agent(True).cuda()
        
        a = agent.get_action_mask(obs)

        
        value, logs = agent.get_value_and_logits(obs)

        assert value.shape == (5,1)
        assert logs.shape == (5,9**3)


        action, logs, entropy, values = agent.get_action_and_value(obs, )

        assert action.shape == (5,)
        assert logs.shape == entropy.shape
        assert logs.shape == (5,)
        assert values.shape == (5,1)

