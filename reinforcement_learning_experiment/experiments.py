import deepsudoku.reinforcement_learning
from single_action_agents import SeperateOnlyConv, MLP, TransformerAgent, SharedOnlyConv, ConvwithFCHead, SpecialSoftmax
import gymnasium as gym
import torch
from reinforcement_learning_experiment.multi_action_agents import SplitMLP, MultiActionSeperateOnlyConv

Single_Action_Agents = {
    "SeperateOnlyConv" :SeperateOnlyConv,
    "SharedOnlyConv" :SharedOnlyConv,
    "MLP" : MLP,
    "TransformerAgent": TransformerAgent,
    "ConvwithFCHead": ConvwithFCHead,
    "SpecialSoftmax": SpecialSoftmax,
}
    
Multi_Action_AGENTS = { 
    'MultiActionMLP' : SplitMLP,
    "MultiActionSeperateOnlyConv": MultiActionSeperateOnlyConv,
}

if __name__ == "__main__":




    env = gym.make_vec('Sudoku-v0', 5, render_mode = "human", upper_bound_missing_digist = 5)
    obs,_ = env.reset()


    obs = torch.tensor(obs).to("cuda")

    print("Single Action Testing")

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
        
        single_obs = torch.randint(0, 10, (1,9,9)).cuda()
        
        value = agent.get_value(single_obs).reshape(1, -1)
        
        assert value.shape == (1,1), f"Print value shape {value.shape}"
        
        
        params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f"Number of parameters: {params}")
        
        params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
        
        if hasattr(agent, "network"):
            params += sum(p.numel() for p in agent.network.parameters() if p.requires_grad)
            
        print(f"Number of actor parameters: {params}\n\n")


    print("Multi Action Testing")

    for name, agent in Multi_Action_AGENTS.items():
        
        agent = agent(True).cuda()
        
        
        a = agent.get_action_mask(obs)

        
        value, logs = agent.get_value_and_logits(obs)

        assert value.shape == (5,1)
        assert len(logs) == 3
        assert logs[0].shape == (5,9)


        action, logs, entropy, values = agent.get_action_and_value(obs, )

        assert action.shape == env.action_space.sample().shape
        assert logs.shape == entropy.shape
        assert logs.shape == (5,)
        assert values.shape == (5,1)

        
        params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f"Number of parameters: {params}")
        
        
        params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
        
        if hasattr(agent, "network"):
            params += sum(p.numel() for p in agent.network.parameters() if p.requires_grad)
            
        print(f"Number of actor parameters: {params}\n\n")
        