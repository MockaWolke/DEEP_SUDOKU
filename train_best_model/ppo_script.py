import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from experiments import Single_Action_Agents
import tqdm


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="Sudoku-nostop0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default= None,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")


    # ---------------------------------------------------------------------------------------------------------------------

    # Algorithm specific arguments
    parser.add_argument("--eval-freq", type=int, default=100,
        help="Frequency to evaluate")
    
    parser.add_argument("--eval-steps", type=int, default=100,
        help="How many rounds to evaluate")
    
    parser.add_argument("--ckpt-dir-load", type = str, default= None, help = "Which ckpt directory")
    
    parser.add_argument("--ckpt-freq", type = int, default= 1000, help = "Frequency of checkpoints")
    
    parser.add_argument("--mask-actions", type=bool, default=True,
        help="Whether to apply action masking")
    
    parser.add_argument("--agent", type=str, default="Single_Action_MLP_Onehot",
        help="Which agent")
    
    parser.add_argument("--upper-bound-missing-digits", type=int, default= None,
        help="Which agent")
    
    parser.add_argument("--use-random-starting-point", type = bool, default = True, help ="Wheter to generate sudkus newly")
    
    parser.add_argument("--cut-off-limit", type = int, default = 10, help ="When to cutoff")
    parser.add_argument("--win-reward", type = float, default = 3, help ="When to cutoff")
    parser.add_argument("--fail-penalty", type = float, default = 0.1, help ="When to cutoff")
    
    # ---------------------------------------------------------------------------------------------------------------------
    
    
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx):
    def thunk():
        
        
        env = gym.make(
            gym_id,
            difficulty="easy",
            upper_bound_missing_digist=args.upper_bound_missing_digits,
            use_random_starting_point = args.use_random_starting_point,
            cut_off_limit = args.cut_off_limit,
            win_reward = args.win_reward,
            fail_penatly = args.fail_penalty,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


def eval_greedy(agent):
    
     
    env = gym.make(
        "Sudoku-x2", render_mode="human", easy_fraq= 0, difficulty="easy", 
    )

    obs, _ = env.reset()
    obs = torch.tensor(obs)[None, :].float().to("cuda")

    terminated = False
    episodic_reward = 0
    episode_length = 1
    win = False

    while not terminated:
        action = agent.get_greedy_action(obs)[0]

        action = np.unravel_index(action.cpu().numpy(), (9, 9, 9))

        obs, reward, terminated, _, _ = env.step(action)

        episodic_reward += reward

        win = (obs == 0).sum() == 0

        if terminated:
            return episodic_reward, episode_length, win

        obs = torch.tensor(obs)[None, :].float().to("cuda")

        episode_length += 1



if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.gym_id,
                args.seed + i,
                i,
            )
            for i in range(args.num_envs)
        ]
    )

    agent = Single_Action_Agents[args.agent](args.mask_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = (
        torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        )
        .to(device)
        .to(torch.int32)
    )
    actions = torch.zeros(
        (
            args.num_steps,
            args.num_envs,
        )
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)



    if args.ckpt_dir_load is not None:
        
        print("Load checkpoint from:", args.ckpt_dir_load)
        
        checkpoint = torch.load(args.ckpt_dir_load)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['learning_rate'] != 0:
        
            optimizer.param_groups[0]["lr"] = checkpoint['learning_rate']
            
        else:
            
            print("Saved learning rate was 0. We will use:", args.learning_rate)
            
            optimizer.param_groups[0]["lr"] = args.learning_rate
            
        global_step = checkpoint['step_count']


    def check_point_model(path):
        
            torch.save({
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.param_groups[0]["lr"], 
            'step_count': global_step   
        }, path)

    chkpt_dir = f"ckpt/{run_name}"
    os.makedirs(chkpt_dir, exist_ok= True)
    

    until_ever = args.total_timesteps is None

    if not until_ever:
        num_updates = args.total_timesteps // args.batch_size

    else:
        num_updates = None

    highest_error = None

    try:
        update = 0
        

        while True:
            
            old_global_step = global_step
            start_time = time.time()
            
            # Annealing the rate if instructed to do so.
            if args.anneal_lr and not until_ever:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            
            if update % args.eval_freq == 0:
                test_rewards = []
                test_lengths = []
                win_rate = []

                for _ in tqdm.tqdm(range(args.eval_steps), desc="Evaluating Agent"):
                    episodic_reward, episode_length, win = eval_greedy(agent)
                    test_rewards.append(episodic_reward)
                    test_lengths.append(episode_length)
                    win_rate.append(win)

                writer.add_histogram(
                    "eval/episodic_length_hist", np.array(test_lengths), global_step
                )
                writer.add_histogram(
                    "eval/episodic_return_hist", np.array(test_rewards), global_step
                )
                writer.add_scalar(
                    "eval/avg_episodic_length", np.mean(test_lengths), global_step
                )
                writer.add_scalar(
                    "eval/avg_episodic_return", np.mean(test_rewards), global_step
                )
                writer.add_scalar("eval/avg_winrate", np.mean(win_rate), global_step)
                
                unique, counts = np.unique(test_lengths, return_counts= True)
                
                highest_error = 27 - unique[np.argmax(counts)] + 1
                
                writer.add_scalar("eval/highest_error", highest_error, global_step)
                
                

            if update % args.ckpt_freq == 0:
                
                check_point_model(os.path.join(chkpt_dir,f"{global_step}.pth"))
            
            
            average_return = []
            average_length = []

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.

                unraveled = np.array(
                    [np.unravel_index(i, (9, 9, 9)) for i in action.cpu().numpy()]
                )

                next_obs, reward, done, _, info = envs.step(unraveled)

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                    done
                ).to(device)

                if "final_info" in info:
                    for item in info["final_info"]:
                        if isinstance(item, dict) and "episode" in item.keys():
                            # print(f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")

                            average_return.append(item["episode"]["r"])
                            average_length.append(item["episode"]["l"])

            writer.add_scalar(
                "charts/avg_episodic_return", np.mean(average_return), global_step
            )
            writer.add_scalar(
                "charts/avg_episodic_length", np.mean(average_length), global_step
            )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + args.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + args.gamma
                            * args.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + args.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar(
                "charts/SPS", int((global_step - old_global_step) / (time.time() - start_time)), global_step
            )



            update += 1

            if update == num_updates:
                break

    except KeyboardInterrupt:
        print("Stopping now. Saving model")

    envs.close()
    writer.close()

    check_point_model(os.path.join(chkpt_dir,f"{global_step}.pth"))