import gymnasium as gym
import tensorflow as tf
import numpy as np
import random


class PPO_MultiDiscrete_Environment_Wrapper:
    def __init__(self, environment_name, NUM_ENVS, action_mask=None, preprocessing_function=lambda x: x, **env_kwargs):
        self.num_envs = NUM_ENVS
        self.action_mask = action_mask
        # We use vectorized environments (Implementation Detail 1)
        #self.envs = envs = gym.vector.make('CartPole-v1', num_envs=NUM_ENVS)
        self.envs = gym.vector.make(environment_name, num_envs=NUM_ENVS, **env_kwargs)
        self.current_state, _ = self.envs.reset()
        self.preprocess = preprocessing_function

    def sample(self, model, epsilon = 1):
        old_observation = self.current_state
        q_values = model(self.preprocess(self.current_state)) #get q values for current state
        # Q_values are logits, we convert them to a categorical distribution to sample
        if self.action_mask:
            masked_logits = np.array([self.action_mask(obs_, logits_, epsilon = epsilon) for obs_, logits_ in zip(old_observation, q_values.numpy())])
        else:
            masked_logits = q_values

        # Implementing Multi-Discrete Action spaces:
        # masked_logits = [environments][action-spaces][logprobs]
        # We need to sample for each environment from all action-spaces
        # NOTE: This assumes that all action-spaces look like [0,1,2,3,...]
        # In particular, for the sudoku case, the third action space represents the number we want to fill in
        # This number will be one less than the actual digit that gets filled in
        # This is accounted for and implemented like this in our custom environment
        probs = tf.nn.softmax(masked_logits).numpy()
        action = np.array([[np.random.choice(np.arange(len(action_space)),p=action_space) for action_space in environment] for environment in probs])
        logprob = tf.cast(tf.reduce_sum(tf.math.log(tf.gather(probs, action, batch_dims=2)), axis=-1), np.float32).numpy()

        new_observation, reward, terminated, _, _ = self.envs.step(action)

        self.current_state = new_observation #update current state after environment did step
        return (old_observation, action, reward, new_observation, terminated, logprob)
    
    def collect_trajectories(self, model, length, epsilon = 1):
        old_obs, act, rew, new_obs, term, log_probs = self.sample(model, epsilon=epsilon)
        data = {"observations": np.expand_dims(old_obs, axis=1), 
                "actions": np.expand_dims(act, axis=1), 
                "rewards": rew, 
                "terminateds": term,
                "log_prob": log_probs}
        for i in range(length-1):
            old_obs, act, rew, new_obs, term, log_probs = self.sample(model, epsilon=epsilon)
            data["observations"] = np.column_stack((data["observations"], np.expand_dims(old_obs, axis=1)))
            data["actions"] = np.column_stack((data["actions"], np.expand_dims(act, axis=1)))
            data["rewards"] = np.column_stack((data["rewards"], rew))
            data["terminateds"] = np.column_stack((data["terminateds"], term))
            data["log_prob"] = np.column_stack((data["log_prob"], log_probs))
        return data, new_obs


class PPO_Discrete_Environment_Wrapper:
    def __init__(self, environment_name, NUM_ENVS, action_mask=None, preprocessing_function=lambda x: x, **env_kwargs):
        self.num_envs = NUM_ENVS
        self.action_mask = action_mask
        # We use vectorized environments (Implementation Detail 1)
        #self.envs = envs = gym.vector.make('CartPole-v1', num_envs=NUM_ENVS)
        self.envs = gym.vector.make(environment_name, num_envs=NUM_ENVS, **env_kwargs)
        self.current_state, _ = self.envs.reset()
        self.preprocess = preprocessing_function

    def sample(self, model, epsilon = 1):
        old_observation = self.current_state
        q_values = model(self.preprocess(self.current_state)) #get q values for current state

        # Q_values are logits, we convert them to a categorical distribution to sample
        if self.action_mask:
            masked_logits = np.array([self.action_mask(obs_, logits_, epsilon=epsilon) for obs_, logits_ in zip(old_observation, q_values.numpy())])
        else:
            masked_logits = q_values

        probs = tf.nn.softmax(masked_logits).numpy()
        action = np.array([np.random.choice(np.arange(len(environment)),p=environment) for environment in probs])
        logprob = tf.cast(tf.math.log(tf.gather(probs, action, batch_dims=1)), np.float32).numpy()

        new_observation, reward, terminated, _, _ = self.envs.step(action)

        self.current_state = new_observation #update current state after environment did step
        return (old_observation, action, reward, new_observation, terminated, logprob)
    
    def collect_trajectories(self, model, length, epsilon = 1):
        old_obs, act, rew, new_obs, term, log_probs = self.sample(model, epsilon=epsilon)
        data = {"observations": np.expand_dims(old_obs, axis=1), 
                "actions": np.expand_dims(act, axis=1), 
                "rewards": rew, 
                "terminateds": term,
                "log_prob": log_probs}
        for i in range(length-1):
            old_obs, act, rew, new_obs, term, log_probs = self.sample(model, epsilon=epsilon)
            data["observations"] = np.column_stack((data["observations"], np.expand_dims(old_obs, axis=1)))
            data["actions"] = np.column_stack((data["actions"], np.expand_dims(act, axis=1)))
            data["rewards"] = np.column_stack((data["rewards"], rew))
            data["terminateds"] = np.column_stack((data["terminateds"], term))
            data["log_prob"] = np.column_stack((data["log_prob"], log_probs))
        return data, new_obs



def PPO(env, pi, V, multi_discrete=False, STEPS_PER_TRAJECTORY = 50, GAMMA = 0.99, LAMBDA = 0.95, CLIP_RATIO = 0.2,
        MAX_GRAD_NORM = 0.5, TRAIN_EPOCHS = 2500, NUM_UPDATE_EPOCHS = 1, MINIBATCH_SIZE = 50, 
        LEARNING_RATE_START = 0.005, LEARNING_RATE_DECAY_PER_EPOCH = None):
    
    if LEARNING_RATE_DECAY_PER_EPOCH == None:
        LEARNING_RATE_DECAY_PER_EPOCH = LEARNING_RATE_START/TRAIN_EPOCHS
    # Implementation Detail 3: The Adam Optimizers Epsilon and beta parameters are already fine by default
    value_optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_START)
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_START)

    kl_approx = 0
    mean_rewards = 0
    eps = 0
    ########################################################################################################
    # 2: for k = 0, 1, 2, ... do
    for k in range(TRAIN_EPOCHS):
        print("epoch: ", k, " ; KL: ", kl_approx, " ; LR: ", pi_optimizer.learning_rate.numpy(), " ; MR: ", mean_rewards, " ; EPS: ", eps)
    ########################################################################################################


        ########################################################################################################
        # 3: Collect set of trajectories D_k = {τ_i} by running policy π_k = pi(θ_k) in the environment.
        print("Collection")
        eps = 1/(1+10*np.exp(-(-7.3*((k/TRAIN_EPOCHS)-0.9)))) #logistic decay
        D, ensuing_observation = env.collect_trajectories(pi, STEPS_PER_TRAJECTORY, epsilon = eps)
        mean_rewards = np.mean(D["rewards"])
        #D = {"observations": [environment][timestep][observation], 
        #     "actions": [environment][timestep][actions], 
        #     "rewards": [environment][timestep], 
        #     "terminateds": [environment][timestep]] }
        ########################################################################################################

        ########################################################################################################
        # 4: Compute rewards-to-go R̂_t
        # 5: Compute advantage estimates, Â_t (using any method of advantage estimation) based on the current value function V_Φ_k
        # We use Generalized Advantage Estimation, since spinning up uses it too (Implementation Detail 5)
        # Accordingly, we also do not really implement rewards-to-go, but use a TD(lambda) estimation where rewards_to_go = advantages + values
        # (derived from advantages = rewards_to_go - values)
        #
        #TODO: Can be made more efficient?
        values = np.zeros_like(D["rewards"], dtype=np.float32)
        for env_ind in range(len(D["observations"])):
            values[env_ind] = tf.reshape(V(env.preprocess(D["observations"][env_ind])), (-1))
        #
        advantages = np.zeros_like(D["rewards"], dtype=np.float32)
        # Most environments will not be terminated at the last step. We estimate the rest-reward after the last step
        # as the value function for the next observation.
        advantages[:,-1] = D["rewards"][:,-1] + GAMMA * tf.reshape(V(env.preprocess(ensuing_observation)), (-1)) * (1-D["terminateds"][:,-1]) - values[:,-1]
        for ind in reversed(range(STEPS_PER_TRAJECTORY-1)):
            # The GAE(t) = delta + (GAMMA*LAMBDA)*GAE(t+1) with delta = r_t + gamma * V(s_t+1) - V(s_t)
            # If a state s_t is terminal, V(s_t+1) should be disregarded, since the next state does not belong to the episode anymore
            delta = D["rewards"][:,ind] + GAMMA * values[:,ind+1] * (1-D["terminateds"][:,ind]) - values[:,ind]
            advantages[:,ind] = delta + GAMMA*LAMBDA*advantages[:,ind+1] * (1-D["terminateds"][:,ind])
        #
        # flatten data

        advantages = np.reshape(advantages, (-1, *advantages.shape[2:]))
        values = np.reshape(values, (-1, *values.shape[2:]))
        for key, val in D.items():
            D[key] = np.reshape(val, (-1, *val.shape[2:]))
        #
        rewards_to_go = advantages + values
        ########################################################################################################

        print("Tapework")

        # We minibatch for increasing the efficiency of the gradient ascent (PPO-implementation details nr. 6)
        batch_size = D["observations"].shape[0]
        batch_inds = np.arange(batch_size)
        for update_epoch in range(NUM_UPDATE_EPOCHS):
            np.random.shuffle(batch_inds)
            for minibatch_start in range(0, batch_size, MINIBATCH_SIZE):
                minibatch_end = min(minibatch_start + MINIBATCH_SIZE, batch_size-1)
                minibatch_inds = batch_inds[minibatch_start:minibatch_end]

                mb_obs = D["observations"][minibatch_inds]
                mb_acts = D["actions"][minibatch_inds]
                mb_old_logits = tf.gather(D["log_prob"], minibatch_inds)
                mb_advantages = tf.gather(advantages, minibatch_inds)
                # zero center advantages (Implementation Detail 7)
                mb_advantages = (mb_advantages - tf.reduce_mean(mb_advantages)) / (tf.math.reduce_std(mb_advantages) + 1e-8)

                ########################################################################################################
                # 6: Update the policy by maximizing the PPO-Clip objective
                with tf.GradientTape() as pi_tape:
                    if multi_discrete:
                    # calculating new logits from multi-discrete action space
                    # pi(D["observations"]) is now an array of [action][action_subspace][logprobs]
                    # D["actions"] is a array of [actions][action_subspace], where [action_subspace] contains the index of the chosen action
                    # We need to gather the logprobs of the chosen subactions and add them to get an array of shape [action]
                    # NOTE: Setting batch_dims=2 only allows for simply nested multi-discrete action spaces
                        new_logits = tf.gather(pi(env.preprocess(mb_obs)), mb_acts, batch_dims=2)
                        new_logits = tf.reduce_sum(new_logits, axis=-1)
                    else:
                        new_logits = tf.gather(pi(env.preprocess(mb_obs)), mb_acts, batch_dims=1)
                    # for ppo clip, we want pi(s,a)/pi_old(s,a) = exp(log(pi(s,a))-log(pi_old(s,a)))
                    logratios = new_logits - mb_old_logits
                    ratios = tf.math.exp(logratios)
                    surrogate_objective1 = ratios * mb_advantages
                    # Implementation Detail 8: clip surrogate objective
                    surrogate_objective2 = tf.clip_by_value(ratios, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * mb_advantages
                    pi_loss = -tf.reduce_mean(tf.minimum(surrogate_objective1, surrogate_objective2))
                pi_gradients = pi_tape.gradient(pi_loss, pi.trainable_variables) #get
                # Implementation Detail 11: clip gradients
                pi_gradients, _ = tf.clip_by_global_norm(pi_gradients, MAX_GRAD_NORM)
                pi_optimizer.apply_gradients(zip(pi_gradients, pi.trainable_variables)) #and apply gradients
                ########################################################################################################

                # Implementation Detail 12: Implementing KL divergence as debug variable:
                kl_approx = tf.reduce_mean((ratios - 1) - logratios).numpy()
                #kl_approx = tf.reduce_mean(ratios).numpy()

                ########################################################################################################
                # 7: Fit value function by regression on mean-squared error:
                with tf.GradientTape() as val_tape:
                    values = V(env.preprocess(mb_obs))
                    # Implementation Detail 9 not implemented, as it may hurt performance
                    value_loss = tf.reduce_mean(tf.square(values - rewards_to_go[minibatch_inds]))
                value_gradients = val_tape.gradient(value_loss, V.trainable_variables) #get
                # Implementation Detail 11 cont.: clip gradients
                value_gradients, _ = tf.clip_by_global_norm(value_gradients, MAX_GRAD_NORM)
                value_optimizer.apply_gradients(zip(value_gradients, V.trainable_variables)) #and apply gradients
                ########################################################################################################
        
        # Anneal Adam Learning Rate (Implementation Detail 4 cont.)
        pi_optimizer.learning_rate.assign(pi_optimizer.learning_rate - LEARNING_RATE_DECAY_PER_EPOCH)
        value_optimizer.learning_rate.assign(value_optimizer.learning_rate - LEARNING_RATE_DECAY_PER_EPOCH)

    return pi, V
