import os
import gymnasium
import gymnasium.spaces.multi_discrete
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from gymnasium import spaces


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        # self.actor = nn.Sequential(
        #         nn.Linear(*input_dims, fc1_dims),
        #         nn.ReLU(),
        #         nn.Linear(fc1_dims, fc2_dims),
        #         nn.ReLU(),
        #         nn.Linear(fc2_dims, n_actions),
        #         nn.Softmax(dim=-1)
        # )
        self.lnn1 = nn.Linear(8, 256)
        self.lnn1a = nn.Linear(256, 256)
        self.lnn1b = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.lnn2 = nn.Linear(64, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"device: {self.device}")
        self.to(self.device)

    def forward(self, state):
        out = self.lnn1(state)
        out = self.relu(out)
        out = self.lnn1a(out)
        out = self.relu(out)
        out = self.lnn1b(out)
        out = self.relu(out)
        out = self.lnn2(out)
        
        
        y = out
        #y = F.softmax(y, dim=-1)
        # dist = self.actor(state)
        # dist = Categorical(dist)
        # Discrete = Categorical(F.softmax(y, dim=-1))
        # Continous = T.distributions.Normal(y, 0.2) where 0.2 is std deviation
        return T.distributions.Normal(y, 0.2)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        # self.critic = nn.Sequential(
        #         nn.Linear(*input_dims, fc1_dims),
        #         nn.ReLU(),
        #         nn.Linear(fc1_dims, fc2_dims),
        #         nn.ReLU(),
        #         nn.Linear(fc2_dims, 1)
        # )
        self.lnn1 = nn.Linear(8, 64)
        self.lnn1a = nn.Linear(64, 64)
        self.lnn1b = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.lnn2 = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        #value = self.critic(state)
        out = self.lnn1(state)
        out = self.relu(out)
        out = self.lnn1a(out)
        out = self.relu(out)
        out = self.lnn1b(out)
        out = self.relu(out)
        out = self.lnn2(out)
        #y = F.softmax(y, dim=-1)
        y = out
        return y
    
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    # n_actions = actions before update
    def __init__(self, n_actions, input_dims, num_envs, action_space, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.num_envs = num_envs
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.env_discrete_action_space = (type(action_space) == gymnasium.spaces.Discrete or type(action_space) == gymnasium.spaces.multi_discrete.MultiDiscrete)
        print(f"Env is using {type(action_space)} action space")

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
    def choose_action_discrete(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action))
        # Discrete action here!!
        action = T.squeeze(action)
        value = T.squeeze(value)

        return action.cpu().detach().numpy(), probs.cpu().detach().numpy(), value.cpu().detach().numpy()
    def choose_action_continous(self, observation):
        #print("observation: ", observation)
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action))
        action = T.squeeze(action)
        value = T.squeeze(value)

        return action.cpu().detach().numpy(), probs.cpu().detach().numpy(), value.cpu().detach().numpy()
    def choose_action(self, observation):
        if self.env_discrete_action_space:
            return self.choose_action_discrete(observation)
        else:
            return self.choose_action_continous(observation)
    def learn_discrete(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros([len(reward_arr), self.num_envs], dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1 - dones_arr[k] * 1.0) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
    def learn_continous(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros([len(reward_arr), self.num_envs], dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1 - dones_arr[k] * 1.0) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            advantage = T.unsqueeze(advantage, dim = -1)

            values = T.tensor(values).to(self.actor.device)
            values = T.unsqueeze(values, dim = -1)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                #critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        pass
    def learn(self):
        if self.env_discrete_action_space:
            self.learn_discrete()
        else:
            self.learn_continous()

