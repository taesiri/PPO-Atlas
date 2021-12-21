import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy 
import math
from collections import namedtuple, deque
from torch.utils.data import TensorDataset, DataLoader

BUFFER_SIZE  = int(2e5)  # replay buffer size
BATCH_SIZE   = 128       # minibatch size
GAMMA        = 0.999     # discount factor
TAU1         = 1e-3      # for soft update of target parameters
TAU2         = 1e-3      # for soft update of target parameters
LR_ACTOR     = 2e-4      # learning rate 
LR_CRITIC    = 2e-4      # learning rate 
LR           = 2e-4      # learning rate 
UPDATE_EVERY = 4         # how often to update the network
PI = math.pi             # 3.1415...
ENTROPY_BETA = 1e-4
PPO_UPDATES  = 4         # Number of PPO updates per Step
ENTROPY_BETA = 0.001     # Entropy Multiplier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, actor_critic_network, seed=0):
    """ PPO Agent - Initialize an Agent object.
    
    Params
    ======
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      actor_critic_network (FCActorCriticBrain): Actor Critic Network
      seed (int): random seed
    """
    self.state_size  = state_size
    self.action_size = action_size
    self.seed        = random.seed(seed)
    
    self.actor_critic_network   = actor_critic_network
    self.optimizer              = optim.Adam(self.actor_critic_network.parameters(), lr=LR)

  def act(self, state):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (array_like): current state
    """
    state = torch.tensor(state).float().to(device)
    action_distribution, value = self.actor_critic_network(state)

    action = action_distribution.sample()
    log_prob = action_distribution.log_prob(action)
    entropy =  action_distribution.entropy().mean()

    return action, log_prob, entropy, value

  def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """Computes GAE """
    gae = 0
    values = values + [next_value]
    returns = []
    for step in reversed(range(len(rewards))):
      delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
      gae = delta + gamma * tau * masks[step] * gae
      returns.insert(0, gae + values[step])
    return returns

  def ppo_optimize(self, state, action, old_log_probs, return_, advantage, eps=0.2):
    """learn - Performs One step gradient descent on a batch of data"""
    dist, value = self.actor_critic_network(state)
    entropy = dist.entropy().mean()
    new_log_probs = dist.log_prob(action)

    ratio = (new_log_probs - old_log_probs).exp()
    surr = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantage

    actor_loss  = - torch.min(ratio * advantage, surr).mean()
    critic_loss = (return_ - value).pow(2).mean()

    loss = 0.5 * critic_loss + actor_loss - ENTROPY_BETA * entropy

    # ------------------- Optimize the Models ------------------- #
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()   

  def learn(self, states, actions, log_probs, returns, advantages):
    """PPO Update - Loop through all collected trajectories and Update the Network."""

    dataset = TensorDataset(states, actions, log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for _ in range(PPO_UPDATES):
      for batch_idx, (state, action, old_log_probs, return_, advantage) in enumerate(loader):
        self.ppo_optimize(state, action, old_log_probs, return_, advantage)

class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, action_size, buffer_size, batch_size, seed):
    """Initialize a ReplayBuffer object.

    Params
    ======
      action_size (int): dimension of each action
      buffer_size (int): maximum size of buffer
      batch_size (int): size of each training batch
      seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)  
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)
  
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
  
  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([1-e.done for e in experiences if e is not None])).float().to(device)

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)