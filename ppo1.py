import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from collections import deque
import random
import time
from stock_trading_env import StockTradingEnv
import torch.multiprocessing as mp
import os

# Metal specific optimizations for Apple Silicon
def setup_metal_optimizations():
    if torch.backends.mps.is_available():
        # Print Metal information
        print(f"MPS (Metal Performance Shader) is available")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: mps")
        return True
    else:
        print("MPS not available. Using CPU.")
        return False

class TensorBatch:
    """Efficient batch processing with memory management"""
    def __init__(self, device):
        self.device = device
        self.tensors = {}
        
    def add(self, name, tensor):
        if isinstance(tensor, np.ndarray):
            # Convert numpy array to float32 before creating tensor
            tensor = tensor.astype(np.float32)
            self.tensors[name] = torch.from_numpy(tensor).to(self.device)
        else:
            # If it's already a tensor, ensure it's float32
            self.tensors[name] = tensor.float().to(self.device)
            
    def get(self, name):
        return self.tensors.get(name)
        
    def clear(self):
        self.tensors.clear()

class ParallelEnvs:
    def __init__(self, env_fn, num_envs=4):
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]
        
    def reset(self):
        return np.stack([env.reset() for env in self.envs])
        
    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
        
    def close(self):
        for env in self.envs:
            env.close()

class Memory:
    def __init__(self, batch_size=256):  # Increased batch size for better GPU utilization
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.batch_size = batch_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tensor_batch = TensorBatch(self.device)
        
    def push(self, states, actions, rewards, next_states, dones, values, log_probs):
        # Handle batch inputs with efficient memory management
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.next_states.extend(next_states)
        self.dones.extend(dones)
        self.values.extend(values)
        self.log_probs.extend(log_probs)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.tensor_batch.clear()
        
    def get_batch(self):
        batch_start = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        # Convert to tensors efficiently using TensorBatch, ensuring float32 dtype
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)  # actions should be long
        rewards = np.array(self.rewards, dtype=np.float32)
        next_states = np.array(self.next_states, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        log_probs = np.array(self.log_probs, dtype=np.float32)
        
        self.tensor_batch.add('states', states)
        self.tensor_batch.add('actions', actions)
        self.tensor_batch.add('rewards', rewards)
        self.tensor_batch.add('next_states', next_states)
        self.tensor_batch.add('dones', dones)
        self.tensor_batch.add('values', values)
        self.tensor_batch.add('log_probs', log_probs)
        
        return (self.tensor_batch.get('states'),
                self.tensor_batch.get('actions'),
                self.tensor_batch.get('rewards'),
                self.tensor_batch.get('next_states'),
                self.tensor_batch.get('dones'),
                self.tensor_batch.get('values'),
                self.tensor_batch.get('log_probs'),
                batches)

class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = 0
        self._var = 1  # Use private variable for variance
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self._var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self._var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self._var)
    
    @std.setter
    def std(self, value):
        self._var = value * value  # Store variance based on provided std

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):  # Increased network capacity
        super(ActorCritic, self).__init__()
        
        # Shared layers with larger capacity
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights with optimized gain
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.001, max_grad_norm=0.5,
                 batch_size=256, epochs=10):
        # Setup Metal if available, otherwise use CPU
        self.use_metal = setup_metal_optimizations()
        self.device = torch.device("mps" if self.use_metal else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        
        # Use Adam optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=lr,
            eps=1e-5
        )
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.memory = Memory(batch_size)
        self.reward_scaler = RunningMeanStd()
        
    def select_action(self, states):
        with torch.no_grad():
            # Convert states to numpy array first if it's a list
            if isinstance(states, list):
                states = np.array(states)
            states = torch.FloatTensor(states).to(self.device)
            
            # Add batch dimension if not present
            if states.dim() == 1:
                states = states.unsqueeze(0)
                
            action_probs, values = self.actor_critic(states)
            dist = Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
        return (actions.cpu().numpy(), 
                values.cpu().numpy().reshape(-1),  # Ensure values is 1D
                log_probs.cpu().numpy())
    
    def scale_reward(self, reward):
        reward_array = np.array([reward], dtype=np.float32)
        self.reward_scaler.update(reward_array)
        return reward / (self.reward_scaler.std + 1e-8)
    
    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        next_value = next_value.to(self.device)
        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
            
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
            
        returns = advantages + values
        return returns, advantages
    
    def update(self):
        states, actions, rewards, _, dones, values, old_log_probs, batches = self.memory.get_batch()
        
        # Scale rewards
        rewards = torch.tensor([self.scale_reward(r) for r in rewards.cpu().numpy()], device=self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            _, next_value = self.actor_critic(states[-1].unsqueeze(0))
            
        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.epochs):
            for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_probs, values = self.actor_critic(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate losses
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * ((values.squeeze() - batch_returns) ** 2).mean()
                entropy_loss = -self.entropy_coef * entropy
                
                total_loss = actor_loss + self.value_coef * critic_loss + entropy_loss
                
                # Optimize
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.memory.clear()

def train(env, agent, num_episodes=1000, max_steps=1000):
    # Create parallel environments
    num_envs = min(mp.cpu_count(), 4)
    parallel_envs = ParallelEnvs(lambda: StockTradingEnv(env.df.copy()), num_envs)
    
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(0, num_episodes, num_envs):
        start_time = time.time()
        states = parallel_envs.reset()
        episode_rewards_batch = np.zeros(num_envs)
        
        for step in range(max_steps):
            actions, values, log_probs = agent.select_action(states)
            next_states, rewards, dones, _ = parallel_envs.step(actions)
            
            agent.memory.push(states, actions, rewards, next_states, dones, values, log_probs)
            episode_rewards_batch += rewards
            
            if len(agent.memory.states) >= agent.memory.batch_size:
                agent.update()
            
            if all(dones):
                break
                
            states = next_states
        
        episode_rewards.extend(episode_rewards_batch)
        
        # Save best model
        current_best = max(episode_rewards_batch)
        if current_best > best_reward:
            best_reward = current_best
            torch.save({
                'model_state_dict': agent.actor_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'episode': episode,
                'best_reward': best_reward,
                'reward_scaler_mean': agent.reward_scaler.mean,
                'reward_scaler_std': agent.reward_scaler.std
            }, 'best_model_ppo1LLBS.pth')
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-num_envs:])
            print(f'Episode {episode}, Average Reward: {avg_reward:.2f}, Best Reward: {best_reward:.2f}')
            print(f'Episode Time: {time.time() - start_time:.2f}s')
    
    # Save rewards data
    np.savetxt('training_rewards_ppo1LLBS.txt', episode_rewards)
    
    parallel_envs.close()
    return episode_rewards

def main():
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    # Enable parallel processing for numpy
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count())
    
    # Load and preprocess data
    df = pd.read_csv('LLBS.csv')
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Create environment
    env = StockTradingEnv(train_df)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.0001,
        max_grad_norm=0.5,
        batch_size=256,
        epochs=10
    )
    
    # Train agent
    start_time = time.time()
    rewards = train(env, agent)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_rewards_ppo1LLBS.png')
    plt.show()

if __name__ == "__main__":
    main()
