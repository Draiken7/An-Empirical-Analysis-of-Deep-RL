import torch
import numpy as np
import os
from tqdm import tqdm
import os


class BaseTrainer:
    """
    Abstract base class for all Trainers. 
    Handles shared state initialization, model checkpointing, and I/O.
    """
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        
        # Shared State Tracking
        self.reward_hist = []
        self.episodes_completed = 0
        self.current_env_rewards = np.zeros(self.config.NUM_ENVS)
        
        # --- DYNAMIC MODEL NAME EXTRACTION ---
        # Gets the literal class name (e.g., "DuelingDQNAgent")
        raw_name = self.agent.__class__.__name__
        
        # This turns "DuelingDQNAgent" into "DuelingDQN"
        self.model_name = raw_name.replace("Agent", "")

    def train(self):
        """Abstract method to be implemented by specific training architectures."""
        raise NotImplementedError("Child trainers must implement the train() method.")

    def save_checkpoint(self, filepath=None):
        """Saves the agent's neural network weights to disk."""
        if filepath is None:
            # 1. Find the absolute path of the directory containing trainer.py (your root folder)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. Safely join the root path with the models directory and filename
            filepath = os.path.join(base_dir, "models", f"{self.model_name}_best_agent.pth")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.agent.policy_net.state_dict(), filepath)
        print(f"Model successfully saved to {filepath}")

    def load_checkpoint(self, filepath=None):
        """Loads neural network weights from disk."""
        if filepath is None:
            # 1. Find the absolute path of the directory containing trainer.py (your root folder)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. Safely join the root path with the models directory and filename
            filepath = os.path.join(base_dir, "models", f"{self.model_name}_best_agent.pth")
            
        if os.path.exists(filepath):
            self.agent.policy_net.load_state_dict(torch.load(filepath))
            
            # If the agent has a target network (Value Agents), sync it immediately
            if hasattr(self.agent, 'target_net'):
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
                
            print(f"Model successfully loaded from {filepath}")
        else:
            print(f"Warning: No checkpoint found at {filepath}")
            
            
class ValueTrainer(BaseTrainer):
    """
    Trainer specific to Off-Policy, Value-Based algorithms (DQN, DDQN, Dueling).
    Manages Epsilon-Greedy exploration, Replay Buffers, and Target Network updates.
    """
    def __init__(self, env, agent, config):
        super().__init__(env, agent, config)
        self.steps = 0
        self.epsilon = self.config.EPSILON_START
        self.use_polyak = self.config.POLYAK

    def train(self):
        """
        Trains the value based agents and returns the reward history.
        """
        print(f"Starting Vectorized {self.model_name} Training on {self.config.ENV_NAME}...")
        states, _  = self.env.reset()

        with tqdm(total=self.config.EPISODES) as pbar:
            while self.episodes_completed < self.config.EPISODES:
                
                # 1. Batched Epsilon-Greedy Action Selection
                actions = self.agent.select_action(states, self.epsilon)
                
                # 2. Step Environments
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                dones = terminated | truncated
                
                self.steps += self.config.NUM_ENVS
                self.current_env_rewards += rewards
                
                # 3. Store transitions and handle completions
                for i in range(self.config.NUM_ENVS):
                    self.agent.replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    
                    if dones[i]:
                        self.episodes_completed += 1
                        self.reward_hist.append(self.current_env_rewards[i])
                        self.current_env_rewards[i] = 0
                        
                        # Decay epsilon per episode completed
                        self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)
                        pbar.update(1)
                
                # 4. Learn & Update Target Networks
                for _ in range(self.config.NUM_ENVS):
                    self.agent.learn()
                    
                    # Target Update Routing
                    if self.use_polyak:
                        self.agent.polyak_update_target_network()
                        
                # Hard update routing (if not using Polyak)
                if not self.use_polyak and (self.steps % self.config.TARGET_UPDATE_FREQ == 0):
                    self.agent.update_target_network()
                
                # 5. Advance states
                states = next_states

        print(f"{self.model_name} Training completed!")
        self.env.close()
        return self.reward_hist
 
    
class PolicyTrainer(BaseTrainer):
    """
    Trainer specific to On-Policy, Policy Gradient algorithms (REINFORCE).
    Manages pure stochastic action selection and chronological episodic updates.
    """
    def __init__(self, env, agent, config):
        super().__init__(env, agent, config)

    def train(self):
        """
        Trains the policy based agents and returns the reward history.
        """
        print(f"Starting Vectorized {self.model_name} Training on {self.config.ENV_NAME}...")
        states, _  = self.env.reset()

        with tqdm(total=self.config.EPISODES) as pbar:
            while self.episodes_completed < self.config.EPISODES:
                
                # 1. Batched Stochastic Action Selection
                actions, _ = self.agent.select_action(states)
                
                # 2. Step Environments
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                dones = terminated | truncated
                
                self.current_env_rewards += rewards
                
                # 3. Store chronological trajectories and learn on completion
                for i in range(self.config.NUM_ENVS):
                    self.agent.store_transition(i, states[i], actions[i], rewards[i], next_states[i], dones[i])
                    
                    if dones[i]:
                        # Policy Gradients learn exactly at the end of the episode
                        self.agent.learn(env_idx=i)
                        
                        self.reward_hist.append(self.current_env_rewards[i])
                        self.current_env_rewards[i] = 0
                        self.episodes_completed += 1
                        pbar.update(1)
                
                # 4. Advance states
                states = next_states

        print("Policy-Based Training completed!")
        self.env.close()
        return self.reward_hist
