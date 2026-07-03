import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# Get Replay Buffer from utils
from utils import ReplayBuffer

# Get Neural Nets from networks
from networks import DQN, DuelingQNetwork, MonteCarloReinforce, CriticNetwork


#█████████ Base Classes for Various RL implementations ██████████████████████████████████████████████████████████████████████████████████████████████████████
#========= Base Class for Q Value based agents ==============================================================================================================
# Abstract Class for common features
class ValueAgent:
    """ Abstract Class for Agent Framework for various Q Value based RL Algorithms with common function implementations."""
    def __init__(self, state_dim, action_dim, config):
        # Initialize values and Replay Buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = ReplayBuffer(size=config.BUFFER_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.lr = config.LEARNING_RATE
        self.gamma = config.GAMMA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tau = config.TAU
    
        # Following are algorithm based implementations
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
    def select_action(self, state, epsilon):        
        """Modified to support vectorized epsilon-greedy action selection"""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # Add batch dimension if it's a single state
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        num_envs = state_tensor.shape[0]
        actions = np.zeros(num_envs, dtype=int)
        
        # 1. Get the greedy actions for all environments at once
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            
        # 2. Roll the dice for each environment independently
        for i in range(num_envs):
            if np.random.rand() < epsilon:
                # Explore
                actions[i] = np.random.randint(0, self.action_dim)
            else:
                # Exploit
                actions[i] = greedy_actions[i]
                
        # Return a scalar if single env, else return the numpy array
        if num_envs == 1:
            return actions[0]
        return actions
    
    def learn(self):
        """Trains the policy network using a batch of transitions from the replay buffer. Shared optimization loop using the Template Method pattern"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 1. Sample and cast (Strict PyTorch Type Casting Firewall)
        states, actions, rewards, next_states, done = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        done = torch.as_tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 2. Compute current Q values
        curr_q_values = self.policy_net(states).gather(1, actions)
        
        # 3. Compute next Q values (DELEGATED TO CHILD CLASS)
        next_q_values = self._compute_next_q_values(next_states)
            
        # 4. Compute Bellman target Q values
        targ_q_values = rewards + (self.gamma * next_q_values * (1 - done))
        
        # 5. Compute Loss and Optimize
        loss = F.mse_loss(curr_q_values, targ_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
    def _compute_next_q_values(self, next_states):
        """Abstract method to calculate the next Q Values based on Implementation."""
        raise NotImplementedError("Child classes must implement this method")
    
    def update_target_network(self):
        """Updates the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def polyak_update_target_network(self):
        """Soft updates the target network via Polyak Averaging"""
        # Zip the parameters of both networks together
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            # Formula: Target = (Tau * Policy) + ((1 - Tau) * Target)
            # We use .data to ensure we are only updating the raw numbers, not the gradient graph
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )    
#============================================================================================================================================================

#========= Base Class for Policy based Agents ===============================================================================================================
class PolicyAgent:
    """ Abstract Class for On-Policy (Policy Gradient) RL Algorithms. """
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.GAMMA
        self.lr = config.LEARNING_RATE
        self.num_envs = config.NUM_ENVS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy gradients require strictly chronological trajectories, not random buffers
        self.memory = [[] for _ in range(self.num_envs)]
        
        # Placeholders to be initialized by child classes
        self.policy_net = None
        self.optimizer = None

    def select_action(self, state):
        """ Stochastic action selection using Categorical distributions (Shared by REINFORCE, A2C, PPO) """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        # We use no_grad() here because we will rebuild the computational graph during learn()
        with torch.no_grad():    
            action_probs = self.policy_net(state_tensor)
            
        action_dist = torch.distributions.Categorical(action_probs)
        actions = action_dist.sample()
        probs = action_dist.log_prob(actions)
        
        if actions.shape[0] == 1:
            return actions.item(), probs[0]
        return actions.cpu().numpy(), probs

    def store_transition(self, env_idx, state, action, reward, next_state, done):
        """ Stores chronological transitions for a specific environment """
        self.memory[env_idx].append((state, action, reward, next_state, done))

    def clear_memory(self, env_idx):
        """ Empties the trajectory memory after an update """
        self.memory[env_idx] = []

    def learn(self, env_idx):
        """ Abstract method for calculating policy gradients and updating weights """
        raise NotImplementedError("Child classes must implement the specific policy gradient math")   
#============================================================================================================================================================
#████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████



#█████████ Implementations of Value Based Algorithms █████████████████████████████████████████████████████████████████████████████████████████████████████████
#======= Deep Q Network ======================================================================================================================================
# Agent class that uses DQN to learn the optimal policy
class DQNAgent(ValueAgent):
    """Deep Q-Network Agent"""
    def __init__(self, state_dim, action_dim, config):
        # Initialize Parent class
        super().__init__(state_dim, action_dim, config)
        
        # Initialize Policy and target networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        
        # load Target Net with policy net weights and set it to evaluation mode only
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
      
    def _compute_next_q_values(self, next_states):
        """Standard DQN based next q value calculation"""
        with torch.no_grad():
            return self.target_net(next_states).max(1)[0].unsqueeze(1)
#============================================================================================================================================================

#========== Double DQN =======================================================================================================================================
# Agent class that uses DQN to learn the optimal policy
class DDQNAgent(ValueAgent):
    """Deep Q-Network Agent"""
    def __init__(self, state_dim, action_dim, config):
        # Initialize Parent class
        super().__init__(state_dim, action_dim, config)
        
        # Initialize Policy and target networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        
        # load Target Net with policy net weights and set it to evaluation mode only
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
    def _compute_next_q_values(self, next_states):
        """Double DQN implementation of next Q value calculation. Splits Action Selection (Policy) from Evaluation (Target)"""
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            return self.target_net(next_states).gather(1, next_actions)
#============================================================================================================================================================

#========== Dueling DQN ===================================================================================================================================== 
# Agent class that uses the Dueling architecture to learn optimal Q values and hence generate the optimal policy
class DuelingDQNAgent(DDQNAgent):
    """Dueling Deep Q-Network Agent. Inherits from the DDQN class."""
    def __init__(self, state_dim, action_dim, config):
        # Initialize Parent class
        super().__init__(state_dim, action_dim, config)
        
        # Initialize Policy and target networks
        self.policy_net = DuelingQNetwork(state_dim, action_dim, config.DUELING_OPT).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, config.DUELING_OPT).to(self.device)
        
        # load Target Net with policy net weights and set it to evaluation mode only
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
#============================================================================================================================================================
#████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████



#█████████ Implementations of Policy Based Algorithms ███████████████████████████████████████████████████████████████████████████████████████████████████████
#========== Monte Carlo REINFORCE ===========================================================================================================================
class REINFORCEAgent(PolicyAgent):
    """ Monte Carlo REINFORCE Agent with optional Baselines """
    def __init__(self, state_dim, action_dim, config):
        # 1. Call parent to handle state, memory arrays, and device setup
        super().__init__(state_dim, action_dim, config)
        self.baseline = config.MONTECARLO_BASELINE_OPT  # 0: No Baseline, 1: Mean Normalization, 2: Critic Network

        # 2. Initialize Actor network
        self.policy_net = MonteCarloReinforce(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 3. Initialize Critic network if baseline is active
        if self.baseline > 1:
            self.critic = CriticNetwork(state_dim).to(self.device)
            self.optimizer_c = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

    def learn(self, env_idx):
        """ Implements the specific Monte Carlo Return (G_t) logic for REINFORCE """
        if len(self.memory[env_idx]) == 0:
            return

        # 1. Unpack chronological memory
        states, actions, rewards, next_states, dones = zip(*self.memory[env_idx]) 
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # 2. Re-build the computational graph for the entire trajectory
        action_probs = self.policy_net(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions) 
        
        # 3. Calculate Monte Carlo Returns (G_t) backwards
        returns = []
        G = 0
        for r in reversed(rewards.tolist()):
            G = r + self.gamma * G 
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # 4. Calculate Advantage and Loss based on Baseline selection
        if self.baseline == 0:
            # No baseline
            actor_loss = -torch.sum(log_probs * returns)
            
        elif self.baseline == 1:
            # Mean normalization baseline
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            actor_loss = -torch.sum(log_probs * returns)
            
        else:           
            # Critic neural network baseline
            values = self.critic(states).squeeze()
            advantage = returns - values.detach()
            actor_loss = -(advantage * log_probs).mean()
            
            # Critic Update via TD(0)
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze()
                td_targets = rewards + self.gamma * next_values * (1 - dones)
            
            critic_loss = F.mse_loss(values, td_targets)
            
            self.optimizer_c.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.optimizer_c.step()
            
        # 5. Optimize the Actor policy network
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 6. Clear chronological memory via the parent class method
        self.clear_memory(env_idx)
#============================================================================================================================================================
#████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████