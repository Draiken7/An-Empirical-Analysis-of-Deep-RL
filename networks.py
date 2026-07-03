import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#========== Simple Q Netwroks for DQN  =============================================================================================================
# Simple feedforward neural network for approximating the Q-function
class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        """The forward pass of the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
#===================================================================================================================================================

#========== Dueling Network  ======================================================================================================================= 
class DuelingQNetwork(nn.Module):
    """Dueling Q-Network architecture"""
    def __init__(self, state_dim, action_dim, opt=0):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Separate streams for value and advantage
        self.value_fc = nn.Linear(64, 32)  
        self.value_stream = nn.Linear(32, 1)  # Outputs a single value representing the state value
        self.advantage_fc = nn.Linear(64, 32)  # Outputs advantages for each action
        self.advantage_stream = nn.Linear(32, action_dim)  # Final advantage stream
        
        # Save option for combining value and advantage
        self.opt = opt  # 0 for mean normalization, 1 for max normalization

    def forward(self, x):
        """The forward pass of the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        value = self.value_stream(self.value_fc(x))  # Shape: (batch_size, 1)
        advantage = self.advantage_stream(self.advantage_fc(x))  # Shape: (batch_size, action_dim)
        
        # Combine value and advantage to get Q-values
        # Option 1: Mean Normalized
        if self.opt == 0:
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        # Option 2: Max Normalized
        else:
            q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
        return q_values  
#===================================================================================================================================================

#========== Monte Carlo Network  ===================================================================================================================
class MonteCarloReinforce(nn.Module):
    """Monte Carlo REINFORCE Agent"""
    def __init__(self, state_dim, action_dim, lr=1e-3):
        super(MonteCarloReinforce, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        """The forward pass of the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1) # Sum across all actions should equal 1, so we apply softmax to get a valid probability distribution over actions
#===================================================================================================================================================

#========== Critic Network  ========================================================================================================================
# Creating a Critic Network for Estimating state values
class CriticNetwork(nn.Module):
    """Critic Network to estimate the state value used for advantage calculation"""
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        return self.net(state)
#===================================================================================================================================================