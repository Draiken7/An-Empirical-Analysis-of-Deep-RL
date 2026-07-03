from dataclasses import dataclass

@dataclass
class RLConfig:
    """Centralized configuration for RL hyperparameters with safe defaults."""
    
    # Environment Settings
    ENV_NAME: str = "Acrobot-v1"
    # ENV_NAME: str = "CartPole-v1"
    NUM_ENVS: int = 10
    
    # Training Loop Limits
    EPISODES: int = 5000
    
    # Replay Buffer & Batch
    BUFFER_SIZE: int = 500000
    BATCH_SIZE: int = 64
    
    # Algorithm Hyperparameters
    GAMMA: float = 0.99
    LEARNING_RATE: float = 1e-3
    TAU: float = 0.005  # For Polyak Averaging
    
    # Exploration (Epsilon-Greedy)
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: float = 0.999
    
    # Dueling Architecture
    DUELING_OPT: int = 1  # 0 for Mean, 1 for Max Normalization
    
    # MonteCarlo REINFOCE 
    MONTECARLO_BASELINE_OPT: int = 0 # 0 for No Baseline, 1 for Mean based and 2 for critic Network based
    
    # Critic network Learning Rate (for Monte Carlo REINFORCE with Critic)
    CRITIC_LR: float = 1e-3
    
    # Target network parameters
    TARGET_UPDATE_FREQ: int = 10
    STEP_BASED_TARGET_UPDATE_FREQ: int = 5000
    POLYAK: bool = True
