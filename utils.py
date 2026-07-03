import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



class ReplayBuffer:
    """Fixed-size replay memory for storing and sampling experience tuples"""
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Saves a transition as a new transition if there is capacity left in the buffer, otherwise it overwrites the oldest transition"""
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.size
        
    def sample(self, batch_size):
        """Randomly samples a batch of transitions from the buffer"""
        indices = torch.randint(0, len(self.buffer), size=(batch_size,))
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, done = zip(*samples)
        return torch.tensor(states), torch.tensor(actions), torch.tensor(rewards), torch.tensor(next_states), torch.tensor(done)
    
    def __len__(self):
        return len(self.buffer)


class RLPlotter:
    """Object-Oriented utility for storing and plotting RL training data."""
    
    def __init__(self, model_name, env_name, reward_hist, **kwargs):
        self.model_name = model_name
        self.env_name = env_name
        self.reward_hist = np.array(reward_hist)
        self.extra_args = kwargs

    def _generate_title(self):
        """Dynamically generates a title based on initialized parameters."""
        title = f"{self.model_name} Training on {self.env_name}"
        if self.extra_args:
            extras = " | ".join([f"{k.replace('_', ' ').title()}: {v}" if not isinstance(v, bool) or not v else k.replace('_', ' ').title() for k, v in self.extra_args.items()])
            title += f"\n({extras})"
        return title

    def plot(self, hyperparameters=None, show=True, save_path=None, color=None):
        """Plots the reward history for this specific model instance."""
        # Use subplots to allow for text box placement outside the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if self.reward_hist.ndim == 2:
            episodes = np.arange(self.reward_hist.shape[1])
            mean_rewards = np.mean(self.reward_hist, axis=0)
            std_rewards = np.std(self.reward_hist, axis=0)
            
            ax.plot(episodes, mean_rewards, label=f"{self.model_name} (Mean)", color='b' if color is None else color)
            ax.fill_between(episodes, 
                             mean_rewards - std_rewards, 
                             mean_rewards + std_rewards, 
                             color='b' if color is None else color, alpha=0.2, label="±1 Standard Deviation")
        else:
            episodes = np.arange(len(self.reward_hist))
            ax.plot(episodes, self.reward_hist, label=self.model_name, color='b' if color is None else color)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(self._generate_title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Render the hyperparameters text box if provided
        if hyperparameters is not None:
            if isinstance(hyperparameters, dict):
                hyperparams_text = "Hyperparameters:\n" + "-"*16 + "\n"
                hyperparams_text += "\n".join([f"{k} = {v}" for k, v in hyperparameters.items()])
            else:
                # Fallback if a pre-formatted string is passed directly
                hyperparams_text = str(hyperparameters)
                
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
            
            ax.text(1.02, 0.95, hyperparams_text, 
                    transform=ax.transAxes, 
                    fontsize=10, 
                    verticalalignment='top', 
                    bbox=bbox_props, 
                    family='monospace')
            
            # Adjust the layout so the text box is visible in the rendering window
            plt.subplots_adjust(right=0.75)
        
        if save_path:
            # bbox_inches='tight' prevents the text box from being cropped in the saved image
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Chart saved to: {os.path.abspath(save_path)}")
            
        if show:
            plt.show()
        
        plt.close()

    @classmethod
    def compare(cls, plotters, global_title=None, hyperparameters=None, show=True, save_path=None):
        """
        Class method to compare multiple RLPlotter instances on a single chart.
        
        Args:
            plotters (list): A list of RLPlotter instances.
            global_title (str): Optional override for the main chart title.
            hyperparameters (dict or str): Optional parameters to display in a text box.
        """
        if not plotters:
            print("Warning: No plotters provided for comparison.")
            return
        
        color_map = ['b' , 'r', 'm', 'g', 'c', 'y', 'k', 'orange', 'purple', 'brown'] 

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Default to the environment name of the first plotter if no global title is provided
        if not global_title:
            env_name = plotters[0].env_name
            global_title = f"Algorithm Comparison on {env_name}"
            
        for plotter in plotters:
            if plotter.reward_hist.ndim == 2:
                episodes = np.arange(plotter.reward_hist.shape[1])
                mean_rewards = np.mean(plotter.reward_hist, axis=0)
                std_rewards = np.std(plotter.reward_hist, axis=0)
                
                line = ax.plot(episodes, mean_rewards, label=f"{plotter.model_name} (Mean)", color=color_map[plotters.index(plotter) % len(color_map)])[0]
                ax.fill_between(episodes, 
                                 mean_rewards - std_rewards, 
                                 mean_rewards + std_rewards, 
                                 color=line.get_color(), alpha=0.2)
            else:
                episodes = np.arange(len(plotter.reward_hist))
                ax.plot(episodes, plotter.reward_hist, label=plotter.model_name, color=color_map[plotters.index(plotter) % len(color_map)])

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title(global_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Render the hyperparameters text box if provided
        if hyperparameters is not None:
            if isinstance(hyperparameters, dict):
                hyperparams_text = "Hyperparameters:\n" + "-"*16 + "\n"
                hyperparams_text += "\n".join([f"{k} = {v}" for k, v in hyperparameters.items()])
            else:
                hyperparams_text = str(hyperparameters)
                
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
            
            ax.text(1.02, 0.95, hyperparams_text, 
                    transform=ax.transAxes, 
                    fontsize=10, 
                    verticalalignment='top', 
                    bbox=bbox_props, 
                    family='monospace')
            
            plt.subplots_adjust(right=0.75)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Comparison chart saved to: {os.path.abspath(save_path)}")
            
        if show:
            plt.show()
            
        plt.close()