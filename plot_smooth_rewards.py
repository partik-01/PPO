import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_smooth_rewards():
    try:
        # Read the original data from the training process
        rewards = []
        with open('training_rewards_ppo1LLBS.txt', 'r') as f:
            for line in f:
                try:
                    rewards.append(float(line.strip()))
                except:
                    continue
        
        rewards = np.array(rewards)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot original data
        plt.plot(rewards, alpha=0.4, color='lightblue', label='Original', linewidth=1)
        
        # Calculate and plot moving averages
        window_50 = np.convolve(rewards, np.ones(50)/50, mode='same')
        plt.plot(window_50, color='red', alpha=0.9, label='50-Episode MA', linewidth=2)
        
        # Apply Savitzky-Golay filter for additional smoothing
        window_length = 101  # Must be odd
        polyorder = 3
        smooth_rewards = savgol_filter(rewards, window_length, polyorder)
        plt.plot(smooth_rewards, color='blue', alpha=0.9, label='Smoothed Trend', linewidth=2)
        
        plt.title('Training Rewards with Moving Average', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig('training_rewards_smooth_LLBS_v2.png', 
                    bbox_inches='tight', dpi=300,
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print("Smoothed visualization saved as 'training_rewards_smooth_LLBS_v2.png'")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure training_rewards_ppo1LLBS.txt exists with reward values")

if __name__ == "__main__":
    plot_smooth_rewards() 