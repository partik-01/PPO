import torch
import pandas as pd
import numpy as np
from stock_trading_env import StockTradingEnv
from ppo1 import PPO
import matplotlib.pyplot as plt
from datetime import datetime
import json
from torch.serialization import add_safe_globals

# Add numpy scalar to safe globals
add_safe_globals(['numpy._core.multiarray.scalar'])

def validate_model(env, agent, model_path):
    # Load the model with weights_only=False for compatibility
    checkpoint = torch.load(model_path, 
                          map_location=torch.device('mps'),
                          weights_only=False)  # Added weights_only parameter
    
    # Rest of the function remains the same
    agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    agent.reward_scaler.mean = checkpoint['reward_scaler_mean']
    agent.reward_scaler.std = checkpoint['reward_scaler_std']
    
    # Set to evaluation mode
    agent.actor_critic.eval()
    
    # Initialize lists to store data
    dates = []
    actions = []
    portfolio_values = []
    positions = []
    stock_prices = []
    
    state = env.reset()
    done = False
    
    while not done:
        with torch.no_grad():
            action, _, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action[0])
            
            # Store data
            dates.append(info['date'])
            actions.append(action[0])
            portfolio_values.append(info['portfolio_value'])
            positions.append(info['shares_held'])
            stock_prices.append(info['stock_price'])
            
            state = next_state
    
    # Create performance DataFrame
    performance_df = pd.DataFrame({
        'Date': dates,
        'Action': actions,
        'Portfolio_Value': portfolio_values,
        'Position': positions,
        'Stock_Price': stock_prices
    })
    
    # Save performance data
    performance_df.to_csv('model_performanceLLBS_3y.csv', index=False)
    
    # Calculate final portfolio value and return
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    return performance_df

def main():
    # Load and preprocess data
    df = pd.read_csv('LLBS.csv')
    
    # Convert dates and filter for last 3 years
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    start_date = '2019-01-01'
    end_date = '2025-05-14'  # Last date in the dataset
    
    # Filter data for the last 3 years
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    filtered_df = df.loc[mask]
    
    print(f"Using data from {filtered_df['Date'].min()} to {filtered_df['Date'].max()}")
    print(f"Total days in validation: {len(filtered_df)}")
    
    # Create environment and agent
    env = StockTradingEnv(filtered_df)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        epochs=10
    )
    
    # Validate model
    performance_df = validate_model(env, agent, 'best_model_ppo1LLBS.pth')

if __name__ == "__main__":
    main() 