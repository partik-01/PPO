import numpy as np
import torch
from ppo1 import PPO
from stock_trading_env import StockTradingEnv

def calculate_price_prediction_mse(model_path, test_data):
    env = StockTradingEnv(test_data)
    agent = PPO(state_dim=env.observation_space.shape[0], 
                    action_dim=env.action_space.n)
    
    checkpoint = torch.load(model_path)
    agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    agent.actor_critic.eval()
    
    actual_prices = []
    predicted_actions = []
    
    obs = env.reset()
    done = False
    
    while not done:
        with torch.no_grad():
            state = torch.FloatTensor(agent.normalize_observation(obs)).unsqueeze(0)
            action_probs, _ = agent.actor_critic(state)
            
        actual_prices.append(env.current_price())
        predicted_actions.append(action_probs.numpy()[0])
        
        action, _, _ = agent.choose_action(obs)
        obs, _, done, _ = env.step(action)
    
    actual_prices = np.array(actual_prices)
    predicted_actions = np.array(predicted_actions)
    
    # Calculate price-based MSE
    price_mse = np.mean((actual_prices[1:] - actual_prices[:-1]) ** 2)
    
    return {
        'price_mse': price_mse,
        'actual_prices': actual_prices,
        'predicted_actions': predicted_actions
    }

if __name__ == "__main__":
    import pandas as pd
    test_data = pd.read_csv('LLBS.csv')
    results = calculate_price_prediction_mse('best_model_ppo1LLBS.pth', test_data)
    print(f"Price-based MSE: {results['price_mse']:.4f}")