import torch
import numpy as np
from ppo1 import PPO, StockTradingEnv
import pandas as pd

def test_mps_availability():
    print("\nTesting MPS (Metal) Availability:")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
def test_environment():
    print("\nTesting Environment Setup:")
    # Create a small test DataFrame
    df = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=100),
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Ltp': np.random.rand(100) * 100,
        'Qty': np.random.rand(100) * 1000,
        'Turnover': np.random.rand(100) * 10000,
        '% Change': np.random.rand(100) * 5 - 2.5
    })
    
    try:
        env = StockTradingEnv(df)
        print("✓ Environment creation successful")
        
        state = env.reset()
        print(f"✓ Environment reset successful (state shape: {state.shape})")
        
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print("✓ Environment step successful")
        return env
    except Exception as e:
        print(f"✗ Environment test failed: {str(e)}")
        return None

def test_ppo_agent(env):
    print("\nTesting PPO Agent:")
    if env is None:
        print("✗ Cannot test PPO agent: Environment not available")
        return
    
    try:
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
            batch_size=32,  # Smaller batch for testing
            epochs=2
        )
        print("✓ PPO agent creation successful")
        
        # Test action selection
        state = env.reset()
        actions, values, log_probs = agent.select_action([state])
        print(f"✓ Action selection successful (action: {actions}, value: {values})")
        
        # Test training step
        next_state, reward, done, _ = env.step(actions[0])
        agent.memory.push([state], actions, [reward], [next_state], [done], [values[0]], [log_probs[0]])
        
        if len(agent.memory.states) >= agent.memory.batch_size:
            agent.update()
            print("✓ Training step successful")
        
    except Exception as e:
        print(f"✗ PPO agent test failed: {str(e)}")

def main():
    print("Starting PPO Implementation Tests")
    print("=" * 50)
    
    test_mps_availability()
    env = test_environment()
    test_ppo_agent(env)
    
    print("\nTest Summary:")
    print("=" * 50)
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal) is available and will be used for training")
    else:
        print("! MPS not available, falling back to CPU")

if __name__ == "__main__":
    main() 