# predict_trades.py

import torch
import pandas as pd
import numpy as np
from stock_trading_env import StockTradingEnv
from ppo1 import PPO
import torch.serialization

class TradingPredictor:
    def __init__(self, model_path, state_dim, action_dim):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize the agent
        self.agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            batch_size=256,
        )
        
        # Add numpy types to safe globals for PyTorch 2.6+
        import numpy
        torch.serialization.add_safe_globals([
            'numpy._core.multiarray.scalar',
            'numpy.ndarray',
            'numpy.dtype',
            numpy.float32,
            numpy.float64,
            numpy.int64,
            numpy.bool_
        ])
        
        # Load the trained model
        try:
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            
            # Load reward scaler parameters
            if 'reward_scaler_mean' in checkpoint and 'reward_scaler_std' in checkpoint:
                self.agent.reward_scaler.mean = checkpoint['reward_scaler_mean']
                self.agent.reward_scaler.std = checkpoint['reward_scaler_std']
                print("Loaded reward scaler parameters")
            else:
                print("Warning: No reward scaler parameters found in checkpoint")
            
            self.agent.actor_critic.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict_action(self, state):
        """
        Predict the next action given the current state
        Returns: action, action_probability, predicted_value
        """
        try:
            # Convert state to numpy array if it's not already
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            # Scale the state using reward_scaler instead of normalize_observation
            if hasattr(self.agent, 'reward_scaler'):
                state = (state - self.agent.reward_scaler.mean) / (self.agent.reward_scaler.std + 1e-8)
            
            # Convert to tensor and add batch dimension
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action_probs, value = self.agent.actor_critic(state)
                
                # Get the action with highest probability
                action = torch.argmax(action_probs).item()
                probability = action_probs[0][action].item()
                
            return action, probability, value.item()
        except Exception as e:
            print(f"Error in predict_action: {str(e)}")
            raise

def calculate_technical_indicators(df):
    """Calculate all required technical indicators"""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate Returns
    df['Returns'] = df['Ltp'].pct_change()
    
    # Calculate MACD
    exp1 = df['Ltp'].ewm(span=12, adjust=False).mean()
    exp2 = df['Ltp'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Ltp'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate SMAs
    df['SMA_20'] = df['Ltp'].rolling(window=20).mean()
    df['SMA_50'] = df['Ltp'].rolling(window=50).mean()
    
    # Calculate EMA
    df['EMA_20'] = df['Ltp'].ewm(span=20, adjust=False).mean()
    
    # Calculate Volatility (20-day standard deviation)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calculate ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Ltp'].shift())
    low_close = np.abs(df['Low'] - df['Ltp'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Calculate Volume Ratio (comparing to 20-day average)
    df['Volume_Ratio'] = df['Qty'] / df['Qty'].rolling(window=20).mean()
    
    # Calculate MFI (Money Flow Index)
    typical_price = (df['High'] + df['Low'] + df['Ltp']) / 3
    money_flow = typical_price * df['Qty']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    money_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    return df

def load_and_preprocess_data(csv_path):
    """Load and preprocess the latest market data"""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    try:
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Convert numeric columns, handling commas
        numeric_columns = ['Open', 'High', 'Low', 'Ltp', '% Change', 'Qty', 'Turnover']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date and reset index
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Forward fill any NaN values using ffill() instead of fillna(method='ffill')
        df = df.ffill()
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        # Reset index again after removing NaN rows
        df = df.reset_index(drop=True)
        
        # Print verification of required columns
        required_columns = ['Ltp', 'Returns', 'MACD', 'Signal', 'RSI', 'SMA_20', 
                          'SMA_50', 'EMA_20', 'Volatility', 'ATR', 'Volume_Ratio', 'MFI']
        print("\nVerifying required columns:")
        for col in required_columns:
            print(f"{col}: {col in df.columns}")
        
        # Create environment to preprocess data
        env = StockTradingEnv(df)
        return env, df
        
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        print("DataFrame columns:", df.columns.tolist())
        raise

def interpret_action(action, probability, value):
    """Convert numerical action to human-readable recommendation"""
    actions = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }
    
    return {
        "recommendation": actions[action],
        "confidence": f"{probability:.2%}",
        "predicted_value": f"{value:.2f}"
    }

def main():
    # Configuration
    MODEL_PATH = "best_model_ppo1LLBS.pth"  # Make sure this matches your model file name
    DATA_PATH = "LLBS.csv"
    STATE_DIM = 16
    ACTION_DIM = 3
    
    try:
        # Load predictor
        print("\nInitializing Trading Predictor...")
        predictor = TradingPredictor(MODEL_PATH, STATE_DIM, ACTION_DIM)
        
        # Load and preprocess latest data
        print("\nLoading and preprocessing market data...")
        env, df = load_and_preprocess_data(DATA_PATH)
        
        # Reset the environment
        print("\nResetting environment...")
        initial_observation = env.reset()
        print(f"Initial observation shape: {initial_observation.shape}")
        
        # Get prediction
        print("\nPredicting next action...")
        action, prob, value = predictor.predict_action(initial_observation)
        
        # Interpret results
        result = interpret_action(action, prob, value)
        
        # Print results
        print("\nTrading Recommendation:")
        print("-" * 50)
        print(f"Date: {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"Current Price: {df['Ltp'].iloc[-1]:.2f}")
        print(f"Action: {result['recommendation']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Predicted Value: {result['predicted_value']}")
        print("-" * 50)
        
        # Print recent market data
        print("\nRecent Market Data:")
        recent_data = df.tail()[['Date', 'Open', 'High', 'Low', 'Ltp', '% Change']]
        recent_data['Date'] = recent_data['Date'].dt.strftime('%Y-%m-%d')
        print(recent_data.to_string())
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()