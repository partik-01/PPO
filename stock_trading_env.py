# stock_trading_env.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
import ta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging
import time
from datetime import datetime
import os

# Configure logging
def setup_logging():
    """Configure logging with timestamps and create logs directory if needed"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/trading_env_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class MarketRegimeDetector:
    def __init__(self, n_states=2, n_iter=1000):
        logger.info("Initializing MarketRegimeDetector with %d states", n_states)
        self.n_states = n_states
        self.n_iter = n_iter
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            init_params="kmeans",  # Use kmeans for better initialization
            tol=1e-5,  # Tighter convergence for better accuracy
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._cached_features = None
        self._cached_predictions = None
        
    def prepare_features(self, returns, volatility):
        """Prepare and cache features for HMM"""
        if self._cached_features is not None and len(returns) == len(self._cached_features):
            return self._cached_features
            
        start_time = time.time()
        features = np.column_stack([returns, volatility])
        if not self.is_fitted:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        self._cached_features = features
        logger.debug("Feature preparation took %.3f seconds", time.time() - start_time)
        return features
        
    def fit(self, returns, volatility):
        """Fit HMM model to the data"""
        start_time = time.time()
        logger.info("Starting HMM model fitting")
        
        features = self.prepare_features(returns, volatility)
        
        try:
            self.hmm.fit(features)
            self.is_fitted = True
            predictions = self.hmm.predict(features)
            self._cached_predictions = predictions
            
            fit_time = time.time() - start_time
            logger.info("HMM fitting completed in %.3f seconds", fit_time)
            logger.info("Model score: %.3f", self.hmm.score(features))
            
            # Log regime statistics
            unique, counts = np.unique(predictions, return_counts=True)
            for regime, count in zip(unique, counts):
                logger.info("Regime %d: %d samples (%.2f%%)", 
                          regime, count, (count/len(predictions))*100)
            
            return predictions
            
        except Exception as e:
            logger.error("Error during HMM fitting: %s", str(e))
            raise
        
    def predict(self, returns, volatility):
        """Predict market regime for new data"""
        if not self.is_fitted:
            raise ValueError("Model needs to be fitted first")
            
        start_time = time.time()
        features = self.prepare_features(returns, volatility)
        
        # Use cached predictions if available and valid
        if (self._cached_predictions is not None and 
            len(self._cached_predictions) == len(features)):
            return self._cached_predictions[-1]
            
        prediction = self.hmm.predict(features)[-1]
        logger.debug("Prediction took %.3f seconds", time.time() - start_time)
        return prediction

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, max_shares=10, transaction_fee_percent=0.001):
        start_time = time.time()
        logger.info("Initializing StockTradingEnv")
        super(StockTradingEnv, self).__init__()
        
        # Make a copy of the dataframe to avoid modifying the original
        self.df = df.copy()
        
        logger.info("Starting data preprocessing with %d rows", len(self.df))
        
        try:
            # Vectorized operations instead of apply for better performance
            # Drop unnecessary columns
            orig_cols = set(self.df.columns)
            self.df = self.df.drop(['S.N.', 'Symbol'], axis=1, errors='ignore')
            dropped_cols = orig_cols - set(self.df.columns)
            if dropped_cols:
                logger.info("Dropped columns: %s", dropped_cols)
            
            # Convert numeric columns using vectorized operations
            numeric_columns = ['Open', 'High', 'Low', 'Ltp', 'Qty', 'Turnover', '% Change']
            
            # Optimize numeric conversion by processing all columns at once
            for col in numeric_columns:
                if col in self.df.columns:
                    try:
                        # First try direct conversion
                        self.df[col] = pd.to_numeric(self.df[col], errors='raise')
                    except (ValueError, TypeError):
                        # If direct conversion fails, try cleaning the string
                        logger.debug("Cleaning string values in column: %s", col)
                        self.df[col] = (self.df[col].astype(str)
                                      .str.replace(',', '')
                                      .str.replace('%', '')
                                      .astype(float))
            
            logger.info("Numeric conversion completed for columns: %s", 
                       [col for col in numeric_columns if col in self.df.columns])
            
            # Convert date and sort
            logger.info("Converting dates and sorting data")
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')
            self.df.sort_values('Date', inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            
            # Initialize market regime detector
            logger.info("Initializing market regime detector")
            self.regime_detector = MarketRegimeDetector()
            
            # Precompute technical indicators using vectorized operations
            logger.info("Computing technical indicators")
            indicator_start_time = time.time()
            self._precompute_indicators()
            logger.info("Technical indicators computed in %.3f seconds", 
                       time.time() - indicator_start_time)
            
            # Fit market regimes
            logger.info("Fitting market regimes")
            regime_start_time = time.time()
            returns = self.df['Returns'].values
            volatility = self.df['Volatility'].values
            self.df['Market_Regime'] = self.regime_detector.fit(returns, volatility)
            logger.info("Market regimes fitted in %.3f seconds", 
                       time.time() - regime_start_time)
            
        except Exception as e:
            logger.error("Error during preprocessing: %s", str(e))
            logger.error("DataFrame columns: %s", self.df.columns.tolist())
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            raise
            
        if len(self.df) < 100:
            logger.error("Insufficient data points: %d (minimum required: 100)", len(self.df))
            raise ValueError("Not enough historical data. Need at least 100 data points.")
        
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.transaction_fee_percent = transaction_fee_percent
        
        # Expand observation space to include market regime
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        self.max_steps = len(self.df) - 1
        self.current_step = None
        self.balance = None
        self.shares_held = None
        self.purchase_prices = None
        self.portfolio_history = None
        self.window_size = 20
        
        # Precompute price scaling factor for normalization
        self.price_scaler = self.df['Ltp'].max() / 1000
        
        # Cache frequently accessed data
        self._cache = {}
        
        init_time = time.time() - start_time
        logger.info("Environment initialization completed in %.3f seconds", init_time)
        logger.info("Data shape: %s, Date range: %s to %s", 
                   self.df.shape,
                   self.df['Date'].min().strftime('%Y-%m-%d'),
                   self.df['Date'].max().strftime('%Y-%m-%d'))

    def _precompute_indicators(self):
        """Precompute all technical indicators using vectorized operations"""
        logger.debug("Starting technical indicator calculations")
        
        # Use numpy operations where possible for better performance
        prices = self.df['Ltp'].values
        high_prices = self.df['High'].values
        low_prices = self.df['Low'].values
        volumes = self.df['Qty'].values
        
        # Price-based calculations
        start_time = time.time()
        self.df['Returns'] = np.concatenate([[0], np.diff(prices) / prices[:-1]])
        logger.debug("Returns calculation: %.3fs", time.time() - start_time)
        
        # MACD (vectorized)
        start_time = time.time()
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        self.df['MACD'] = (exp1 - exp2).fillna(0)
        self.df['Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean().fillna(0)
        logger.debug("MACD calculation: %.3fs", time.time() - start_time)
        
        # RSI (vectorized with numpy)
        start_time = time.time()
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs)).fillna(50)
        logger.debug("RSI calculation: %.3fs", time.time() - start_time)
        
        # Moving averages (numpy for speed)
        start_time = time.time()
        self.df['SMA_20'] = pd.Series(prices).rolling(window=20, min_periods=1).mean()
        self.df['SMA_50'] = pd.Series(prices).rolling(window=50, min_periods=1).mean()
        self.df['EMA_20'] = pd.Series(prices).ewm(span=20, adjust=False).mean()
        logger.debug("Moving averages calculation: %.3fs", time.time() - start_time)
        
        # Volatility measures (numpy operations)
        start_time = time.time()
        returns = self.df['Returns'].values
        vol_window = 20
        vol = np.array([np.std(returns[max(0, i-vol_window):i+1]) 
                       for i in range(len(returns))])
        self.df['Volatility'] = vol
        logger.debug("Volatility calculation: %.3fs", time.time() - start_time)
        
        # ATR (vectorized)
        start_time = time.time()
        high_low = high_prices - low_prices
        high_close = np.abs(high_prices - np.roll(prices, 1))
        low_close = np.abs(low_prices - np.roll(prices, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        self.df['ATR'] = pd.Series(tr).rolling(window=14).mean().fillna(0)
        logger.debug("ATR calculation: %.3fs", time.time() - start_time)
        
        # Volume indicators (numpy operations)
        start_time = time.time()
        self.df['Volume_SMA'] = pd.Series(volumes).rolling(window=20).mean().fillna(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio = volumes / self.df['Volume_SMA'].values
            vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 1)
        self.df['Volume_Ratio'] = vol_ratio
        logger.debug("Volume indicators calculation: %.3fs", time.time() - start_time)
        
        # MFI (vectorized)
        start_time = time.time()
        typical_price = (high_prices + low_prices + prices) / 3
        money_flow = typical_price * volumes
        
        delta_tp = np.diff(typical_price, prepend=typical_price[0])
        pos_flow = np.where(delta_tp > 0, money_flow, 0)
        neg_flow = np.where(delta_tp < 0, money_flow, 0)
        
        pos_mf = pd.Series(pos_flow).rolling(window=14).sum()
        neg_mf = pd.Series(neg_flow).rolling(window=14).sum()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        self.df['MFI'] = mfi.fillna(50)
        logger.debug("MFI calculation: %.3fs", time.time() - start_time)
        
        logger.debug("All technical indicators computed successfully")

    def calculate_reward(self, action, current_price, next_price, shares_bought=0, shares_sold=0):
        """Enhanced reward function with market regime awareness"""
        reward = 0
        
        # Get current market regime
        current_regime = self.df.iloc[self.current_step]['Market_Regime']
        
        # Time decay factor to encourage earlier profitable actions
        time_factor = 0.99 ** (self.current_step / self.max_steps)
        
        # Calculate smoothed price change
        price_change_pct = (next_price - current_price) / current_price
        smooth_price_change = np.tanh(price_change_pct)  # Limit extreme values
        
        # Calculate portfolio value change
        portfolio_value = self.balance + (self.shares_held * current_price)
        prev_portfolio_value = self.portfolio_history[-1]
        portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Regime-based reward scaling
        regime_multiplier = 1.0
        if current_regime == 0:  # Low volatility regime
            regime_multiplier = 1.2 if action == 1 else 0.8  # Encourage buying in stable markets
        else:  # High volatility regime
            regime_multiplier = 1.2 if action == 2 else 0.8  # Encourage selling in volatile markets
        
        # Base reward calculation with regime awareness
        if action == 0:  # Hold
            if self.shares_held > 0:
                reward = smooth_price_change * self.shares_held * regime_multiplier
            else:
                reward = -smooth_price_change * regime_multiplier  # Reward for avoiding losses in downtrend
        
        elif action == 1:  # Buy
            if shares_bought > 0:
                # Transaction cost penalty
                reward -= self.transaction_fee_percent
                
                # Price momentum component with regime awareness
                reward += smooth_price_change * regime_multiplier
                
                # Buy timing reward based on technical indicators and regime
                current_rsi = self.df.iloc[self.current_step]['RSI']
                if current_rsi < 30 and current_regime == 0:  # Oversold in stable market
                    reward += 0.2
                elif current_rsi < 20 and current_regime == 1:  # Deeply oversold in volatile market
                    reward += 0.1
                
                # Volume consideration
                volume_ratio = self.df.iloc[self.current_step]['Volume_Ratio']
                if volume_ratio > 1.5:  # High volume buy
                    reward += 0.1 * regime_multiplier
        
        elif action == 2:  # Sell
            if shares_sold > 0:
                # Calculate profit/loss percentage
                avg_purchase_price = np.mean(self.purchase_prices) if self.purchase_prices else current_price
                profit_pct = (current_price - avg_purchase_price) / avg_purchase_price
                reward += np.tanh(profit_pct) * regime_multiplier
                
                # Sell timing reward based on technical indicators and regime
                current_rsi = self.df.iloc[self.current_step]['RSI']
                if current_rsi > 70 and current_regime == 0:  # Overbought in stable market
                    reward += 0.1
                elif current_rsi > 80 and current_regime == 1:  # Strongly overbought in volatile market
                    reward += 0.2
        
        # Portfolio value change component with regime awareness
        reward += np.tanh(portfolio_change) * regime_multiplier
        
        # Scale reward based on position size and time
        position_size = (self.shares_held * current_price) / portfolio_value if portfolio_value > 0 else 0
        reward *= (1 + position_size) * time_factor
        
        # Add regime transition penalty/reward
        if self.current_step > 0:
            prev_regime = self.df.iloc[self.current_step - 1]['Market_Regime']
            if prev_regime != current_regime:
                # Penalize holding through regime changes
                if action == 0 and self.shares_held > 0:
                    reward -= 0.1
                # Reward appropriate action during regime change
                elif (action == 1 and current_regime == 0) or (action == 2 and current_regime == 1):
                    reward += 0.1
        
        return reward

    def _next_observation(self):
        """Get the next observation with caching for performance"""
        cache_key = f"obs_{self.current_step}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        start_time = time.time()
        
        # Get the current window of data
        obs = np.array([
            self.df.iloc[self.current_step]['Ltp'],
            self.df.iloc[self.current_step]['Returns'],
            self.df.iloc[self.current_step]['MACD'],
            self.df.iloc[self.current_step]['Signal'],
            self.df.iloc[self.current_step]['RSI'],
            self.df.iloc[self.current_step]['SMA_20'],
            self.df.iloc[self.current_step]['SMA_50'],
            self.df.iloc[self.current_step]['EMA_20'],
            self.df.iloc[self.current_step]['Volatility'],
            self.df.iloc[self.current_step]['ATR'],
            self.df.iloc[self.current_step]['Volume_Ratio'],
            self.df.iloc[self.current_step]['MFI'],
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held / self.max_shares,   # Normalized position size
            self.portfolio_value() / self.initial_balance,  # Normalized portfolio value
            self.df.iloc[self.current_step]['Market_Regime']
        ], dtype=np.float32)
        
        # Normalize the observation values
        obs[0] = obs[0] / self.price_scaler  # Price
        obs[4] = obs[4] / 100   # RSI
        obs[5] = obs[5] / 1000  # SMA_20
        obs[6] = obs[6] / 1000  # SMA_50
        
        # Cache the observation
        self._cache[cache_key] = obs
        
        logger.debug("Observation generation took %.3fs", time.time() - start_time)
        return obs

    def portfolio_value(self):
        return self.balance + (self.shares_held * self.current_price())

    def current_price(self):
        return float(self.df.iloc[self.current_step]['Ltp'])

    def reset(self):
        """Reset the environment"""
        start_time = time.time()
        logger.info("Resetting environment")
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.purchase_prices = []
        self.portfolio_history = [self.initial_balance]
        
        # Clear cache
        self._cache.clear()
        
        logger.debug("Environment reset completed in %.3fs", time.time() - start_time)
        return self._next_observation()

    def step(self, action):
        """Execute one step in the environment"""
        start_time = time.time()
        logger.debug("Starting step %d with action %d", self.current_step, action)
        
        self.current_step += 1
        # Clear cache for the new step
        self._cache.clear()
        
        current_price = self.current_price()
        next_price = float(self.df.iloc[min(self.current_step + 1, len(self.df)-1)]['Ltp'])
        
        shares_bought = 0
        shares_sold = 0
        
        # Execute action
        if action == 1 and self.balance >= current_price and self.shares_held < self.max_shares:  # Buy
            shares_to_buy = min(self.max_shares - self.shares_held, 
                              self.balance // current_price)
            shares_to_buy = int(shares_to_buy)
            shares_bought = shares_to_buy
            
            purchase_cost = shares_to_buy * current_price * (1 + self.transaction_fee_percent)
            self.balance -= purchase_cost
            self.shares_held += shares_to_buy
            self.purchase_prices.extend([current_price] * shares_to_buy)
            
            logger.debug("Bought %d shares at $%.2f each", shares_bought, current_price)
            
        elif action == 2 and self.shares_held > 0:  # Sell
            shares_to_sell = self.shares_held
            shares_sold = shares_to_sell
            
            sell_revenue = shares_to_sell * current_price * (1 - self.transaction_fee_percent)
            self.balance += sell_revenue
            self.shares_held = 0
            self.purchase_prices = []
            
            logger.debug("Sold %d shares at $%.2f each", shares_sold, current_price)
        
        # Calculate reward
        reward = self.calculate_reward(action, current_price, next_price, shares_bought, shares_sold)
        
        # Update portfolio history
        portfolio_value = self.portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Calculate done flag
        done = self.current_step >= self.max_steps - 1
        
        info = {
            'portfolio_value': portfolio_value,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'current_price': current_price,
            'action': action,
            'reward': reward,
            'date': self.df.iloc[self.current_step]['Date'].strftime('%Y-%m-%d'),
            'stock_price': current_price,
            'position': self.shares_held
        }
        
        step_time = time.time() - start_time
        logger.debug("Step completed in %.3fs. Portfolio value: $%.2f, Reward: %.3f", 
                    step_time, portfolio_value, reward)
        
        if done:
            logger.info("Episode finished. Final portfolio value: $%.2f (%.2f%% return)", 
                       portfolio_value, 
                       ((portfolio_value - self.initial_balance) / self.initial_balance) * 100)
        
        return self._next_observation(), reward, done, info

    def render(self, mode='human'):
        profit = self.portfolio_value() - self.initial_balance
        
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Shares held: {self.shares_held}')
        print(f'Current price: ${self.current_price():.2f}')
        print(f'Portfolio value: ${self.portfolio_value():.2f}')
        print(f'Profit: ${profit:.2f} ({(profit/self.initial_balance)*100:.2f}%)')
        print(f'Current RSI: {self.df.iloc[self.current_step]["RSI"]:.2f}')
        print('-' * 50)