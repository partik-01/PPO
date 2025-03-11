import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.dates import MonthLocator, YearLocator, DateFormatter
import seaborn as sns
import json

def calculate_metrics(performance_df, stock_data):
    """Calculate trading performance metrics"""
    # Trading metrics
    total_trades = len(performance_df[performance_df['Action'].isin([1, 2])])
    buy_trades = len(performance_df[performance_df['Action'] == 1])
    sell_trades = len(performance_df[performance_df['Action'] == 2])
    hold_trades = len(performance_df[performance_df['Action'] == 0])
    
    # Portfolio performance
    initial_value = float(performance_df['Portfolio_Value'].iloc[0])
    final_value = float(performance_df['Portfolio_Value'].iloc[-1])
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    # Stock performance for comparison
    stock_return = ((float(stock_data['Ltp'].iloc[-1]) - float(stock_data['Ltp'].iloc[0])) 
                   / float(stock_data['Ltp'].iloc[0])) * 100
    
    # Calculate daily returns
    performance_df['Daily_Return'] = performance_df['Portfolio_Value'].pct_change()
    stock_data['Daily_Return'] = pd.to_numeric(stock_data['Ltp'], errors='coerce').pct_change()
    
    # Risk metrics
    portfolio_volatility = performance_df['Daily_Return'].std() * np.sqrt(252) * 100
    stock_volatility = stock_data['Daily_Return'].std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    excess_returns = performance_df['Daily_Return'] - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / performance_df['Daily_Return'].std()
    
    return {
        'Total_Trades': total_trades,
        'Buy_Trades': buy_trades,
        'Sell_Trades': sell_trades,
        'Hold_Trades': hold_trades,
        'Total_Return': total_return,
        'Stock_Return': stock_return,
        'Portfolio_Volatility': portfolio_volatility,
        'Stock_Volatility': stock_volatility,
        'Sharpe_Ratio': sharpe_ratio
    }

def visualize_trading_actions():
    try:
        # Read the performance data
        performance_df = pd.read_csv('model_performanceLLBS_3y.csv')  # Updated filename
        performance_df['Date'] = pd.to_datetime(performance_df['Date'])
        
        # Read the stock data
        stock_data = pd.read_csv('LLBS.csv')
        stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%d/%m/%Y')
        
        # Filter stock data for the same period as performance data
        start_date = performance_df['Date'].min()
        end_date = performance_df['Date'].max()
        stock_data = stock_data[
            (stock_data['Date'] >= start_date) & 
            (stock_data['Date'] <= end_date)
        ]
        
        # Convert numeric columns
        numeric_columns = ['Ltp', 'Open', 'High', 'Low', 'Qty', 'Turnover', '% Change']
        for col in numeric_columns:
            if col in stock_data.columns:
                stock_data[col] = pd.to_numeric(stock_data[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Sort data by date
        stock_data = stock_data.sort_values('Date')
        performance_df = performance_df.sort_values('Date')
        
        # Calculate metrics
        metrics = calculate_metrics(performance_df, stock_data)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
        
        # Main trading plot
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot stock price
        ax1.plot(stock_data['Date'], stock_data['Ltp'], 
                color='black', alpha=0.7, label='Stock Price', linewidth=1.5)
        
        # Plot portfolio value
        ax1.plot(performance_df['Date'], performance_df['Portfolio_Value'],
                color='blue', alpha=0.7, label='Portfolio Value', linewidth=1.5)
        
        # Plot buy/sell markers
        buy_points = performance_df[performance_df['Action'] == 1]
        sell_points = performance_df[performance_df['Action'] == 2]
        
        if not buy_points.empty:
            ax1.scatter(buy_points['Date'], buy_points['Stock_Price'],
                       color='green', marker='^', s=100, label='Buy', alpha=0.7)
        if not sell_points.empty:
            ax1.scatter(sell_points['Date'], sell_points['Stock_Price'],
                       color='red', marker='v', s=100, label='Sell', alpha=0.7)
        
        # Format main plot
        ax1.set_title('Trading Actions and Portfolio Performance', pad=20, fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Returns comparison plot
        ax2 = fig.add_subplot(gs[1, 0])
        performance_df['Cumulative_Return'] = (performance_df['Portfolio_Value'] / performance_df['Portfolio_Value'].iloc[0] - 1) * 100
        stock_data['Cumulative_Return'] = (stock_data['Ltp'] / stock_data['Ltp'].iloc[0] - 1) * 100
        
        ax2.plot(performance_df['Date'], performance_df['Cumulative_Return'],
                label='Portfolio Returns', color='blue')
        ax2.plot(stock_data['Date'], stock_data['Cumulative_Return'],
                label='Stock Returns', color='black', alpha=0.7)
        ax2.set_title('Cumulative Returns Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Position plot
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(performance_df['Date'], performance_df['Position'],
                label='Position Size', color='purple')
        ax3.set_title('Position Size Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Shares')
        ax3.grid(True, alpha=0.3)
        
        # Add performance metrics text
        metrics_text = f"""Performance Metrics:
• Trading Activity:
  - Total Trades: {metrics['Total_Trades']}
  - Buys: {metrics['Buy_Trades']}
  - Sells: {metrics['Sell_Trades']}
  - Holds: {metrics['Hold_Trades']}
        
• Returns:
  - Portfolio Return: {metrics['Total_Return']:.2f}%
  - Stock Return: {metrics['Stock_Return']:.2f}%
        
• Risk Metrics:
  - Portfolio Volatility: {metrics['Portfolio_Volatility']:.2f}%
  - Stock Volatility: {metrics['Stock_Volatility']:.2f}%
  - Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}"""
        
        plt.figtext(1.02, 0.5, metrics_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, pad=5))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # Save the plot
        plt.savefig('trading_performance_LLBS_3y.png', 
                    bbox_inches='tight', dpi=300,
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print("Visualization saved as 'trading_performance_LLBS_3y.png'")
        
        with open('trading_metrics_LLBS_3y.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved to 'trading_metrics_LLBS_3y.json'")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    visualize_trading_actions()