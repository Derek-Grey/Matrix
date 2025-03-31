import pandas as pd
import numpy as np
import os
import pymongo
from urllib.parse import quote_plus
from typing import List, Tuple
import random
from portfolio_metrics import DataChecker

def get_client_U():
    user, pwd = 'Tom', 'tom'
    return pymongo.MongoClient(f"mongodb://{quote_plus(user)}:{quote_plus(pwd)}@192.168.1.99:29900/")

def get_valid_trading_dates(start_date, end_date):
    """获取指定日期范围内的有效交易日"""
    checker = DataChecker()
    all_dates = pd.date_range(start_date, end_date, freq='D')
    valid_dates = [d.strftime('%Y-%m-%d') for d in all_dates 
                  if d.strftime('%Y-%m-%d') in checker.trading_dates]
    return valid_dates

def get_random_stocks_and_returns(date: str, client) -> Tuple[List[str], List[float]]:
    """从数据库中获取指定日期的随机股票及其收益率"""
    daily_data = list(client.basic_wind.w_vol_price.find(
        {"date": date},
        {"_id": 0, "code": 1, "pct_chg": 1}
    ))
    
    num_stocks = random.randint(40, 50)
    selected_stocks = random.sample(daily_data, min(num_stocks, len(daily_data)))
    
    # Ensure the lists have the same length
    if len(selected_stocks) == 0:
        return [], []
    
    codes = [stock['code'] for stock in selected_stocks]
    returns = [float(stock['pct_chg']) for stock in selected_stocks]
    
    # Check if lengths are equal
    assert len(codes) == len(returns), "Codes and returns lists must have the same length"
    
    return codes, returns

def create_minute_data(date: str, stocks: List[str], is_weight=True) -> pd.DataFrame:
    """创建单个交易日的分钟频数据
    
    Args:
        date: 交易日期
        stocks: 股票代码列表
        is_weight: 是否创建权重数据
    """
    data = []
    # 创建该日期的分钟时间序列
    full_day = pd.date_range(
        start=pd.Timestamp(date).replace(hour=9, minute=30),
        end=pd.Timestamp(date).replace(hour=15, minute=0),
        freq='1min'
    )
    
    # 只保留有效交易时间
    valid_times = []
    for t in full_day:
        hour = t.hour
        minute = t.minute
        if ((9 < hour < 11) or 
            (hour == 9 and minute >= 30) or 
            (13 <= hour < 15) or 
            (hour == 11 and minute < 30)):
            valid_times.append(t)
    
    for t in valid_times:
        time_str = t.strftime('%H:%M:%S')
        if is_weight:
            # Equal weights for all stocks
            values = [1.0 / len(stocks)] * len(stocks)
        else:
            values = np.random.normal(0.0001, 0.001, len(stocks))
            
        for stock, value in zip(stocks, values):
            data.append({
                'date': date,
                'time': time_str,
                'code': stock,
                'weight' if is_weight else 'return': value
            })
    
    return pd.DataFrame(data)

def generate_all_data(start_date='2024-01-02', end_date='2024-12-31', 
                     minute_start_date='2024-01-02', minute_end_date='2024-01-31'):
    """生成所有数据并保存到CSV文件"""
    # 创建目录
    output_dir = 'd:/Derek/Code/Matrix/Daily/data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置分钟频数据的日期范围
    if minute_start_date is None:
        minute_start_date = start_date
    if minute_end_date is None:
        minute_end_date = (pd.Timestamp(minute_start_date) + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')
    
    # 获取有效交易日
    daily_dates = get_valid_trading_dates(start_date, end_date)
    minute_dates = get_valid_trading_dates(minute_start_date, minute_end_date)
    
    if not daily_dates:
        raise ValueError("未找到有效交易日")
    
    client = get_client_U()
    
    # 分别存储权重和收益率数据
    daily_weights = []
    daily_returns = []
    minute_weights_data = []
    minute_returns_data = []
    
    # 生成日频数据
    for date in daily_dates:
        codes, returns = get_random_stocks_and_returns(date, client)
        
        # 生成日频权重数据 with dynamic weights
        total_stocks = len(codes)
        weights = np.random.dirichlet(np.ones(total_stocks), size=1).flatten()
        daily_weights.extend([{
            'date': date,
            'code': code,
            'weight': weight  # Add weight to daily data
        } for code, weight in zip(codes, weights)])
        
        # 生成日频收益率数据
        daily_returns.extend([{
            'date': date,
            'code': code,
            'return': ret
        } for code, ret in zip(codes, returns)])
        
        # 只在指定的分钟频日期范围内生成分钟频数据
        if date in minute_dates:
            minute_weights = create_minute_data(date, codes, is_weight=True)
            minute_returns = create_minute_data(date, codes, is_weight=False)
            minute_weights_data.append(minute_weights)
            minute_returns_data.append(minute_returns)
    
    client.close()
    
    # 转换为DataFrame
    daily_weights_df = pd.DataFrame(daily_weights)
    daily_returns_df = pd.DataFrame(daily_returns)
    minute_weights_df = pd.concat(minute_weights_data, ignore_index=True)
    minute_returns_df = pd.concat(minute_returns_data, ignore_index=True)
    
    # 保存到CSV
    daily_weights_df.to_csv(os.path.join(output_dir, 'test_daily_weight.csv'), index=False)
    daily_returns_df.to_csv(os.path.join(output_dir, 'test_daily_return.csv'), index=False)
    minute_weights_df.to_csv(os.path.join(output_dir, 'test_minute_weight.csv'), index=False)
    minute_returns_df.to_csv(os.path.join(output_dir, 'test_minute_return.csv'), index=False)
    
    print(f"已保存日频权重数据，共 {len(daily_weights_df)} 行")
    print(f"已保存日频收益率数据，共 {len(daily_returns_df)} 行")
    print(f"已保存分钟频权重数据，共 {len(minute_weights_df)} 行")
    print(f"已保存分钟频收益率数据，共 {len(minute_returns_df)} 行")

if __name__ == "__main__":
    generate_all_data()