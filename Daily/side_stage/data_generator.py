import pandas as pd
import numpy as np
import os
from portfolio_metrics import DataChecker

def get_valid_trading_dates(start_date, end_date):
    """获取指定日期范围内的有效交易日
    
    Args:
        start_date (str): 开始日期，格式：'YYYY-MM-DD'
        end_date (str): 结束日期，格式：'YYYY-MM-DD'
    
    Returns:
        list: 有效交易日列表
    """
    checker = DataChecker()
    all_dates = pd.date_range(start_date, end_date, freq='D')
    valid_dates = [d.strftime('%Y-%m-%d') for d in all_dates 
                  if d.strftime('%Y-%m-%d') in checker.trading_dates]
    return valid_dates

def create_minute_data(is_weight=True):
    """创建分钟频数据示例
    
    Args:
        is_weight: 是否创建权重数据，False则创建收益率数据
    Returns:
        pd.DataFrame: 包含以下列的数据框：
            - date: 交易日期
            - time: 交易时间
            - code: 股票代码
            - weight/return: 权重或收益率数据
    """
    # 获取有效交易日
    valid_dates = get_valid_trading_dates('2024-01-02', '2024-01-03')
    if not valid_dates:
        raise ValueError("未找到有效交易日")
    
    stocks = ['300001.SZ', '300002.SZ', '300003.SZ']
    
    # 创建数据框
    data = []
    for date in valid_dates:
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
                values = np.random.dirichlet(np.ones(len(stocks)))
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

def create_daily_data(is_weight=True):
    """创建日频数据示例
    
    Args:
        is_weight: 是否创建权重数据，False则创建收益率数据
    Returns:
        pd.DataFrame: 包含以下列的数据框：
            - date: 交易日期
            - code: 股票代码
            - weight/return: 权重或收益率数据
    """
    # 获取有效交易日
    valid_dates = get_valid_trading_dates('2023-01-02', '2024-01-31')
    if not valid_dates:
        raise ValueError("未找到有效交易日")
    
    stocks = ['600001.SH', '600002.SH', '600003.SH']
    
    # 创建数据框
    data = []
    for date in valid_dates:
        if is_weight:
            # 为每个日期生成随机权重
            values = np.random.dirichlet(np.ones(len(stocks)))
        else:
            # 为每个日期生成随机收益率
            values = np.random.normal(0.0002, 0.02, len(stocks))
            
        for stock, value in zip(stocks, values):
            data.append({
                'date': date,
                'code': stock,
                'weight' if is_weight else 'return': value
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 创建目录（如果不存在）
    output_dir = 'D:\\Derek\\Code\\Matrix\\Daily\\data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成分钟频数据 - 权重
    minute_weight = create_minute_data(is_weight=True)
    minute_weight.to_csv(os.path.join(output_dir, 'test_minute_weight.csv'), index=False)
    print(f"已保存分钟频权重数据，共 {len(minute_weight)} 行")
    
    # 生成分钟频数据 - 收益率
    minute_return = create_minute_data(is_weight=False)
    minute_return.to_csv(os.path.join(output_dir, 'test_minute_return.csv'), index=False)
    print(f"已保存分钟频收益率数据，共 {len(minute_return)} 行")
    
    # 生成日频数据 - 权重
    daily_weight = create_daily_data(is_weight=True)
    daily_weight.to_csv(os.path.join(output_dir, 'test_daily_weight.csv'), index=False)
    print(f"已保存日频权重数据，共 {len(daily_weight)} 行")
    
    # 生成日频数据 - 收益率
    daily_return = create_daily_data(is_weight=False)
    daily_return.to_csv(os.path.join(output_dir, 'test_daily_return.csv'), index=False)
    print(f"已保存日频收益率数据，共 {len(daily_return)} 行")