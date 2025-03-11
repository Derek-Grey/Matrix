import sys
import os

# 将项目根目录添加到 Python 路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import pandas as pd
import numpy as np
from db_client import get_client_U
from urllib.parse import quote_plus

class DataChecker:
    def __init__(self):
        try:
            # 使用已有的数据库连接函数
            client = get_client_U('r')  # 使用只读权限
            db = client['economic']
            collection = db['trade_dates']
            
            # 获取所有交易日期
            trading_dates = list(collection.find({}, {'_id': 0, 'trade_date': 1}))
            client.close()
            
            # 将交易日期转换为字符串格式的集合，方便查找
            self.trading_dates = set(d['trade_date'] for d in trading_dates)
            
            # 输出交易日信息
            print("\n=== 交易日信息 ===")
            print(f"交易日总数: {len(self.trading_dates)}")
            print("交易日示例(前5个):", sorted(list(self.trading_dates))[:5])
            print("最近的交易日(后5个):", sorted(list(self.trading_dates))[-5:])
            print("=================\n")
            
        except Exception as e:
            print(f"\n=== MongoDB连接错误 ===")
            print(f"错误信息: {str(e)}")
            print("===================\n")
            raise
        
    def check_time_format(self, df):
        """检查时间列格式是否符合HH:MM:SS格式，并验证是否在交易时间内
        
        Args:
            df (pd.DataFrame): 包含 'time' 列的数据框
            
        Raises:
            ValueError: 当时间格式不正确或不在交易时间范围内时抛出异常
        """
        if 'time' not in df.columns:
            return
            
        print("检测到time列，开始检查时间")
        try:
            # 转换时间格式并检查
            times = pd.to_datetime(df['time'], format='%H:%M:%S')
            invalid_times = df[pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').isna()]
            if not invalid_times.empty:
                raise ValueError(f"发现不符合格式的时间: \n{invalid_times['time'].unique()}")
            
            # 定义交易时间段
            morning_start = pd.to_datetime('09:30:00').time()
            morning_end = pd.to_datetime('11:29:00').time()
            afternoon_start = pd.to_datetime('13:00:00').time()
            afternoon_end = pd.to_datetime('14:59:00').time()
            
            # 检查是否在交易时间内
            times_outside_trading = df[~(
                ((times.dt.time >= morning_start) & (times.dt.time <= morning_end)) |
                ((times.dt.time >= afternoon_start) & (times.dt.time <= afternoon_end))
            )]
            
            if not times_outside_trading.empty:
                non_trading_times = times_outside_trading['time'].unique()
                raise ValueError(
                    f"发现非交易时间数据：\n"
                    f"{non_trading_times}\n"
                    f"交易时间为 09:30:00-11:29:00 和 13:00:00-14:59:00"
                )
            
            print("时间格式和交易时间范围检查通过")
            
        except ValueError as e:
            print("时间检查失败")
            raise ValueError(f"时间检查错误: {str(e)}")

    def check_time_frequency(self, df):
        """检查时间切片的频率是否一致，并检查是否存在缺失的时间点
        
        检查规则：
        1. 对于日频数据，每个交易日应该只有一条数据（每个股票）
        2. 对于分钟频数据：
           - 相邻时间点之间的间隔应该一致
           - 在交易时段内不应该有缺失的时间点
        
        Args:
            df (pd.DataFrame): 包含 'date'、'time'、'code' 列的数据框
            
        Raises:
            ValueError: 当时间频率不一致或存在缺失时间点时抛出异常
        """
        if 'time' not in df.columns:
            # 日频数据检查：检查每个股票在每个交易日是否只有一条数据
            date_code_counts = df.groupby(['date', 'code']).size()
            invalid_records = date_code_counts[date_code_counts > 1]
            if not invalid_records.empty:
                raise ValueError(
                    f"发现日频数据中存在重复记录：\n"
                    f"日期-股票对及其出现次数：\n{invalid_records}"
                )
            return
        
        # 分钟频数据检查
        print("开始检查时间频率一致性")
        
        # 合并日期和时间列创建完整的时间戳
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # 获取唯一的时间点
        unique_times = sorted(df['datetime'].unique())
        
        # 计算时间间隔
        time_diffs = []
        for i in range(1, len(unique_times)):
            # 只在同一个交易时段内计算时间差
            curr_time = unique_times[i]
            prev_time = unique_times[i-1]
            
            # 跳过跨天和午休时段的时间差
            if (curr_time.date() != prev_time.date() or  # 跨天
                (prev_time.time() <= pd.to_datetime('11:30:00').time() and 
                 curr_time.time() >= pd.to_datetime('13:00:00').time())):  # 跨午休
                continue
            
            time_diffs.append((curr_time - prev_time).total_seconds())
        
        if not time_diffs:
            raise ValueError("没有足够的数据来确定时间频率")
        
        # 计算众数作为标准频率
        freq_seconds = pd.Series(time_diffs).mode()
        if len(freq_seconds) == 0:
            raise ValueError("无法确定标准时间频率")
        
        freq_minutes = freq_seconds[0] / 60
        if freq_minutes <= 0:
            raise ValueError(
                f"计算得到的时间频率异常: {freq_minutes} 分钟\n"
                f"时间差统计：{pd.Series(time_diffs).value_counts()}"
            )
        
        # 确保频率是整数分钟
        if not freq_minutes.is_integer():
            raise ValueError(f"时间频率必须是整数分钟，当前频率为: {freq_minutes} 分钟")
        
        freq_minutes = int(freq_minutes)
        print(f"检测到数据频率为: {freq_minutes} 分钟")
        
        # 检查是否存在异常的时间间隔
        invalid_diffs = [diff for diff in time_diffs if abs(diff - freq_seconds[0]) > 1]
        if invalid_diffs:
            raise ValueError(
                f"发现不规则的时间间隔：\n"
                f"标准频率为: {freq_minutes} 分钟\n"
                f"异常间隔（秒）：{invalid_diffs}"
            )
        
        # 生成理论上应该存在的所有时间点
        all_dates = pd.to_datetime(df['date']).unique()
        expected_times = []
        
        for date in all_dates:
            try:
                # 生成上午的时间序列
                morning_times = pd.date_range(
                    f"{date.strftime('%Y-%m-%d')} 09:30:00",
                    f"{date.strftime('%Y-%m-%d')} 11:29:00",
                    freq=f"{freq_minutes}min"
                )
                # 生成下午的时间序列
                afternoon_times = pd.date_range(
                    f"{date.strftime('%Y-%m-%d')} 13:00:00",
                    f"{date.strftime('%Y-%m-%d')} 14:59:00",
                    freq=f"{freq_minutes}min"
                )
                expected_times.extend(morning_times)
                expected_times.extend(afternoon_times)
            except Exception as e:
                raise ValueError(f"生成时间序列时出错，日期: {date}, 频率: {freq_minutes}分钟\n错误信息: {str(e)}")
        
        expected_times = pd.DatetimeIndex(expected_times)
        actual_times = pd.DatetimeIndex(unique_times)
        
        # 找出缺失的时间点
        missing_times = expected_times[~expected_times.isin(actual_times)]
        if len(missing_times) > 0:
            raise ValueError(
                f"发现缺失的时间点：\n"
                f"共计缺失 {len(missing_times)} 个时间点\n"
                f"部分缺失时间点示例（最多显示10个）：\n"
                f"{missing_times[:10].strftime('%Y-%m-%d %H:%M:%S').tolist()}"
            )
        
        print(f"时间频率检查通过，数据频率为: {freq_minutes} 分钟")

    def check_trading_dates(self, df):
        """检查数据是否包含非交易日
        
        Args:
            df (pd.DataFrame): 包含 'date' 列的数据框
            
        Raises:
            ValueError: 当数据包含非交易日时抛出异常
        """
        # 将输入数据的日期转换为与数据库相同的字符串格式 (YYYY-MM-DD)
        dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').unique()
        
        print("\n=== 检查日期信息 ===")
        print("待检查的日期:", dates)
        
        # 检查非交易日
        invalid_dates = [d for d in dates if d not in self.trading_dates]
        if invalid_dates:
            print("交易日集合中包含的部分日期:", sorted(list(self.trading_dates))[-10:])
            raise ValueError(f"数据包含非交易日: {invalid_dates}")
        
        # 检查时间格式（如果存在time列）
        self.check_time_format(df)
        
        # 检查时间频率
        self.check_time_frequency(df)
        
        print("=================\n")

def get_returns_from_db(dates, codes):
    """从数据库获取指定日期和股票代码的收益率数据
    
    Args:
        dates (list): 日期列表
        codes (list): 股票代码列表
        
    Returns:
        pd.DataFrame: 包含date、code、return列的收益率数据框
        
    Raises:
        Exception: 当数据库查询出错或数据不完整时抛出异常
    """
    try:
        client = get_client_U('r')
        returns_data = []
        
        for date in dates:
            # 查询该日期所有股票的收益率
            query = {
                "date": date,
                "code": {"$in": list(codes)}
            }
            daily_returns = list(client.basic_wind.w_vol_price.find(
                query,
                {"_id": 0, "code": 1, "pct_chg": 1, "date": 1}
            ))
            
            # 转换为DataFrame格式
            for record in daily_returns:
                returns_data.append({
                    'date': record['date'],
                    'code': record['code'],
                    'return': float(record['pct_chg']) / 100  # 转换为小数
                })
        
        client.close()
        returns = pd.DataFrame(returns_data)
        
        # 检查是否所有需要的数据都获取到了
        missing_dates = set(dates) - set(returns['date'].unique())
        if missing_dates:
            raise ValueError(f"数据库中缺少以下日期的收益率数据: {missing_dates}")
        
        missing_codes = set(codes) - set(returns['code'].unique())
        if missing_codes:
            raise ValueError(f"数据库中缺少以下股票的收益率数据: {missing_codes}")
        
        return returns
        
    except Exception as e:
        raise Exception(f"从数据库获取收益率数据时出错: {str(e)}")

def calculate_portfolio_metrics(weight_file, return_file=None):
    """计算投资组合的收益率和换手率
    
    Args:
        weight_file (str): 权重数据文件路径
        return_file (str, optional): 收益率数据文件路径。如果为None，将从数据库获取收益率数据
        
    Returns:
        tuple: (portfolio_returns, turnover)
            - portfolio_returns (pd.Series): 投资组合每期收益率
            - turnover (pd.Series): 投资组合每期换手率
    """
    # 读取权重数据
    weights = pd.read_csv(weight_file)
    
    # 如果没有提供收益率文件，从数据库获取数据
    if return_file is None:
        print("\n未提供收益率数据文件，将从数据库获取收益率数据...")
        unique_dates = weights['date'].unique()
        unique_codes = weights['code'].unique()
        returns = get_returns_from_db(unique_dates, unique_codes)
        print(f"成功从数据库获取了 {len(returns)} 条收益率记录")
    else:
        returns = pd.read_csv(return_file)
    
    # 判断是否为分钟频数据
    is_minute = 'time' in weights.columns
    
    # 设置索引
    if is_minute:
        weights['datetime'] = pd.to_datetime(weights['date'] + ' ' + weights['time'])
        returns['datetime'] = pd.to_datetime(returns['date'] + ' ' + returns['time'])
        index_cols = ['datetime', 'code']
    else:
        weights['date'] = pd.to_datetime(weights['date'])
        returns['date'] = pd.to_datetime(returns['date'])
        index_cols = ['date', 'code']
    
    # 将数据转换为宽格式
    weights_wide = weights.pivot(
        index=index_cols[0],
        columns='code',
        values='weight'
    )
    returns_wide = returns.pivot(
        index=index_cols[0],
        columns='code',
        values='return'
    )
    
    # 计算组合收益率
    portfolio_returns = (weights_wide * returns_wide).sum(axis=1)
    
    # 计算换手率
    weights_shift = weights_wide.shift(1)
    
    # 处理第一个时间点
    turnover = pd.Series(index=weights_wide.index)
    turnover.iloc[0] = weights_wide.iloc[0].abs().sum()  # 第一个时间点的换手率为权重绝对值之和
    
    # 计算其他时间点的换手率
    for i in range(1, len(weights_wide)):
        # 获取当前和前一时间点的权重
        curr_weights = weights_wide.iloc[i]
        prev_weights = weights_wide.iloc[i-1]
        
        # 计算前一时间点权重在当前时间点的理论值
        returns_t = returns_wide.iloc[i-1]
        theoretical_weights = prev_weights * (1 + returns_t)
        theoretical_weights = theoretical_weights / theoretical_weights.sum()  # 归一化
        
        # 计算换手率
        turnover.iloc[i] = np.abs(curr_weights - theoretical_weights).sum() / 2
    
    # 保存结果
    results = pd.DataFrame({
        'portfolio_return': portfolio_returns,
        'turnover': turnover
    })
    
    output_prefix = 'minute' if is_minute else 'daily'
    results.to_csv(f'csv_folder/test_{output_prefix}_portfolio_metrics.csv')
    
    print(f"已保存{output_prefix}频投资组合指标数据，共 {len(results)} 行")
    
    return portfolio_returns, turnover

if __name__ == "__main__":
    # 读取数据
    weights = pd.read_csv('csv_folder/test_minute_weight.csv')
    returns = pd.read_csv('csv_folder/test_minute_return.csv')
    
    # 检查交易日
    checker = DataChecker()
    checker.check_trading_dates(weights)
    checker.check_trading_dates(returns)
    
    # 计算分钟频的投资组合指标
    calculate_portfolio_metrics(
        'csv_folder/test_minute_weight.csv',
        'csv_folder/test_minute_return.csv'
    )
    
    # 读取日频数据并计算指标
    weights = pd.read_csv('csv_folder/test_daily_weight.csv')
    
    # 检查交易日
    checker.check_trading_dates(weights)
    
    # 计算日频的投资组合指标（将从数据库获取收益率数据）
    calculate_portfolio_metrics(
        'csv_folder/test_daily_weight.csv'
    ) 