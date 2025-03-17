import pandas as pd
import numpy as np
from db_client import get_client_U

class DataChecker:
    def __init__(self):
        try:
            # 连接数据库获取交易日期
            client = get_client_U('r')
            db = client['economic']
            collection = db['trade_dates']
            trading_dates = list(collection.find({}, {'_id': 0, 'trade_date': 1}))
            client.close()
            
            self.trading_dates = set(d['trade_date'] for d in trading_dates)
            print(f"\n=== 交易日信息 ===")
            print(f"交易日总数: {len(self.trading_dates)}")
            print(f"最近的交易日(后5个):", sorted(list(self.trading_dates))[-5:])
            print("=================\n")
            
        except Exception as e:
            print(f"\n=== MongoDB连接错误 ===")
            print(f"错误信息: {str(e)}")
            print("===================\n")
            raise

    def check_data_format(self, df):
        """检查DataFrame的格式是否符合要求
        
        Args:
            df (pd.DataFrame): 输入的数据框
            
        Raises:
            ValueError: 当数据格式不符合要求时抛出异常
        """
        required_cols = {'date', 'code'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"数据框必须包含以下列: {required_cols}")
        
        # 检查是否为分钟频数据
        is_minute = 'time' in df.columns
        if is_minute:
            if not df['time'].dtype == 'object':
                raise ValueError("time列必须是字符串格式")
        
        # 检查date列格式
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                raise ValueError("date列必须是可转换为日期格式的数据")
        
        return is_minute

    def check_trading_dates(self, df):
        """检查日期是否为有效交易日
        
        Args:
            df (pd.DataFrame): 输入的数据框
            
        Raises:
            ValueError: 当包含非交易日数据时抛出异常
        """
        dates = df['date'].dt.strftime('%Y-%m-%d').unique()
        invalid_dates = [d for d in dates if d not in self.trading_dates]
        if invalid_dates:
            raise ValueError(f"数据包含非交易日: {invalid_dates}")

def calculate_portfolio_metrics_df(weights_df, returns_df=None):
    """使用DataFrame格式计算投资组合的收益率和换手率
    
    Args:
        weights_df (pd.DataFrame): 包含权重数据的DataFrame，必须包含date、code、weight列
        returns_df (pd.DataFrame, optional): 包含收益率数据的DataFrame，必须包含date、code、return列
            如果为None，将从数据库获取收益率数据
            
    Returns:
        pd.DataFrame: 包含portfolio_return和turnover的结果数据框
    """
    # 初始化数据检查器
    checker = DataChecker()
    
    # 检查权重数据格式
    is_minute = checker.check_data_format(weights_df)
    checker.check_trading_dates(weights_df)
    
    # 如果没有提供收益率数据，从数据库获取
    if returns_df is None:
        print("\n从数据库获取收益率数据...")
        unique_dates = weights_df['date'].dt.strftime('%Y-%m-%d').unique()
        unique_codes = weights_df['code'].unique()
        
        print(f"\n待查询的日期范围: {unique_dates}")
        print(f"待查询的股票代码: {unique_codes[:5]}... (共{len(unique_codes)}个)")
        
        try:
            client = get_client_U('r')
            returns_data = []
            
            for date in unique_dates:
                query = {
                    "date": date,
                    "code": {"$in": list(unique_codes)}
                }
                print(f"\n正在查询日期 {date} 的数据...")
                print(f"查询条件: {query}")
                
                # 先检查数据是否存在
                count = client.basic_wind.w_vol_price.count_documents(query)
                print(f"找到 {count} 条记录")
                
                daily_returns = list(client.basic_wind.w_vol_price.find(
                    query,
                    {"_id": 0, "code": 1, "pct_chg": 1, "date": 1}
                ))
                
                print(f"成功获取 {len(daily_returns)} 条数据")
                if daily_returns:
                    print("数据示例:", daily_returns[0])
                
                for record in daily_returns:
                    returns_data.append({
                        'date': pd.to_datetime(record['date']),
                        'code': record['code'],
                        'return': float(record['pct_chg']) / 100
                    })
            
            client.close()
            
            if not returns_data:
                # 尝试获取一些示例数据来检查数据库内容
                print("\n尝试获取数据库中的示例数据:")
                sample_data = list(client.basic_wind.w_vol_price.find().limit(5))
                print("数据库中的数据示例:", sample_data)
                raise ValueError("未能从数据库获取到任何收益率数据")
            
            returns_df = pd.DataFrame(returns_data)
            print(f"\n成功获取 {len(returns_df)} 条收益率记录")
            print("\n收益率数据示例:")
            print(returns_df.head())
            print("\n收益率数据列名:", returns_df.columns.tolist())
            
        except Exception as e:
            print("\n获取收益率数据时出现错误:")
            print(f"权重数据日期范围: {unique_dates}")
            print(f"权重数据股票代码: {unique_codes}")
            
            # 检查数据库连接
            try:
                print("\n检查数据库连接...")
                client = get_client_U('r')
                db_names = client.list_database_names()
                print(f"可用的数据库: {db_names}")
                if 'basic_wind' in db_names:
                    collections = client.basic_wind.list_collection_names()
                    print(f"basic_wind 数据库中的集合: {collections}")
                client.close()
            except Exception as db_error:
                print(f"数据库连接检查失败: {str(db_error)}")
            
            raise Exception(f"获取收益率数据失败: {str(e)}")
    else:
        # 检查收益率数据格式
        checker.check_data_format(returns_df)
        checker.check_trading_dates(returns_df)
    
    # 设置索引列
    if is_minute:
        weights_df['datetime'] = pd.to_datetime(weights_df['date'].astype(str) + ' ' + weights_df['time'])
        returns_df['datetime'] = pd.to_datetime(returns_df['date'].astype(str) + ' ' + returns_df['time'])
        index_col = 'datetime'
    else:
        index_col = 'date'
    
    # 转换为宽格式
    weights_wide = weights_df.pivot(
        index=index_col,
        columns='code',
        values='weight'
    )
    returns_wide = returns_df.pivot(
        index=index_col,
        columns='code',
        values='return'
    )
    
    # 计算组合收益率
    portfolio_returns = (weights_wide * returns_wide).sum(axis=1)
    
    # 计算换手率
    turnover = pd.Series(index=weights_wide.index)
    turnover.iloc[0] = weights_wide.iloc[0].abs().sum()
    
    for i in range(1, len(weights_wide)):
        curr_weights = weights_wide.iloc[i]
        prev_weights = weights_wide.iloc[i-1]
        
        # 计算前一时间点权重在当前时间点的理论值
        returns_t = returns_wide.iloc[i-1]
        theoretical_weights = prev_weights * (1 + returns_t)
        theoretical_weights = theoretical_weights / theoretical_weights.sum()
        
        # 计算换手率
        turnover.iloc[i] = np.abs(curr_weights - theoretical_weights).sum() / 2
    
    # 合并结果
    results = pd.DataFrame({
        'portfolio_return': portfolio_returns,
        'turnover': turnover
    })
    
    # 保存结果
    output_prefix = 'minute' if is_minute else 'daily'
    results.to_csv(f'csv_folder/test_{output_prefix}_portfolio_metrics_df.csv')
    print(f"\n已保存{output_prefix}频投资组合指标数据，共 {len(results)} 行")
    
    return results

if __name__ == "__main__":
    # 测试代码
    try:
        print("\n=== 测试分钟频数据 ===")
        weights_df = pd.read_csv('csv_folder/test_minute_weight.csv')
        returns_df = pd.read_csv('csv_folder/test_minute_return.csv')
        
        print("\n权重数据前几行:")
        print(weights_df.head())
        
        weights_df['date'] = pd.to_datetime(weights_df['date'])
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        
        results = calculate_portfolio_metrics_df(weights_df, returns_df)
        print("\n分钟频结果示例:")
        print(results.head())
        
        print("\n=== 测试日频数据 ===")
        daily_weights = pd.read_csv('csv_folder/test_daily_weight.csv')
        
        print("\n日频权重数据前几行:")
        print(daily_weights.head())
        
        daily_weights['date'] = pd.to_datetime(daily_weights['date'])
        
        results_daily = calculate_portfolio_metrics_df(daily_weights)
        print("\n日频结果示例:")
        print(results_daily.head())
        
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        print("\n请检查:")
        print("1. 数据库连接是否正常")
        print("2. 数据库中是否存在对应日期的数据")
        print("3. 股票代码格式是否匹配")
        print("4. 输入文件格式是否正确")
        raise