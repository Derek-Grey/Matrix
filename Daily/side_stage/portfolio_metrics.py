import sys
import os
import pandas as pd
import numpy as np
import time
from Utils.db_client import get_client_U
from urllib.parse import quote_plus

class DataChecker:
    def __init__(self):
        self.trading_dates = self._fetch_trading_dates()

    def _fetch_trading_dates(self):
        """从数据库获取交易日"""
        try:
            client = get_client_U('r')
            db = client['economic']
            collection = db['trade_dates']
            trading_dates = list(collection.find({}, {'_id': 0, 'trade_date': 1}))
            client.close()
            trading_dates_set = set(d['trade_date'] for d in trading_dates)
            self._print_trading_dates_info(trading_dates_set)
            return trading_dates_set
        except Exception as e:
            print(f"\n=== MongoDB连接错误 ===\n错误信息: {str(e)}\n===================\n")
            raise

    def _print_trading_dates_info(self, trading_dates):
        """打印交易日信息"""
        print("\n=== 交易日信息 ===")
        print(f"交易日总数: {len(trading_dates)}")
        print("交易日示例(前5个):", sorted(list(trading_dates))[:5])
        print("最近的交易日(后5个):", sorted(list(trading_dates))[-5:])
        print("=================\n")

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
        """检查数据是否包含非交易日"""
        dates = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d').unique()
        invalid_dates = [d for d in dates if d not in self.trading_dates]
        if invalid_dates:
            raise ValueError(f"数据包含非交易日: {invalid_dates}")

class PortfolioMetrics:
    def __init__(self, weight_file, return_file=None, use_equal_weights=True):
        """初始化投资组合指标计算器
        
        Args:
            weight_file (str): 权重文件路径，必须包含 'date' 和 'code' 列，可选包含 'weight' 列
            return_file (str, optional): 收益率文件路径。如果为None，将从数据库获取收益率数据
            use_equal_weights (bool): 当权重文件不包含 'weight' 列时，是否使用等权重。默认为True
        """
        self.weight_file = weight_file
        self.return_file = return_file
        self.use_equal_weights = use_equal_weights
        self.weights = None  # 存储权重数据
        self.returns = None  # 存储收益率数据
        self.index_cols = None  # 存储索引列
        self.is_minute = None  # 标记是否为分钟频数据
        self.prepare_data()

    def prepare_data(self):
        """为投资组合指标计算准备数据。"""
        start_time = time.time()
        
        # 读取并转换权重数据
        weights_df = pd.read_csv(self.weight_file)
        self._validate_weights(weights_df)
        
        # 获取收益率数据
        returns_df = self._fetch_returns(weights_df)
        
        # 转换为numpy数组
        self.dates, self.codes, self.weights_arr, self.returns_arr = self._convert_to_arrays(weights_df, returns_df)
        
        # 设置数据频率标志
        self.is_minute = 'time' in weights_df.columns
        
        print(f"数据准备总耗时: {time.time() - start_time:.2f}秒\n")

    def _validate_weights(self, weights_df):
        """验证权重数据"""
        required_weight_columns = ['date', 'code']
        missing_weight_columns = [col for col in required_weight_columns if col not in weights_df.columns]
        if missing_weight_columns:
            raise ValueError(f"权重表缺少必要的列: {missing_weight_columns}")

    def _fetch_returns(self, weights_df):
        """从文件或数据库获取收益率数据"""
        if self.return_file is None:
            print("\n未提供收益率数据文件，将从数据库获取收益率数据...")
            unique_dates = weights_df['date'].unique()
            unique_codes = weights_df['code'].unique()
            returns = self.get_returns_from_db(unique_dates, unique_codes)
            print(f"成功从数据库获取了 {len(returns)} 条收益率记录")
            return returns
        else:
            return pd.read_csv(self.return_file)

    def _convert_to_arrays(self, weights_df, returns_df):
        """将DataFrame转换为numpy数组，并处理等权重"""
        # 获取唯一的日期和股票代码
        dates = weights_df['date'].unique()
        codes = weights_df['code'].unique()
        
        # 创建空的权重和收益率矩阵
        n_dates = len(dates)
        n_codes = len(codes)
        weights_arr = np.zeros((n_dates, n_codes))
        returns_arr = np.zeros((n_dates, n_codes))
        
        # 创建日期和代码的映射字典以加速查找
        date_idx = {date: i for i, date in enumerate(dates)}
        code_idx = {code: i for i, code in enumerate(codes)}
        
        # 填充权重矩阵
        if 'weight' in weights_df.columns:
            for _, row in weights_df.iterrows():
                i = date_idx[row['date']]
                j = code_idx[row['code']]
                weights_arr[i, j] = row['weight']
        else:
            # 使用等权重
            if self.use_equal_weights:
                print("权重列缺失，使用等权重")
                weights_per_date = 1.0 / weights_df.groupby('date')['code'].transform('count').values
                for idx, row in weights_df.iterrows():
                    i = date_idx[row['date']]
                    j = code_idx[row['code']]
                    weights_arr[i, j] = weights_per_date[idx]
            else:
                raise ValueError("权重列缺失，且未设置使用等权重")
        
        # 填充收益率矩阵
        for _, row in returns_df.iterrows():
            i = date_idx[row['date']]
            j = code_idx[row['code']]
            returns_arr[i, j] = row['return']
        
        return dates, codes, weights_arr, returns_arr

    def get_returns_from_db(self, dates, codes):
        """从数据库获取收益率数据"""
        start_time = time.time()
        try:
            client = get_client_U('r')
            returns_data = []
            
            # 查询数据
            for date in dates:
                query = {"date": date, "code": {"$in": list(codes)}}
                daily_returns = list(client.basic_wind.w_vol_price.find(query, {"_id": 0, "code": 1, "pct_chg": 1, "date": 1}))
                for record in daily_returns:
                    returns_data.append({'date': record['date'], 'code': record['code'], 'return': float(record['pct_chg']) / 100})

            # 转换为DataFrame
            returns = pd.DataFrame(returns_data)
            
            # 验证数据
            self._validate_fetched_returns(returns, dates, codes)
            
            client.close()
            print(f"获取收益率数据总耗时: {time.time() - start_time:.2f}秒\n")
            return returns
            
        except Exception as e:
            raise Exception(f"从数据库获取收益率数据时出错: {str(e)}")

    def _validate_fetched_returns(self, returns, dates, codes):
        """验证从数据库获取的收益率数据"""
        if returns.empty:
            raise ValueError("从数据库获取的收益率数据为空")
        missing_dates = set(dates) - set(returns['date'].unique())
        if missing_dates:
            raise ValueError(f"数据库中缺少以下日期的收益率数据: {missing_dates}")
        missing_codes = set(codes) - set(returns['code'].unique())
        if missing_codes:
            raise ValueError(f"数据库中缺少以下股票的收益率数据: {missing_codes}")

    def calculate_portfolio_metrics(self):
        """计算投资组合收益率和换手率"""
        start_time = time.time()
        
        # 计算组合收益率
        portfolio_returns = np.sum(self.weights_arr * self.returns_arr, axis=1)

        # 计算换手率
        turnover = self._calculate_turnover(self.weights_arr, self.returns_arr)
        
        # 保存结果
        self._save_results_array(self.dates, portfolio_returns, turnover)
        
        print(f"计算指标总耗时: {time.time() - start_time:.2f}秒\n")
        return portfolio_returns, turnover

    def _calculate_turnover(self, weights_arr, returns_arr):
        """计算换手率
        
        计算方法：
        1. 第一期换手率为权重绝对值之和
        2. 后续期间：
           a. 计算理论权重（前一期权重考虑收益率变化后的权重）
           b. 计算当前实际权重与理论权重的差异
           c. 换手率 = 差异绝对值之和 / 2
        
        Args:
            weights_arr (numpy.ndarray): 权重矩阵，形状为 (n_periods, n_stocks)
            returns_arr (numpy.ndarray): 收益率矩阵，形状为 (n_periods, n_stocks)
        
        Returns:
            numpy.ndarray: 换手率序列，长度为 n_periods
        """
        n_periods = len(weights_arr)
        turnover = np.zeros(n_periods)
        
        # 第一期换手率就是权重之和
        turnover[0] = np.sum(np.abs(weights_arr[0]))
        
        # 计算后续期间的换手率
        for i in range(1, n_periods):
            # 获取当前权重和前一期权重
            curr_weights = weights_arr[i]
            prev_weights = weights_arr[i-1]
            prev_returns = returns_arr[i-1]
            
            # 计算理论权重
            theoretical_weights = prev_weights * (1 + prev_returns)
            theoretical_weights = theoretical_weights / np.sum(theoretical_weights)
            
            # 计算换手率
            turnover[i] = np.sum(np.abs(curr_weights - theoretical_weights)) / 2
        
        return turnover

    def _save_results_array(self, dates, portfolio_returns, turnover):
        """将结果保存到CSV文件和NPY文件
        
        保存格式：
        1. CSV文件：包含日期、组合收益率和换手率三列
        2. NPY文件：包含日期数组、组合收益率数组和换手率数组的字典
        
        保存路径：
        - CSV: output/test_{频率}_portfolio_metrics.csv
        - NPY: output/test_{频率}_portfolio_metrics.npy
        
        Args:
            dates (array-like): 日期序列
            portfolio_returns (numpy.ndarray): 组合收益率序列
            turnover (numpy.ndarray): 换手率序列
        """
        output_prefix = 'minute' if self.is_minute else 'daily'
        
        # 保存CSV格式
        results = np.column_stack((dates, portfolio_returns, turnover))
        np.savetxt(
            f'output/test_{output_prefix}_portfolio_metrics.csv',
            results,
            delimiter=',',
            header='date,portfolio_return,turnover',
            fmt=['%s', '%.6f', '%.6f'],
            comments=''
        )
        
        # 保存NPY格式
        np_results = {
            'dates': np.array(dates),
            'portfolio_returns': portfolio_returns,
            'turnover': turnover
        }
        np.save(f'output/test_{output_prefix}_portfolio_metrics.npy', np_results)
        
        print(f"已保存{output_prefix}频投资组合指标数据，共 {len(results)} 行")
        print(f"数据已同时保存为 CSV 和 NPY 格式")

def main():
    """主函数，支持交互式和命令行参数两种调用方式"""
    from pathlib import Path
    
    print("=== 投资组合指标计算器 ===")
    weight_path = input("请输入权重文件路径 (默认csv/test_daily_weight.csv): ") or 'csv/test_daily_weight.csv'
    
    # 处理路径解析
    abs_weight_path = str(Path(__file__).parent.parent / weight_path)
    weights_df = pd.read_csv(abs_weight_path)
    
    # 自动检测权重列并提示
    if 'weight' in weights_df.columns:
        print("\n检测到权重列存在，将直接使用文件中的权重")
        use_equal = 'n'
    else:
        print("\n未检测到权重列")
        use_equal = input("是否使用等权重？[y/n] (默认y): ").lower() or 'y'
    
    return_file = input("请输入收益率文件路径（留空则从数据库获取）: ") or None

    # 初始化检查器
    checker = DataChecker()
    weights = pd.read_csv(abs_weight_path)
    checker.check_trading_dates(weights)
    
    # 初始化组合指标计算器
    portfolio = PortfolioMetrics(
        weight_file=abs_weight_path,
        return_file=return_file,
        use_equal_weights=use_equal == 'y'
    )
    
    # 执行计算并保存结果
    returns, turnover = portfolio.calculate_portfolio_metrics()
    print(f"\n计算完成！结果已保存至 output/ 目录")

if __name__ == "__main__":
    main()

# 新增模块调用接口
def calculate_from_args(weight_path, return_file=None, use_equal=True):
    """
    编程式调用接口
    
    Args:
        weight_path: 权重文件路径（相对或绝对路径）
        return_file: 收益率文件路径（可选）
        use_equal: 是否使用等权重（默认为True）
    """
    from pathlib import Path
    
    abs_weight_path = str(Path(__file__).parent.parent / weight_path)
    
    checker = DataChecker()
    weights = pd.read_csv(abs_weight_path)
    checker.check_trading_dates(weights)
    
    portfolio = PortfolioMetrics(
        weight_file=abs_weight_path,
        return_file=return_file,
        use_equal_weights=use_equal
    )
    return portfolio.calculate_portfolio_metrics()
