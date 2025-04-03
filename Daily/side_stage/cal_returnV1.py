import os
import pandas as pd
import numpy as np
import time
from loguru import logger
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
OUTPUT_DIR = Path(__file__).parent / 'output'  # 使用当前文件所在目录下的output文件夹

# NPQ数据结构定义
D1_11_dtype = np.dtype([
    ('date', 'S64'),
    ('code', 'S64'),
    ('code_w', 'S64'),
    ('pct_chg', 'f8'),
    ('volume', 'f8'),
], align=True)

D1_11_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote', D1_11_dtype),
], align=True)

def read_npq_file(file_path):
    """读取NPQ文件并返回DataFrame"""
    npq_data = np.fromfile(file_path, dtype=D1_11_numpy_dtype)
    
    # 构建列名
    columns = [field for field in D1_11_numpy_dtype.fields if field != 'quote']
    columns.extend(D1_11_dtype.fields)
    
    # 处理数据
    rows = []
    for item in npq_data:
        row_data = {}
        # 处理非quote字段
        for field in D1_11_numpy_dtype.fields:
            if field != 'quote':
                row_data[field] = item[field]
        # 处理quote字段
        for quote_field in D1_11_dtype.fields:
            value = item['quote'][quote_field]
            if isinstance(value, bytes):
                try: value = value.decode('utf-8')
                except UnicodeDecodeError: value = None
            row_data[quote_field] = value
        rows.append(row_data)
    
    # 只保留date, code, pct_chg字段
    df = pd.DataFrame(rows, columns=columns)
    return df[['date', 'code', 'pct_chg']]

def read_all_npq_files(data_root, start_date=None, end_date=None):
    """遍历时间段目录读取NPQ文件"""
    load_start_time = time.time()  # 数据加载开始时间记录
    data_path = Path(data_root)
    all_dfs = []
    
    # 遍历所有日期子目录
    for date_dir in data_path.glob('*'):
        if date_dir.is_dir():
            npq_file = date_dir / "1" / "11.npq"
            try:
                df = read_npq_file(str(npq_file))
                df['date'] = date_dir.name  # 保留日期作为索引
                
                # 日期过滤
                if start_date and end_date:
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"跳过{date_dir}，加载失败: {str(e)}")
                continue
                
    load_end_time = time.time()  # 数据加载结束时间记录
    logger.info(f"数据加载完成，耗时: {load_end_time - load_start_time:.4f}s")  # 输出数据加载耗时

    return pd.concat(all_dfs).sort_values('date')

# 定义数据检查器
class DataChecker:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.trading_dates = self._fetch_trading_dates()

    def _fetch_trading_dates(self):
        """从CSV文件获取交易日"""
        try:
            # 使用 data_directory 构建 CSV 文件路径
            csv_path = os.path.join(self.data_directory, 'trade_dates_all.csv')
            trading_dates_df = pd.read_csv(csv_path)
            
            # 转换日期格式
            trading_dates_df['trade_date'] = pd.to_datetime(trading_dates_df['trade_date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')
            
            trading_dates_set = set(trading_dates_df['trade_date'])
            self._print_trading_dates_info(trading_dates_set)
            return trading_dates_set
        except Exception as e:
            print(f"\n=== CSV文件读取错误 ===\n错误信息: {str(e)}\n===================\n")
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
    def __init__(self, weight_file, return_file=None, use_equal_weights=True, data_directory='D:\\Data'):
        """初始化投资组合指标计算器"""
        self.weight_file = weight_file
        self.return_file = return_file
        self.use_equal_weights = use_equal_weights
        self.data_directory = data_directory  # 确保使用用户提供的路径或默认路径
        self.weights = None
        self.returns = None
        self.index_cols = None
        self.is_minute = None
        self.prepare_data()

    def prepare_data(self):
        """为投资组合指标计算准备数据。"""
        start_time = time.time()
        
        # 读取并转换权重数据
        weights_df = pd.read_csv(self.weight_file)
        self._validate_weights(weights_df)
        
        date_range = weights_df['date'].unique()# 获取日期范围
        returns_df = self._fetch_returns(weights_df) # 获取收益率数据
        self.dates, self.codes, self.weights_arr, self.returns_arr = self._convert_to_arrays(weights_df, returns_df)# 转换为numpy数组
        self.is_minute = 'time' in weights_df.columns# 设置数据频率标志
        
        print(f"数据准备总耗时: {time.time() - start_time:.2f}秒\n")

    def _fetch_returns(self, weights_df):
        """从文件或数据库获取收益率数据"""
        if self.return_file is None:
            print("\n未提供收益率数据文件，将从数据库获取收益率数据...")
            unique_dates = weights_df['date'].unique()
            unique_codes = weights_df['code'].unique()
            # 修改此处，传递正确的参数
            start_date = min(unique_dates)
            end_date = max(unique_dates)
            returns = self.get_returns_from_db(start_date, end_date, unique_codes)
            print(f"成功从数据库获取了 {len(returns)} 条收益率记录")
            return returns
        else:
            return pd.read_csv(self.return_file)

    def get_returns_from_db(self, start_date, end_date, codes):
        """从数据库获取收益率数据"""
        start_time = time.time()
        try:
            returns_data = []
            # 查询数据
            for date in pd.date_range(start=start_date, end=end_date):
                npq_file_path = Path(self.data_directory) / date.strftime('%Y-%m-%d') / "1" / "11.npq"
                if not npq_file_path.exists():
                    continue

                df = read_npq_file(str(npq_file_path))
                daily_returns = df[df['code'].isin(codes)]
                for _, record in daily_returns.iterrows():
                    returns_data.append({'date': record['date'], 'code': record['code'], 'return': float(record['pct_chg']) / 100})
        
            returns = pd.DataFrame(returns_data)
        
            print(f"获取收益率数据总耗时: {time.time() - start_time:.2f}秒\n")
            return returns
        
        except Exception as e:
            raise Exception(f"从NPQ文件获取收益率数据时出错: {str(e)}")

    def _validate_weights(self, weights_df):
        """验证权重数据"""
        required_weight_columns = ['date', 'code']
        missing_weight_columns = [col for col in required_weight_columns if col not in weights_df.columns]
        if missing_weight_columns:
            raise ValueError(f"权重表缺少必要的列: {missing_weight_columns}")

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

    def calculate_portfolio_metrics(self):
        """计算投资组合收益率和换手率"""
        start_time = time.time()
        portfolio_returns = np.sum(self.weights_arr * self.returns_arr, axis=1)  # 计算组合收益率
        turnover = self._calculate_turnover(self.weights_arr, self.returns_arr)  # 计算换手率
        self._save_results_array(self.dates, portfolio_returns, turnover)   # 保存结果
        
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
        """将结果保存到CSV文件
        
        保存格式：
        1. CSV文件：包含日期、组合收益率和换手率三列
        
        保存路径：
        - CSV: output/test_{频率}_portfolio_metrics.csv
        
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
        
        print(f"已保存{output_prefix}频投资组合指标数据，共 {len(results)} 行")

def main(data_directory, frequency, weight_path, return_file, use_equal_weights):
    """主函数，支持交互式和命令行参数两种调用方式"""
    from pathlib import Path
    
    print("=== 投资组合指标计算器 ===")
    
    abs_weight_path = str(Path(__file__).parent.parent / weight_path)
    weights_df = pd.read_csv(abs_weight_path)
    
    use_equal = 'y'
    if frequency == 'daily':
        if 'weight' in weights_df.columns:
            print("\n检测到权重列存在，将直接使用文件中的权重")
            use_equal = 'n'
        else:
            print("\n未检测到权重列")
            use_equal = input("是否使用等权重？[y/n] (默认y): ").lower() or 'y'
    
    # 初始化检查器，传递 data_directory 参数
    checker = DataChecker(data_directory)
    weights = pd.read_csv(abs_weight_path)
    checker.check_trading_dates(weights)
    if frequency == 'minute':
        checker.check_time_frequency(weights)
    
    # 数据验证通过提示
    print("数据验证通过")
    
    # 初始化组合指标计算器
    portfolio = PortfolioMetrics(
        weight_file=abs_weight_path,
        return_file=return_file,
        use_equal_weights=use_equal == 'y',
        data_directory=data_directory
    )
    
    # 执行计算并保存结果
    returns, turnover = portfolio.calculate_portfolio_metrics()
    print(f"\n计算完成！结果已保存至 output/ 目录")

# 修改 main 函数调用
if __name__ == "__main__":
    main(
        data_directory='D:\\Data',
        frequency='daily',
        weight_path='csv/test_daily_weight.csv',
        return_file=None,
        use_equal_weights=True
    )
