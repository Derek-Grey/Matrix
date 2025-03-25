"""
回测模块
包含回测类和绘图类
"""
import os
import time
import re  # 新增正则表达式模块导入
import pandas as pd
import numpy as np
from loguru import logger
import plotly.graph_objects as go
from pathlib import Path
from functools import wraps

# === 新增装饰器定义 ===
def log_function_call(func):
    """记录函数调用信息的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 跳过密码参数的日志记录
        sanitized_kwargs = {k: '*****' if 'password' in k else v for k, v in kwargs.items()}
        
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {sanitized_kwargs}")
        start_time = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} executed in {time.time()-start_time:.4f}s")
        return result
    return wrapper

def cache_data(func):
    """数据缓存装饰器"""
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 排除verbose参数对缓存键的影响
        key = (args, frozenset({k:v for k,v in kwargs.items() if k != 'verbose'}.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

OUTPUT_DIR = Path(__file__).parent / 'output'  # 使用当前文件所在目录下的output文件夹

# 新增NPQ数据结构定义
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

# 新增NPQ读取函数
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
    
    return pd.DataFrame(rows, columns=columns)

def read_all_npq_files(data_root):
    """遍历时间段目录读取NPQ文件"""
    data_path = Path(data_root)
    all_dfs = []
    
    # 遍历所有日期子目录
    for date_dir in data_path.glob('*'):
        if date_dir.is_dir():
            npq_file = date_dir / "1" / "11.npq"
            try:
                df = read_npq_file(str(npq_file))
                df['date'] = date_dir.name  # 保留日期作为索引
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"跳过{date_dir}，加载失败: {str(e)}")
                continue
                
    return pd.concat(all_dfs).sort_values('date')

class Backtest:
    def __init__(self, data_root, output_dir=Path('output')):
        """初始化支持时间段回测""" 
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 直接加载核心数据
        full_df = read_all_npq_files(data_root)
        self.stocks_matrix = full_df.pivot_table(
            index='date',
            columns='code',
            values='pct_chg'
        ).fillna(0)
        
        # 新增矩阵保存逻辑
        matrix_path = self.output_dir / 'stocks_matrix.csv'
        self.stocks_matrix.to_csv(matrix_path)
        logger.success(f"收益率矩阵已保存至: {matrix_path}")

        # 简化矩阵加载（保持原有功能）
        self._load_matrix('limit_matrix', r'D:\Derek\Code\Matrix\csv\aligned_limit_matrix.csv')
        self._load_matrix('risk_warning_matrix', r'D:\Derek\Code\Matrix\csv\aligned_riskwarning_matrix.csv')
        self._load_matrix('trade_status_matrix', r'D:\Derek\Code\Matrix\csv\aligned_trade_status_matrix.csv')
        self._load_matrix('score_matrix', r'D:\Derek\Code\Matrix\csv\aligned_score_matrix.csv')
        
        # 添加基础矩阵形状日志
        logger.debug(f"基础矩阵形状 | 涨跌停矩阵: {self.limit_matrix.shape}")
        logger.debug(f"基础矩阵形状 | 风险警示矩阵: {self.risk_warning_matrix.shape}")
        logger.debug(f"基础矩阵形状 | 交易状态矩阵: {self.trade_status_matrix.shape}")
        logger.debug(f"基础矩阵形状 | 评分矩阵: {self.score_matrix.shape}")

        # 将有效性矩阵生成移动到__init__方法内
        self._generate_validity_matrices()

    # ====================== 数据加载方法 ======================
    def _load_matrix(self, attr_name, file_path):
        """统一带错误处理的矩阵加载方法"""
        try:
            matrix = pd.read_csv(file_path, index_col=0)
            setattr(self, attr_name, matrix)
            logger.success(f"成功加载 {attr_name}: {file_path}")
        except Exception as e:
            logger.error(f"加载失败 {attr_name}: {str(e)}")
            setattr(self, attr_name, pd.DataFrame())

    # ====================== 核心矩阵生成 ====================== 
    def _generate_validity_matrices(self):
        """生成时间对齐后的有效性矩阵（最终版）"""
        common_index = self._get_common_index()
        
        # 对齐各矩阵时间范围
        self.risk_warning_validity = self._align_matrix(self.risk_warning_matrix, common_index)
        self.trade_status_validity = self._align_matrix(self.trade_status_matrix, common_index)
        self.limit_validity = self._align_matrix(self.limit_matrix, common_index)
        
        # 生成有效性矩阵（使用对齐后的矩阵）
        self.valid_stocks_matrix = (
            self.risk_warning_validity * 
            self.trade_status_validity * 
            self.limit_validity
        )

    # ====================== 工具方法 ======================
    def _get_common_index(self):
        """获取所有矩阵的交集时间索引（最终版）"""
        matrices = [
            self.stocks_matrix,
            self.limit_matrix,
            self.risk_warning_matrix,
            self.trade_status_matrix,
            self.score_matrix
        ]
        common_index = matrices[0].index
        for mat in matrices[1:]:
            if not mat.empty:
                common_index = common_index.intersection(mat.index)
        return common_index

    def _align_matrix(self, matrix, new_index):
        """矩阵时间对齐方法（最终版）"""
        if not matrix.empty:
            return matrix.reindex(index=new_index, method='ffill').fillna(0).astype(int)
        return pd.DataFrame(0, index=new_index, columns=matrix.columns)

    def _create_stocks_matrix(self, df):
        """从NPQ数据创建收益率矩阵"""
        # 筛选所需列并重塑为宽表格式
        return df.pivot_table(
            index='date',
            columns='code',
            values='pct_chg',
            aggfunc='first'
        ).fillna(0)

    def _generate_validity_matrices(self):
        """生成有效性矩阵和受限股票矩阵"""
        # 生成有效性矩阵：股票必须同时满足三个条件
        self.risk_warning_validity = (self.risk_warning_matrix == 0).astype(int)
        self.trade_status_validity = (self.trade_status_matrix == 1).astype(int)
        self.limit_validity = (self.limit_matrix == 0).astype(int)
        
        self.valid_stocks_matrix = (
            self.risk_warning_validity * 
            self.trade_status_validity * 
            self.limit_validity
        )
        
        # 受限股票矩阵：只考虑交易状态和涨跌停限制
        self.restricted_stocks_matrix = (
            self.trade_status_validity * self.limit_validity
        )


    def _get_common_index(self):
        """获取所有矩阵的交集时间索引"""
        matrices = [
            self.stocks_matrix,
            self.limit_matrix,
            self.risk_warning_matrix,
            self.trade_status_matrix,
            self.score_matrix
        ]
        common_index = matrices[0].index
        for mat in matrices[1:]:
            if not mat.empty:
                common_index = common_index.intersection(mat.index)
        return common_index

    
    @log_function_call
    def _update_positions(self, position_history, day, hold_count, rebalance_frequency):
        """更新持仓策略的持仓"""
        previous_positions = position_history.iloc[day - 1]["hold_positions"]
        current_date = position_history.index[day]

        # 评分矩阵空值检查
        if self.score_matrix.iloc[day - 1].isna().all():
            position_history.loc[current_date, "hold_positions"] = previous_positions
            return

        # 处理前一天的持仓
        previous_positions = (set() if pd.isna(previous_positions) 
                            else set(previous_positions.split(',')))
        previous_positions = {stock for stock in previous_positions 
                            if isinstance(stock, str) and stock.isalnum()}

        # 计算有效股票和受限股票
        valid_stocks = self.valid_stocks_matrix.iloc[day].astype(bool)
        restricted = self.restricted_stocks_matrix.iloc[day].astype(bool)
        previous_date = position_history.index[day - 1]
        valid_scores = self.score_matrix.loc[previous_date]

        # 处理受限股票
        restricted_stocks = [stock for stock in previous_positions 
                           if not restricted[stock]]

        # 再平衡处理
        if (day - 1) % rebalance_frequency == 0:
            sorted_stocks = valid_scores.sort_values(ascending=False)
            try:
                # 只保留固定策略逻辑
                top_stocks = sorted_stocks.iloc[:hold_count]

                retained_stocks = list(set(previous_positions) & 
                                    set(top_stocks) | 
                                    set(restricted_stocks))
                new_positions_needed = hold_count - len(retained_stocks)
                final_positions = set(retained_stocks)

                if new_positions_needed > 0:
                    new_stocks = sorted_stocks[valid_stocks].index
                    new_stocks = [stock for stock in new_stocks 
                                if stock not in final_positions]
                    final_positions.update(new_stocks[:new_positions_needed])
            except IndexError:
                logger.warning(f"日期 {current_date}: 可用股票数量不足，使用所有有效股票")
                final_positions = set(sorted_stocks[valid_stocks].index[:hold_count])
        else:
            final_positions = set(previous_positions)

        # 更新持仓信息
        position_history.loc[current_date, "hold_positions"] = ','.join(final_positions)

        # 计算每日收益率和换手率
        if previous_date in self.stocks_matrix.index:
            daily_returns = self.stocks_matrix.loc[current_date, 
                                                 list(final_positions)].astype(float)
            position_history.loc[current_date, "daily_return"] = daily_returns.mean()

        turnover_rate = (len(previous_positions - final_positions) / 
                        max(len(previous_positions), 1))
        position_history.at[current_date, "turnover_rate"] = turnover_rate

    @log_function_call
    def run_fixed_strategy(self, hold_count, rebalance_frequency, start_date=None, end_date=None):
        """运行带时间范围的策略""" 
        # 新增时间范围过滤
        if start_date and end_date:
            self.stocks_matrix = self.stocks_matrix.loc[start_date:end_date]
        
        start_time = time.time()
        position_history = self._initialize_position_history()
    
        for day in range(1, len(self.stocks_matrix)):
            self._update_positions(position_history, day, hold_count, rebalance_frequency)
        
        return self._process_results(position_history, start_time)

    def _process_results(self, position_history, start_time):
        """处理回测结果"""
        position_history = position_history.dropna(subset=["hold_positions"])
        position_history.loc[:, 'hold_count'] = position_history['hold_positions'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
        
        # 保存结果CSV
        csv_file = self.output_dir / 'strategy_results_fixed.csv'
        position_history.to_csv(csv_file)
        
        # 计算并返回核心指标
        stats = {
            'cumulative_return': (1 + position_history['daily_return']).cumprod().iloc[-1] - 1,
            'annualized_return': (1 + position_history['daily_return']).mean() ** 252 - 1,
            'avg_turnover': position_history['turnover_rate'].mean(),
            'avg_holdings': position_history['hold_count'].mean()
        }
        
        logger.info(f"\n=== 固定策略统计 ===")
        logger.info(f"累计收益率: {stats['cumulative_return']:.2%}")
        logger.info(f"年化收益率: {stats['annualized_return']:.2%}")
        logger.info(f"平均换手率: {stats['avg_turnover']:.2%}")
        logger.info(f"平均持仓量: {stats['avg_holdings']:.1f}")
        
        return stats

# 移除StrategyPlotter类

    def _show_loaded_matrices(self):
        """显示已加载矩阵的时间范围信息"""
        matrix_info = [
            ('收益率矩阵', self.stocks_matrix),
            ('涨跌停矩阵', self.limit_matrix),
            ('风险警示矩阵', self.risk_warning_matrix),
            ('交易状态矩阵', self.trade_status_matrix),
            ('评分矩阵', self.score_matrix)
        ]
        
        logger.info("\n=== 加载矩阵时间范围 ===")
        for name, mat in matrix_info:
            if not mat.empty:
                # 新增数据预览
                logger.info(f"{name:.<15} 日期范围: {mat.index[0]} 至 {mat.index[-1]} 股票数量: {len(mat.columns)}")
                logger.debug(f"{name}前3行数据:\n{mat.head(3).to_string()}")
            else:
                logger.warning(f"{name:.<15} 未成功加载数据")

        # 新增完整结构输出
        logger.debug("\n=== 矩阵结构预览 ===")
        for name, mat in matrix_info:
            if not mat.empty:
                logger.debug(f"{name} shape: {mat.shape}\nColumns: {mat.columns.tolist()[:5]}...")
            else:
                logger.warning(f"{name:.<15} 未成功加载数据")

    def _initialize_position_history(self):
        """初始化持仓历史记录"""
        # 假设持仓历史记录是一个DataFrame，初始化时包含日期索引和空的持仓信息
        position_history = pd.DataFrame(index=self.stocks_matrix.index)
        position_history['hold_positions'] = None
        position_history['daily_return'] = 0.0
        position_history['turnover_rate'] = 0.0
        return position_history

def run_strategy(backtest, strategy_name, hold_count, rebalance_frequency):
    """
    运行回测策略并保存结果
    
    Args:
        backtest: Backtest 实例
        strategy_name: 策略名称（仅支持fixed）
        hold_count: 固定持仓数量
        rebalance_frequency: 再平衡频率（天数）
    """
    logger.info(f"运行{strategy_name}策略...")
    try:
        if strategy_name == "fixed":
            results = backtest.run_fixed_strategy(
                hold_count=hold_count,
                rebalance_frequency=rebalance_frequency
            )
            logger.info(f"{strategy_name}策略完成")
            return results
        else:
            logger.error(f"不支持的策略类型: {strategy_name}")
            return None
    except Exception as e:
        logger.error(f"{strategy_name}策略执行失败: {e}")
        return None

def main(start_date="2010-08-02", end_date="2020-07-31", 
         hold_count=50, rebalance_frequency=1,
         data_directory='csv', output_directory='output'):
    """
    主函数，执行回测策略
    
    Args:
        start_date: 回测开始日期，格式："YYYY-MM-DD"
        end_date: 回测结束日期，格式："YYYY-MM-DD"
        hold_count: 持仓数量，默认50只股票
        rebalance_frequency: 再平衡频率（天数），默认每天再平衡
        data_directory: 数据目录，存放原始数据和中间数据
        output_directory: 输出目录，存放回测结果和图表
        
    Returns:
        DataFrame: 固定持仓的回测结果
    """
    try:
        logger.info(f"开始回测 - 时间范围: {start_date} 至 {end_date}")
        
        # 第一步：数据准备
        logger.info("加载数据...")
        data_loader = LoadData(start_date, end_date, data_directory)
        matrices = process_data(data_loader)
        
        # 第二步：初始化回测实例
        backtest = Backtest(*matrices, output_dir=output_directory)
        
        # 第三部：执行固定持仓策略回测
        fixed_results = run_strategy(backtest, "fixed", hold_count, rebalance_frequency)
        
        logger.info("回测完成")
        return fixed_results
    
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        raise

if __name__ == "__main__":
    # 更新测试代码（仅运行固定策略）
    backtester = Backtest(r"D:\Data")
    
    fixed_results = run_strategy(
        backtester, 
        strategy_name="fixed",
        hold_count=10,
        rebalance_frequency=5
    )