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
    """遍历数据根目录读取所有NPQ文件"""
    all_dfs = []
    data_root = Path(data_root)
    
    # 新增路径解析调试
    logger.debug(f"解析后的数据根目录: {data_root.resolve()}")
    logger.debug(f"根目录存在: {data_root.exists()}")
    logger.debug(f"根目录是文件夹: {data_root.is_dir()}")

    # 直接处理单个日期目录
    sub_dir = data_root / "1"
    logger.debug(f"子目录路径: {sub_dir.resolve()}")
    
    if not sub_dir.is_dir():
        # 添加详细目录列表
        logger.error(f"子目录内容: {[p.name for p in data_root.glob('*')]}")
        raise FileNotFoundError(f"缺失子目录: {sub_dir}")
    
    npq_path = sub_dir / "11.npq"
    # 新增路径净化处理
    clean_npq_path = Path(os.path.normcase(os.path.abspath(npq_path)))  # 标准化路径格式
    logger.debug(f"净化后路径: {clean_npq_path}")
    
    # 新增文件扩展名检查
    actual_files = [f.name for f in sub_dir.glob('*')]
    logger.debug(f"实际文件名列表: {actual_files}")
    
    if not clean_npq_path.exists():
        # 新增相似文件匹配
        similar_files = [f for f in sub_dir.glob('*11*') if f.is_file()]
        logger.error(f"发现相似文件: {[f.name for f in similar_files]}")
        raise FileNotFoundError(f"NPQ文件不存在，请检查文件名是否准确: {clean_npq_path}")
    
    logger.debug(f"NPQ文件绝对路径: {npq_path.resolve()}")
    logger.debug(f"文件存在: {npq_path.exists()}")
    
    if not npq_path.exists():
        # 添加文件系统检查
        logger.error(f"该路径下实际文件: {[f.name for f in sub_dir.glob('*')]}")
        raise FileNotFoundError(f"NPQ文件不存在: {npq_path}")
    
    try:
        df = read_npq_file(str(npq_path))
        df['date'] = data_root.name
        all_dfs.append(df)
        logger.success(f"成功加载单日数据: {npq_path}")
    except Exception as e:
        logger.error(f"加载失败 {npq_path}: {str(e)}")
        raise
    
    return pd.concat(all_dfs).sort_values('date')

class Backtest:
    def __init__(self, data_root, output_dir=OUTPUT_DIR):
        """初始化回测类（支持批量读取）"""

        # 从本地读取其他矩阵（增加错误处理）
        self._load_matrix_with_fallback('limit_matrix', r'D:\Derek\Code\Matrix\csv\aligned_limit_matrix.csv')
        self._load_matrix_with_fallback('risk_warning_matrix', r'D:\Derek\Code\Matrix\csv\aligned_riskwarning_matrix.csv')
        self._load_matrix_with_fallback('trade_status_matrix', r'D:\Derek\Code\Matrix\csv\aligned_trade_status_matrix.csv')
        self._load_matrix_with_fallback('score_matrix', r'D:\Derek\Code\Matrix\csv\aligned_score_matrix.csv')

        # 批量读取NPQ数据
        full_df = read_all_npq_files(data_root)
        # 生成股票收益率矩阵（保持独立时间）
        self.stocks_matrix = self._create_stocks_matrix(full_df)
        
        # 加载其他矩阵并保持各自时间范围
        self._load_matrix_with_fallback('limit_matrix', r'D:\Derek\Code\Matrix\csv\aligned_limit_matrix.csv')
        self._load_matrix_with_fallback('risk_warning_matrix', r'D:\Derek\Code\Matrix\csv\aligned_riskwarning_matrix.csv')
        self._load_matrix_with_fallback('trade_status_matrix', r'D:\Derek\Code\Matrix\csv\aligned_trade_status_matrix.csv')
        self._load_matrix_with_fallback('score_matrix', r'D:\Derek\Code\Matrix\csv\aligned_score_matrix.csv')

        # 显示独立时间范围
        self._show_loaded_matrices()  # <-- 这里被调用
        
        # 生成对齐后的有效性矩阵
        self._generate_validity_matrices()

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
        """生成时间对齐后的有效性矩阵"""
        # 获取所有矩阵的共同时间索引
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

    def _align_matrix(self, matrix, new_index):
        """矩阵时间对齐方法"""
        if not matrix.empty:
            return matrix.reindex(index=new_index, method='ffill').fillna(0).astype(int)
        return pd.DataFrame(0, index=new_index, columns=matrix.columns)

    def _load_matrix_with_fallback(self, attr_name, file_path):
        """带错误处理的矩阵加载方法"""
        try:
            matrix = pd.read_csv(file_path, index_col=0)
            setattr(self, attr_name, matrix)
            logger.success(f"成功加载 {attr_name}: {file_path}")
        except Exception as e:
            logger.error(f"加载失败 {attr_name}: {str(e)}")
            setattr(self, attr_name, pd.DataFrame())  # 创建空DataFrame防止后续崩溃

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
    
    @log_function_call
    def _update_positions(self, position_history, day, hold_count, 
                         rebalance_frequency):
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
    def run_fixed_strategy(self, hold_count, rebalance_frequency):
        """运行固定持仓策略"""
        start_time = time.time()
        position_history = self._initialize_position_history()

        for day in range(1, len(self.stocks_matrix)):
            # 移除不再需要的参数
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
                logger.info(f"{name:.<15} 日期范围: {mat.index[0]} 至 {mat.index[-1]} 股票数量: {len(mat.columns)}")
            else:
                logger.warning(f"{name:.<15} 未成功加载数据")

if __name__ == "__main__":
    backtester = Backtest(r"D:\Data\2025-02-25")
    # 运行策略
    stats = backtester.run_fixed_strategy(
        hold_count=10,
        rebalance_frequency=5
    )
    print(f"回测结果: {stats}")