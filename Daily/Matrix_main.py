
# %% 
# 导入所需的库
import os
import time
import math
import pandas as pd
import numpy as np
from loguru import logger
from utils import trans_str_to_float64, get_client_U
import plotly.graph_objects as go
from Matrix_backtest import Backtest   # 导入回测类

# 设置选择加入未来行为的选项
pd.set_option('future.no_silent_downcasting', True)

# %% 
# 数据处理类
# 定义对齐矩阵的函数
def align_and_fill_matrix(target_matrix: pd.DataFrame, reference_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    将目标矩阵与参考矩阵的列对齐，并用0填充缺失值
    
    Args:
        target_matrix: 需要对齐的目标矩阵
        reference_matrix: 作为参考的矩阵
        
    Returns:
        aligned_matrix: 对齐后的矩阵
    """
    try:
        # 重新索引目标矩阵，确保列与参考矩阵一致，缺失值填充为0
        aligned_matrix = target_matrix.reindex(columns=reference_matrix.columns, fill_value=0)
        return aligned_matrix
    except Exception as e:
        logger.error(f"对齐矩阵失败: {e}")
        raise

# 文件检查和时间范围的确认函数
def process_data(data_loader):
    try:
        # 获取开始和结束日期
        start_date = pd.to_datetime(data_loader.date_s)
        end_date = pd.to_datetime(data_loader.date_e)

        # 创建数据目录（如果不存在）
        os.makedirs(data_loader.data_folder, exist_ok=True)

        # 检查文件是否存在
        raw_files = ['raw_wind_data.csv', 'raw_stocks_info.csv', 'raw_limit_info.csv']
        aligned_files = [
            'aligned_stocks_matrix.csv', 'aligned_limit_matrix.csv',
            'aligned_riskwarning_matrix.csv', 'aligned_trade_status_matrix.csv',
            'aligned_score_matrix.csv'
        ]
        
        raw_files_exist = all(os.path.exists(os.path.join(data_loader.data_folder, f)) for f in raw_files)
        aligned_files_exist = all(os.path.exists(os.path.join(data_loader.data_folder, f)) for f in aligned_files)

        # 尝试从现有文件加载数据
        if raw_files_exist and aligned_files_exist:
            logger.debug('发现现有对齐数据文件，尝试加载...')
            try:
                # 读取原始文件的日期范围
                wind_df = pd.read_csv(os.path.join(data_loader.data_folder, 'raw_wind_data.csv'))
                wind_df['date'] = pd.to_datetime(wind_df['date'])
                raw_start_date = wind_df['date'].min()
                raw_end_date = wind_df['date'].max()

                # 检查日期范围是否符合要求
                if start_date >= raw_start_date and end_date <= raw_end_date:
                    logger.debug('设定日期范围在原始文件日期范围内，加载对齐数据...')
                    
                    # 加载对齐文件
                    aligned_stocks_matrix = pd.read_csv(
                        os.path.join(data_loader.data_folder, 'aligned_stocks_matrix.csv'), 
                        index_col=0, parse_dates=True
                    )
                    aligned_limit_matrix = pd.read_csv(
                        os.path.join(data_loader.data_folder, 'aligned_limit_matrix.csv'), 
                        index_col=0, parse_dates=True
                    )
                    aligned_riskwarning_matrix = pd.read_csv(
                        os.path.join(data_loader.data_folder, 'aligned_riskwarning_matrix.csv'), 
                        index_col=0, parse_dates=True
                    )
                    aligned_trade_status_matrix = pd.read_csv(
                        os.path.join(data_loader.data_folder, 'aligned_trade_status_matrix.csv'), 
                        index_col=0, parse_dates=True
                    )
                    score_matrix = pd.read_csv(
                        os.path.join(data_loader.data_folder, 'aligned_score_matrix.csv'), 
                        index_col=0, parse_dates=True
                    )

                    # 截取所需日期范围的数据
                    date_mask = lambda df: (df.index >= start_date) & (df.index <= end_date)
                    aligned_stocks_matrix = aligned_stocks_matrix[date_mask(aligned_stocks_matrix)]
                    aligned_limit_matrix = aligned_limit_matrix[date_mask(aligned_limit_matrix)]
                    aligned_riskwarning_matrix = aligned_riskwarning_matrix[date_mask(aligned_riskwarning_matrix)]
                    aligned_trade_status_matrix = aligned_trade_status_matrix[date_mask(aligned_trade_status_matrix)]
                    score_matrix = score_matrix[date_mask(score_matrix)]

                    return (aligned_stocks_matrix, aligned_limit_matrix,
                           aligned_riskwarning_matrix, aligned_trade_status_matrix,
                           score_matrix)

            except Exception as e:
                logger.warning(f"加载现有对齐数据失败: {e}")
                logger.debug("将重新生成对齐数据...")

        # 如果无法从现有文件加载，则重新生成数据
        logger.debug('生成新的对齐数据...')
        df_stocks, trade_status_matrix, riskwarning_matrix, limit_matrix = data_loader.get_stocks_info()
        score_matrix = data_loader.generate_score_matrix('stra_V3_11.csv')

        # 对齐数据
        aligned_stocks_matrix = align_and_fill_matrix(df_stocks, score_matrix)
        aligned_limit_matrix = align_and_fill_matrix(limit_matrix, score_matrix)
        aligned_riskwarning_matrix = align_and_fill_matrix(riskwarning_matrix, score_matrix)
        aligned_trade_status_matrix = align_and_fill_matrix(trade_status_matrix, score_matrix)

        # 保存对齐后的矩阵
        for matrix, filename in [
            (aligned_stocks_matrix, 'aligned_stocks_matrix.csv'),
            (aligned_limit_matrix, 'aligned_limit_matrix.csv'),
            (aligned_riskwarning_matrix, 'aligned_riskwarning_matrix.csv'),
            (aligned_trade_status_matrix, 'aligned_trade_status_matrix.csv'),
            (score_matrix, 'aligned_score_matrix.csv')
        ]:
            matrix.to_csv(os.path.join(data_loader.data_folder, filename))

        return (aligned_stocks_matrix, aligned_limit_matrix,
                aligned_riskwarning_matrix, aligned_trade_status_matrix,
                score_matrix)

    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise
    
# %%
# 定义数据加载类
class LoadData:
    # 保存原始数据为CSV文件
    def save_raw_data_to_csv(self, df: pd.DataFrame, file_name: str):
        try:
            logger.debug('将原始数据保存为CSV文件...')
            csv_file_path = os.path.join(self.data_folder, file_name)
            df.to_csv(csv_file_path, header=True, index=False)
            logger.debug(f'原始数据已保存到 {csv_file_path}')
        except Exception as e:
            logger.error(f"保存原始数据到CSV文件失败: {e}")
            raise

    # 初始化加载数据类
    def __init__(self, date_s: str, date_e: str, data_folder: str):
        if date_s is None or date_e is None:
            raise ValueError('必须指定起止日期！！！')
        self.client_U = get_client_U()  # 获取客户端连接
        self.date_s, self.date_e = date_s, date_e  # 存储开始和结束日期
        self.data_folder = data_folder  # 存储数据目录
        os.makedirs(self.data_folder, exist_ok=True)  # 创建数据目录

    # 获取WIND的日频涨跌幅数据
    def get_chg_wind(self) -> pd.DataFrame:
        try:
            logger.debug('加载WIND的日频涨跌幅数据...')
            df = pd.DataFrame(self.client_U.basic_wind.w_vol_price.find(
                {"date": {"$gte": self.date_s, "$lte": self.date_e}},
                {"_id": 0, "date": 1, "code": 1, "pct_chg": 1},
                batch_size=1000000))
            df = trans_str_to_float64(df, trans_cols=['pct_chg'])  # 转换数据类型
            df['date'] = pd.to_datetime(df['date'])  # 转期格式
            pivot_df = df.pivot_table(index='date', columns='code', values='pct_chg')  # 创建透视表

            # 保存原始数据
            self.save_raw_data_to_csv(df, 'raw_wind_data.csv')

            return pivot_df  # 返回透视表
        except Exception as e:
            logger.error(f"加载WIND数据失败: {e}")
            raise

    # 获取股票信息
    def get_stocks_info(self) -> tuple:
        try:
            logger.debug('从数据库加载股票信息...')
            t_info = self.client_U.basic_wind.w_basic_info
            t_limit = self.client_U.basic_jq.jq_daily_price_none

            # 从数据库查询股票基本信息
            df_info = pd.DataFrame(t_info.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                            {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
                                            batch_size=1000000))
            df_info['date'] = pd.to_datetime(df_info['date'])  # 转换日期格式
            df_stocks = self.get_chg_wind()  # 获取涨跌幅数据

            # 保存股票信息原始数据
            self.save_raw_data_to_csv(df_info, 'raw_stocks_info.csv')

            # 加载涨跌停信息
            use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
            df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": self.date_s, "$lte": self.date_e}}, use_cols, batch_size=1000000))
            df_limit['date'] = pd.to_datetime(df_limit['date'])  # 转换日期格式
            df_limit['limit'] = df_limit.apply(lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"], axis=1)
            df_limit['limit'] = df_limit['limit'].astype('int')  # 转换为整数类型

            # 保存涨跌停原始数据
            self.save_raw_data_to_csv(df_limit, 'raw_limit_info.csv')

            # 创建各类矩阵
            limit_matrix = df_limit.pivot(index='date', columns='code', values='limit')
            trade_status_matrix = df_info.pivot(index='date', columns='code', values='trade_status')
            riskwarning_matrix = df_info.pivot(index='date', columns='code', values='riskwarning')

            return df_stocks, trade_status_matrix, riskwarning_matrix, limit_matrix  # 返回四个数据框
        except Exception as e:
            logger.error(f"加载股票信息失败: {e}")
            raise

    # 从CSV文件生成评分矩阵
    def generate_score_matrix(self, file_name: str) -> pd.DataFrame:
        try:
            csv_file_path = os.path.join(self.data_folder, file_name)
            logger.debug(f"从本地文件 {csv_file_path} 加载数据...")
            df = pd.read_csv(csv_file_path)  # 读取CSV文件
            df['date'] = pd.to_datetime(df['date'])  # 转换日期格式
            df.set_index('date', inplace=True)  # 设置日期为索引
            score_matrix = df.pivot_table(index='date', columns='code', values='F1')  # 创建透视表
            return score_matrix  # 返回评分矩阵
        except Exception as e:
            logger.error(f"加载评分矩阵失败: {e}")
            raise

    # 保存矩阵数据为CSV文件
    def save_matrix_to_csv(self, df: pd.DataFrame, file_name: str):
        try:
            logger.debug('将数据转换为阵格式并保存为CSV文件...')
            csv_file_path = os.path.join(self.data_folder, file_name)
            df.to_csv(csv_file_path, header=True)  # 保存数据
            logger.debug(f'数据已保存到 {csv_file_path}')
        except Exception as e:
            logger.error(f"保存矩阵到CSV文件失败: {e}")
            raise

    # 获取指定日期范围内的所有交易日
    def _get_all_trade_days(self):
        """
        获取指定日期范围内的所有交易日。
        """
        try:
            logger.debug('获取所有交易日...')
            df_dates = pd.DataFrame(self.client_U.economic.trade_dates.find(
                {"trade_date": {"$gte": self.date_s, "$lte": self.date_e}},
                {'_id': 0, 'trade_date': 1}
            ))
            trade_days = df_dates['trade_date'].tolist()
            logger.debug(f'获取到的交易日数量: {len(trade_days)}')
            return trade_days
        except Exception as e:
            logger.error(f"获取交易日失败: {e}")
            raise

# %% 
# 主函数
def main(start_date="2010-08-02", end_date="2020-07-31", 
         hold_count=50, rebalance_frequency=1,
         data_directory='data', output_directory='output',
         run_fixed=True, run_dynamic=True):
    """
    主函数，执行回测策略
    
    Args:
        start_date: 回测开始日期，格式："YYYY-MM-DD"
        end_date: 回测结束日期，格式："YYYY-MM-DD"
        hold_count: 持仓数量，默认50只股票
        rebalance_frequency: 再平衡频率（天数），默认每天再平衡
        data_directory: 数据目录，存放原始数据和中间数据
        output_directory: 输出目录，存放回测结果和图表
        run_fixed: 是否运行固定持仓策略，默认True
        run_dynamic: 是否运行动态持仓策略，默认True
        
    Returns:
        tuple: (fixed_results, dynamic_results) 包含固定持仓和动态持仓的回测结果
    """
    try:
        # 记录回测开始信息
        logger.info(f"开始回测 - 时间范围: {start_date} 至 {end_date}")
        
        # 第一步：数据准备
        logger.info("加载数据...")
        data_loader = LoadData(start_date, end_date, data_directory)  # 初始化数据加载器
        matrices = process_data(data_loader)  # 处理数据，返回所需的矩阵
        
        # 第二步：初始化回测实例
        backtest = Backtest(*matrices, output_dir=output_directory)  # 创建回测对象
        
        # 用于存储回测结果的字典
        results = {}
        
        # 第三步：执行固定持仓策略回测
        if run_fixed:
            logger.info("运行固定持仓策略...")
            try:
                # 运行固定持仓回测
                fixed_results = backtest.run_fixed_strategy(
                    hold_count=hold_count,  # 固定持仓数量
                    rebalance_frequency=rebalance_frequency,  # 再平衡频率
                    strategy_name="fixed"  # 策略名称
                )
                # 绘制回测结果图表
                backtest.plot_results(fixed_results, "固定持仓")
                results['fixed'] = fixed_results
                logger.info("固定持仓策略完成")
            except Exception as e:
                logger.error(f"固定持仓策略执行失败: {e}")
                results['fixed'] = None
        
        # 第四步：执行动态持仓策略回测
        if run_dynamic:
            logger.info("运行动态持仓策略...")
            try:
                # 运行动态持仓回测
                dynamic_results = backtest.run_dynamic_strategy(
                    hold_count=hold_count,  # 基础持仓数量
                    rebalance_frequency=rebalance_frequency,  # 再平衡频率
                    start_sorted=100  # 排序起始位置
                )
                # 绘制回测结果图表
                backtest.plot_results(dynamic_results, "动态持仓")
                results['dynamic'] = dynamic_results
                logger.info("动态持仓策略完成")
            except Exception as e:
                logger.error(f"动态持仓策略执行失败: {e}")
                results['dynamic'] = None
        
        logger.info("回测完成")
        return results.get('fixed'), results.get('dynamic')
    
    except Exception as e:
        logger.error(f"回测执行失败: {e}")
        raise

if __name__ == "__main__":
    # 设置回测参数
    params = {
        'start_date': "2010-08-02",    # 回测起始日期
        'end_date': "2020-07-31",      # 回测结束日期
        'hold_count': 50,              # 持仓数量
        'rebalance_frequency': 1,      # 每天再平衡
        'data_directory': 'data',      # 数据目录
        'output_directory': 'output',  # 输出目录
        'run_fixed': True,             # 运行固定持仓策略
        'run_dynamic': False           # 运行动态持仓策略
    }
    
    # 执行回测
    try:
        # 运行主函数，获取回测结果
        fixed_results, dynamic_results = main(**params)
        
        # 输出回测结果摘要
        if fixed_results is not None:
            logger.info("\n=== 固定持仓策略结果摘要 ===")
            # 计算累计收益率
            cumulative_return = (1 + fixed_results['daily_return']).cumprod().iloc[-1] - 1
            logger.info(f"累计收益率: {cumulative_return:.2%}")
            
        if dynamic_results is not None:
            logger.info("\n=== 动态持仓策略结果摘要 ===")
            # 计算累计收益率
            cumulative_return = (1 + dynamic_results['daily_return']).cumprod().iloc[-1] - 1
            logger.info(f"累计收益率: {cumulative_return:.2%}")
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")

# %%