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

# Set the option to opt-in to the future behavior
pd.set_option('future.no_silent_downcasting', True)

# %% 
# 数据处理类
# 定义对齐矩阵的函数
def align_and_fill_matrix(target_matrix: pd.DataFrame, reference_matrix: pd.DataFrame) -> pd.DataFrame:
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

        # 检查本地原文件是否存在
        raw_files_exist = all(os.path.exists(os.path.join(data_loader.data_folder, file)) for file in [
            'raw_wind_data.csv', 'raw_stocks_info.csv', 'raw_limit_info.csv'
        ])

        if raw_files_exist:
            # 读取原始文件的日期范围
            wind_df = pd.read_csv(os.path.join(data_loader.data_folder, 'raw_wind_data.csv'))
            wind_df['date'] = pd.to_datetime(wind_df['date'])
            raw_start_date = wind_df['date'].iloc[0]
            raw_end_date = wind_df['date'].iloc[-1]

            # 检查设定的日期范围
            if start_date >= raw_start_date and end_date <= raw_end_date:
                logger.debug('设定日期范围在原始文件日期范围内，直接加载对齐数据...')
                
                # 直接加载对齐文件
                aligned_stocks_matrix = pd.read_csv(os.path.join(data_loader.data_folder, 'aligned_stocks_matrix.csv'), index_col=0, parse_dates=True)
                aligned_limit_matrix = pd.read_csv(os.path.join(data_loader.data_folder, 'aligned_limit_matrix.csv'), index_col=0, parse_dates=True)
                aligned_riskwarning_matrix = pd.read_csv(os.path.join(data_loader.data_folder, 'aligned_riskwarning_matrix.csv'), index_col=0, parse_dates=True)
                aligned_trade_status_matrix = pd.read_csv(os.path.join(data_loader.data_folder, 'aligned_trade_status_matrix.csv'), index_col=0, parse_dates=True)
                score_matrix = pd.read_csv(os.path.join(data_loader.data_folder, 'aligned_score_matrix.csv'), index_col=0, parse_dates=True)

                if aligned_stocks_matrix.index.min() == start_date and aligned_stocks_matrix.index.max() == end_date:
                    logger.debug('对齐文件日期完全匹配，直接加载数据...')
                    return aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, aligned_trade_status_matrix, score_matrix
                
                logger.debug('对齐文件日期不完全匹配，截取符合日期的数据...')
                # 截取符合日期的数据
                aligned_stocks_matrix = aligned_stocks_matrix[(aligned_stocks_matrix.index >= start_date) & (aligned_stocks_matrix.index <= end_date)]
                aligned_limit_matrix = aligned_limit_matrix[(aligned_limit_matrix.index >= start_date) & (aligned_limit_matrix.index <= end_date)]
                aligned_riskwarning_matrix = aligned_riskwarning_matrix[(aligned_riskwarning_matrix.index >= start_date) & (aligned_riskwarning_matrix.index <= end_date)]
                aligned_trade_status_matrix = aligned_trade_status_matrix[(aligned_trade_status_matrix.index >= start_date) & (aligned_trade_status_matrix.index <= end_date)]
                score_matrix = score_matrix[(score_matrix.index >= start_date) & (score_matrix.index <= end_date)]

                return aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, aligned_trade_status_matrix, score_matrix

        logger.debug('原始文件不存在或设定日期范围不在原始文件日期范围内，完全重新加载数据...')
        # 完全重新加载所有数据
        df_stocks, trade_status_matrix, riskwarning_matrix, limit_matrix = data_loader.get_stocks_info()
        score_matrix = data_loader.generate_score_matrix('stra_V3_11.csv')

        # 对齐数据
        aligned_stocks_matrix = align_and_fill_matrix(df_stocks, score_matrix)
        aligned_limit_matrix = align_and_fill_matrix(limit_matrix, score_matrix)
        aligned_riskwarning_matrix = align_and_fill_matrix(riskwarning_matrix, score_matrix)
        aligned_trade_status_matrix = align_and_fill_matrix(trade_status_matrix, score_matrix)

        # 截取符合日期的数据（可选）
        aligned_stocks_matrix = aligned_stocks_matrix[(aligned_stocks_matrix.index >= start_date) & (aligned_stocks_matrix.index <= end_date)]
        aligned_limit_matrix = aligned_limit_matrix[(aligned_limit_matrix.index >= start_date) & (aligned_limit_matrix.index <= end_date)]
        aligned_riskwarning_matrix = aligned_riskwarning_matrix[(aligned_riskwarning_matrix.index >= start_date) & (aligned_riskwarning_matrix.index <= end_date)]
        aligned_trade_status_matrix = aligned_trade_status_matrix[(aligned_trade_status_matrix.index >= start_date) & (aligned_trade_status_matrix.index <= end_date)]
        score_matrix = score_matrix[(score_matrix.index >= start_date) & (score_matrix.index <= end_date)]

        # 保存对齐后的矩阵
        aligned_stocks_matrix.to_csv(os.path.join(data_loader.data_folder, 'aligned_stocks_matrix.csv'))
        aligned_limit_matrix.to_csv(os.path.join(data_loader.data_folder, 'aligned_limit_matrix.csv'))
        aligned_riskwarning_matrix.to_csv(os.path.join(data_loader.data_folder, 'aligned_riskwarning_matrix.csv'))
        aligned_trade_status_matrix.to_csv(os.path.join(data_loader.data_folder, 'aligned_trade_status_matrix.csv'))
        score_matrix.to_csv(os.path.join(data_loader.data_folder, 'aligned_score_matrix.csv'))

        return aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, aligned_trade_status_matrix, score_matrix

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
# 回测类
# 固定持仓策略
def run_backtest(stocks_matrix, limit_matrix, risk_warning_matrix, trade_status_matrix, score_matrix, 
                 hold_count, rebalance_frequency, strategy_name):
    start_time = time.time()
    
    # 生成有效性矩阵
    risk_warning_validity = (risk_warning_matrix == 0).astype(int)
    trade_status_validity = (trade_status_matrix == 1).astype(int)
    limit_validity = (limit_matrix == 0).astype(int)
    valid_stocks_matrix = risk_warning_validity * trade_status_validity * limit_validity
    restricted_stocks_matrix = trade_status_validity * limit_validity

    # 创建 DataFrame 保存持仓股票和收益率
    position_history = pd.DataFrame(index=stocks_matrix.index, columns=["hold_positions", "daily_return", "strategy"])
    position_history["strategy"] = strategy_name

    for day in range(1, len(stocks_matrix)):
        previous_positions = position_history.iloc[day - 1]["hold_positions"]
        current_date = position_history.index[day]
        
        # 如果评分矩阵的前一天数据全为NaN，保持前一天的持仓
        if score_matrix.iloc[day - 1].isna().all():
            position_history.loc[current_date, "hold_positions"] = previous_positions
            continue

        # 解析前一天的持仓
        previous_positions = set() if pd.isna(previous_positions) else set(previous_positions.split(','))
        previous_positions = {stock for stock in previous_positions if isinstance(stock, str) and stock.isalnum()}

        # 计算有效股票和受限股票
        valid_stocks = valid_stocks_matrix.iloc[day].astype(bool)
        restricted = restricted_stocks_matrix.iloc[day].astype(bool)
        previous_date = position_history.index[day - 1]
        valid_scores = score_matrix.loc[previous_date]
        
        # 受限股票
        restricted_stocks = [stock for stock in previous_positions if not restricted[stock]]

        # 每隔 rebalance_frequency 天重新平衡持仓
        if (day - 1) % rebalance_frequency == 0:
            sorted_stocks = valid_scores.sort_values(ascending=False)
            try:
                top_stocks = sorted_stocks.iloc[:hold_count].index
                retained_stocks = list(set(previous_positions) & set(top_stocks) | set(restricted_stocks))

                new_positions_needed = hold_count - len(retained_stocks)
                final_positions = set(retained_stocks)

                if new_positions_needed > 0:
                    new_stocks = sorted_stocks[valid_stocks].index
                    new_stocks = [stock for stock in new_stocks if stock not in final_positions]
                    final_positions.update(new_stocks[:new_positions_needed])
            except IndexError:
                logger.warning(f"日期 {current_date}: 可用股票数量不足，使用所有有效股票")
                final_positions = set(sorted_stocks[valid_stocks].index[:hold_count])
        else:
            final_positions = set(previous_positions)

        # 更新持仓
        position_history.loc[current_date, "hold_positions"] = ','.join(final_positions)

        # 计算每日收益率
        if previous_date in stocks_matrix.index:
            daily_returns = stocks_matrix.loc[current_date, list(final_positions)].astype(float)
            daily_return = daily_returns.mean()
            position_history.loc[current_date, "daily_return"] = daily_return

        # 计算换手率
        previous_positions_set = previous_positions
        current_positions_set = final_positions
        turnover_rate = len(previous_positions_set - current_positions_set) / max(len(previous_positions_set), 1)
        position_history.at[current_date, "turnover_rate"] = turnover_rate

    # 删除没有持仓记录的行
    position_history = position_history.dropna(subset=["hold_positions"])

    # 计算持仓数量
    position_history['hold_count'] = position_history['hold_positions'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) else 0
    )
    
    # 保存结果
    results = position_history[['hold_positions', 'hold_count', 'turnover_rate', 'daily_return']]
    results.index.name = 'date'
    
    csv_file = os.path.join('output', f'strategy_results_{strategy_name}.csv')
    results.to_csv(csv_file)
    
    # 计算统计指标
    cumulative_return = (1 + results['daily_return']).cumprod().iloc[-1] - 1
    avg_daily_return = results['daily_return'].mean()
    avg_turnover = results['turnover_rate'].mean()
    avg_holdings = results['hold_count'].mean()
    
    # 输出统计信息
    logger.info(f"\n=== 固定持仓策略统计 ===")
    logger.info(f"累计收益率: {cumulative_return:.2%}")
    logger.info(f"平均日收益率: {avg_daily_return:.2%}")
    logger.info(f"平均换手率: {avg_turnover:.2%}")
    logger.info(f"平均持仓量: {avg_holdings:.1f}")
    logger.info(f"结果已保存到: {csv_file}")
    logger.info(f"策略运行耗时: {time.time() - start_time:.2f} 秒")
    
    return results

# %% 
# 动态持仓策略
def run_backtest_dynamic(stocks_matrix, limit_matrix, risk_warning_matrix, trade_status_matrix, score_matrix, 
                        hold_count, rebalance_frequency, start_sorted=100, the_end_month=None, fixed_by_month=True):
    start_time = time.time()
    
    # 计算每月的股票数量
    monthly_counts = stocks_matrix.resample('ME').apply(lambda x: x.notna().sum())
    if isinstance(monthly_counts, pd.Series):
        df_dynamic = pd.DataFrame({'count': monthly_counts})
    else:
        df_dynamic = pd.DataFrame({'count': monthly_counts.sum(axis=1)})
    
    # 计算结束月份的股票数量
    if the_end_month is None:
        the_end_count = df_dynamic.iloc[-1]['count']
    else:
        the_end_count = df_dynamic.loc[the_end_month]['count']
    
    # 计算持仓开始和结束位置
    df_dynamic['hold_s'] = (df_dynamic['count'] * (start_sorted / the_end_count)).fillna(start_sorted).apply(lambda x: math.floor(x))
    df_dynamic['hold_e'] = (df_dynamic['count'] * ((start_sorted + hold_count) / the_end_count)).fillna(start_sorted + hold_count).apply(lambda x: math.floor(x))
    df_dynamic['hold_num'] = (df_dynamic.hold_e - df_dynamic.hold_s).fillna(hold_count)
    df_dynamic['num_pre'] = df_dynamic.hold_num.shift(-1)
    
    # 向前填充数据
    df_dynamic = df_dynamic.ffill().infer_objects(copy=False)
    
    # 确保持仓数量不超过下一个月全部的数量
    df_dynamic['hold_num'] = df_dynamic.apply(
        lambda dx: dx.hold_num if dx.hold_num <= dx.num_pre else dx.num_pre, 
        axis=1
    ).astype(int)
    
    # 如果指定了结束月份，调整持仓策略
    if the_end_month is not None:
        df_dynamic.loc[the_end_month:, 'hold_s'] = start_sorted
        if fixed_by_month:
            df_dynamic.loc[the_end_month:, 'hold_num'] = hold_count
    
    # 计算每日的持仓数量和开始位置
    daily_hold_nums = df_dynamic['hold_num'].reindex(stocks_matrix.index, method='ffill').fillna(hold_count).astype(int)
    daily_hold_starts = df_dynamic['hold_s'].reindex(stocks_matrix.index, method='ffill').fillna(start_sorted).astype(int)

    # 生成有效性矩阵
    risk_warning_validity = (risk_warning_matrix == 0).astype(int)
    trade_status_validity = (trade_status_matrix == 1).astype(int)
    limit_validity = (limit_matrix == 0).astype(int)
    valid_stocks_matrix = risk_warning_validity * trade_status_validity * limit_validity
    restricted_stocks_matrix = trade_status_validity * limit_validity

    # 创建 DataFrame 保存持仓股票和收益率
    position_history = pd.DataFrame(index=stocks_matrix.index, 
                                  columns=["hold_positions", "daily_return", "turnover_rate", "strategy"])
    position_history["strategy"] = "dynamic_hold"

    for day in range(1, len(stocks_matrix)):
        previous_positions = position_history.iloc[day - 1]["hold_positions"]
        current_date = position_history.index[day]
        
        current_hold_count = daily_hold_nums[current_date]
        current_start_pos = daily_hold_starts[current_date]

        # 如果评分矩阵的前一天数据全为NaN，保持前一天的持仓
        if score_matrix.iloc[day - 1].isna().all():
            position_history.loc[current_date, "hold_positions"] = previous_positions
            continue

        # 解析前一天的持仓
        previous_positions = set() if pd.isna(previous_positions) else set(previous_positions.split(','))
        previous_positions = {stock for stock in previous_positions if isinstance(stock, str) and stock.isalnum()}

        # 计算有效股票和受限股票
        valid_stocks = valid_stocks_matrix.iloc[day].astype(bool)
        restricted = restricted_stocks_matrix.iloc[day].astype(bool)
        previous_date = position_history.index[day - 1]
        valid_scores = score_matrix.loc[previous_date]
        restricted_stocks = [stock for stock in previous_positions if not restricted[stock]]

        # 每隔 rebalance_frequency 天重新平衡持仓
        if (day - 1) % rebalance_frequency == 0:
            sorted_stocks = valid_scores.sort_values(ascending=False)
            start_pos = max(0, int(current_start_pos))
            hold_num = max(1, int(current_hold_count))
            
            try:
                limited_stocks = sorted_stocks.iloc[start_pos:start_pos + hold_num].index
                retained_stocks = list(set(previous_positions) & set(limited_stocks) | set(restricted_stocks))

                new_positions_needed = hold_num - len(retained_stocks)
                final_positions = set(retained_stocks)

                if new_positions_needed > 0:
                    new_stocks = sorted_stocks[valid_stocks].iloc[start_pos:].index
                    new_stocks = [stock for stock in new_stocks if stock not in final_positions]
                    final_positions.update(new_stocks[:new_positions_needed])
            except IndexError:
                logger.warning(f"日期 {current_date}: 可用股票数量不足，使用所有有效股票")
                final_positions = set(sorted_stocks[valid_stocks].index[:hold_num])
        else:
            final_positions = set(previous_positions)

        # 更新持仓
        position_history.loc[current_date, "hold_positions"] = ','.join(final_positions)

        # 计算每日收益率
        if previous_date in stocks_matrix.index:
            daily_returns = stocks_matrix.loc[current_date, list(final_positions)].astype(float)
            daily_return = daily_returns.mean()
            position_history.loc[current_date, "daily_return"] = daily_return

        # 计算换手率
        previous_positions_set = previous_positions
        current_positions_set = final_positions
        turnover_rate = len(previous_positions_set - current_positions_set) / max(len(previous_positions_set), 1)
        position_history.at[current_date, "turnover_rate"] = turnover_rate

    # 删除没有持仓记录的行
    position_history = position_history.dropna(subset=["hold_positions"])

    # 计算持仓数量
    position_history['hold_count'] = position_history['hold_positions'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) else 0
    )
    
    # 保存结果
    results = position_history[['hold_positions', 'hold_count', 'turnover_rate', 'daily_return']]
    results.index.name = 'date'
    
    csv_file = os.path.join('output', 'strategy_results_dynamic.csv')
    results.to_csv(csv_file)
    
    # 计算统计指标
    cumulative_return = (1 + results['daily_return']).cumprod().iloc[-1] - 1
    avg_daily_return = results['daily_return'].mean()
    avg_turnover = results['turnover_rate'].mean()
    avg_holdings = results['hold_count'].mean()
    
    # 输出统计信息
    logger.info("\n=== 动态持仓策略统计 ===")
    logger.info(f"累计收益率: {cumulative_return:.2%}")
    logger.info(f"平均日收益率: {avg_daily_return:.2%}")
    logger.info(f"平均换手率: {avg_turnover:.2%}")
    logger.info(f"平均持仓数量: {avg_holdings:.1f}")
    logger.info(f"结果已保存到: {csv_file}")
    logger.info(f"策略运行耗时: {time.time() - start_time:.2f} 秒")
    
    return results

# 绘制累计净值和回撤曲线的函数
def _plot_net_value(df: pd.DataFrame, text: str, turn_loss):
    """绘制累计净值和回撤曲线。"""
    df.reset_index(inplace=True)
    df.set_index('date', inplace=True)
    start_date = df.index[0]
    
    # 确保 'daily_return' 列存在
    if 'daily_return' not in df.columns:
        logger.error("DataFrame 不包含 'daily_return' 列。")
        return

    # 设置固定成本，并针对特定日期进行调整
    df['loss'] = 0.0013  # 初始固定成本
    df.loc[df.index > '2023-08-31', 'loss'] = 0.0008  # 特定日期后的调整成本
    df['loss'] += float(turn_loss)  # 加上换手损失

    # 计算调整后的变动和累计净值
    df['chg_'] = df['daily_return'] - df['turnover_rate'] * df['loss']
    df['net_value'] = (df['chg_'] + 1).cumprod()

    # 计算最大净值和回撤
    dates = df.index.unique().tolist()
    for date in dates:
        df.loc[date, 'max_net'] = df.loc[:date].net_value.max()
    df['back_net'] = df['net_value'] / df['max_net'] - 1

    # 计算年化收益和月波动率
    s_ = df.iloc[-1]
    ana = format(s_.net_value ** (252 / df.shape[0]) - 1, '.2%')
    vola = format(df.net_value.pct_change().std() * 21 ** 0.5, '.2%')

    # 创建净值和回撤的plotly图形对象
    g1 = go.Scatter(x=df.index.unique().tolist(), y=df['net_value'], name='净值')
    g2 = go.Scatter(x=df.index.unique().tolist(), y=df['back_net'] * 100, name='回撤', xaxis='x', yaxis='y2', mode="none",
                    fill="tozeroy")

    # 配置并显示图表
    fig = go.Figure(
        data=[g1, g2],
        layout={
            'height': 1122,
            "title": f"{text}，<br>净值（左）& 回撤（右），<br>全期：{start_date} ~ {s_.name}，<br>年化收益：{ana}，月波动：{vola}",
            "font": {"size": 22},
            "yaxis": {"title": "累计净值"},
            "yaxis2": {"title": "最大回撤", "side": "right", "overlaying": "y", "ticksuffix": "%", "showgrid": False},
        }
    )
    fig.show()

# %% 
# 主函数
def main():
    # 初始化参数
    start_date = "2010-08-02"
    end_date = "2020-07-31"
    data_directory = 'data'
    hold_count = 50
    rebalance_frequency = 1

    # 数据加载
    data_loader = LoadData(start_date, end_date, data_directory)
    aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, aligned_trade_status_matrix, score_matrix = process_data(data_loader)

    # 运行固定持仓回测
    fixed_results = run_backtest(
        aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, 
        aligned_trade_status_matrix, score_matrix, hold_count, rebalance_frequency, "fixed"
    )
    
    # 运行动态持仓回测
    dynamic_results = run_backtest_dynamic(
        aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, 
        aligned_trade_status_matrix, score_matrix, hold_count, rebalance_frequency
    )
    
    # 确保每个结果 DataFrame 包含 'strategy' 列
    fixed_results['strategy'] = 'fixed'
    dynamic_results['strategy'] = 'dynamic'

    # 合并结果
    all_position_history = pd.concat([fixed_results, dynamic_results])
    
    # 计算累计收益率
    all_position_history['cumulative_return'] = (1 + all_position_history['daily_return']).cumprod() - 1
    
    # 按策略分组计算统计指标
    strategy_stats = all_position_history.groupby('strategy').agg({
        'daily_return': ['mean', 'std', 'count'],
        'turnover_rate': 'mean',
        'cumulative_return': 'last'
    })
    
    # 输出结果
    print("\n=== 回测结果统计 ===")
    print(strategy_stats)
    
    # 绘制净值曲线图
    _plot_net_value(fixed_results, "固定持仓策略净值曲线", 0.003)
    _plot_net_value(dynamic_results, "动态持仓策略净值曲线", 0.003)
    
    return fixed_results, dynamic_results

if __name__ == "__main__":
    fixed_results, dynamic_results = main()

# %%
