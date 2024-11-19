# %% 
# 导入所需的库
import os
import time
import pandas as pd
import numpy as np
from loguru import logger
from utils import trans_str_to_float64, get_client_U

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
            df['date'] = pd.to_datetime(df['date'])  # 转换日期格式
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
            logger.debug('将数据转换为矩阵格式并保存为CSV文件...')
            csv_file_path = os.path.join(self.data_folder, file_name)
            df.to_csv(csv_file_path, header=True)  # 保存数据
            logger.debug(f'数据已保存到 {csv_file_path}')
        except Exception as e:
            logger.error(f"保存矩阵到CSV文件失败: {e}")
            raise

# %% 
# 回测类
def run_backtest(stocks_matrix, limit_matrix, risk_warning_matrix, trade_status_matrix, score_matrix, hold_count, rebalance_frequency):
    start_time = time.time()  # 记录开始时间
    
    # 生成有效性矩阵
    risk_warning_validity = (risk_warning_matrix == 0).astype(int)  # 风险警告有效性
    trade_status_validity = (trade_status_matrix == 1).astype(int)  # 交易状态有效性
    limit_validity = (limit_matrix == 0).astype(int)  # 涨跌停有效性

    # 计算有效股票矩阵
    valid_stocks_matrix = risk_warning_validity * trade_status_validity * limit_validity  # 计算有效股票
    restricted_stocks_matrix = trade_status_validity * limit_validity  # 计算限制股票

    # 创建 DataFrame 保存持仓股票和收益率
    position_history = pd.DataFrame(index=stocks_matrix.index, columns=["hold_positions", "daily_return"])

    # 从第二天开始进行回测
    for day in range(1, len(stocks_matrix)):
        previous_positions = position_history.iloc[day - 1]["hold_positions"]

        if score_matrix.iloc[day - 1].isna().all():  # 检查前一天评分是否全部缺失
            position_history.loc[position_history.index[day], "hold_positions"] = previous_positions
            continue

        previous_positions = set() if pd.isna(previous_positions) else set(previous_positions.split(','))
        previous_positions = {stock for stock in previous_positions if isinstance(stock, str) and stock.isalnum()}

        valid_stocks = valid_stocks_matrix.iloc[day].astype(bool)  # 获取有效股票
        restricted = restricted_stocks_matrix.iloc[day].astype(bool)  # 获取限制股票

        previous_date = position_history.index[day - 1]  # 获取前一天日期
        valid_scores = score_matrix.loc[previous_date]  # 获取前一天评分

        restricted_stocks = [stock for stock in previous_positions if not restricted[stock]]  # 获取限制股票

        # 判断是否需要换仓
        if (day - 1) % rebalance_frequency == 0:
            limited_stocks = valid_scores.nlargest(hold_count).index  # 获取得分最高的股票
            retained_stocks = list(set(previous_positions) & set(limited_stocks) | set(restricted_stocks))  # 保留股票

            new_positions_needed = hold_count - len(retained_stocks)  # 计算新需要的持仓
            final_positions = set(retained_stocks)  # 最终持仓

            if new_positions_needed > 0:
                new_stocks = valid_scores[valid_stocks].nlargest(hold_count).index  # 获取新股票
                new_stocks = [stock for stock in new_stocks if stock not in final_positions]  # 去除已有股票
                final_positions.update(new_stocks[:new_positions_needed])  # 更新最终持仓
        else:
            final_positions = set(previous_positions)  # 保持原有持仓

        position_history.loc[position_history.index[day], "hold_positions"] = ','.join(final_positions)  # 保存持仓

        if previous_date in stocks_matrix.index:
            daily_returns = stocks_matrix.loc[position_history.index[day], list(final_positions)].astype(float)  # 获取当天持仓收益
            daily_return = daily_returns.mean()  # 计算日均收益
            position_history.loc[position_history.index[day], "daily_return"] = daily_return  # 保存日收益

    # 计算换仓率
    position_history["turnover_rate"] = np.nan
    for day in range(1, len(position_history)):
        previous_positions = position_history.iloc[day - 1]["hold_positions"]
        current_positions = position_history.iloc[day]["hold_positions"]

        previous_positions_set = {stock for stock in previous_positions.split(',') if isinstance(previous_positions, str) and stock.isalnum()} if isinstance(previous_positions, str) else set()
        current_positions_set = {stock for stock in current_positions.split(',') if isinstance(current_positions, str) and stock.isalnum()} if isinstance(current_positions, str) else set()
        
        turnover_rate = len(previous_positions_set - current_positions_set) / max(len(previous_positions_set), 1)  # 计算换仓率
        position_history.at[position_history.index[day], "turnover_rate"] = turnover_rate

    position_history = position_history.dropna(subset=["hold_positions"])  # 删除持仓为空的记录

    end_time = time.time()  # 记录结束时间
    print(f"回测总耗时: {end_time - start_time:.2f} 秒")

    position_history.to_csv("output/position_holdings.csv")  # 保存持仓记录
    return position_history  # 返回持仓记录

# %%
# 主函数
def main():
    # 初始化参数
    start_date = "2010-08-02"
    end_date = "2020-07-31"
    data_directory = 'data'  # 数据存储目录
    hold_count = 20  # 持仓数
    rebalance_frequency = 1  # 换仓周期

    # 数据加载
    data_loader = LoadData(start_date, end_date, data_directory)  # 实例化数据加载类

    # 调用数据处理逻辑
    aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, aligned_trade_status_matrix, score_matrix = process_data(data_loader)

    # 运行回测
    position_history = run_backtest(aligned_stocks_matrix, aligned_limit_matrix, aligned_riskwarning_matrix, aligned_trade_status_matrix, score_matrix, hold_count, rebalance_frequency)

    # 输出每日持仓的股票代码和收益率
    print(position_history)

if __name__ == "__main__":
    main()  # 执行主函数
