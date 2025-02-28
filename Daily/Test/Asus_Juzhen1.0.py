# %% Derek 2024/9/19 矩阵相乘，二次提速
# 导入相关的库
import os
import time
import pandas as pd
import numpy as np
from loguru import logger
from utils import trans_str_to_float64, get_client_U

# %% 
# loaddata类，用于加载数据
class LoadData:
    def __init__(self, date_s: str, date_e: str, data_folder: str):
        if date_s is None or date_e is None:
            raise ValueError('必须指定起止日期！！！')
        self.client_U = get_client_U()
        self.date_s, self.date_e = date_s, date_e
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)

    def get_chg_wind(self) -> pd.DataFrame:
        try:
            logger.debug('加载WIND的日频涨跌幅数据...')
            df = pd.DataFrame(self.client_U.basic_wind.w_vol_price.find(
                {"date": {"$gte": self.date_s, "$lte": self.date_e}},
                {"_id": 0, "date": 1, "code": 1, "pct_chg": 1},
                batch_size=1000000))
            df = trans_str_to_float64(df, trans_cols=['pct_chg'])
            df['date'] = pd.to_datetime(df['date'])
            pivot_df = df.pivot_table(index='date', columns='code', values='pct_chg')
            return pivot_df
        except Exception as e:
            logger.error(f"加载WIND数据失败: {e}")
            raise

    def get_stocks_info(self) -> tuple:
        try:
            logger.debug('从数据库加载股票信息...')
            t_info = self.client_U.basic_wind.w_basic_info
            t_limit = self.client_U.basic_jq.jq_daily_price_none

            df_info = pd.DataFrame(t_info.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                               {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
                                               batch_size=1000000))
            df_info['date'] = pd.to_datetime(df_info['date'])
            df_stocks = self.get_chg_wind()

            # 加载涨跌停信息
            use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
            df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": self.date_s, "$lte": self.date_e}}, use_cols, batch_size=1000000))
            df_limit['date'] = pd.to_datetime(df_limit['date'])
            df_limit['limit'] = df_limit.apply(lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"], axis=1)
            df_limit['limit'] = df_limit['limit'].astype('int')
            limit_matrix = df_limit.pivot(index='date', columns='code', values='limit')

            trade_status_matrix = df_info.pivot(index='date', columns='code', values='trade_status')
            riskwarning_matrix = df_info.pivot(index='date', columns='code', values='riskwarning')

            return df_stocks, trade_status_matrix, riskwarning_matrix, limit_matrix
        except Exception as e:
            logger.error(f"加载股票信息失败: {e}")
            raise

    def generate_score_matrix(self, file_name: str) -> pd.DataFrame:
        try:
            csv_file_path = os.path.join(self.data_folder, file_name)
            logger.debug(f"从本地文件 {csv_file_path} 加载数据...")
            df = pd.read_csv(csv_file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            score_matrix = df.pivot_table(index='date', columns='code', values='F1')
            return score_matrix
        except Exception as e:
            logger.error(f"加载评分矩阵失败: {e}")
            raise

    def save_matrix_to_csv(self, df: pd.DataFrame, file_name: str):
        try:
            logger.debug('将数据转换为矩阵格式并保存为CSV文件...')
            csv_file_path = os.path.join(self.data_folder, file_name)
            df.to_csv(csv_file_path, header=True)
            logger.debug(f'数据已保存到 {csv_file_path}')
        except Exception as e:
            logger.error(f"保存矩阵到CSV文件失败: {e}")
            raise
        
# 定义对齐矩阵的函数
def align_and_fill_matrix(target_matrix: pd.DataFrame, reference_matrix: pd.DataFrame) -> pd.DataFrame:
    try:
        # 对齐 target_matrix 的列，使其与 reference_matrix 的列一致，并用 0 填充
        aligned_matrix = target_matrix.reindex(columns=reference_matrix.columns, fill_value=0)
        return aligned_matrix
    except Exception as e:
        logger.error(f"对齐矩阵失败: {e}")
        raise

# 数据处理函数
def process_data(data_loader: LoadData):
    try:
        # 检查是否存在本地文件并加载数据
        stocks_info_file = 'data/aligned_stocks_matrix.csv'
        limit_info_file = 'data/aligned_limit_matrix.csv'
        riskwarning_info_file = 'data/aligned_riskwarning_matrix.csv'
        trade_status_info_file = 'data/aligned_trade_status_matrix.csv'
        score_info_file = 'data/aligned_score_matrix.csv'

        # 如果所有本地文件存在，则直接读取
        if all(os.path.exists(f) for f in [stocks_info_file, limit_info_file, riskwarning_info_file, trade_status_info_file, score_info_file]):
            logger.debug('所有本地文件存在，直接加载...')
            aligned_stocks_matrix = pd.read_csv(stocks_info_file, index_col=0, parse_dates=True)
            aligned_limit_matrix = pd.read_csv(limit_info_file, index_col=0, parse_dates=True)
            aligned_riskwarning_matrix = pd.read_csv(riskwarning_info_file, index_col=0, parse_dates=True)
            aligned_trade_status_matrix = pd.read_csv(trade_status_info_file, index_col=0, parse_dates=True)
            score_matrix = pd.read_csv(score_info_file, index_col=0, parse_dates=True)
        else:
            logger.debug('本地文件不存在，加载数据...')
            # 从数据加载器中获取股票信息、交易状态、风险警示和涨跌停数据
            df_stocks, trade_status_matrix, riskwarning_matrix, limit_matrix = data_loader.get_stocks_info()
            
            # 从评分矩阵文件中生成评分矩阵
            score_matrix = data_loader.generate_score_matrix('stra_V3_11.csv')

            # 对齐股票信息矩阵、涨跌停矩阵、风险警示矩阵、交易状态矩阵与评分矩阵
            aligned_stocks_matrix = align_and_fill_matrix(df_stocks, score_matrix)
            aligned_limit_matrix = align_and_fill_matrix(limit_matrix, score_matrix)
            aligned_riskwarning_matrix = align_and_fill_matrix(riskwarning_matrix, score_matrix)
            aligned_trade_status_matrix = align_and_fill_matrix(trade_status_matrix, score_matrix)

            # 将对齐后的数据保存到本地CSV文件
            data_loader.save_matrix_to_csv(aligned_stocks_matrix, 'aligned_stocks_matrix.csv')
            data_loader.save_matrix_to_csv(aligned_limit_matrix, 'aligned_limit_matrix.csv')
            data_loader.save_matrix_to_csv(aligned_riskwarning_matrix, 'aligned_riskwarning_matrix.csv')
            data_loader.save_matrix_to_csv(aligned_trade_status_matrix, 'aligned_trade_status_matrix.csv')
            data_loader.save_matrix_to_csv(score_matrix, 'aligned_score_matrix.csv')  # 保存对齐后的评分矩阵

    except Exception as e:
        # 捕捉并记录异常
        logger.error(f"数据处理失败: {e}")
        raise

# %%
# 回测类
class Backtest:
    def __init__(self, aligned_stocks_matrix: pd.DataFrame, aligned_limit_matrix: pd.DataFrame, aligned_score_matrix: pd.DataFrame, aligned_riskwarning_matrix: pd.DataFrame, aligned_trade_status_matrix: pd.DataFrame, hold_num: int = 20, hold_sorted: bool = True, period: int = 1):
        self.stocks_info = aligned_stocks_matrix
        self.limit_matrix = aligned_limit_matrix
        self.score_matrix = aligned_score_matrix
        self.riskwarning_matrix = aligned_riskwarning_matrix
        self.trade_status_matrix = aligned_trade_status_matrix
        self.hold_num = hold_num
        self.hold_sorted = hold_sorted
        self.period = period
        self.initial_positions = None

    def initialize_positions(self, start_date: pd.Timestamp):
        # 获取当天的股票得分矩阵，并将 NaN 填充为 0
        today_scores = self.score_matrix.loc[start_date].fillna(0)
        
        # 构建每个条件的布尔矩阵，将布尔值转换为 1 和 0
        risk_warning_matrix = (self.riskwarning_matrix.loc[start_date] == 0).astype(int)  # 没有风险警告为 1
        trade_status_matrix = (self.trade_status_matrix.loc[start_date] == 1).astype(int)  # 正常交易为 1
        limit_matrix = (self.limit_matrix.loc[start_date] == 0).astype(int)  # 没有涨跌停限制为 1
        
        # 通过逐元素乘法的方式组合所有条件，得到最终的有效股票矩阵
        valid_matrix = risk_warning_matrix * trade_status_matrix * limit_matrix
        
        # 使用逐元素相乘的方式筛选有效股票的得分
        valid_scores = today_scores * valid_matrix
        
        # 根据筛选后的得分进行排序，选择得分最高的前 hold_num 只股票
        if self.hold_sorted:
            # 只筛选出有效得分大于 0 的股票
            sorted_scores = valid_scores[valid_scores > 0].sort_values(ascending=False, kind='mergesort')
            
            # 选取得分最高的 top N 个股票
            top_stocks = sorted_scores.head(self.hold_num)
            
            # 获取最后一个选中的股票得分，处理平分情况
            last_score_value = top_stocks.iloc[-1]
            tied_stocks = sorted_scores[sorted_scores == last_score_value]
            
            # 初始化头寸，将选中的股票加入持仓列表
            self.initial_positions = top_stocks.index.tolist()

    def run_backtest(self, start_date: pd.Timestamp, output_folder: str = 'output'):
        try:
            logger.debug('开始回测...')
            os.makedirs(output_folder, exist_ok=True)

            returns = []  # 存储每个日期的收益
            holdings = []  # 存储每个日期的持仓股票

            start_time = time.time()
            self.initialize_positions(start_date)
            current_positions = self.initial_positions
            dates = self.stocks_info.index

            limit_matrix = self.limit_matrix
            trade_status_matrix = self.trade_status_matrix
            riskwarning_matrix = self.riskwarning_matrix
            score_matrix = self.score_matrix

            hold_num = self.hold_num
            period = self.period

            for i, date in enumerate(dates[1:]):
                daily_returns = self.stocks_info.loc[date, current_positions].mean()
                returns.append(daily_returns)
                holdings.append((date, current_positions))

                if (i + 1) % period == 0:
                    next_date = dates[i + 2] if (i + 2) < len(dates) else None
                    if next_date:
                        next_day_trade_status = trade_status_matrix.loc[next_date, current_positions]
                        next_day_limit_status = limit_matrix.loc[next_date, current_positions]
                        cannot_trade_stocks = next_day_trade_status[
                            (next_day_trade_status == 0) | (next_day_limit_status == 1)
                        ].index.tolist()

                        today_scores = score_matrix.loc[date].fillna(0)

                        # 计算有效股票的布尔矩阵
                        riskwarning_valid = riskwarning_matrix.loc[next_date] == 0
                        trade_status_valid = trade_status_matrix.loc[next_date] == 1
                        limit_status_valid = limit_matrix.loc[next_date] == 0

                        # 逐元素乘法得到最终有效股票矩阵
                        valid_stocks_matrix = riskwarning_valid & trade_status_valid & limit_status_valid
                        valid_stocks_scores = today_scores[valid_stocks_matrix]

                        # 选择前 hold_num 名股票
                        top_today_stocks = today_scores.nlargest(hold_num).index.tolist()

                        # 计算当前持仓与今天可交易股票的交集
                        intersection_stocks = list(set(current_positions) & set(top_today_stocks))
                        new_positions = list(set(cannot_trade_stocks) | set(intersection_stocks))

                        if len(new_positions) < hold_num:
                            # 计算剩余股票
                            remaining_stocks = valid_stocks_scores.index.difference(new_positions)
                            # 从剩余股票中选择前 hold_num - len(new_positions) 名
                            additional_stocks = valid_stocks_scores.reindex(remaining_stocks).nlargest(hold_num - len(new_positions)).index.tolist()
                            new_positions.extend(additional_stocks)

                        current_positions = new_positions[:hold_num]

            df_returns = pd.DataFrame(returns, index=dates[1:], columns=['chg'])
            df_returns.to_csv(os.path.join(output_folder, 'results.csv'), index=True, index_label='date')

            df_holdings = pd.DataFrame(holdings, columns=['dat', 'hold_positions'])
            df_holdings['hold_positions'] = [','.join(x) for x in df_holdings['hold_positions']]
            df_holdings.to_csv(os.path.join(output_folder, 'holdings.csv'), index=False)

            logger.debug(f'回测结果已保存到 {output_folder}/results.csv 和 {output_folder}/holdings.csv')

            end_time = time.time()
            logger.debug(f"回测总耗时: {end_time - start_time:.4f} 秒")

            return df_returns, df_holdings
        except Exception as e:
            logger.error(f"回测失败: {e}")
            raise

# %%
# 主程序
def main():
    start_time = time.time()  # 开始时间
    try:
        # 初始化数据加载器和数据处理过程
        data_loader = LoadData(date_s="2010-08-02", date_e="2020-08-01", data_folder='data')
        process_data(data_loader)

        # 读取处理后的数据
        aligned_stocks_matrix = pd.read_csv('data/aligned_stocks_matrix.csv', index_col=0, parse_dates=True)
        aligned_limit_matrix = pd.read_csv('data/aligned_limit_matrix.csv', index_col=0, parse_dates=True)
        aligned_score_matrix = pd.read_csv('data/aligned_score_matrix.csv', index_col=0, parse_dates=True)
        aligned_riskwarning_matrix = pd.read_csv('data/aligned_riskwarning_matrix.csv', index_col=0, parse_dates=True)
        aligned_trade_status_matrix = pd.read_csv('data/aligned_trade_status_matrix.csv', index_col=0, parse_dates=True)

        # 初始化回测对象
        start_date = pd.Timestamp("2010-08-02")
        backtest = Backtest(
            aligned_stocks_matrix,
            aligned_limit_matrix,
            aligned_score_matrix,
            aligned_riskwarning_matrix,
            aligned_trade_status_matrix,
            hold_num=20,
            hold_sorted=True
        )

        # 执行回测
        backtest.run_backtest(start_date=start_date, output_folder='output')
        logger.debug('回测完成！')

    except Exception as e:
        logger.error(f"主程序运行失败: {e}")

    finally:
        total_time = time.time() - start_time  # 计算总时间
        logger.info(f"总运行时间: {total_time:.2f}秒")

if __name__ == "__main__":
    main()
