# %%
# 导入所需的库
import os
import time
import pandas as pd
import numpy as np
from loguru import logger
from utils import trans_str_to_float64, get_client_U

#%%
# 定义数据加载类
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
# %%
#数据处理类
# 定义对齐矩阵的函数
def align_and_fill_matrix(target_matrix: pd.DataFrame, reference_matrix: pd.DataFrame) -> pd.DataFrame:
    try:
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
def run_backtest(stocks_matrix, limit_matrix, risk_warning_matrix, trade_status_matrix, score_matrix, hold_count=20):
    start_time = time.time()  # 记录开始时间
    
    # 生成有效性矩阵
    risk_warning_validity = (risk_warning_matrix == 0).astype(int)
    trade_status_validity = (trade_status_matrix == 1).astype(int)
    limit_validity = (limit_matrix == 0).astype(int)

    # 计算有效股票矩阵
    valid_stocks_matrix = risk_warning_validity * trade_status_validity * limit_validity
    restricted_stocks_matrix = trade_status_validity * limit_validity

    # 创建 DataFrame 保存持仓股票和收益率
    position_history = pd.DataFrame(index=stocks_matrix.index, columns=["hold_positions", "daily_return"])

    # 从第二天开始进行回测
    for day in range(1, len(stocks_matrix)):
        previous_positions = position_history.iloc[day - 1]["hold_positions"]

        if score_matrix.iloc[day - 1].isna().all():
            position_history.loc[position_history.index[day], "hold_positions"] = previous_positions
            continue

        previous_positions = set() if pd.isna(previous_positions) else set(previous_positions.split(','))
        previous_positions = {stock for stock in previous_positions if isinstance(stock, str) and stock.isalnum()}

        valid_stocks = valid_stocks_matrix.iloc[day].astype(bool)
        restricted = restricted_stocks_matrix.iloc[day].astype(bool)

        previous_date = position_history.index[day - 1]
        valid_scores = score_matrix.loc[previous_date]

        restricted_stocks = [stock for stock in previous_positions if not restricted[stock]]

        limited_stocks = valid_scores.nlargest(hold_count).index
        retained_stocks = list(set(previous_positions) & set(limited_stocks) | set(restricted_stocks))

        new_positions_needed = hold_count - len(retained_stocks)
        final_positions = set(retained_stocks)

        if new_positions_needed > 0:
            new_stocks = valid_scores[valid_stocks].nlargest(hold_count).index
            new_stocks = [stock for stock in new_stocks if stock not in final_positions]
            final_positions.update(new_stocks[:new_positions_needed])

        position_history.loc[position_history.index[day], "hold_positions"] = ','.join(final_positions)

        if previous_date in stocks_matrix.index:
            daily_returns = stocks_matrix.loc[position_history.index[day], list(final_positions)].astype(float)
            daily_return = daily_returns.mean()
            position_history.loc[position_history.index[day], "daily_return"] = daily_return

    position_history["turnover_rate"] = np.nan
    for day in range(1, len(position_history)):
        previous_positions = position_history.iloc[day - 1]["hold_positions"]
        current_positions = position_history.iloc[day]["hold_positions"]

        previous_positions_set = {stock for stock in previous_positions.split(',') if isinstance(previous_positions, str) and stock.isalnum()} if isinstance(previous_positions, str) else set()
        current_positions_set = {stock for stock in current_positions.split(',') if isinstance(current_positions, str) and stock.isalnum()} if isinstance(current_positions, str) else set()
        
        turnover_rate = len(previous_positions_set - current_positions_set) / max(len(previous_positions_set), 1)
        position_history.at[position_history.index[day], "turnover_rate"] = turnover_rate

    position_history = position_history.dropna(subset=["hold_positions"])

    end_time = time.time()
    print(f"回测总耗时: {end_time - start_time:.2f} 秒")

    position_history.to_csv("output/position_holdings.csv")
    return position_history

# %%
#主函数
def main():
    # 初始化参数
    start_date = "2010-08-02"
    end_date = "2020-08-01"
    data_directory = 'data'

    # 数据加载
    data_loader = LoadData(start_date, end_date, data_directory)
    
    # 调用数据处理函数，确保生成所需的文件
    process_data(data_loader)
    
    # 读取对齐后的数据
    stocks_matrix = pd.read_csv(os.path.join(data_directory, 'aligned_stocks_matrix.csv'), index_col=0, parse_dates=True)
    limit_matrix = pd.read_csv(os.path.join(data_directory, 'aligned_limit_matrix.csv'), index_col=0, parse_dates=True)
    risk_warning_matrix = pd.read_csv(os.path.join(data_directory, 'aligned_riskwarning_matrix.csv'), index_col=0, parse_dates=True)
    trade_status_matrix = pd.read_csv(os.path.join(data_directory, 'aligned_trade_status_matrix.csv'), index_col=0, parse_dates=True)
    score_matrix = pd.read_csv(os.path.join(data_directory, 'aligned_score_matrix.csv'), index_col=0, parse_dates=True)

    # 运行回测
    position_history = run_backtest(stocks_matrix, limit_matrix, risk_warning_matrix, trade_status_matrix, score_matrix)

    # 输出每日持仓的股票代码和收益率
    print(position_history)

if __name__ == "__main__":
    main()
