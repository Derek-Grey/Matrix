#%%
import pymongo
import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from loguru import logger
from urllib.parse import quote_plus

#%%
# 数据转换函数，将字符串转换为浮点数
def trans_str_to_float64(df: pd.DataFrame, exp_cols: list = None, trans_cols: list = None) -> pd.DataFrame:
    if trans_cols is None and exp_cols is None:
        trans_cols = df.columns
    if not exp_cols is None:
        trans_cols = list(set(df.columns) - set(exp_cols))
    df[trans_cols] = df[trans_cols].astype('float64')
    return df

#%%
# 获取MongoDB客户端连接函数（只读）
def get_client_U():
    user, pwd = 'Tom', 'tom'  # 此处应使用实际的用户名和密码
    return pymongo.MongoClient(f"mongodb://{quote_plus(user)}:{quote_plus(pwd)}@192.168.1.99:29900/")

#%%
# LoadData类，用于从MongoDB获取数据
class LoadData:
    def __init__(self, date_s: str, date_e: str):
        if date_s is None or date_e is None:
            raise Exception('必须指定起止日期！！！')
        self.client_U = get_client_U()
        self.date_s, self.date_e = date_s, date_e

    def get_chg_wind(self) -> pd.DataFrame:
        logger.info('加载基于WIND的日频涨跌幅数据...')
        df = pd.DataFrame(self.client_U.basic_wind.w_vol_price.find(
            {"date": {"$gte": self.date_s, "$lte": self.date_e}},
            {"_id": 0, "date": 1, "code": 1, "pct_chg": 1},
            batch_size=1000000))
        return trans_str_to_float64(df, trans_cols=['pct_chg']).set_index(['date', 'code']).sort_index()

    def get_data_from_csv_or_db(self, csv_file_path: str, force_reload: bool = False) -> pd.DataFrame:
        if os.path.exists(csv_file_path) and not force_reload:
            logger.info(f"从本地文件 {csv_file_path} 加载数据...")
            df = pd.read_csv(csv_file_path, index_col=[0, 1], parse_dates=True)
        else:
            logger.info("从数据库加载数据...")
            df = self.get_chg_wind()
            df.to_csv(csv_file_path, header=True)
            logger.info(f"数据已保存到本地文件 {csv_file_path}")

        return df

#%%
# BackTest类，用于执行回测
class BackTest:
    def __init__(self, df_stk: pd.DataFrame, stra_name="等权重交易", initial_capital=1):
        self.df_stocks = df_stk
        self.stra_name = stra_name
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.dates = self.df_stocks.index.get_level_values('date').unique()
        self.total_value = pd.Series([initial_capital] * len(self.dates), index=self.dates)
        self.max_value = initial_capital
        self.drawdowns = pd.Series([0.0] * len(self.dates), index=self.dates)

    def run_backtest(self):
        stock_count = 500
        for i, date in enumerate(self.dates):
            if date in self.df_stocks.index:
                date_data = self.df_stocks.loc[date].dropna()
                if not date_data.empty:
                    stock_returns = date_data['pct_chg']
                    stock_returns = stock_returns[~np.isnan(stock_returns) & ~np.isinf(stock_returns)]
                    if len(stock_returns) >= stock_count:
                        stock_returns = stock_returns.head(stock_count)
                        stock_value = self.portfolio_value / stock_count
                        daily_gain = (stock_returns * stock_value).sum()
                        if np.isfinite(daily_gain):
                            self.portfolio_value += daily_gain
                            self.total_value.at[date] = self.portfolio_value
                            if self.portfolio_value > self.max_value:
                                self.max_value = self.portfolio_value
                            drawdown = (self.max_value - self.portfolio_value) / self.max_value
                            self.drawdowns.at[date] = drawdown*-1

    def plot_net_value_and_drawdown(self):
        # 创建具有两个y轴的子图
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 绘制净值曲线
        fig.add_trace(
            go.Scatter(x=self.dates, y=self.total_value, mode='lines', name='净值'),
            secondary_y=False
        )

        # 绘制回撤面积图
        fig.add_trace(
            go.Scatter(x=self.dates, y=self.drawdowns, mode='lines', fill='tozeroy', name='回撤'),
            secondary_y=True
        )

        # 设置图表布局
        fig.update_layout(
            title=f'{self.stra_name} 净值与回撤图',
            xaxis_title='日期',
            yaxis_title='净值',
            yaxis2_title='回撤',
            font=dict(
                family='Arial, sans-serif',
                size=18,
                color='black'
            ),
            showlegend=True
        )

        # 显示图表
        fig.show()
#%%
# 主执行部分
if __name__ == "__main__":
    data_csv_path = 'path_to_your_data.csv'
    ld = LoadData(date_s='2001-1-1', date_e='2020-12-31')
    df_chg = ld.get_data_from_csv_or_db(csv_file_path=data_csv_path, force_reload=False)
    backtest = BackTest(df_stk=df_chg, stra_name="等权重交易", initial_capital=1)
    backtest.run_backtest()
    backtest.plot_net_value_and_drawdown()
# %%
