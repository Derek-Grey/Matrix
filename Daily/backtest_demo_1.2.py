# %%
import pymongo
import pandas as pd
from loguru import logger
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
from pymongo import InsertOne
import numpy as np
import plotly.graph_objs as go

# %% [markdown]
# ## 1.自定义功能函数

# %%
def _thread_insert2db(table: pymongo.collection, df: pd.DataFrame) -> None:
    # 丢弃DF索引后插入数据库
    data = list(json.loads(df.reset_index(drop=True).transpose().to_json()).values())
    try:
        table.insert_many(data, ordered=False)
    except Exception as e:
        logger.warning(f'{str(e)[:300]}')
        requests = list(map(lambda d: InsertOne(d), data))
        result = table.bulk_write(requests, ordered=False)
        logger.warning(f'异常情况下写入了 -->> {result.inserted_count} 条，PS:总共（{len(data)}）条')

def insert_db_from_df(table: pymongo.collection, df: pd.DataFrame) -> None:
    if table is None or df is None:
        raise Exception("必须传入数据表，数据(df格式)")
    if df.empty:
        raise Exception("数据 df 为空，请检查！目标table：{}".format(table))
    df_len = df.shape[0]
    if df_len > 1500000:
        cpus = int(psutil.cpu_count(logical=False) * 0.7)
        logger.info(f'数据量为：{df_len}，将分拆成 {cpus} 个线程 分布入库')
        df_list = np.array_split(df, cpus)
        arg_list = [(table, df_) for df_ in df_list]
        with ThreadPoolExecutor(max_workers=cpus) as pool:
            pool.map(lambda arg: _thread_insert2db(*arg), arg_list)
    else:
        _thread_insert2db(table=table, df=df)

def df_trans_to_str_and_insert_db(table: pymongo.collection = None, df: pd.DataFrame = None) -> None:
    df_str = df.astype(str)
    insert_db_from_df(table, df_str)
    
def trans_str_to_float64(df: pd.DataFrame, exp_cols: list = None, trans_cols: list = None) -> pd.DataFrame:
    """如果给定 exp_cols 就不会考虑 trans_cols ，如果都不指定，就全部转"""
    if (trans_cols is None) and (exp_cols is None):
        trans_cols = df.columns
    if not (exp_cols is None):
        trans_cols = list(set(df.columns) - set(exp_cols))
    df[trans_cols] = df[trans_cols].astype('float64')
    return df

# %%

from urllib.parse import quote_plus
def get_client(c_from='local'):
    client_dict = {
        'local': '127.0.0.1:27017',
    }
    client_name = client_dict.get(c_from, None)
    if client_name is None:
        raise Exception(f'传入的数据库目标服务器有误 {client_name}，请检查 {client_dict}')
    return pymongo.MongoClient("mongodb://{}".format(client_name))

def get_client_U(m='r'):
    
    # 这是原始数据库 刀片机那台机器
    
    if m == 'r': # 只读
        user, pwd = 'Tom', 'tom'
    else:
        logger.warning(f'你传入的参数 {m} 有毛病，那就返回默认的只读权限吧！')
        user, pwd = 'Tom', 'tom'
    return pymongo.MongoClient("mongodb://%s:%s@%s" % (quote_plus(user),
                                                       quote_plus(pwd),
                                                       '192.168.1.99:29900/'))


class LoadData:
    def __init__(self, date_s: str, date_e: str):
        if date_s is None or date_e is None:
            raise Exception(f'必须指定起止日期！！！')
        self.client_U = get_client_U(m='r')
        # self.client_dev = get_client(c_from='dev')
        # self.client_neo = get_client(c_from='neo')

        self.date_s, self.date_e = date_s, date_e


    def get_stocks_info(self, rt=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        股票状态[ST、停牌]由WIND反馈，涨跌停根据聚宽信息判断
        :param rt: 如果盘中动态出结果&动态换仓，那么涨跌停信息在涨跌幅数据中动态判断
        :return:set_index(['date', 'code'])
        """
        t_info = self.client_U.basic_wind.w_basic_info
        t_limit = self.client_U.basic_jq.jq_daily_price_none

        df_info = pd.DataFrame(t_info.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                           {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
                                           batch_size=1000000))
        if rt:  # 盘中动态出结果，动态换仓，那么涨跌停信息在涨跌幅数据中动态判断
            logger.warning(f'加载完成 ST & 停牌数据，注意这里没有涨跌停数据')
            return df_info.set_index(['date', 'code']).sort_index(), pd.DataFrame()
        use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
        df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": self.date_s, "$lte": self.date_e}}, use_cols, batch_size=1000000))
        df_limit['limit'] = df_limit.apply(lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"], axis=1)
        df_limit['limit'] = df_limit['limit'].astype('int')
        df_limit = df_limit[['date', 'code', 'limit']]

        return df_info.set_index(['date', 'code']).sort_index(), df_limit.set_index(['date', 'code']).sort_index()


    def get_chg_wind(self) -> pd.DataFrame:
        """
        :return: 基于WIND的涨跌幅数据'pct_chg'，set_index(['date', 'code'])
        """
        logger.info(f'加载 基于WIND的日频涨跌幅数据...')
        df = pd.DataFrame(self.client_U.basic_wind.w_vol_price.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                                    {"_id": 0, 'date': 1, 'code': 1, 'pct_chg': 1},
                                                                    batch_size=1000000))
        return trans_str_to_float64(df, trans_cols=['pct_chg', ]).set_index(['date', 'code']).sort_index()


# %% [markdown]
# ## 2.回测类

# %%
class BackTest:

    def __init__(self, df_stk: pd.DataFrame = None, df_info: pd.DataFrame = None, df_limit: pd.DataFrame = None,
                 period=1, hold_num=50, db: pymongo.database.Database = None, stra_name="策略名"):

        """
            __init__:

            此方法用于初始化回测所需的各种参数和数据。
            df_stk: 股票数据DataFrame。
            df_info: 股票信息DataFrame，包括风险警示和交易状态。
            df_limit: 股票涨跌停信息DataFrame。
            period: 调仓周期。
            hold_num: 目标持仓数量。
            db: MongoDB数据库对象。
            stra_name: 策略名称
        """
        if not isinstance(db, pymongo.database.Database): #这行代码检查变量 db 是否是 pymongo.database.Database 类型的实例。如果不是，就会执行条件语句中的代码块。
            raise Exception(f'必须指定 db=pymongo.collection.Database 类型')

        self.df_stocks = df_stk.copy()  # 区间内股票F1信息
        self.df_info = df_info.copy() # 股票信息
        self.df_limit = df_limit.copy() # 停牌信息

        self.period = period  # 调仓周期
        self.hold_num = hold_num  # 目标持仓数量

        self.db = db     # 数据库
        self.stra_name = stra_name  # 策略名称

    def fixed_pos2db(self, t_name=None, hold_sorted=0):
        """基础功能，固定持仓数量，可以设定择股排名区间(如果想从排名101开始，hold_sorted=100)，
        得到每天真实具体持仓的票并入库"""
        if t_name is None:
            t_name = 'default_bt_table'
        table = self.db[t_name]

        #首先检查并创建数据库索引。
        table.create_index([("date", pymongo.ASCENDING), ("code", pymongo.ASCENDING)], background=True, unique=True)
        logger.warning(f"开始回测建仓，起始排名：{hold_sorted}，固定持仓数：{self.hold_num}，策略名：{self.stra_name}")
        #获取股票数据索引列表。
        index_list = self.df_stocks.index.unique().tolist()
        
        """ 循环每个交易日进行持仓调整：
                获取当天的股票推荐列表。
                根据排名和限制条件筛选出实际持仓的股票。
                将持仓股票信息存入数据库。"""
        
        cnt = 1  # 持仓第一天
        target_ = self.df_stocks.loc[index_list[0]].code.tolist() #从股票数据中获取第一个交易日的选股结果， 总计200只
        df_info_ = self.df_info.loc[index_list[1][1]] #从股票信息数据中获取第二个交易日的信息（因为第一个交易日没有信息）。
        df_limit_ = self.df_limit.loc[index_list[1][1]] #从涨跌停数据中获取第二个交易日的信息。

        target_position = target_[hold_sorted:] # 根据排名区间获取目标股票位置。
        target_position = df_info_.loc[target_position].loc[(df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list() #根据股票信息筛选出交易状态正常且没有风险警示的股票。
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list() #根据涨跌停信息筛选出没有涨跌停的股票。
        real_hold_position = target_position[:self.hold_num]  # 得到最终的选股结果池
        

        for i in range(2, len(index_list)):
            # 首先，通过循环迭代每个交易日（从第三个交易日开始，因为前两个交易日用于初始化持仓）。
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{len(real_hold_position)}，策略：{self.stra_name}") # 打印日志记录当前交易日的信息，包括交易日日期、持仓股票数量以及策略名称。
            df_tmp = pd.DataFrame(data=real_hold_position, columns=['code', ]) #将当前持仓股票信息转换为DataFrame，并插入数据库中。
            df_tmp.insert(0, 'date', index_list[i - 1][1])
            df_trans_to_str_and_insert_db(table, df_tmp)
            if cnt < self.period:  # 非调仓日 判断是否为调仓日：
                cnt = cnt + 1 #如果不是调仓日，则增加计数器 cnt。
            else:  # 调仓日 先处理已持仓股票，保留涨跌停和停牌，保留排名但去除今日被ST，；再处理推荐买入股票，去除涨跌停&停牌&ST
                # 如果是调仓日，则进行持仓调整
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy() #根据前一个交易日的股票推荐列表，获取今日的股票数据。
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist() #目标持仓
                df_info_ = self.df_info.loc[index_list[i][1]]
                df_limit_ = self.df_limit.loc[index_list[i][1]]

                # 根据前一个交易日的股票推荐列表，获取今日的股票数据。
                # 根据排名和限制条件筛选出实际持仓的股票，并保留涨跌停和停牌的股票。
                # 获取推荐买入的股票列表，并去除涨跌停和停牌的股票。
                # 更新持仓股票列表为调整后的股票列表。
                today_stock_list = self.df_info.loc[index_list[i][1]].index.tolist()
                
                        
                real_hold_position = list(set(real_hold_position) & set(today_stock_list)) # 剔除不在今天股票池中的票
                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()
                pre_list = target_[:hold_sorted + self.hold_num]  # 当前持仓中进入目标 前 hold_num + start_sorted
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))  # 按排名保留的票中剔除今日被ST的
                limits = list(set(hold_limit + hold_paused + hold_in_pre))  # 已持仓股票处理完成,这里可能还有ST，因为在跌停里面

                df_target_ = df_target_.iloc[hold_sorted:].copy()  # 拿到目标起始位置之后的票
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()  # 先剔除上面需要保留的票
                not_ = df_info_.loc[real_].loc[(df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()  # ST 停牌
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()  # 涨跌停
                real_target_position = target_no_[:self.hold_num - len(limits)] + limits

                real_hold_position = real_target_position

    def fixed_chg2db(self, t_name=None, hold_sorted=0, df_chg=None):
        """
            t_name 数据库中的集合名称
        """
        # 固定收益换手率方法 fixed_chg2db  此方法用于根据固定持仓数量的策略，计算收益和换手率，并将结果存入数据库。
        """固定持仓，直接算好收益、换手率入库 可以设定择股排名区间(如果想从排名101开始，hold_sorted=100)"""
        if t_name is None:
            t_name = 'default_bt_table'
        # 新建数据库
        table = self.db[t_name]
        table.create_index([("date", pymongo.ASCENDING), ], background=True, unique=True)
        logger.warning(f"开始回测建仓，起始排名：{hold_sorted}，固定持仓数：{self.hold_num}，策略名：{self.stra_name}")
        turnover = []
        account_chg = []

        index_list = self.df_stocks.index.unique().tolist()
        cnt = 1  # 持仓第一天
        target_ = self.df_stocks.loc[index_list[0]].code.tolist()
        
        # 第二天看涨跌停信息 
        
        df_info_ = self.df_info.loc[index_list[1][1]]
        df_limit_ = self.df_limit.loc[index_list[1][1]]

        """     循环每个交易日进行持仓调整：
                    计算前一交易日持仓股票的平均收益。
                    根据排名和限制条件筛选出实际持仓的股票。
                    计算换手率。
                    将收益和换手率信息存入数据库。"""
        
        
        target_position = target_[hold_sorted:]
        
        target_position = df_info_.loc[target_position].loc[
            (df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list() # 没有st、不停牌
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list() # 不涨停、不跌停
        real_hold_position = target_position[:self.hold_num]  # 选取一定数量股票池
        
        turnover.append((index_list[1][1], 0))  # 初始化持仓这天，换手为0

        for i in range(2, len(index_list)):
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{len(real_hold_position)}，策略：{self.stra_name}")
            # 取出昨日持仓的收益
            chg_today = df_chg.loc[index_list[i - 1][1]].loc[real_hold_position].pct_chg.mean()  # 默认等权重
            account_chg.append((index_list[i - 1][1], chg_today))

            if cnt < self.period:  # 非调仓日
                cnt = cnt + 1
                turnover.append((index_list[i][1], 0))
            else:  # 调仓日 先处理已持仓股票，去除今日被ST，保留涨跌停和停牌；再处理推荐买入股票，去除涨跌停&停牌&ST
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy()
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist()
                df_info_ = self.df_info.loc[index_list[i][1]]
                df_limit_ = self.df_limit.loc[index_list[i][1]]
                today_stock_list = self.df_info.loc[index_list[i][1]].index.tolist()
                
                        
                real_hold_position = list(set(real_hold_position) & set(today_stock_list)) # 剔除不在今天股票池中的票
                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()
                pre_list = target_[:hold_sorted + self.hold_num]  # 当前持仓中进入目标 前 hold_num + start_sorted
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))  # 按排名保留的票中剔除今日被ST的
                limits = list(set(hold_limit + hold_paused + hold_in_pre))  # 已持仓股票处理完成,这里可能还有ST，因为在跌停里面

                df_target_ = df_target_.iloc[hold_sorted:].copy()  # 拿到目标起始位置之后的票
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()  # 先剔除上面需要保留的票
                not_ = df_info_.loc[[r for r in real_ if r in df_info_.index]].loc[(df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()  # ST 停牌 还有退市
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()  # 涨跌停
                real_target_position = target_no_[:self.hold_num - len(limits)] + limits

                turnover.append((index_list[i][1], len(list(set(real_hold_position) - set(real_target_position))) / self.hold_num))
                real_hold_position = real_target_position

        df_chg = pd.DataFrame(account_chg, columns=['date', 'chg'])
        df_turnover = pd.DataFrame(turnover, columns=['date', 'turnover'])
        df_stra = pd.merge(df_turnover, df_chg, on='date')
        df_stra.insert(1, 'hold_num', self.hold_num)
        
        # insert_db_from_df(table, df_stra)
        return df_stra

# %% [markdown]
# ## 3.实例化示例



# %%
ld = LoadData(date_s='2020-01-01', date_e='2021-12-31')
db_bt = get_client(c_from='local')['bt_test_demo_today'] #mongo 要写入的集合
df_chg = ld.get_chg_wind()
 
def calc_past_5d_pct_chg(group):
    # 对每个股票代码进行分组并计算过去5日涨幅
    group['past_5d_pct_chg'] = group['pct_chg'].rolling(window=5).sum().shift(1)
    return group
ds = df_chg.groupby('code').apply(calc_past_5d_pct_chg)
ds.index = ds.index.droplevel()
ds.reset_index(inplace=True)
ds['month'] = ds.date.str[:7]
ds.set_index(['month', 'date'], inplace=True)
df_stk = ds.dropna().sort_values(by=['date', 'past_5d_pct_chg'], ascending=[True, False]).drop(columns=['pct_chg'])


# %%
df_stk

# %%
ld = LoadData(date_s='2020-01-09', date_e='2021-12-31')
df_info, df_limit = ld.get_stocks_info(rt=False) #获取股票信息，字段可以问小胡
df_chg = ld.get_chg_wind()
# df_stk = ld.get_stra_res(c_from='model', db_name='stra_V3_1', table_name='stocks_list')
bt = BackTest(df_stk=df_stk, df_info=df_info, df_limit=df_limit, hold_num=20, db=db_bt, stra_name='v31_fixed_20')
# bt.fixed_chg2db(t_name=None, hold_sorted=0, df_chg=df_chg) #涨跌幅入库
df_stra = bt.fixed_chg2db(t_name=None, hold_sorted=0, df_chg=df_chg) #涨跌幅入库

# %%


# %%
df_info.head()

# %%
df_limit.head()

# %%
df_chg.head()

# %%
df_stk.head()

# %% [markdown]
# ## 4.净值计算示例

# %%
def _plot_net_value(df: pd.DataFrame, text: str,turn_loss):
    """累计净值曲线图"""

    df.reset_index(inplace=True)
    df.set_index('date',inplace=True)
    start_date = df.index[0]
    df['loss'] = 0.0013 # 固定成本
    df.loc[df.index > '2023-08-31', 'loss'] = 0.0008 
    df['loss'] = df['loss'] + float(turn_loss)#调整成本
    df['chg_'] = df.chg - df.turnover * df.loss#计算净变化
    df['net_value'] = (df.chg_ + 1).cumprod()#计算累计净值
    dates = df.index.unique().tolist()#计算最大净值
    for date in dates:
        df.loc[date, 'max_net'] = df.loc[:date].net_value.max()
    df['back_net'] = df['net_value'] / df['max_net'] - 1#计算回撤
    s_ = df.iloc[-1]
    
    ana = format(s_.net_value ** (252 / df.shape[0]) - 1, '.2%')
    vola = format(df.net_value.pct_change().std() * 21 ** 0.5, '.2%')
    g1 = go.Scatter(x=df.index.unique().tolist(), y=df['net_value'], name='净值')
    g2 = go.Scatter(x=df.index.unique().tolist(), y=df['back_net'] * 100, name='回撤', xaxis='x', yaxis='y2', mode="none",
                    fill="tozeroy")

    fig = go.Figure(
                    data=[g1, g2, ],
                    # data=[g1, ],
                    layout={
                        'height': 1122,
                        "title": f"{text}，净值（左）& 回撤（右），全期：{start_date} ~ {s_.name}，年化收益：{ana}，月波动：{vola}",
                        "font": {"size": 22, },
                        "yaxis": {"title": "累计净值", },
                        "yaxis2": {"title": "最大回撤", "side": "right", "overlaying": "y", "ticksuffix": "%", "showgrid": False, },
                    })
    fig.show()
# _plot_net_value(bt.fixed_chg2db(t_name=None, hold_sorted=0, df_chg=df_chg),'111',0.003)

# %%


# %%
_plot_net_value(df_stra,'111',0.003)

# %%


# %%


# %%


# %%


# %%



