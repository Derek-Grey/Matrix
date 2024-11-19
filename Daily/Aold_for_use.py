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
import os
import time

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

class BackTest:

    def __init__(self, df_stk: pd.DataFrame = None, df_info: pd.DataFrame = None, df_limit: pd.DataFrame = None,
                 period=1, hold_num=50, db: pymongo.database.Database = None, stra_name="策略名"):
        """
        初始化回测所需的各种参数和数据。
        """
        if not isinstance(db, pymongo.database.Database):
            raise Exception(f'必须指定 db=pymongo.collection.Database 类型')

        self.df_stocks = df_stk.copy()
        self.df_info = df_info.copy()
        self.df_limit = df_limit.copy()

        self.period = period
        self.hold_num = hold_num

        self.db = db
        self.stra_name = stra_name

    def fixed_chg2db(self, t_name=None, hold_sorted=0, df_chg=None):
        """
        计算收益和换手率，并将结果存入数据库，同时输出每日持有的股票。
        
        参数:
        t_name (str): 数据库表的名称。如果未提供，使用默认值 'default_bt_table'。
        hold_sorted (int): 控制初始持仓排序的变量。
        df_chg (DataFrame): 股票每日涨跌幅数据。
        
        返回:
        df_stra (DataFrame): 包含每日收益和换手率的策略结果。
        df_hold_positions (DataFrame): 每日持仓股票数据。
        """
         # 记录回测开始时间
        start_time = time.time()
        if t_name is None:
            t_name = 'default_bt_table'  # 如果未提供表名，使用默认值
        table = self.db[t_name]
        table.create_index([("date", pymongo.ASCENDING)], background=True, unique=True)  # 为日期创建索引以提高查询效率
        logger.warning(f"开始回测建仓，起始排名：{hold_sorted}，固定持仓数：{self.hold_num}，策略名：{self.stra_name}")

        turnover = []  # 换手率记录列表
        account_chg = []  # 收益记录列表
        hold_positions = []  # 每日持仓股票记录列表

        index_list = self.df_stocks.index.unique().tolist()  # 获取唯一日期列表
        cnt = 1
        target_ = self.df_stocks.loc[index_list[0]].code.tolist()  # 获取第一天的股票代码列表
        df_info_ = self.df_info.loc[index_list[1][1]]  # 获取第二天的股票信息
        df_limit_ = self.df_limit.loc[index_list[1][1]]  # 获取第二天的涨跌停信息

        # 计算初始持仓
        target_position = target_[hold_sorted:]
        target_position = df_info_.loc[target_position].loc[
            (df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list()  # 剔除停牌和风险警示的股票
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list()  # 剔除涨跌停股票
        real_hold_position = target_position[:self.hold_num]  # 确保持仓数不超过指定数量

        turnover.append((index_list[1][1], 0))  # 初始换手率为0
        hold_positions.append((index_list[1][1], real_hold_position))  # 记录初始持仓

        for i in range(2, len(index_list)):
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{len(real_hold_position)}，策略：{self.stra_name}")
            chg_today = df_chg.loc[index_list[i - 1][1]].loc[real_hold_position].pct_chg.mean()  # 计算当日持仓股票的平均涨跌幅
            account_chg.append((index_list[i - 1][1], chg_today))  # 记录每日收益

            if cnt < self.period:
                cnt += 1
                turnover.append((index_list[i][1], 0))  # 在持仓周期内换手率为0
                hold_positions.append((index_list[i][1], real_hold_position))  # 保持现有持仓不变
            else:
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy()  # 获取上一交易日的股票数据
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist()  # 获取上一交易日的股票代码列表
                df_info_ = self.df_info.loc[index_list[i][1]]  # 获取当前交易日的股票信息
                df_limit_ = self.df_limit.loc[index_list[i][1]]  # 获取当前交易日的涨跌停信息
                today_stock_list = self.df_info.loc[index_list[i][1]].index.tolist()  # 获取当天股票代码列表

                real_hold_position = list(set(real_hold_position) & set(today_stock_list))  # 保留当日仍在交易的持仓股票
                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()  # 找出被涨跌停限制的股票
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()  # 找出停牌的股票
                pre_list = target_[:hold_sorted + self.hold_num]  # 获取前N只股票
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()  # 去掉风险警示的股票
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))  # 计算在前N只股票中的非风险警示股票
                limits = list(set(hold_limit + hold_paused + hold_in_pre))  # 生成持仓限制列表

                df_target_ = df_target_.iloc[hold_sorted:].copy()
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()  # 剔除受限股票
                not_ = df_info_.loc[[r for r in real_ if r in df_info_.index]].loc[
                    (df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()  # 剔除停牌和风险警示的股票
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()  # 剔除涨跌停股票
                real_target_position = target_no_[:self.hold_num - len(limits)] + limits  # 生成新的持仓股票列表

                # 计算换手率并记录
                turnover.append((index_list[i][1], len(list(set(real_hold_position) - set(real_target_position))) / self.hold_num))
                hold_positions.append((index_list[i][1], real_target_position))  # 记录新的持仓股票
                real_hold_position = real_target_position  # 更新持仓股票

        # 创建收益和换手率的数据框
        df_chg = pd.DataFrame(account_chg, columns=['date', 'chg'])
        df_turnover = pd.DataFrame(turnover, columns=['date', 'turnover'])
        df_hold_positions = pd.DataFrame(hold_positions, columns=['date', 'hold_positions'])
        df_stra = pd.merge(df_turnover, df_chg, on='date')  # 将换手率和收益合并成一个数据框
        df_stra.insert(1, 'hold_num', self.hold_num)  # 插入持仓数量列

        # 确保 DataFile 文件夹存在
        os.makedirs('DataFile', exist_ok=True)

        # 保存每日持仓数据为 CSV 文件
        df_hold_positions['hold_positions'] = df_hold_positions['hold_positions'].apply(lambda x: ','.join(x))  # 将持仓列表转换为逗号分隔的字符串
        df_hold_positions.to_csv('DataFile/daily_holdings.csv', index=False)

        # 保存策略结果为 CSV 文件
        df_stra.to_csv('DataFile/strategy_results.csv', index=False)
         # 记录回测结束时间
        end_time = time.time()
        logger.debug(f"回测总耗时: {end_time - start_time:.4f} 秒")
        return df_stra, df_hold_positions  # 返回策略结果和每日持仓数据

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

# %% [markdown]
# ## 3.实例化示例

# %%
ld = LoadData(date_s='2022-08-01', date_e='2022-09-01')
db_bt = get_client(c_from='local')['bt_test_demo_today'] #mongo 要写入的集合
df_chg = ld.get_chg_wind()
 
# def calc_past_5d_pct_chg(group):
#     # 对每个股票代码进行分组并计算过去5日涨幅
#     group['past_5d_pct_chg'] = group['pct_chg'].rolling(window=5).sum().shift(1)
#     return group
# ds = df_chg.groupby('code').apply(calc_past_5d_pct_chg)
# ds.index = ds.index.droplevel()
# ds.reset_index(inplace=True)
# ds['month'] = ds.date.str[:7]
# ds.set_index(['month', 'date'], inplace=True)
# df_stk = ds.dropna().sort_values(by=['date', 'past_5d_pct_chg'], ascending=[True, False]).drop(columns=['pct_chg'])


# %%
# df_stk=pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\stra_V3_1.stocks_list.csv',date_format='%Y-%m-%d')
df_stk=pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\stra_V3_11.csv')
# df_stk['date'] = df_stk['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
df_stk = df_stk.loc[pd.to_datetime(df_stk.date)>=pd.to_datetime('2022-08-01')]
df_stk = df_stk.loc[pd.to_datetime(df_stk.date)<=pd.to_datetime('2022-09-01')]
# df_stk['month'] = pd.to_datetime(df_stk['date']).dt.strftime('%Y-%m')
df_stk.set_index(['month', 'date'], inplace=True)
df_stk = df_stk[['code','F1']]
df_stk.sort_index()

# df_stk.set_index(['date'], inplace=True)

# %%
ld = LoadData(date_s='2022-08-01', date_e='2022-09-01')
df_info, df_limit = ld.get_stocks_info(rt=False) #获取股票信息，字段可以问小胡
df_chg = ld.get_chg_wind()
# df_stk = ld.get_stra_res(c_from='model', db_name='stra_V3_1', table_name='stocks_list')
bt = BackTest(df_stk=df_stk, df_info=df_info, df_limit=df_limit, hold_num=20, db=db_bt, stra_name='v31_fixed_20')
# bt.fixed_chg2db(t_name=None, hold_sorted=0, df_chg=df_chg) #涨跌幅入库

df_stra, df_hold_positions = bt.fixed_chg2db(t_name=None, hold_sorted=0, df_chg=df_chg)

# %%
