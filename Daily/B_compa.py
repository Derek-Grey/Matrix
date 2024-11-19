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
from functools import reduce
from pymongo.database import Database
import math
# %% [markdown]
# ## 1.自定义功能函数
# 不参与RANK的行业
USELESS_INDUS = ["证券、期货业", "银行业", "货币金融服务", "其他金融业", "资本市场服务", "保险业",
                 "燃气生产和供应业", "电力、蒸汽、热水的生产和供应业", "煤气生产和供应业", "电力、热力生产和供应业",
                 "水的生产和供应业", "自来水的生产和供应业", "卫生", "公共设施服务业",
                 "房屋建筑业", "房地产业", "房地产中介服务业", "房地产管理业", "房地产开发与经营业",
                 '金融信托业',
                 ]
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

    def _get_dynamic_nums(self):
        # 取得每个交易月的第一个交易日
        df_dates = pd.DataFrame(self.client_U.economic.trade_dates.find({"trade_date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                                        {'_id': 0, 'trade_date': 1}))
        df_dates['month'] = df_dates.trade_date.str[:7]
        df_dates = df_dates.loc[~(df_dates.duplicated(subset=['month', ], keep='first'))]
        trade_first_m_day = df_dates.trade_date.tolist()

        # 取得剔除行业、ST、次新后每个月第一天的股票数量
        pipeline = [
            {'$match': {'date': {'$in': trade_first_m_day},
                        'industry': {'$nin': USELESS_INDUS},
                        'trade_days': {'$gte': 365}}},
            {'$group': {'_id': '$date', 'count': {'$sum': 1}}},
        ]
        df_m_day_count = pd.DataFrame(list(self.client_U.basic_wind.w_basic_info.aggregate(pipeline)), columns=['_id', 'count'])
        df_m_day_count['month'] = df_m_day_count['_id'].str[:7]
        df_m_day_count.set_index('month', inplace=True)
        df_m_day_count.sort_index(inplace=True)
        return df_m_day_count

    def get_hold_num(self, hold_num=50, start_sorted=100, the_end_month=None, fixed_by_month=True):
        """
        :param hold_num:
        :param start_sorted:
        :param the_end_month: 指定这个月起 start_sorted 为对应值，后面就不变了
        :param fixed_by_month:用这个月去固定 the_end_month 的持仓数量是否不变(固定为指定的hold_num)
        :return: set_index('month') df[['hold_s', 'hold_num']]
        """
        df = self._get_dynamic_nums()
        if the_end_month is None:
            the_end_count = df.iloc[-1]['count']
        else:
            the_end_count = df.loc[the_end_month]['count']

        df['hold_s'] = (df['count'] * (start_sorted / the_end_count)).apply(lambda x: math.floor(x))
        df['hold_e'] = (df['count'] * ((start_sorted + hold_num) / the_end_count)).apply(lambda x: math.floor(x))
        df['hold_num'] = df.hold_e - df.hold_s
        df['num_pre'] = df.hold_num.shift(-1)
        df.ffill(inplace=True)
        # df.fillna(method='ffill', inplace=True)
        df['hold_num'] = df.apply(lambda dx: dx.hold_num if dx.hold_num <= dx.num_pre else dx.num_pre, axis=1).astype(int)
        df.loc[the_end_month:, 'hold_s'] = start_sorted
        if fixed_by_month:
            df.loc[the_end_month:, 'hold_num'] = hold_num
        return df[['hold_s', 'hold_num']]
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

    def process_backtest(self, t_name=None, hold_sorted=0, df_chg=None):
        """
        合并 fixed_pos2db 和 fixed_chg2db 的功能。
        将持仓信息插入数据库，并计算收益和换手率。
        """
        if t_name is None:
            t_name = 'default_bt_table'
        table = self.db[t_name]
        # 创建数据库索引以提高查询效率
        table.create_index([("date", pymongo.ASCENDING), ("code", pymongo.ASCENDING)], background=True, unique=True)
        logger.warning(f"开始回测建仓，起始排名：{hold_sorted}，固定持仓数：{self.hold_num}，策略名：{self.stra_name}")

        index_list = self.df_stocks.index.unique().tolist()
        cnt = 1  # 持仓第一天
        target_ = self.df_stocks.loc[index_list[0]].code.tolist()
        
        df_info_ = self.df_info.loc[index_list[1][1]]
        df_limit_ = self.df_limit.loc[index_list[1][1]]

        # 根据排名区间获取目标股票位置
        target_position = target_[hold_sorted:]
        # 筛选出交易状态正常且没有风险警示的股票
        target_position = df_info_.loc[target_position].loc[
            (df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list()
        # 筛选出没有涨跌停的股票
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list()
        real_hold_position = target_position[:self.hold_num]

        turnover = [(index_list[1][1], 0)]  # 初始化持仓这天，换手为0
        account_chg = []

        for i in range(2, len(index_list)):
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{len(real_hold_position)}，策略：{self.stra_name}")
            # 将当前持仓股票信息转换为DataFrame，并插入数据库中
            df_tmp = pd.DataFrame(data=real_hold_position, columns=['code', ])
            df_tmp.insert(0, 'date', index_list[i - 1][1])
            df_trans_to_str_and_insert_db(table, df_tmp)

            if df_chg is not None:
                # 计算前一交易日持仓股票的平均收益
                chg_today = df_chg.loc[index_list[i - 1][1]].loc[real_hold_position].pct_chg.mean()
                account_chg.append((index_list[i - 1][1], chg_today))

            if cnt < self.period:  # 非调仓日
                cnt += 1
                turnover.append((index_list[i][1], 0))
            else:  # 调仓日
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy()
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist()
                df_info_ = self.df_info.loc[index_list[i][1]]
                df_limit_ = self.df_limit.loc[index_list[i][1]]
                today_stock_list = self.df_info.loc[index_list[i][1]].index.tolist()

                # 剔除不在今天股票池中的票
                real_hold_position = list(set(real_hold_position) & set(today_stock_list))
                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()
                pre_list = target_[:hold_sorted + self.hold_num]
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))
                limits = list(set(hold_limit + hold_paused + hold_in_pre))

                df_target_ = df_target_.iloc[hold_sorted:].copy()
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()
                not_ = df_info_.loc[[r for r in real_ if r in df_info_.index]].loc[
                    (df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()
                real_target_position = target_no_[:self.hold_num - len(limits)] + limits

                # 计算换手率
                turnover.append((index_list[i][1], len(list(set(real_hold_position) - set(real_target_position))) / self.hold_num))
                real_hold_position = real_target_position

        if df_chg is not None:
            df_chg = pd.DataFrame(account_chg, columns=['date', 'chg'])
            df_turnover = pd.DataFrame(turnover, columns=['date', 'turnover'])
            df_stra = pd.merge(df_turnover, df_chg, on='date')
            df_stra.insert(1, 'hold_num', self.hold_num)
            return df_stra
class BackTestRegion(BackTest):
    """

    模拟盘主要调用

    1、换仓日T 用T-1日的选股结果；
    2、内部实现的为持仓数量动态计算的；
    3、_dyn 结尾的模块 用于参考盘中价换仓
    """

    def __init__(self, df_stk: pd.DataFrame, df_info: pd.DataFrame, df_limit: pd.DataFrame,
                 df_mv: pd.DataFrame, db: Database, stra_name: str):

        super().__init__(df_stk=df_stk, df_info=df_info, df_limit=df_limit, db=db, stra_name=stra_name)

        if df_mv.empty:
            raise Exception(f'动态持仓数量 DF 必须要有！！')

        self.df_m = df_mv.copy()

    def range_pos2db(self, t_name: str = None):
        if t_name is None:
            t_name = 'default_bt_table'
        table = self.db[t_name]
        table.create_index([("date", pymongo.ASCENDING), ("code", pymongo.ASCENDING)], background=True, unique=True)

        index_list = self.df_stocks.index.unique().tolist()
        month_stain = index_list[0][0]

        df_m_ = self.df_m.loc[month_stain]
        hold_num = df_m_.hold_num
        start_num = df_m_.hold_s

        logger.warning("开始回测，浮动持仓,交易月：{}，目标持仓数：{}，起始位：{}".format(month_stain, hold_num, start_num))

        cnt = 1  # 持仓第一天
        target_ = self.df_stocks.loc[index_list[0]].code.tolist()
        df_info_ = self.df_info.loc[index_list[1][1]]
        df_limit_ = self.df_limit.loc[index_list[1][1]]

        target_position = target_[start_num:]
        target_position = df_info_.loc[target_position].loc[
            (df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list()
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list()
        real_hold_position = target_position[:hold_num]  # 选取一定数量股票池

        for i in range(2, len(index_list)):
            df_tmp = pd.DataFrame(data=real_hold_position, columns=['code', ])
            df_tmp.insert(0, 'date', index_list[i - 1][1])
            df_trans_to_str_and_insert_db(table, df_tmp)

            month = index_list[i][0]
            if month != month_stain:  # 换月，重新计算目标持仓
                df_m_ = self.df_m.loc[month_stain]
                hold_num = df_m_.hold_num
                start_num = df_m_.hold_s
                logger.info("交易月：{}，目标持仓数：{}".format(month, hold_num))
                month_stain = month
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{hold_num} ~ {len(real_hold_position)}，策略：{self.stra_name}")

            if cnt < self.period:  # 非调仓日
                cnt = cnt + 1
            else:  # 调仓日 处理今日推荐买入股票，去除涨跌停，再处理持仓股票，保留涨跌停和停牌
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy()
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist()
                df_info_ = self.df_info.loc[index_list[i][1]]
                df_limit_ = self.df_limit.loc[index_list[i][1]]

                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()
                pre_list = target_[:start_num + hold_num]  # 当前持仓中进入目标 前 hold_num + start_num
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))  # 按排名保留的票中剔除今日被ST的
                limits = list(set(hold_limit + hold_paused + hold_in_pre))  # 已持仓股票处理完成,这里可能还有ST，因为在跌停里面

                df_target_ = df_target_.iloc[start_num:].copy()  # 拿到目标起始位置之后的票
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()  # 先剔除上面需要保留的票
                not_ = df_info_.loc[real_].loc[(df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()  # ST 停牌
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()  # 涨跌停
                real_target_position = target_no_[:hold_num - len(limits)] + limits

                real_hold_position = real_target_position

    def range_chg2db(self, t_name: str = None, df_chg: pd.DataFrame = None):
        if t_name is None:
            t_name = 'default_bt_table'
        table = self.db[t_name]
        table.create_index([("date", 1), ], background=True, unique=True)

        index_list = self.df_stocks.index.unique().tolist()
        month_stain = index_list[0][0]

        df_ = self.df_m.loc[month_stain]
        hold_num = df_.hold_num
        start_num = df_.hold_s

        logger.warning("开始回测，浮动持仓,交易月：{}，目标持仓数：{}，起始位：{}".format(month_stain, hold_num, start_num))

        turnover_list = []  # 记录每次交易的换手率
        hold_num_list = []  # 记录每次持仓数量
        account_chg_list = []  # 记录策略每日涨跌幅数据，由(日期，涨跌幅)元组构成,算交易成本

        cnt = 1  # 持仓第一天
        target_ = self.df_stocks.loc[index_list[0]].code.tolist()
        df_info_ = self.df_info.loc[index_list[1][1]]
        df_limit_ = self.df_limit.loc[index_list[1][1]]

        target_position = target_[start_num:]
        target_position = df_info_.loc[target_position].loc[
            (df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list()
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list()
        real_hold_position = target_position[:hold_num]  # 选取一定数量股票池

        turnover_list.append((index_list[1][1], 0))  # 初始化持仓这天，换手为0

        for i in range(2, len(index_list)):
            month = index_list[i][0]
            if month != month_stain:  # 换月，重新计算目标持仓
                df_ = self.df_m.loc[month_stain]
                hold_num = df_.hold_num
                start_num = df_.hold_s
                logger.info("交易月：{}，目标持仓数：{}".format(month, hold_num))
                month_stain = month
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{hold_num} ~ {len(real_hold_position)}，策略：{self.stra_name}")

            # # 取出昨日持仓的收益
            chg_today = df_chg.loc[index_list[i - 1][1]].loc[real_hold_position].pct_chg.mean()  # 默认等权重
            account_chg_list.append((index_list[i - 1][1], chg_today))
            hold_num_list.append((index_list[i - 1][1], hold_num))
            # # 取出昨日持仓的收益
            # chg_today = df_chg.loc[index_list[i ][1]].loc[real_hold_position].pct_chg.mean()  # 默认等权重
            # account_chg_list.append((index_list[i][1], chg_today))
            # hold_num_list.append((index_list[i][1], hold_num))

            if cnt < self.period:  # 非调仓日
                cnt = cnt + 1
                turnover_list.append((index_list[i][1], 0))
            else:
                # if hold_num>18:
                #     print('111')
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy()
                # df_target_ = self.df_stocks.loc[index_list[i]].copy()
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist()
                df_info_ = self.df_info.loc[index_list[i][1]]
                df_limit_ = self.df_limit.loc[index_list[i][1]]
                today_stock_list = self.df_info.loc[index_list[i][1]].index.tolist()
                
                        
                real_hold_position = list(set(real_hold_position) & set(today_stock_list)) # 剔除不在今天股票池中的票
                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()
                pre_list = target_[:start_num + hold_num]  # 当前持仓中进入目标 前 hold_num + start_num
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))  # 按排名保留的票中剔除今日被ST的
                limits = list(set(hold_limit + hold_paused + hold_in_pre))  # 已持仓股票处理完成,这里可能还有ST，因为在跌停里面

                df_target_ = df_target_.iloc[start_num:].copy()  # 拿到目标起始位置之后的票
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()  # 先剔除上面需要保留的票
                not_ = df_info_.loc[real_].loc[(df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()  # ST 停牌
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()  # 涨跌停
                real_target_position = target_no_[:hold_num - len(limits)] + limits

                stocks_sell = list(set(real_hold_position) - set(real_target_position))
                turnover_list.append((index_list[i][1], len(stocks_sell) / hold_num))
                # turnover_list.append((index_list[i][1], 0))
                # real_target_position = target_[:hold_num]
                real_hold_position = real_target_position

        df = reduce(lambda df1, df2: pd.merge(df1, df2, on='date'), [
            pd.DataFrame(hold_num_list, columns=['date', 'hold_num']),
            pd.DataFrame(turnover_list, columns=['date', 'turnover']),
            pd.DataFrame(account_chg_list, columns=['date', 'chg'])
        ])
        insert_db_from_df(table, df)
        return df
    
class BackTestRegion(BackTest):
    """

    模拟盘主要调用

    1、换仓日T 用T-1日的选股结果；
    2、内部实现的为持仓数量动态计算的；
    3、_dyn 结尾的模块 用于参考盘中价换仓
    """

    def __init__(self, df_stk: pd.DataFrame, df_info: pd.DataFrame, df_limit: pd.DataFrame,
                 df_mv: pd.DataFrame, db: Database, stra_name: str):

        super().__init__(df_stk=df_stk, df_info=df_info, df_limit=df_limit, db=db, stra_name=stra_name)

        if df_mv.empty:
            raise Exception(f'动态持仓数量 DF 必须要有！！')

        self.df_m = df_mv.copy()

    def range_pos2db(self, t_name: str = None):
        if t_name is None:
            t_name = 'default_bt_table'
        table = self.db[t_name]
        table.create_index([("date", pymongo.ASCENDING), ("code", pymongo.ASCENDING)], background=True, unique=True)

        index_list = self.df_stocks.index.unique().tolist()
        month_stain = index_list[0][0]

        df_m_ = self.df_m.loc[month_stain]
        hold_num = df_m_.hold_num
        start_num = df_m_.hold_s

        logger.warning("开始回测，浮动持仓,交易月：{}，目标持仓数：{}，起始位：{}".format(month_stain, hold_num, start_num))

        cnt = 1  # 持仓第一天
        target_ = self.df_stocks.loc[index_list[0]].code.tolist()
        df_info_ = self.df_info.loc[index_list[1][1]]
        df_limit_ = self.df_limit.loc[index_list[1][1]]

        target_position = target_[start_num:]
        target_position = df_info_.loc[target_position].loc[
            (df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list()
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list()
        real_hold_position = target_position[:hold_num]  # 选取一定数量股票池

        for i in range(2, len(index_list)):
            df_tmp = pd.DataFrame(data=real_hold_position, columns=['code', ])
            df_tmp.insert(0, 'date', index_list[i - 1][1])
            df_trans_to_str_and_insert_db(table, df_tmp)

            month = index_list[i][0]
            if month != month_stain:  # 换月，重新计算目标持仓
                df_m_ = self.df_m.loc[month_stain]
                hold_num = df_m_.hold_num
                start_num = df_m_.hold_s
                logger.info("交易月：{}，目标持仓数：{}".format(month, hold_num))
                month_stain = month
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{hold_num} ~ {len(real_hold_position)}，策略：{self.stra_name}")

            if cnt < self.period:  # 非调仓日
                cnt = cnt + 1
            else:  # 调仓日 处理今日推荐买入股票，去除涨跌停，再处理持仓股票，保留涨跌停和停牌
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy()
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist()
                df_info_ = self.df_info.loc[index_list[i][1]]
                df_limit_ = self.df_limit.loc[index_list[i][1]]

                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()
                pre_list = target_[:start_num + hold_num]  # 当前持仓中进入目标 前 hold_num + start_num
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))  # 按排名保留的票中剔除今日被ST的
                limits = list(set(hold_limit + hold_paused + hold_in_pre))  # 已持仓股票处理完成,这里可能还有ST，因为在跌停里面

                df_target_ = df_target_.iloc[start_num:].copy()  # 拿到目标起始位置之后的票
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()  # 先剔除上面需要保留的票
                not_ = df_info_.loc[real_].loc[(df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()  # ST 停牌
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()  # 涨跌停
                real_target_position = target_no_[:hold_num - len(limits)] + limits

                real_hold_position = real_target_position

    def range_chg2db(self, t_name: str = None, df_chg: pd.DataFrame = None):
        if t_name is None:
            t_name = 'default_bt_table'
        table = self.db[t_name]
        table.create_index([("date", 1), ], background=True, unique=True)

        index_list = self.df_stocks.index.unique().tolist()
        month_stain = index_list[0][0]

        df_ = self.df_m.loc[month_stain]
        hold_num = df_.hold_num
        start_num = df_.hold_s

        logger.warning("开始回测，浮动持仓,交易月：{}，目标持仓数：{}，起始位：{}".format(month_stain, hold_num, start_num))

        turnover_list = []  # 记录每次交易的换手率
        hold_num_list = []  # 记录每次持仓数量
        account_chg_list = []  # 记录策略每日涨跌幅数据，由(日期，涨跌幅)元组构成,算交易成本

        cnt = 1  # 持仓第一天
        target_ = self.df_stocks.loc[index_list[0]].code.tolist()
        df_info_ = self.df_info.loc[index_list[1][1]]
        df_limit_ = self.df_limit.loc[index_list[1][1]]

        target_position = target_[start_num:]
        target_position = df_info_.loc[target_position].loc[
            (df_info_.trade_status == 1) & (df_info_.riskwarning == 0)].index.to_list()
        target_position = df_limit_.loc[target_position].loc[df_limit_.limit == 0].index.to_list()
        real_hold_position = target_position[:hold_num]  # 选取一定数量股票池

        turnover_list.append((index_list[1][1], 0))  # 初始化持仓这天，换手为0

        for i in range(2, len(index_list)):
            month = index_list[i][0]
            if month != month_stain:  # 换月，重新计算目标持仓
                df_ = self.df_m.loc[month_stain]
                hold_num = df_.hold_num
                start_num = df_.hold_s
                logger.info("交易月：{}，目标持仓数：{}".format(month, hold_num))
                month_stain = month
            logger.info(f"交易日：{index_list[i][1]}，持仓数：{hold_num} ~ {len(real_hold_position)}，策略：{self.stra_name}")

            # # 取出昨日持仓的收益
            chg_today = df_chg.loc[index_list[i - 1][1]].loc[real_hold_position].pct_chg.mean()  # 默认等权重
            account_chg_list.append((index_list[i - 1][1], chg_today))
            hold_num_list.append((index_list[i - 1][1], hold_num))
            # # 取出昨日持仓的收益
            # chg_today = df_chg.loc[index_list[i ][1]].loc[real_hold_position].pct_chg.mean()  # 默认等权重
            # account_chg_list.append((index_list[i][1], chg_today))
            # hold_num_list.append((index_list[i][1], hold_num))

            if cnt < self.period:  # 非调仓日
                cnt = cnt + 1
                turnover_list.append((index_list[i][1], 0))
            else:
                # if hold_num>18:
                #     print('111')
                cnt = 1
                df_target_ = self.df_stocks.loc[index_list[i - 1]].copy()
                # df_target_ = self.df_stocks.loc[index_list[i]].copy()
                df_target_.set_index('code', inplace=True)
                target_ = df_target_.index.tolist()
                df_info_ = self.df_info.loc[index_list[i][1]]
                df_limit_ = self.df_limit.loc[index_list[i][1]]
                today_stock_list = self.df_info.loc[index_list[i][1]].index.tolist()
                
                        
                real_hold_position = list(set(real_hold_position) & set(today_stock_list)) # 剔除不在今天股票池中的票
                hold_limit = df_limit_.loc[real_hold_position].loc[df_limit_.limit == 1].index.to_list()
                hold_paused = df_info_.loc[real_hold_position].loc[df_info_.trade_status == 0].index.to_list()
                pre_list = target_[:start_num + hold_num]  # 当前持仓中进入目标 前 hold_num + start_num
                hold_not_st_list = df_info_.loc[real_hold_position].loc[df_info_.riskwarning == 0].index.to_list()
                hold_in_pre = list(set(pre_list) & set(hold_not_st_list))  # 按排名保留的票中剔除今日被ST的
                limits = list(set(hold_limit + hold_paused + hold_in_pre))  # 已持仓股票处理完成,这里可能还有ST，因为在跌停里面

                df_target_ = df_target_.iloc[start_num:].copy()  # 拿到目标起始位置之后的票
                real_ = df_target_.loc[~(df_target_.index.isin(limits))].index.tolist()  # 先剔除上面需要保留的票
                not_ = df_info_.loc[real_].loc[(df_info_.riskwarning == 0) & (df_info_.trade_status == 1)].index.to_list()  # ST 停牌
                target_no_ = df_limit_.loc[not_].loc[df_limit_.limit == 0].index.to_list()  # 涨跌停
                real_target_position = target_no_[:hold_num - len(limits)] + limits

                stocks_sell = list(set(real_hold_position) - set(real_target_position))
                turnover_list.append((index_list[i][1], len(stocks_sell) / hold_num))
                # turnover_list.append((index_list[i][1], 0))
                # real_target_position = target_[:hold_num]
                real_hold_position = real_target_position

        df = reduce(lambda df1, df2: pd.merge(df1, df2, on='date'), [
            pd.DataFrame(hold_num_list, columns=['date', 'hold_num']),
            pd.DataFrame(turnover_list, columns=['date', 'turnover']),
            pd.DataFrame(account_chg_list, columns=['date', 'chg'])
        ])
        insert_db_from_df(table, df)
        return df
# %% [markdown]
# ## 3.实例化示例



# %%
ld = LoadData(date_s='2009-01-01', date_e='2011-12-31')
db_bt = get_client(c_from='local')['bt_test_demo_today'] #mongo 要写入的集合
df_chg = ld.get_chg_wind()

def calc_past_5d_pct_chg(group):
    # 对每个股票代码进行分组并计算过去5日涨幅
    group['past_5d_pct_chg'] = group['pct_chg'].rolling(window=5).sum()
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
# ld = LoadData(date_s='2009-01-01', date_e='2011-12-31')
# df_info, df_limit = ld.get_stocks_info(rt=False) #获取股票信息，字段可以问小胡
# df_chg = ld.get_chg_wind()
# # df_stk = ld.get_stra_res(c_from='model', db_name='stra_V3_1', table_name='stocks_list')
# bt = BackTest(df_stk=df_stk, df_info=df_info, df_limit=df_limit, hold_num=20, db=db_bt, stra_name='v31_fixed_20')
# # bt.fixed_pos2db(t_name=None, hold_sorted=0) #涨跌幅入库
# bt.fixed_chg2db(t_name=None, hold_sorted=0, df_chg=df_chg) #涨跌幅入库
# # df_stra = bt.fixed_chg2db(t_name=None, hold_sorted=0, df_chg=df_chg) #涨跌幅入库

# %%
# 按比例持仓
n=50
ld = LoadData(date_s='2009-01-01', date_e='2020-02-27')
logger.info(f'加载选股结果...')
db_bt = get_client(c_from='local')['bt_test_demo_today1']
logger.info(f'加载ST、停牌、涨跌停信息...')
df_info, df_limit = ld.get_stocks_info(rt=False)
df_chg = ld.get_chg_wind()
df_mv = ld.get_hold_num(hold_num=n, start_sorted=0, the_end_month='2021-12')
bt = BackTestRegion(df_stk=df_stk, df_info=df_info, df_limit=df_limit, df_mv=df_mv, db=db_bt, stra_name=f'{n}')
bt.range_chg2db(t_name=f'{n}', df_chg=df_chg)

# %%
# df_info.head()

# # %%
# df_limit.head()

# # %%
# df_chg.head()

# # %%
# df_stk.head()

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
    df['loss'] = df['loss'] + float(turn_loss)
    df['chg_'] = df.chg - df.turnover * df.loss
    df['net_value'] = (df.chg_ + 1).cumprod()
    dates = df.index.unique().tolist()
    for date in dates:
        df.loc[date, 'max_net'] = df.loc[:date].net_value.max()
    df['back_net'] = df['net_value'] / df['max_net'] - 1
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


# # %%
# _plot_net_value(df_stra,'111',0.003)

# %%


# %%


# %%


# %%


# %%



