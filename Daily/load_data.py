# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: load_data.py
@time: 2023-08-03 19:07
说明:
"""

import os
import math
import pandas as pd
from functools import reduce
import multiprocessing as mp
from loguru import logger
from settings import USELESS_INDUS, DB_U_MINUTE, END_MONTH
from Utils.utils import trans_str_to_float64
from Utils.db import get_trading_days
from db_client import get_client_U, get_client
from Utils.my_errors import LogExceptions
from datetime import datetime
from functools import wraps
import numpy as np
# from BackTest.hedge_final_extra import Edg

mp.log_to_stderr()
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 14)



def merge_pct_data(start_date_1, end_date_1, start_date_2, end_date_2):
    
    t_ic = get_client_U(m='r')['basic_wind']['w_bench']
    t_futures = get_client_U(m='r')['basic_wind']['w_futures_day']
    max_date_905 = '2015-04-15'
    hedge_index_code = "000905.SH"
    df_ic = pd.DataFrame(t_ic.find({'date': {'$lte': max_date_905, '$gte': '2009-01-01'}, 'code': f'{hedge_index_code}'},
                            {'_id': 0, 'date': 1, 'close':1,'pct_chg': 1,'volume': 1}, batch_size=1000000))


    IC00 = 'IC00.CFE'
    df_11 = pd.DataFrame(t_futures.find({'date': {'$lte': end_date_1, '$gte': start_date_1}, 'code': f'{IC00}'},
                            {'_id': 0, 'date': 1, 'close':1,'pct_chg': 1,'volume': 1}, batch_size=1000000)).sort_values(by='date')

    df_1_filtered = df_11

    IM00 = 'IM00.CFE'
    df_22 = pd.DataFrame(t_futures.find({'date': {'$lte': end_date_2, '$gte': start_date_2}, 'code': f'{IM00}'},
                            {'_id': 0, 'date': 1, 'close':1,'pct_chg': 1,'volume': 1}, batch_size=1000000)).sort_values(by='date')
    df_2_filtered = df_22
    # 筛选第二个文件的指定日期范围数据


    # 将两个DataFrame拼接起来
    edg_data = pd.concat([df_ic,df_1_filtered, df_2_filtered])
    return edg_data




def merge_signal(start_date_1,end_date_1,start_date_2,end_date_2):
    xx = Edg(index_code="000905.SH",dates='2009-01-01',logger_display=False)
    df_1 = xx.cal_hedge_init(show=False)
    df_1.reset_index(inplace=True)
    # 筛选第一个文件的指定日期范围数据
    df_1_filtered = df_1.query(f"date >= @start_date_1 and date <= @end_date_1")

    xxf = Edg(index_code="000852.SH",dates='2009-01-01',logger_display=True)
    df_2 = xxf.cal_hedge_init(show=False)
    df_2.reset_index(inplace=True)

    # 筛选第二个文件的指定日期范围数据
    df_2_filtered = df_2.query(f"date >= @start_date_2 and date <= @end_date_2")

    # 将两个DataFrame拼接起来
    edg_data = pd.concat([df_1_filtered, df_2_filtered])
    return edg_data


def proc_minute_chg_info(year: int, rt=False, time: str = '11:30:00'):
    """
    多进程拉取所有股票指定分钟的数据，每年一个进程。
    'pct_chg',为当天的涨跌幅，'limit'，为涨跌停信息（根据RT返回）
    :param year:
    :param rt:是否包含当前时点的涨跌停信息
    :param time: 默认 '11:30:00'
    :return:set_index(['date', 'code'] [['pct_chg', 'chg_pre', 'chg_close', 'limit']]
    """
    logger.info(f'Do {year} ~ {time},PID = {os.getpid()}')
    client_u = get_client_U(m='r')
    db_minute = client_u[DB_U_MINUTE]

    df = pd.DataFrame(db_minute['jq_minute_none_' + str(year)].find({'time': time},
                                                                    {'_id': 0, 'date': 1, 'code': 1, 'pre_day_close': 1,
                                                                     'today_close': 1, 'close': 1, 'chg_pre': 1, 'chg_close': 1},
                                                                    batch_size=2000000))
    df = trans_str_to_float64(df, trans_cols=['chg_pre', 'chg_close'])
    df['pct_chg'] = df.today_close / df.pre_day_close - 1
    df.set_index(['date', 'code'], inplace=True)

    if rt:
        t_limit = client_u.basic_jq.jq_daily_price_none
        use_cols = {"_id": 0, "date": 1, "code": 1, "high_limit": 1, "low_limit": 1}
        date_s, date_e = str(year) + '-01-01', str(year) + '-12-31'
        df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": date_s, "$lte": date_e}}, use_cols, batch_size=3000000))
        df_limit.set_index(['date', 'code'], inplace=True)

        df = pd.merge(df_limit, df, left_index=True, right_index=True)
        df['limit'] = df.apply(lambda x: (x["close"] == x["high_limit"]) or (x["close"] == x["low_limit"]), axis=1)
        df['limit'] = df['limit'].astype('int')
        return df[['pct_chg', 'chg_pre', 'chg_close', 'limit']].sort_index()

    return df[['pct_chg', 'chg_pre', 'chg_close']].sort_index()

def mp_minute_chg_info(y1=2009, y2=2024, time_='15:00:00'):
    """
    多进程拉取所有股票指定分钟的数据，每年一个进程。
    :param y1:
    :param y2:
    :param time_:'11:30:00'
    :return:set_index(['date', 'code'] [['pct_chg', 'chg_pre', 'chg_close', 'limit']]
    """
    res_list = []
    pool = mp.Pool(16)
    for year in range(y1, y2):
        res_list.append(pool.apply_async(LogExceptions(proc_minute_chg_info), args=(year, True, time_)))
    pool.close()
    pool.join()
    df = reduce(lambda x, y: pd.concat((x, y)), [res.get() for res in res_list])
    return df.sort_index()


# 装饰器工厂函数，接受csv文件名作为参数

def data_loader_decorator(csv_filename, suffix=None):
    """CSV数据加载装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            file_name, file_extension = os.path.splitext(csv_filename)
            decorator_filename = f"{file_name}_{suffix}{file_extension}" if suffix else csv_filename
            
            if not hasattr(self, 'csv_folder') or not self.csv_folder:
                return func(self, *args, **kwargs)
                
            csv_file_path = os.path.join(self.csv_folder, decorator_filename)
            if not os.path.exists(csv_file_path):
                result = func(self, *args, **kwargs)
                result.to_csv(csv_file_path)
                return result
                
            try:
                df_csv = pd.read_csv(csv_file_path, index_col=[0, 1])
                dates = df_csv.index.get_level_values('date')
                date_s = datetime.strptime(self.date_s, '%Y-%m-%d')
                date_e = datetime.strptime(self.date_e, '%Y-%m-%d')
                
                if (datetime.strptime(dates[0], '%Y-%m-%d') <= date_s and 
                    datetime.strptime(dates[-1], '%Y-%m-%d') >= date_e):
                    time_range = slice(
                        datetime.strftime(date_s, "%Y-%m-%d"),
                        datetime.strftime(date_e, "%Y-%m-%d")
                    )
                    date_index = df_csv.index.names.index('date')
                    slicer = [slice(None)] * len(df_csv.index.names)
                    slicer[date_index] = time_range
                    return df_csv.loc[tuple(slicer), :]
                    
            except Exception as e:
                logger.error(f'读取CSV文件时出错: {e}')
                
            result = func(self, *args, **kwargs)
            result.to_csv(csv_file_path)
            return result
            
        return wrapper
    return decorator


class LoadData:
    # def __init__(self, date_s: str, date_e: str, csv_folder: str = r"E:\\workplacejie\\A_workplace_data"):
    def __init__(self, date_s: str, date_e: str, csv_folder: str = None):
        if date_s is None or date_e is None:
            raise Exception(f'必须指定起止期！！！')
        self.client_U = get_client_U(m='r')
        # self.client_dev = get_client(c_from='dev')
        # self.client_dev = get_client(c_from='dev')

        self.date_s, self.date_e = date_s, date_e
        self.csv_folder = csv_folder  # 默认为None
        
        # start_date_1 = '2015-04-16'
        self.end_date_1 = '2022-07-21'
        self.start_date_2 = '2022-07-22'
        # end_date_2 = '2024-04-23'
        
        
    def _get_stra(self, t_,n=None) -> pd.DataFrame:
        """给定结果表，取所有选股结果"""

        df = pd.DataFrame(t_.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                  {"_id": 0}, batch_size=2000000).sort([('date', 1), ('F1', -1)]))
        if n:
            df.F1=df.F1.astype(float)
            top_results = df.groupby('date').apply(lambda x: x.nlargest(n, 'F1'))

            # top_results 现在包含每个日期下 F1 最大的200个记录
            # 如果你想要重置索引，可以使用以下代码：
            # top_results = top_results.reset_index()
            df = top_results
        df['month'] = df.date.str[:7]
        df.set_index(['month', 'date'], inplace=True)
        return df
    
    
    # def _get_stra(self, t_) -> pd.DataFrame:
    #     # ""“给定结果表，取每天 F1 倒序排列最大的200个结果”""
    #     # 使用聚合框架来分组、排序和限制每天的结果数量
    #     pipeline = [
    #         {
    #             "$match": {"date": {"$gte": self.date_s, "$lte": self.date_e}}
    #         },
    #         {
    #             "$sort": {"date": 1, "F1": -1}  # 首先按日期升序，然后按 F1 降序
    #         },
    #         {
    #             "$group": {
    #                 "_id": "$date",  # 按日期分组
    #                 "results": {
    #                     "$push": "$$ROOT",  # 将所有文档推送到数组
    #                 }
    #             }
    #         },
    #         {
    #             "$project": {
    #                 "date": "$_id",  # 将分组后的 _id 重命名为 date
    #                 "results": {
    #                     "$slice": ["$results", 0, 200]  # 限制每个日期的结果为200个
    #                 }
    #             }
    #         },
    #         {"$replaceRoot": {"newRoot": "$results"}}  # 将结果数组提升为文档的根
    #     ]

    #     # 执行聚合查询
    #     cursor = self.t_.aggregate(pipeline)
    #     # 将查询结果转换为 DataFrame
    #     df = pd.DataFrame(list(cursor))

    #     # 确保日期列是字符串类型，然后提取年月部分
    #     df['date'] = pd.to_datetime(df['date'])
    #     df['month'] = df['date'].dt.strftime('%Y-%m')

    #     # 设置多层索引
    #     df.set_index(['month', 'date'], inplace=True)

    #     return df

    def get_stra_res(self, c_from='dev', db_name='strategy', table_name='stocks_list',n=200) -> pd.DataFrame:
        """
        拉取策略结果
        :param c_from:
        :param db_name:
        :param table_name:
        :return:set_index(['month', 'date']) ['code','F1']
        """
        if c_from in ['neo', 'dev', 'stra','model','xnrj']:
            t_ = get_client(c_from=c_from)[db_name][table_name]
        else:
            raise Exception(f'目标服务器只能是[dev or neo]。。。')
        return self._get_stra(t_=t_,n=n)

    # def get_stocks_info(self, rt=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     股票状态[ST、停牌]由WIND反馈，涨跌停根据聚宽信息判断
    #     :param rt: 如果盘中动态出结果&动态换仓，那么涨跌停信息在涨跌幅数据中动态判断
    #     :return:set_index(['date', 'code'])
    #     """
    #     t_info = self.client_U.basic_wind.w_basic_info
    #     t_limit = self.client_U.basic_jq.jq_daily_price_none

    #     df_info = pd.DataFrame(t_info.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
    #                                        {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
    #                                        batch_size=1000000))
    #     if rt:  # 盘中动态出结果，动态换仓，那么涨跌停信息在涨跌幅数据中动态判断
    #         logger.warning(f'加载完成 ST & 停牌数据，注意这里没有涨跌停数据')
    #         return df_info.set_index(['date', 'code']).sort_index(), pd.DataFrame()
    #     use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
    #     df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": self.date_s, "$lte": self.date_e}}, use_cols, batch_size=1000000))
    #     df_limit['limit'] = df_limit.apply(lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"], axis=1)
    #     df_limit['limit'] = df_limit['limit'].astype('int')
    #     df_limit = df_limit[['date', 'code', 'limit']]

    #     return df_info.set_index(['date', 'code']).sort_index(), df_limit.set_index(['date', 'code']).sort_index()

    # @data_loader_decorator(csv_filename='df_info.csv', suffix='_realtime')
    # def get_stocks_info_realtime(self) -> tuple[pd.DataFrame, pd.DataFrame]:


    #     # 非实时情的股票信息查询处理逻辑
    #     t_info = self.client_U.basic_wind.w_basic_info
    #     df_info = pd.DataFrame(t_info.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
    #                                        {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
    #                                        batch_size=1000000))
    #     logger.warning(f'加载完成 ST & 停牌数据，注意这里没有涨跌停数据')
    #     # df_info = df_info.set_index(['date', 'code']).sort_index()
    #     # return df_info.set_index(['date', 'code']).sort_index(), pd.DataFrame()
    #     return df_info.set_index(['date', 'code']).sort_index()

    @data_loader_decorator(csv_filename='df_limit.csv')
    def get_stocks_limit_normal(self) -> pd.DataFrame:
        
        t_limit = self.client_U.basic_jq.jq_daily_price_none
        use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
        df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                             use_cols,
                                             batch_size=1000000))
        df_limit['limit'] = df_limit.apply(lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"], axis=1)
        df_limit['limit'] = df_limit['limit'].astype('int')
        df_limit = df_limit[['date', 'code', 'limit']]
        return df_limit.set_index(['date', 'code']).sort_index()
    
    
    @data_loader_decorator(csv_filename='df_info.csv')
    def get_stocks_info_normal(self) -> pd.DataFrame:
        # 实时情况的股票信息查询和处理逻辑
        t_info = self.client_U.basic_wind.w_basic_info
        # t_limit = self.client_U.basic_jq.jq_daily_price_none
        df_info = pd.DataFrame(t_info.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                            {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
                                            batch_size=1000000))

        return df_info.set_index(['date', 'code']).sort_index()

    def get_stocks_info(self, rt=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 根据rt参数调用对应的辅助函数
        if rt:
            return self.get_stocks_info_normal(), pd.DataFrame()
        else:
            return self.get_stocks_info_normal(),self.get_stocks_limit_normal()


    def get_stocks_info_for_paper_trade(self) -> pd.DataFrame:
        """
        因为聚宽的量价数据为当天盘后更新
        所以专为收盘后 PAPER_TRADE 准备的
        """
        t_st = self.client_U.basic_jq.jq_base_info
        t_limit = self.client_U.basic_jq.jq_daily_price_none

        df_st = pd.DataFrame(t_st.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                       {"_id": 0, 'date': 1, 'code': 1, 'is_st': 1, }, batch_size=1000000))

        use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1, 'paused': 1}
        df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                             use_cols, batch_size=1000000))
        df_limit['limit'] = df_limit.apply(lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"], axis=1)
        df_limit['limit'] = df_limit['limit'].astype('int')
        df_limit = df_limit[['date', 'code', 'paused', 'limit']]
        df = pd.merge(df_st.set_index(['date', 'code']), df_limit.set_index(['date', 'code']), left_index=True, right_index=True)
        df.dropna(inplace=True)
        return df

    
    @data_loader_decorator(csv_filename='df_chg.csv')
    def get_chg_wind(self) -> pd.DataFrame:
        """
        :return: 基于WIND的涨跌幅数据'pct_chg'，set_index(['date', 'code'])
        """
        logger.info(f'加载 基于WIND的日频涨跌幅数据...')
        df = pd.DataFrame(self.client_U.basic_wind.w_vol_price.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                                    {"_id": 0, 'date': 1, 'code': 1, 'pct_chg': 1},
                                                                    batch_size=1000000))
        return trans_str_to_float64(df, trans_cols=['pct_chg', ]).set_index(['date', 'code']).sort_index()

    def get_chg_minute(self, the_date: str = None, minutes=None, codes: list = None) -> pd.DataFrame:
        """
        基于聚宽分钟数据的涨跌幅，当前点前面的收益&当前点后面的收益，注意：无当天的整体收益情况
        :param the_date: 交易日
        :param minutes: 分钟列表，['15:00:00',]形式，如无则全部分钟数据
        :param codes: 指定代码列表 ['SH600001',]形式
        :return: DF,['date', 'time', 'code', 'chg_pre', 'chg_close']
        """
        t_minute = self.client_U['minute_jq']['jq_minute_none_' + str(the_date)[:4]]
        if minutes is None:
            df = pd.DataFrame(t_minute.find({'date': the_date, 'code': {'$in': codes}},
                                            {'_id': 0, 'date': 1, 'time': 1, 'code': 1, 'chg_pre': 1, 'chg_close': 1}))
        else:
            df = pd.DataFrame(t_minute.find({'date': the_date, 'time': {'$in': minutes}, 'code': {'$in': codes}},
                                            {'_id': 0, 'date': 1, 'time': 1, 'code': 1, 'chg_pre': 1, 'chg_close': 1}))
        if df.empty:
            raise Exception('数据为空，请检查！！！')
        return trans_str_to_float64(df, trans_cols=['chg_pre', 'chg_close'])

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

    def get_hold_num_per(self,percentage_1=0,percentage_2=0.03):
        
        df_info = self.get_stocks_info_normal()
        
        # Filter df_info to include rows where trade_status is 1 and riskwarning is 0   剔除st 停牌、涨跌停因为盘前不知道，所以不搞
        filtered_df = df_info[(df_info['trade_status'] == 1) & (df_info['riskwarning'] == 0)]
        
        # daily_code_count = df_info.groupby('date').size()
        daily_code_count = filtered_df.groupby('date').size()
        # 计算 hold_s 列
        hold_s = daily_code_count * (percentage_1)
        hold_s = np.ceil(hold_s).astype(int)
        # 计算 hold_num 列
        hold_num = hold_s + (daily_code_count * percentage_2)
        hold_num = np.ceil(hold_num).astype(int)
        # 创建 DataFrame
        df_mv = pd.DataFrame({'hold_s': hold_s, 'hold_num': hold_num})
        return df_mv


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
        df['num_pre'] = df['hold_num']
        df.ffill(inplace=True)
        # df.fillna(method='ffill', inplace=True)
        df['hold_num'] = df.apply(lambda dx: dx.hold_num if dx.hold_num <= dx.num_pre else dx.num_pre, axis=1).astype(int)
        df.loc[the_end_month:, 'hold_s'] = start_sorted
        if fixed_by_month:
            df.loc[the_end_month:, 'hold_num'] = hold_num
        return df[['hold_s', 'hold_num']]
    
    def get_hold_num_per_2(self,start_per=0.01,end_per = 0.03, the_end_month=None, fixed_by_month=True):
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

        df['hold_s'] = (df['count'] * (start_per)).apply(lambda x: math.floor(x))
        
        df['hold_e'] = (df['count'] * (start_per + end_per)).apply(lambda x: math.floor(x))
        df['hold_num'] = df.hold_e - df.hold_s
        df['num_pre'] = df.hold_num.shift(-1)
        df.ffill(inplace=True)
        # df.fillna(method='ffill', inplace=True)
        df['hold_num'] = df.apply(lambda dx: dx.hold_num if dx.hold_num <= dx.num_pre else dx.num_pre, axis=1).astype(int)
        # df.loc[the_end_month:, 'hold_s'] = start_sorted
        # if fixed_by_month:
        #     df.loc[the_end_month:, 'hold_num'] = hold_num
        return df[['hold_s', 'hold_num']]
    
    
    
    def get_moving_hold_num(self,start_per=0.01,end_per = 0.03, hold_num=50, start_sorted=100, the_end_month=None, fixed_by_month=True):
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

        start_sorted = (df['count'] * start_per).ceil()
        end_sorted = (df['count'] * end_per).ceil()
        
        # df['hold_s'] = (df['count'] * (start_sorted / the_end_count)).apply(lambda x: math.floor(x))
        # df['hold_e'] = (df['count'] * ((start_sorted + hold_num) / the_end_count)).apply(lambda x: math.floor(x))
        df['hold_num'] = start_sorted - end_sorted
        df['num_pre'] = df.hold_num.shift(-1)
        df.ffill(inplace=True)
        # df.fillna(method='ffill', inplace=True)
        df['hold_num'] = df.apply(lambda dx: dx.hold_num if dx.hold_num <= dx.num_pre else dx.num_pre, axis=1).astype(int)
        df.loc[the_end_month:, 'hold_s'] = start_sorted
        if fixed_by_month:
            df.loc[the_end_month:, 'hold_num'] = hold_num
        return df[['hold_s', 'hold_num']]
    
    @data_loader_decorator(csv_filename='edg.csv')
    def proc_edg_csv(self):
        edg_csv = merge_signal(self.date_s,self.end_date_1,self.start_date_2,self.date_e)
        return edg_csv
    
    @data_loader_decorator(csv_filename='ic_data.csv')
    def proc_ic_data_csv(self):
        ic_data_csv = merge_pct_data(self.date_s,self.end_date_1,self.start_date_2,self.date_e)
        return ic_data_csv


class LoadDataMargin(LoadData):
    """
    为多空策略准备数据
    """

    def __init__(self, date_s: str, date_e: str, csv_folder: str = None):
        if date_s is None or date_e is None:
            raise Exception(f'必须指定起止日期！！！')
        super().__init__(date_s, date_e,csv_folder)

 #代码重复了，判断csv是否存在，直接装饰器里玩就好了 ，函数还修改了 原有的500，使之直接成为top_num参数， 后期可以多恐怖不一致，再定义。
    def get_stra_FC(self,top_num=500):
        if self.csv_folder:
            buy_data = self.get_stra_FC_buy(top_num)
            sell_data = self.get_stra_FC_sell(top_num)

            # 可以选择在这里进行额外的处理，如果需要的话

            return buy_data, sell_data
        else:
            db = get_client(c_from='dev')['stra_V3_1']
            df = pd.DataFrame(db['stocks_all'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                    {"_id": 0, 'mv_vol': 0, 'F1': 0},
                                                    batch_size=1000000))
            # df = df[~df.code.str.startswith('SH688')].set_index(['date', 'code']).sort_index()  # 剔除科创板

            # 拿到多头组合
            ds_buy = df.groupby('date', as_index=False,
                                group_keys=False).apply(lambda dx: dx.sort_values('F_c',
                                                                                ascending=False)[['F_c', ]].iloc[:top_num]).reset_index()
            ds_buy['month'] = ds_buy.date.str[:7]

            # ds = df.groupby('date', as_index=False,
            #                 group_keys=False).apply(lambda dx: dx.sort_values('F_c')[['F_c', ]].iloc[:int(dx.shape[0] / 2)])
            # ds_sell = ds.groupby('date', as_index=False, group_keys=False).apply(lambda dx: dx.sample(min(800, dx.shape[0]))).reset_index()
            
            # 取800或者 更小的数目
            
            # db_sell = get_client(c_from='dev')['stra_V31_MARGIN']
            db_sell = get_client(c_from='dev')['stra_V31_MARGIN']
            # db_sell = get_client(c_from='neo')['stra_V3_1']
            df = pd.DataFrame(db_sell['stocks_all'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                        {"_id": 0, 'mv_vol': 0, 'F1': 0},
                                                        batch_size=1000000))
            # df = df[~df.code.str.startswith('SH688')].set_index(['date', 'code']).sort_index()  # 剔除科创板
            ds_sell = df.groupby('date', as_index=False,
                                group_keys=False).apply(lambda dx: dx.sort_values('F_c')[['F_c', ]].iloc[:top_num]).reset_index()
            ds_sell['month'] = ds_sell.date.str[:7]

            return (ds_buy.set_index(['month', 'date']),
                    ds_sell.set_index(['month', 'date']))
    
    def get_stra_FC_2(self,top_num=500):


        db = get_client(c_from='dev')['stra_V3_1']
        df = pd.DataFrame(db['stocks_all'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                    {"_id": 0, 'mv_vol': 0, 'F1': 0},
                                                    batch_size=1000000))
        df = df[~df.code.str.startswith('SH688')].set_index(['date', 'code']).sort_index()  # 剔除科创板

        # 拿到多头组合
        ds_buy = df.groupby('date', as_index=False,
                                group_keys=False).apply(lambda dx: dx.sort_values('F_c',
                                                                                ascending=False)[['F_c', ]].iloc[:top_num]).reset_index()
        ds_buy['month'] = ds_buy.date.str[:7]

            # ds = df.groupby('date', as_index=False,
            #                 group_keys=False).apply(lambda dx: dx.sort_values('F_c')[['F_c', ]].iloc[:int(dx.shape[0] / 2)])
            # ds_sell = ds.groupby('date', as_index=False, group_keys=False).apply(lambda dx: dx.sample(min(800, dx.shape[0]))).reset_index()
            
            # 取800或者 更小的数目
            
        db_sell = get_client(c_from='dev')['stra_V3_1']
            # db_sell = get_client(c_from='neo')['stra_V3_1']
        df = pd.DataFrame(db_sell['stocks_all'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                        {"_id": 0, 'mv_vol': 0, 'F1': 0},
                                                        batch_size=1000000))
        df = df[~df.code.str.startswith('SH688')].set_index(['date', 'code']).sort_index()  # 剔除科创板
        ds_sell = df.groupby('date', as_index=False,
                                group_keys=False).apply(lambda dx: dx.sort_values('F_c')[['F_c', ]].iloc[:top_num]).reset_index()
        ds_sell['month'] = ds_sell.date.str[:7]

        return (ds_buy.set_index(['month', 'date']),
                ds_sell.set_index(['month', 'date']))


    # 使用data_loader_decorator装饰器加载买入数据
    @data_loader_decorator(csv_filename='FC_buy.csv')
    def get_stra_FC_buy(self, top_num=500):
        # 执行买入数据的查询和处理逻辑

        # db = get_client(c_from='dev')['stra_V3_1']
        db = get_client(c_from='dev')['stra_V3_1']
        df = pd.DataFrame(db['stocks_all'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                {"_id": 0, 'mv_vol': 0, 'F1': 0},
                                                batch_size=1000000))
        df = df.set_index(['date', 'code']).sort_index()
        # df = df[~df.code.str.startswith('SH688')].set_index(['date', 'code']).sort_index()  # 剔除科创板

        # 拿到多头组合
        ds_buy = df.groupby('date', as_index=False,
                            group_keys=False).apply(lambda dx: dx.sort_values('F_c',
                                                                              ascending=False)[['F_c', ]].iloc[:top_num]).reset_index()
        ds_buy['month'] = ds_buy.date.str[:7]
        
        # ds = df.groupby('date', as_index=False,
        #                 group_keys=False).apply(lambda dx: dx.sort_values('F_c')[['F_c', ]].iloc[:int(dx.shape[0] / 2)])
        # ds_sell = ds.groupby('date', as_index=False, 
        #                      group_keys=False).apply(lambda dx: dx.sample(min(800, dx.shape[0]))).reset_index()

        return ds_buy.set_index(['month', 'date'])

    # 使用data_loader_decorator装饰器加载卖出数据
    @data_loader_decorator(csv_filename='FC_sell.csv')
    def get_stra_FC_sell(self, top_num=500):
        # 执行卖出数据的查询和处理逻辑
        # db_sell = get_client(c_from='dev')['stra_V31_MARGIN']
        db_sell = get_client(c_from='dev')['stra_V31_MARGIN']
        df = pd.DataFrame(db_sell['stocks_all'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                     {"_id": 0, 'mv_vol': 0, 'F1': 0},
                                                     batch_size=1000000))
        df = df.set_index(['date', 'code']).sort_index()
        # df = df[~df.code.str.startswith('SH688')].set_index(['date', 'code']).sort_index()  # 剔除科创板
        ds_sell = df.groupby('date', as_index=False,
                             group_keys=False).apply(lambda dx: dx.sort_values('F_c')[['F_c', ]].iloc[:top_num]).reset_index()
        ds_sell['month'] = ds_sell.date.str[:7]
        return ds_sell.set_index(['month', 'date'])

    # def get_stra_FC(self, top_num=500):
    #     # 直接调用装饰后的buy和sell方法，由装饰器决定数据加载或执行逻辑
    #     # 随机数没有被考虑
    #     buy_data, sell_data = self.get_stra_FC_buy(top_num), self.get_stra_FC_sell(top_num)
    #     return buy_data, sell_data

    def get_stra_buy_sell(self, c_from: str, db_name: str, ex_black: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        拉取策略结果，包括多头&空头
        :param ex_black:
        :param c_from:
        :param db_name:
        :return:
        """
        if c_from in ['neo', 'dev', 'stra']:
            db = get_client(c_from=c_from)[db_name]
        else:
            raise Exception(f'目标服务器只能是[dev or neo]。。。')

        df = pd.DataFrame(db['stocks_all'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                {"_id": 0, 'mv_vol': 0}, batch_size=1000000))
        df = df[~df.code.str.startswith('SH688')].set_index(['date', 'code']).sort_index()  # 剔除科创板

        blacks = pd.DataFrame(db['blacklist'].find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                                   {"_id": 0, 'date': 1, 'code': 1}, batch_size=1000000))
        # 全部剔除黑名单
        df1 = df.loc[~df.index.isin(blacks.set_index(['date', 'code']).sort_index().index.tolist())]
        # 拿到多头组合
        ds_buy = df1.groupby('date', as_index=False,
                             group_keys=False).apply(lambda dx: dx.sort_values('F1',
                                                                               ascending=False)[['F1', ]].iloc[:800]).reset_index()
        ds_buy['month'] = ds_buy.date.str[:7]

        if not ex_black:
            df1 = df.copy()

        ds_sell1 = df1.groupby('date', as_index=False,
                               group_keys=False).apply(lambda dx: dx.sort_values('F1')[['F1', ]].iloc[:100]).reset_index()
        ds_sell1['month'] = ds_sell1.date.str[:7]

        ds_sell2 = df1.groupby('date', as_index=False,
                               group_keys=False).apply(lambda dx: dx.sort_values('F_c')[['F_c', ]].iloc[:100]).reset_index()
        ds_sell2['month'] = ds_sell2.date.str[:7]

        return (ds_buy.set_index(['month', 'date']),
                ds_sell1.set_index(['month', 'date']),
                ds_sell2.set_index(['month', 'date']))

    # def get_stocks_info(self,short_db='huatai',short_collection='t_index', rt=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     df_info 中增加 两融标识  指数成分股标识（300、500、1000）
    #     股票状态[ST、停牌]由WIND反馈，涨跌停根据聚宽信息判断
    #     :param rt: 如果盘中动态出结果&动态换仓，那么涨跌停信息在涨跌幅数据中动态判断
    #     :return:set_index(['date', 'code'])
    #     """
    #     t_info = self.client_U.basic_wind.w_basic_info
    #     t_limit = self.client_U.basic_jq.jq_daily_price_none
    #     t_margin = self.client_U.basic_jq.jq_base_info
        
        
        
    #     # 为在三大指数内的股票进行打标签
    #     t_index = get_client(c_from='dev')[short_db][short_collection]
    #     index_df = pd.DataFrame(t_index.find({}, {'_id': 0})).set_index('date').sort_index()
    #     index_df['code'] = index_df.apply(lambda sx: eval(sx['000300.XSHG']) + eval(sx['000905.XSHG']) + eval(sx['399852.XSHE']), axis=1)
    #     # index_df['len'] = index_df.code.transform(lambda x: len(x))
    #     df_index = index_df[['code']].explode(column='code').reset_index().set_index(['date', 'code']).sort_index()



    #     df_info = pd.DataFrame(t_info.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
    #                                        {"_id": 0, 'date': 1, 'code': 1, 'riskwarning': 1, 'trade_status': 1},
    #                                        batch_size=1000000)).set_index(['date', 'code']).sort_index()
        
    #     df_margin = pd.DataFrame(t_margin.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
    #                                            {"_id": 0, 'date': 1, 'code': 1, 'margin': 1},
    #                                            batch_size=1000000)).set_index(['date', 'code']).sort_index()
        
        
    #     df_margin['in_index'] = 0
    #     df_margin.loc[df_margin.index.isin(df_index.index.tolist()), 'in_index'] = 1

    #     df_info = pd.merge(df_info, df_margin, left_index=True, right_index=True)

    #     if rt:  # 盘中动态出结果，动态换仓，那么涨跌停信息在涨跌幅数据中动态判断
    #         logger.warning(f'加载完成 ST & 停牌数据，注意这里没有涨跌停数据')
    #         return df_info.set_index(['date', 'code']).sort_index(), pd.DataFrame()

    #     use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
    #     df_limit = pd.DataFrame(t_limit.find({"date": {"$gte": self.date_s, "$lte": self.date_e}}, use_cols, batch_size=1000000))
    #     df_limit['limit'] = df_limit.apply(lambda x: x["close"] == x["high_limit"] or x["close"] == x["low_limit"], axis=1)
    #     df_limit['limit'] = df_limit['limit'].astype('int')
    #     df_limit = df_limit[['date', 'code', 'limit']].set_index(['date', 'code']).sort_index()

    #     return df_info, df_limit

    @data_loader_decorator('df_info.csv', suffix='margin')
    def get_stocks_info_normal(self)-> pd.DataFrame:
        # 非实时股票信息查询逻辑
        df_info = super().get_stocks_info_normal()
        df_info = self._add_margin_and_index_flags(df_info)

        return df_info
    
    def _add_margin_and_index_flags(self, df_info: pd.DataFrame) -> pd.DataFrame:
        # 添加两融标识和指数成分股标识
        t_margin = self.client_U.basic_jq.jq_base_info
        
        t_index = get_client(c_from='dev')['huatai']['t_index']
        index_df = pd.DataFrame(t_index.find({}, {'_id': 0})).set_index('date').sort_index()
        index_df['code'] = index_df.apply(lambda sx: eval(sx['000300.XSHG']) + eval(sx['000905.XSHG']) + eval(sx['399852.XSHE']), axis=1)
        df_index = index_df[['code']].explode(column='code').reset_index().set_index(['date', 'code']).sort_index()

        df_margin = pd.DataFrame(t_margin.find({"date": {"$gte": self.date_s, "$lte": self.date_e}},
                                               {"_id": 0, 'date': 1, 'code': 1, 'margin': 1},
                                               batch_size=1000000)).set_index(['date', 'code']).sort_index()
        df_margin['in_index'] = 0
        df_margin.loc[df_margin.index.isin(df_index.index.tolist()), 'in_index'] = 1

        df_info = pd.merge(df_info, df_margin, left_index=True, right_index=True)
        
        
        
        dates = df_info.index.get_level_values(0).unique().tolist()
        codes = df_info.index.get_level_values(1).unique().tolist()
        jq_info = self.client_U.basic_jq.jq_daily_valuation
        df_circulating_market_cap = pd.DataFrame(jq_info.find({"date": {"$in": dates},'code':{"$in": codes}},
                                               {"_id": 0, 'date': 1, 'code': 1, 'circulating_market_cap': 1},
                                               batch_size=1000000)).set_index(['date', 'code']).sort_index()
        df_info = pd.merge(df_info, df_circulating_market_cap, left_index=True, right_index=True)
        
        
        return df_info

    # def get_stocks_info(self, rt=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     # 根据rt参数调用对应的辅助函数
    #     if rt:
    #         logger.warning('加载完成 ST & 停牌数据，注意这里没有涨跌停数据')
    #         return self.get_stocks_info_normal(),pd.DataFrame()
    #     else:
    #         return self.get_stocks_info_normal(),self.get_stocks_limit_normal()


if __name__ == '__main__':
    
    # 载入数据实例化测试
    xx = LoadData(date_s='2023-01-01', date_e='2024-01-31')
    xx.get_hold_num_per_2(0,0.001)
    # xx = LoadDataMargin(date_s='2023-01-01', date_e='2024-01-31')
    # xx.get_stocks_info()
    # df = xx.get_stra_res(c_from='neo', db_name='strategy', table_name='stocks_list')
