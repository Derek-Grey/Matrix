# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: db.py
@time: 2023-07-27 16:42
说明:
"""
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
import pymongo
from pymongo import InsertOne
import pandas as pd
import numpy as np
from loguru import logger
from db_client import get_client_U
from pymongo.collection import Collection

def _thread_insert2db(table: Collection, df: pd.DataFrame) -> None:
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
        cpus = int(psutil.cpu_count(logical=False) * 0.9)
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


def series2db(table: pymongo.collection, series: pd.Series) -> None:
    """
    pymongo 写入 Series
    :param series:
    :param table:
    :return:
    """
    if not series.empty:
        try:
            table.insert_one(series.to_dict())
        except Exception as e:
            logger.warning(f'{str(e)[:300]}\n写入出错\nSeries：{series}')
    else:
        logger.error(f'Series为空，清检查！！')
        
def get_trading_days(date_s, date_e,list2=None):
    """
    获取给定日期范围内的所有交易日。
    
    :param date_s: 开始日期，格式为 'YYYY-MM-DD'
    :param date_e: 结束日期，格式为 'YYYY-MM-DD'
    :return: 交易日的列表
    """
    # date_s = '2013-03-01'
    # date_e = '2024-05-01'
    trade_date = get_client_U()['economic']['trade_dates']
    trading_days = list(trade_date.find(
    {"trade_date": {"$gte": date_s, "$lte": date_e}},
    {"_id": 0, "trade_date": 1}
    ))
    date_s, date_e = trading_days[0]['trade_date'],trading_days[-1]['trade_date']
    if list2:
        return trading_days
    else:
        return date_s, date_e
    
    
if __name__ == '__main__':
    date_s = '2013-03-01'
    date_e = '2024-05-01'
    date_s, date_e = get_trading_days(date_s, date_e)
    print(date_s)