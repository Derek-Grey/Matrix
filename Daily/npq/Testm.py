import os
import sys
import time
import math
import pandas as pd
import numpy as np
from loguru import logger

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
            
            # 记录读取数据开始时间
            read_start_time = time.time()
            df = pd.DataFrame(self.client_U.basic_wind.w_vol_price.find(
                {"date": {"$gte": self.date_s, "$lte": self.date_e}},
                {"_id": 0, "date": 1, "code": 1, "pct_chg": 1},
                batch_size=1000000))
            # 记录读取数据结束时间
            read_end_time = time.time()

            # 记录转换数据开始时间
            trans_start_time = time.time()
            df = trans_str_to_float64(df, trans_cols=['pct_chg'])  # 转换数据类型
            df['date'] = pd.to_datetime(df['date'])  # 转期格式
            pivot_df = df.pivot_table(index='date', columns='code', values='pct_chg')  # 创建透视表
            # 记录转换数据结束时间
            trans_end_time = time.time()

            # 记录保存数据开始时间
            save_start_time = time.time()
            # 保存原始数据
            self.save_raw_data_to_csv(df, 'raw_wind_data.csv')
            # 记录保存数据结束时间
            save_end_time = time.time()

            # 输出详细时间统计
            read_time = read_end_time - read_start_time
            trans_time = trans_end_time - trans_start_time
            save_time = save_end_time - save_start_time
            logger.info(f"读取数据时间: {read_time:.2f} 秒")
            logger.info(f"转换数据时间: {trans_time:.2f} 秒")
            logger.info(f"保存数据时间: {save_time:.2f} 秒")

            return pivot_df  # 返回透视表
        except Exception as e:
            logger.error(f"加载WIND数据失败: {e}")
            raise

# 定义转换函数
def trans_str_to_float64(df: pd.DataFrame, exp_cols: list = None, trans_cols: list = None) -> pd.DataFrame:
    """
    将DataFrame中的字符串列转换为float64类型
    :param df: 输入DataFrame
    :param exp_cols: 排除的列
    :param trans_cols: 需要转换的列
    :return: 转换后的DataFrame
    """
    if trans_cols is None and exp_cols is None:
        trans_cols = df.columns
    if exp_cols is not None:
        trans_cols = list(set(df.columns) - set(exp_cols))
    df[trans_cols] = df[trans_cols].astype('float64')
    return df

# 定义获取客户端连接的函数
def get_client_U():
    # 这里应该包含连接到本地数据库的逻辑
    # 例如，使用pymongo连接到MongoDB
    from pymongo import MongoClient
    client = MongoClient('localhost', 27017)
    return client

def main():
    # 记录开始时间
    start_time = time.time()

    # 定义起始和结束日期
    date_start = "2015-01-05"
    date_end = "2024-12-31"
    data_folder = r"D:\Derek\Code\Matrix\csv"  # 使用原始字符串

    # 创建LoadData类的实例
    data_loader = LoadData(date_start, date_end, data_folder)

    # 获取WIND的日频涨跌幅数据
    wind_data = data_loader.get_chg_wind()

    # 打印获取的数据
    print(wind_data)

    # 记录结束时间
    end_time = time.time()

    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time:.2f} 秒")

if __name__ == "__main__":
    main()