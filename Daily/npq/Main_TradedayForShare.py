# =============================================================================
# Author: Keven Wang
# Date: 2025-02-07
# Illustration: 
# 1. 检查万得数据库是否正常摘录数据
# 2. 检查聚宽数据库是否正常摘录数据
# 3. 检查选股列表是否正常
# 4. 如果以上三个条件都满足，则发送邮件通知
# 5. 如果以上三个条件有一个不满足，则发送邮件通知报错
# 6. 在本地log文件中记录日志
# =============================================================================
import pymongo
from datetime import datetime, timedelta
import pymongo
from datetime import datetime, timedelta
from loguru import logger
import os
import pandas as pd
import pandas as pd


# 连接mongodb数据库
def get_client(c_from='local'):
    client_dict = {
        'local': '127.0.0.1:27017',
        'bob': '192.168.1.87:27017',
        'db_u': 'Tom:tom@192.168.1.99:29900',
    }
    client_uri = f"mongodb://{client_dict.get(c_from, '')}"
    if not client_uri:
        raise ValueError(f'传入的数据库目标服务器有误 {c_from}，请检查 {list(client_dict.keys())}')
    try:
        return pymongo.MongoClient(client_uri)
    except pymongo.errors.PyMongoError as e:
        print(f"无法连接到MongoDB服务器: {e}")
        raise

if __name__ == '__main__':
    # 读取Excel文件
    df = pd.read_csv('D:\\KevenCode\\获取交易日\\trade_dates_all.csv')    # 确保trade_date列是日期格式
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    FirstDate = "2000-01-04"
    LastDate = "2024-12-31"

    #在df中查找符合条件的记录，即trade_date大于等于FirstDate且小于等于LastDate
    dfnew = df[(df['trade_date'] >= FirstDate) & (df['trade_date'] <= LastDate)]

    #打印
    print(dfnew)