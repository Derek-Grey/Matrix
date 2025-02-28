# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: db_client.py
@time: 2023-09-10 13:35
说明: 对再用数据库服务器精选剥离，并对基础库服务器做权限控制
"""
import pymongo
from urllib.parse import quote_plus
from loguru import logger

def get_client(c_from='local'):
    client_dict = {
        'local': 'localhost:27017',  # 本地 MongoDB 服务器
        'dev': '192.168.1.78:27017', # 开发环境 MongoDB 服务器
    }

    client_name = client_dict.get(c_from, None)
    if client_name is None:
        raise Exception(f'传入的数据库目标服务器有误 {c_from}，请检查 {client_dict}')
    
    # 连接到指定的 MongoDB 服务器
    client = pymongo.MongoClient(f"mongodb://{client_name}")
    return client


def get_client_U(m='r'):
    
    # 这是原始数据库 刀片机那台机器
    
    if m == 'r': # 只读
        user, pwd = 'Tom', 'tom'
    elif m == 'rw': # 读写
        user, pwd = 'Amy', 'amy'
    elif m == 'Neo': # 管理员
        user, pwd = 'Neo', 'neox'
    else:
        logger.warning(f'你传入的参数 {m} 有毛病，那就返回默认的只读权限吧！')
        user, pwd = 'Tom', 'tom'
    return pymongo.MongoClient("mongodb://%s:%s@%s" % (quote_plus(user),
                                                       quote_plus(pwd),
                                                       '192.168.1.99:29900/'))
