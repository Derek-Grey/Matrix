# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: db_client.py
@time: 2023-09-10 13:35
说明: MongoDB数据库连接客户端
"""
import pymongo
from urllib.parse import quote_plus
from loguru import logger

def get_client(c_from='local'):
    """
    获取MongoDB客户端连接
    :param c_from: 连接目标('local'/'dev')
    :return: MongoDB客户端实例
    """
    client_dict = {
        'local': 'localhost:27017',  # 本地 MongoDB 服务器
        'dev': '192.168.1.78:27017', # 开发环境 MongoDB 服务器
    }

    client_name = client_dict.get(c_from)
    if client_name is None:
        raise Exception(f'传入的数据库目标服务器有误 {c_from}，请检查 {client_dict}')
    
    return pymongo.MongoClient(f"mongodb://{client_name}")

def get_client_U(m='r'):
    """
    获取带用户认证的MongoDB客户端连接
    :param m: 权限类型('r'/'rw'/'Neo')
    :return: MongoDB客户端实例
    """
    # 用户权限配置
    auth_config = {
        'r': ('Tom', 'tom'),      # 只读权限
        'rw': ('Amy', 'amy'),     # 读写权限
        'Neo': ('Neo', 'neox'),   # 管理员权限
    }
    
    user, pwd = auth_config.get(m, ('Tom', 'tom'))  # 默认只读权限
    if m not in auth_config:
        logger.warning(f'传入的参数 {m} 有误，使用默认只读权限')
        
    return pymongo.MongoClient(
        "mongodb://%s:%s@%s" % (
            quote_plus(user),
            quote_plus(pwd),
            '192.168.1.99:29900/'
        )
    )
