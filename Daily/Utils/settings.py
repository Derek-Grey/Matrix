# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: settings.py
@time: 2023-07-27 10:30
说明:
"""
from pathlib import Path

# 基础路径配置
BASE_DIR = str(Path(__file__).resolve().parent.parent)
BASE_DIR_P = str(Path(__file__).resolve().parent.parent.parent)

# DEBUG = False
# DEV = True
# NEO_DEV = True
# SERVER = False

# MongoDB连接配置
CLIENT_DEV = 'mongodb://localhost:27017/'
CLIENT_NEO = 'mongodb://192.168.1.77:27017/'
CLIENT_SERVER = 'mongodb://192.168.1.76:27017/'

# 数据库名称配置
DB_U_WIND = 'basic_wind'
DB_U_JQ = 'basic_jq'
DB_U_ECO = 'economic'
DB_U_MINUTE = 'minute_jq'

# 以下数据库在 NEO_DEV 上
DB_NEO_PAPER_TRADE = 'bt_paper_trade'  # NEO上 模拟盘的持仓数据 & 模拟盘涨跌幅等数据
DB_NEO_BASIC = 'stocks_basic_info'
DB_NEO_FACTOR = "base_factors"
DB_NEO_STRATEGY = 'strategy'
DB_NEO_BACKTEST = 'backtest'
DB_NEO_STAT = 'statistic'
DB_NEO_ECO = 'economic'
DB_NEO_TRADE_DETAIL = 'trade_detail'
DB_NEO_BASIC_WIND = 'basic_wind'
DB_NEO_BASIC_JQ = 'basic_jq'
DB_NEO_STRATEGY_DEBUG = 'debug_strategy'

# 以下数据库在本地DEV上
DB_DEV_BASIC_WIND = 'basic_wind'
DB_DEV_BASIC = 'basic_info_test'

DB_DEV_FACTOR = "factors_untouched"  # 原始数据生成的因子库，在DEV本地上,用来验证本地环境跑出来的基础因子是否一致
DB_DEV_FACTOR_TEST = 'factor_test'

DB_DEV_STRATEGY = "strategy_untouched"  # 原始数据生成的因子库 & 策略库，在DEV本地上,用来验证本地环境跑出来的策略是否一致
DB_DEV_STRATEGY_TEST = 'strategy_test'

DB_DEV_BACKTEST = 'backtest_pos'  # 产品选股结果，回测数据库，在DEV本地上
DB_DEV_BACKTEST_TEST = 'backtest_pos_test'  # 产品选股结果，回测数据库，在DEV本地上，测试用
DB_DEV_BACKTEST_CHG = 'backtest_chg'  # 产品选股结果对应的涨跌幅数据，回测数据库，在DEV本地上，
DB_DEV_BACKTEST_CHG_TEST = 'backtest_chg_test'  # 产品选股结果对应的涨跌幅数据，回测数据库，在DEV本地上，测试用

# 不参与RANK的行业列表
USELESS_INDUS = [
    "证券、期货业", "银行业", "货币金融服务", "其他金融业", 
    "资本市场服务", "保险业", "燃气生产和供应业",
    "电力、蒸汽、热水的生产和供应业", "煤气生产和供应业",
    "电力、热力生产和供应业", "水的生产和供应业",
    "自来水的生产和供应业", "卫生", "公共设施服务业",
    "房屋建筑业", "房地产业", "房地产中介服务业",
    "房地产管理业", "房地产开发与经营业", "金融信托业",
]

# 策略回测配置
DICT_STRATEGY_BT = {
    "alpha_first": ("bt_alpha_first", 0, 50),
    'yk_first': ('bt_yk_first', 100, 50),
}

END_MONTH = '2022-07'
