# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: my_logger.py
@time: 2023/3/6 12:06
说明:
"""
import datetime
from loguru import logger
from Utils.utils import get_log_path


class MyLogger(object):
    def __init__(self, check_t=0):
        today = datetime.date.today()
        self.this_week_start = today - datetime.timedelta(days=today.weekday())
        self.check_t = check_t

    def _my_logger(self, txt=None, name=None, level=None):
        file_name = str(get_log_path()) + name + self.this_week_start.isoformat() + ".log"
        x = logger.add(
            file_name, encoding="utf-8", level=level,
            enqueue=True,  # 要记录的消息在到达接收器之前是否应该首先通过多进程安全队列
            rotation="1 week",  # 分隔日志文件
            retention="3 months",  # 可配置旧日志的最长保留时间，例如，"1 week, 3 days"、"2 months"
            format='{time:YYYY-MM-DD HH:mm:ss} - '
                   '{level}-{message}',
        )
        if level == 'INFO':
            logger.info(txt)
        elif level == 'ERROR':
            logger.error(txt)
        elif level == 'WARN':
            logger.warning(txt)
        logger.remove(x)
        return logger

    def bt_logger(self, txt=None, file_name="模拟交易-持仓50-", level='INFO'):
        if txt:
            self._my_logger(txt=txt, name=file_name, level=level)

    def check_logger(self, txt=None, level='INFO'):
        name_list = ['数据自检-基础信息-', '数据自检-基本因子-', '数据自检-策略-', '数据自检-回测-']
        if txt:
            self._my_logger(txt=txt, name=name_list[self.check_t], level=level)

    def db_insert_error_logger(self, txt=None):
        if txt:
            self._my_logger(txt=txt, name="数据库插入错入-", level='ERROR')

    def info_logger(self, txt=None):
        if txt:
            self._my_logger(txt=txt, name="info-", level='INFO')

    def error_logger(self, txt=None):
        if txt:
            self._my_logger(txt=txt, name="error-", level='ERROR')

    def info(self, txt: str):
        """
        用于替换 info_logger ,其他类同
        :param txt:
        :return: 保持到 info-日期格式文件
        """
        self._my_logger(txt=txt, name="info-", level='INFO')

    def error(self, txt: str):
        self._my_logger(txt=txt, name="error-", level='ERROR')

    def warning(self, txt: str):
        self._my_logger(txt=txt, name="warning-", level='WARN')
