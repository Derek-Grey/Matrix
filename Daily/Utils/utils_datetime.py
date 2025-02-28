# -*- coding: utf-8 -*-
"""
@author: Neo
@software: PyCharm
@file: utils_datetime.py
@time: 2023-07-27 10:22
说明:
"""

import time
import datetime
from dateutil.relativedelta import relativedelta
from Utils.my_errors import ParamsError


def get_week_month_date(the_date=None, week=False, month=False):
    """获取本周（本月）的第一天和最后一天"""
    if the_date is None:
        today = datetime.datetime.today().date()
    else:
        today = datetime.datetime.fromisoformat(the_date)
    if week:
        this_week_start = today - datetime.timedelta(days=today.weekday())
        this_week_end = today + datetime.timedelta(days=6 - today.weekday())
        return str(this_week_start.date()), str(this_week_end.date())
    elif month:
        t_m = today.month + 1 if today.month < 12 else 1

        this_month_start = datetime.datetime(today.year, today.month, 1)
        this_month_end = datetime.datetime(today.year, t_m, 1) - datetime.timedelta(
            days=1) + datetime.timedelta(
            hours=23, minutes=59, seconds=59)
        return str(this_month_start.date()), str(this_month_end.date())
    return str(today), str(today)


def stamp2time(timeStamp, fmt='T'):
    """
    功能：将时间戳转换成日期函数 例如：1606708276268 ==》2020-11-30 11:51:16
    参数：timeStamp 时间戳，类型 double 例如：1606708276268
    返回值：日期， 类型：字符串 2020-11-30 11:51:16
    """
    time_local = time.localtime(timeStamp / 1000)
    if fmt == 'T':
        dt = time.strftime("%Y-%m-%d", time_local)
    elif fmt == 'D':
        dt = time.strftime("%H:%M:%S", time_local)
    else:
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt


def get_season_key_day(today):
    """
    :param today: 为 datetime.datetime.date
    :return:
    """
    this_month_first_day = datetime.date(today.year, today.month - (today.month - 1) % 3 + 2, 1)  # 本月第一天
    this_season_end_day = this_month_first_day + relativedelta(months=1, days=-1)  # 本季度 最后一天
    last_seaon_end_day = this_season_end_day - relativedelta(months=3, )  # 上季度 最后一天
    last_seaon_end_day_2nd = last_seaon_end_day - relativedelta(months=3, )  # 上上季度 最后一天
    last_seaon_end_day_3nd = last_seaon_end_day_2nd - relativedelta(months=3, )  # 上上上季度 最后一天
    res_list = list(map(deal_with_31, [last_seaon_end_day, last_seaon_end_day_2nd, last_seaon_end_day_3nd]))

    year_end_day = datetime.date(today.year, 12, 31)  # 本年最后一天
    last_year_end_day = year_end_day + relativedelta(years=-1)  # 去年 最后一天
    res_list.append(last_year_end_day.isoformat())
    last_year_end_day_2nd = year_end_day + relativedelta(years=-2)  # 去去年 最后一天
    res_list.append(last_year_end_day_2nd.isoformat())
    return res_list


def deal_with_31(date):
    if date.month == 3 or date.month == 12:
        date = datetime.date(date.year, date.month, 31)
    return date.isoformat()


def to_date(date):
    """
    >>> convert_date('2015-1-1')
    datetime.date(2015, 1, 1)

    >>> convert_date('2015-01-01 00:00:00')
    datetime.date(2015, 1, 1)

    >>> convert_date(datetime.datetime(2015, 1, 1))
    datetime.date(2015, 1, 1)

    >>> convert_date(datetime.date(2015, 1, 1))
    datetime.date(2015, 1, 1)
    """
    if date:
        if ':' in date:
            date = date[:10]
        return datetime.datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, datetime.date):
        return date
    elif date is None:
        return None
    raise ParamsError("type error")
