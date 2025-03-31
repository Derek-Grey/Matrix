import pymongo
from datetime import datetime, timedelta
from smtplib import SMTP_SSL, SMTPException
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from loguru import logger
import os
import pandas as pd
from bson import ObjectId  # 导入ObjectId以处理_id字段（如果需要）
import copy
import numpy as np
# 在文件顶部导入模块部分添加（如果尚未导入）
import time

D1_11_dtype = np.dtype([
    ('date', 'S64'),
    ('code', 'S64'),
    ('code_w', 'S64'),
    ('pct_chg', 'f8'),
    ('volume', 'f8'),
], align=True)

D1_11_numpy_dtype = np.dtype([
    ('serial', 'i4'),
    ('mi_type', 'i4'), 
    ('local_time', 'i8'),
    ('exchange_time', 'i8'),
    ('quote', D1_11_dtype),
], align=True)

if __name__ == '__main__':
    start_total = time.time()  # 总耗时统计开始
    
    target_file = "D:\\Data\\All_11.npq"
    
    # 记录数据加载开始时间
    start_load = time.time()
    # 读取NPQ文件
    npq_data = np.fromfile(target_file, dtype=D1_11_numpy_dtype)

    # 构建DataFrame的列名
    columns = [field for field in D1_11_numpy_dtype.fields if field != 'quote']
    columns.extend(D1_11_dtype.fields)

    # 初始化一个空的列表来存储处理后的行数据
    rows = []

    # 遍历npq_data
    for item in npq_data:
        row_data = {}
        for field in D1_11_numpy_dtype.fields:
            if field != 'quote':
                row_data[field] = item[field]
        for quote_field in D1_11_dtype.fields:
            value = item['quote'][quote_field]
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    value = None
            row_data[quote_field] = value
        rows.append(row_data)

    # 使用处理后的行数据创建DataFrame
    df = pd.DataFrame(rows, columns=columns)
    load_time = time.time() - start_load  # 记录数据加载结束时间

    # 保存DataFrame为CSV文件
    start_save = time.time()
    output_file = 'readdata11_1.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    save_time = time.time() - start_save

    # 输出统计结果
    total_time = time.time() - start_total
    print(f"\n[性能统计] 数据加载:{load_time:.2f}s 保存:{save_time:.2f}s 总耗时:{total_time:.2f}s")