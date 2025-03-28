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
    
    # 记录文件读取时间
    start_read = time.time()
    npq_data = np.fromfile(target_file, dtype=D1_11_numpy_dtype)
    read_time = time.time() - start_read

    # 构建DataFrame的列名
    # 首先添加非'quote'字段
    columns = [field for field in D1_11_numpy_dtype.fields if field != 'quote']
    # 然后添加'quote'内部的字段
    columns.extend(D1_11_dtype.fields)

    # 初始化一个空的列表来存储处理后的行数据
    rows = []

    # 遍历npq_data
    for item in npq_data:
        # 初始化一个空字典来存储当前行的数据
        row_data = {}

        # 添加非'quote'字段的数据
        for field in D1_11_numpy_dtype.fields:
            if field != 'quote':
                row_data[field] = item[field]

        # 添加'quote'内部字段的数据
        for quote_field in D1_11_dtype.fields:
            value = item['quote'][quote_field]
            # 对于字符串类型的字段，尝试解码字节串
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    value = None
            # 对于浮点数类型，直接赋值
            row_data[quote_field] = value

        # 将处理后的行数据添加到列表中
        rows.append(row_data)

    # 使用处理后的行数据创建DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # 保存DataFrame为CSV文件
    # 记录数据处理时间
    start_process = time.time()
    # 原有的数据处理循环代码
    process_time = time.time() - start_process

    # 记录保存时间
    start_save = time.time()
    output_file = 'readdata11_1.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    save_time = time.time() - start_save

    # 输出统计结果
    total_time = time.time() - start_total
    print(f"\n[性能统计] 读取:{read_time:.2f}s 处理:{process_time:.2f}s 保存:{save_time:.2f}s 总耗时:{total_time:.2f}s")