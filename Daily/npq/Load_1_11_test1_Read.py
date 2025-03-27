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
    target_file = "D:\\data\\2025-02-25\\1\\11.npq"
    npq_data = np.fromfile(target_file, dtype=D1_11_numpy_dtype)

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
    output_file = 'readdata11_1.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 打印DataFrame以验证数据（可选）
    print(df)