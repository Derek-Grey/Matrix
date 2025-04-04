"""
回测模块
包含回测类和绘图类
"""
import os
import time
import re  # 新增正则表达式模块导入
import pandas as pd
import numpy as np
from loguru import logger
import plotly.graph_objects as go
from pathlib import Path
from functools import wraps
import sys

OUTPUT_DIR = Path(__file__).parent / 'output'  # 使用当前文件所在目录下的output文件夹

# NPQ数据结构定义
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

def read_npq_file(file_path):
    """读取NPQ文件并返回DataFrame"""
    npq_data = np.fromfile(file_path, dtype=D1_11_numpy_dtype)
    quote = npq_data['quote']
    df = pd.DataFrame(quote)  # 直接使用quote字段构建DataFrame
    
    # 只保留date, code, pct_chg字段
    return df[['date', 'code', 'pct_chg']]

def read_all_npq_files(data_root, start_date=None, end_date=None):
    """遍历时间段目录读取NPQ文件"""
    load_start_time = time.time()  # 数据加载开始时间记录
    data_path = Path(data_root)
    all_dfs = []
    
    # 遍历所有日期子目录
    for date_dir in data_path.glob('*'):
        if date_dir.is_dir():
            npq_file = date_dir / "1" / "11.npq"
            try:
                df = read_npq_file(str(npq_file))
                df['date'] = date_dir.name  # 保留日期作为索引
                
                # 日期过滤
                if start_date and end_date:
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"跳过{date_dir}，加载失败: {str(e)}")
                continue
                
    load_end_time = time.time()  # 数据加载结束时间记录
    logger.info(f"数据加载完成，耗时: {load_end_time - load_start_time:.4f}s")  # 输出数据加载耗时

    return pd.concat(all_dfs).sort_values('date')

def main(data_root, start_date=None, end_date=None):
    """主函数，读取NPQ文件并打印结果"""
    logger.info("开始读取NPQ文件...")
    try:
        df = read_all_npq_files(data_root, start_date, end_date)
        
        # 记录保存数据开始时间
        save_start_time = time.time()
        # 保存数据
        output_path = OUTPUT_DIR / "trade_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        save_end_time = time.time()  # 记录保存数据结束时间

        # 输出保存数据耗时
        logger.info(f"数据保存完成，耗时: {save_end_time - save_start_time:.4f}s")

        print(df.head())  # 打印前几行数据
    except Exception as e:
        logger.error(f"读取NPQ文件失败: {e}")

if __name__ == "__main__":
    data_directory = r"D:\Data"  # 设置数据目录路径
    main(data_directory, start_date="2015-01-05", end_date="2024-12-31")
