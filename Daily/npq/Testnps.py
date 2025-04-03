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

def read_all_npq_files(csv_path, start_date=None, end_date=None):
    """通过CSV路径列表读取NPQ文件"""
    start_time = time.time()  # 记录开始时间
    all_dfs = []
    missing_dates = []  # 用于收集找不到数据的日期和路径
    
    try:
        # 记录数据加载开始时间
        load_start_time = time.time()
        # 读取路径CSV文件
        path_df = pd.read_csv(csv_path)
        logger.info(f"成功加载路径文件，共 {len(path_df)} 条记录")
        
        # 转换日期格式
        path_df['trade_date'] = pd.to_datetime(path_df['trade_date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

        # 日期范围过滤
        if start_date and end_date:
            path_df = path_df[(path_df['trade_date'] >= start_date) & (path_df['trade_date'] <= end_date)]

        # 处理过滤后的数据
        for _, row in path_df.iterrows():
            file_path = Path(row['path'].replace('\\', '\\\\'))  
            trade_date = str(row['trade_date'])
            
            try:
                if file_path.exists():
                    df = read_npq_file(str(file_path))
                    df['trade_date'] = trade_date  
                    all_dfs.append(df)
                else:
                    print(f"文件不存在: {file_path}")  # 使用print代替logger
                    missing_dates.append((trade_date, file_path))  # 收集找不到数据的日期和路径
            except Exception as e:
                print(f"处理文件失败 {file_path}: {str(e)}")  # 使用print代替logger
                missing_dates.append((trade_date, file_path))  # 收集找不到数据的日期和路径
                continue
        load_end_time = time.time()  # 记录数据加载结束时间
                
        # 检查是否有数据帧可供合并
        if not all_dfs:
            print("没有找到任何有效数据帧进行合并")
            return pd.DataFrame()

        # 输出找不到数据的日期和路径
        if missing_dates:
            print("找不到数据的日期和路径:")
            for date, path in missing_dates:
                print(f"日期: {date}, 路径: {path}")

        # 修正日期处理逻辑
        final_df = pd.concat(all_dfs)
        try:
            # 转换为日期格式时指定明确格式
            final_df['trade_date'] = pd.to_datetime(
                final_df['trade_date'], 
                format='%Y/%m/%d'  # 根据图片调整日期格式
            )
        except ValueError:
            # 尝试不带格式转换（处理可能存在的字符串日期）
            final_df['trade_date'] = pd.to_datetime(final_df['trade_date'])
            
        final_df = final_df.sort_values('trade_date').reset_index(drop=True)  # 重置索引确保顺序
        
        # 记录保存数据开始时间
        save_start_time = time.time()
        # 保存数据
        output_path = OUTPUT_DIR / "trade_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        save_end_time = time.time()  # 记录保存数据结束时间

        # 输出详细时间统计
        load_time = load_end_time - load_start_time
        save_time = save_end_time - save_start_time
        print(f"数据加载时间: {load_time:.2f} 秒")
        print(f"保存数据时间: {save_time:.2f} 秒")

        return final_df.drop('trade_date', axis=1)
        
    except Exception as e:
        # 移除日志记录
        return pd.DataFrame()
    finally:
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算时间差
        print(f"处理文件所花费的时间: {elapsed_time:.2f} 秒")

def main():
    """主函数，读取NPQ文件并打印结果"""
    logger.info("开始读取NPQ文件...")
    csv_path = r"D:\Derek\Code\Matrix\Daily\csv\trade_all.csv"
    try:
        df = read_all_npq_files(csv_path, start_date="2015-01-05", end_date="2024-12-31")
        
        # ====== 新增保存逻辑 ======
        output_path = OUTPUT_DIR / "trade_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"数据已保存至: {output_path}")
        # ====== 保存结束 ======
        
        print(df.head() if not df.empty else "没有找到有效数据")
    except Exception as e:
        logger.error(f"读取NPQ文件失败: {e}")

if __name__ == "__main__":
    main()  # 移除旧版参数调用
