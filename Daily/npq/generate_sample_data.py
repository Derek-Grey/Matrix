"""
生成示例交易日路径数据
生成格式：trade_date列值为 YYYY-MM-DD/1/11.npq
"""
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

def read_trade_dates(csv_path):
    """读取交易日文件并添加路径列"""
    try:
        # 直接读取字符串格式的日期列
        df = pd.read_csv(csv_path, usecols=['trade_date'])
        
        # 生成路径列（确保日期格式为YYYY-MM-DD）
        df['path'] = df['trade_date'].apply(
            lambda x: str(Path(r"D:\Data") / f"{pd.to_datetime(x).strftime('%Y-%m-%d')}" / "1" / "11.npq")
        )
        return df
    
    except Exception as e:
        print(f"处理失败: {e}")
        return pd.DataFrame()

def save_sample_data():
    """保存处理后的数据"""
    # 输入文件路径
    input_file = r"D:\Derek\Code\Matrix\Daily\csv\trade_dates_all.csv"
    
    # 输出目录处理
    output_dir = Path(__file__).parent.parent / 'csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成并保存数据
    result_df = read_trade_dates(input_file)
    output_file = output_dir / "trade_all.csv"
    result_df.to_csv(output_file, index=False)
    print(f"文件已生成: {output_file}")

if __name__ == "__main__":
    save_sample_data()