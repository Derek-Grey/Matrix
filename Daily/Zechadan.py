import pandas as pd
import glob


file_pattern = r"data\*.csv"

# 获取所有匹配模式的文件路径列表
file_list = glob.glob(file_pattern)

# 定义要查找的日期和股票代码
date_value = "2014-03-24"
stock_column = "SZ002015"

# 遍历文件并查找值
for file_path in file_list:
    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 确认文件中是否包含所需的列
        if 'date' in df.columns and stock_column in df.columns:
            # 查找指定行和列的值
            if date_value in df['date'].values:
                value = df.loc[df['date'] == date_value, stock_column].values[0]
                print(f"{file_path} - {date_value} 对应 {stock_column} 的值是: {value}")
            else:
                print(f"{date_value} 在 {file_path} 中未找到")
        else:
            print(f"文件 {file_path} 不包含所需的列")
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {e}")
