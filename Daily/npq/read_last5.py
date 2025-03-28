import pandas as pd
from tabulate import tabulate

file_path = r"d:\Derek\Code\Matrix\Daily\npq\output\trade_results.csv"

# 读取CSV文件并输出最后5行
try:
    df = pd.read_csv(file_path)
    last5 = df.tail(5)
    print(tabulate(last5, headers='keys', tablefmt='psql', showindex=False))
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"读取文件失败: {str(e)}")