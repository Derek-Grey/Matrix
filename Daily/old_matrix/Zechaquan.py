import pandas as pd


df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\aligned_limit_matrix.csv')
rows, columns = df.shape
print(f"aligned_limit_matrix.csv,行数: {rows}, 列数: {columns}")


df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\aligned_riskwarning_matrix.csv')
rows, columns = df.shape
print(f"aligned_riskwarning_matrix.csv,行数: {rows}, 列数: {columns}")


df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\aligned_score_matrix.csv')
rows, columns = df.shape
print(f"aligned_score_matrix.csv,行数: {rows}, 列数: {columns}")


df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\aligned_trade_status_matrix.csv')
rows, columns = df.shape
print(f"aligned_trade_status_matrix.csv,行数: {rows}, 列数: {columns}")


df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\aligned_stocks_matrix.csv')
rows, columns = df.shape
print(f"aligned_stocks_matrix.csv,行数: {rows}, 列数: {columns}")


# df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\limit_matrix.csv')
# rows, columns = df.shape
# print(f"limit_matrix.csv,行数: {rows}, 列数: {columns}")


# df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\riskwarning.csv')
# rows, columns = df.shape
# print(f"riskwarning.csv,行数: {rows}, 列数: {columns}")


# df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\score_matrix.csv')
# rows, columns = df.shape
# print(f"score_matrix.csv,行数: {rows}, 列数: {columns}")


# df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\trade_status.csv')
# rows, columns = df.shape
# print(f"trade_status.csv,行数: {rows}, 列数: {columns}")


# df = pd.read_csv(r'C:\Users\Christopher\Desktop\shuidui\code\Daily\data\stocks_info.csv')
# rows, columns = df.shape
# print(f"stocks_info.csv,行数: {rows}, 列数: {columns}")
