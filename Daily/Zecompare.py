import pandas as pd

#%%
# 第一个CSV文件
daily_holdings = pd.read_csv(r'DataFile/daily_holdings.csv')
daily_holdings_split = daily_holdings['hold_positions'].str.split(',', expand=True)
daily_holdings_split.insert(0, 'date', daily_holdings['date'])
daily_holdings_split.to_csv('output/daily_holdings_split1.csv', index=False)

# 第二个CSV文件
daily_holdings1 = pd.read_csv(r'output/position_holdings.csv')
daily_holdings1_split = daily_holdings1['hold_positions'].str.split(',', expand=True)
daily_holdings1_split.insert(0, 'date', daily_holdings1['date'])
daily_holdings1_split.to_csv('output/daily_holdings_split2.csv', index=False)

#%%
# 读取两个CSV文件
daily_holdings = pd.read_csv('output/daily_holdings_split1.csv')
backtest_holdings = pd.read_csv('output/daily_holdings_split2.csv')

# 确保两个 DataFrame 按照日期对齐
merged = pd.merge(daily_holdings, backtest_holdings, on='date', suffixes=('_daily', '_backtest'))

# 创建一个空的列表，用于存储每一行的结果
results = []

# 遍历每一行
for i in range(len(merged)):
    date = merged.loc[i, 'date']
    
    # 获取当日的持仓数据并转换为集合
    daily_stocks = set(merged.loc[i, merged.columns[1:]].filter(like='_daily').dropna())
    backtest_stocks = set(merged.loc[i, merged.columns[1:]].filter(like='_backtest').dropna())

    # 求交集
    intersection = daily_stocks.intersection(backtest_stocks)
    
    # 求差集
    daily_only = daily_stocks - backtest_stocks
    backtest_only = backtest_stocks - daily_stocks

    # 将结果添加到列表中
    results.append({
        'date': date,
        'intersection': ', '.join(sorted(intersection)),
        'daily_only': ', '.join(sorted(daily_only)),
        'backtest_only': ', '.join(sorted(backtest_only))
    })

# 将结果列表转换为DataFrame
result_df = pd.DataFrame(results)

# 保存交集与差集结果到新的CSV文件
result_df.to_csv('output/comparison_results.csv', index=False)

# %%
