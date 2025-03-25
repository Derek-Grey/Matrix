from pymongo import MongoClient
import csv
import os

def fetch_data(start_date, end_date):
    # 连接到MongoDB
    client = MongoClient('mongodb://192.168.1.87:27017/')
    db = client['stra_V31_MARGIN']
    collection = db['stocks_main']
    
    # 定义CSV文件路径
    csv_directory = 'd:\\Derek\\Code\\Matrix\\csv'
    csv_file_path = os.path.join(csv_directory, 'stra_V3_11.csv')
    
    # 如果目录不存在，则创建目录
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    
    # 查询数据，添加日期范围过滤条件
    documents = collection.find(
        {'date': {'$gte': start_date, '$lte': end_date}},
        {'date': 1, 'code': 1, 'F1': 1}
    )
    
    # 打开CSV文件进行写入（如果文件存在则替换）
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV文件头
        writer.writerow(['date', 'code', 'F1'])
        
        # 处理数据并写入CSV文件
        for document in documents:
            date = document.get('date')
            code = document.get('code')
            f1 = document.get('F1')
            writer.writerow([date, code, f1])
            print(f"Date: {date}, Code: {code}, F1: {f1}")

# 指定日期范围
fetch_data("2025-01-02", "2025-01-27")
