from pymongo import MongoClient
import csv
import os
from datetime import datetime
import pandas as pd

def fetch_data(start_date, end_date):
    client = MongoClient('mongodb://192.168.1.87:27017/')
    db = client['stra_V31_MARGIN']
    collection = db['stocks_all']
     
    csv_directory = 'd:\\Derek\\Code\\Matrix\\csv'
    csv_file_path = os.path.join(csv_directory, 'stra_V3_11.csv')
    
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    try:
        client.server_info()
        print("成功连接到MongoDB服务器")
    except Exception as e:
        print(f"连接MongoDB失败: {e}")
        return

    doc_count = collection.count_documents({
        'date': {'$gte': start_date, '$lte': end_date}
    })
    print(f"找到 {doc_count} 条符合条件的文档")

    documents = collection.find({
        'date': {'$gte': start_date, '$lte': end_date}
    }, {'date': 1, 'code': 1, 'F1': 1})

    # 检查文件是否存在以避免重复写入标题
    file_exists = os.path.isfile(csv_file_path)
    
    # 将新数据转换为DataFrame并合并
    new_data = pd.DataFrame(list(documents), columns=['date', 'code', 'F1'])
    new_data['date'] = pd.to_datetime(new_data['date'])

    # 读取现有数据并排序
    if file_exists:
        existing_data = pd.read_csv(csv_file_path, parse_dates=['date'])
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        combined_data = new_data

    # 按日期排序
    combined_data.sort_values(by='date', inplace=True)

    # 写入CSV文件
    combined_data.to_csv(csv_file_path, index=False)
    print(f"数据已写入 {csv_file_path}，共写入 {len(combined_data)} 条记录")

if __name__ == "__main__":
    fetch_data("2015-08-03", "2020-07-31")
    input("按回车键退出...")
