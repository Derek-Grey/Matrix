import pandas as pd
import json
import time

def read_json_file(file_path):
    """读取JSON文件并返回DataFrame"""
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 假设JSON数据结构与NPQ类似
    rows = []
    for item in json_data:
        row_data = {}
        for key, value in item.items():
            if isinstance(value, str):
                try:
                    value = value.encode('utf-8').decode('utf-8')
                except UnicodeDecodeError:
                    value = None
            row_data[key] = value
        rows.append(row_data)
    
    df = pd.DataFrame(rows)
    return df

if __name__ == '__main__':
    start_total = time.time()
    target_file = "D:\\Data\\data.json"
    
    # 文件读取
    start_read = time.time()
    df = read_json_file(target_file)
    read_time = time.time() - start_read

    # 保存结果
    start_save = time.time()
    df.to_csv('readdata_json.csv', index=False, encoding='utf-8-sig')
    save_time = time.time() - start_save

    # 性能统计
    total_time = time.time() - start_total
    print(f"\n[性能统计] 读取:{read_time:.2f}s 保存:{save_time:.2f}s 总耗时:{total_time:.2f}s")