import pandas as pd
import numpy as np
import time

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

if __name__ == '__main__':
    start_total = time.time()
    target_file = "D:\\Data\\All_11.npq"
    
    # 文件读取
    start_read = time.time()
    npq_data = np.fromfile(target_file, dtype=D1_11_numpy_dtype)
    read_time = time.time() - start_read

    # 构建列名
    columns = [f for f in D1_11_numpy_dtype.fields if f != 'quote']
    columns.extend(D1_11_dtype.fields)

    # 数据处理
    start_process = time.time()
    rows = []
    for item in npq_data:
        row_data = {
            f: item[f] for f in D1_11_numpy_dtype.fields if f != 'quote'
        }
        for qf in D1_11_dtype.fields:
            val = item['quote'][qf]
            row_data[qf] = val.decode('utf-8') if isinstance(val, bytes) else val
        rows.append(row_data)
    
    df = pd.DataFrame(rows, columns=columns)
    process_time = time.time() - start_process

    # 保存结果
    start_save = time.time()
    df.to_csv('readdata11_1.csv', index=False, encoding='utf-8-sig')
    save_time = time.time() - start_save

    # 性能统计
    total_time = time.time() - start_total
    print(f"\n[性能统计] 读取:{read_time:.2f}s 处理:{process_time:.2f}s 保存:{save_time:.2f}s 总耗时:{total_time:.2f}s")