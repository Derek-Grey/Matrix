#%%
import pymongo
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from loguru import logger

#%%
# 数据转换函数，将字符串转换为浮点数
def trans_str_to_float64(df: pd.DataFrame, exp_cols: list = None, trans_cols: list = None) -> pd.DataFrame:
    if trans_cols is None and exp_cols is None:
        trans_cols = df.columns
    if not exp_cols is None:
        trans_cols = list(set(df.columns) - set(exp_cols))
    df[trans_cols] = df[trans_cols].astype('float64')
    return df

#%%
# 获取MongoDB客户端连接（只读）
def get_client_U():
    user, pwd = 'Tom', 'tom'  # 替换为实际的用户名和密码
    return pymongo.MongoClient(f"mongodb://{quote_plus(user)}:{quote_plus(pwd)}@192.168.1.99:29900/")

#%%
# LoadData类用于从MongoDB中获取数据
class LoadData:
    def __init__(self, date_s: str, date_e: str):
        if date_s is None or date_e is None:
            raise Exception('必须指定开始和结束日期！')
        self.date_s = date_s
        self.date_e = date_e
        self.client_U = get_client_U()

    def get_chg_wind(self, specific_codes: list) -> pd.DataFrame:
        logger.info('正在加载基于WIND的每日百分比变化数据...')
        query = {
            "date": {"$gte": self.date_s, "$lte": self.date_e},
            "code": {"$in": specific_codes}
        }
        
        projection = {
            "_id": 0,
            "date": 1,
            "code": 1,
            "pct_chg": 1
        }

        df = pd.DataFrame(self.client_U.basic_wind.w_vol_price.find(query, projection, batch_size=1000000))
        df = trans_str_to_float64(df, trans_cols=['pct_chg']).set_index(['date', 'code']).sort_index()
        return df
    
#%%
if __name__ == "__main__":
    start_date = '2023-12-29'
    end_date = '2024-01-02'
    
    # 创建LoadData类的实例
    data_loader = LoadData(start_date, end_date)
    
    # 指定多个股票代码
    specific_codes = ['SZ300649','SZ301197','SZ300716','SH603177','SZ300756','SH603685','SZ300834','SH605177','SZ300807','SZ002732','SZ300721','SH603045','SH600193','SZ301093','SZ300577','SZ300501','SH603955','SZ300651','SH605318','SZ002731','SZ300521','SH603017','SZ002899','SZ300731','SH600182','SH605100','SZ301043','SZ301227','SH605003','SH603165','SZ301182','SZ300635','SH603639','SH603579','SZ000020','SZ002868','SZ002620','SZ300823','SZ300511','SZ300245','SZ002679','SZ002694','SZ300683','SH605268','SZ002669','SZ300647','SZ301228','SH603161','SZ002986','SZ300622','SZ002667','SZ001215','SZ002998','SZ301283','SZ300851','SH603676','SZ002860','SZ300430','SZ300243','SZ301223','SZ300615','SZ300578','SZ300154','SH603326','SH603790','SZ003003','SZ002829','SZ300561','SZ002112','SH603048','SZ300673','SZ002394','SH603331','SZ300743','SZ300076','SH603330','SZ000017','SZ300535','SZ300491','SH603900','SZ301196','SH603130','SZ002963','SH603180','SZ300191','SZ002278','SZ301137','SZ002231','SH600620','SZ301380','SZ300819','SZ300700','SH600858','SZ002774','SZ301122','SH603843','SZ002813','SH603970','SZ300905','SZ300246','SH601566','SZ300824','SZ301361','SH600826','SH603683','SZ300286','SH600694','SH605033','SZ300920','SH605338','SZ300326','SZ301009','SZ300789','SH600855','SZ001207','SZ002159','SZ000159','SZ300092','SZ300668','SH605189','SZ002403','SZ300084','SH603041','SZ000554','SH605001','SZ001316','SH605377','SZ002627','SZ300161','SZ301199','SH603590','SZ300349','SZ002188','SH600230','SH603818','SZ300380','SH603303','SZ002950','SH603990','SH603357','SH603351','SH600793','SZ300539','SH605288','SZ300141','SZ002879','SH600628','SZ000663','SH603611','SZ002930','SH600493','SH603617','SZ000659','SZ002919','SH600262','SZ300695','SH603163','SH603848','SZ300120','SZ002818','SZ002381','SH603859','SZ002367','SZ300016','SH603058','SH600784','SZ300732','SH600650','SH603159','SZ002695','SZ002474','SZ002880','SZ002029','SZ300971','SZ000952','SZ002918','SZ300013','SZ000633','SZ300625','SZ300755','SH605300','SZ002763','SZ301277','SZ300452','SZ002616','SZ300609','SZ300249','SZ300826','SZ001202','SZ001288','SZ300265','SZ002795','SH603978','SH603356','SZ301036','SZ002427','SH603321','SH600955','SZ002687','SZ300810','SZ001218','SZ002823','SH603186','SZ300025','SH600897','SH603214','SH603429','SH603776','SZ002917','SH600861','SZ300779','SZ300817','SZ300606','SZ002859','SZ002877','SZ300626','SZ300829','SZ002796','SZ002098','SH603507','SH605222','SZ002981','SZ300800','SH600689','SZ301359','SZ002636','SZ301156','SZ300571','SZ002483','SZ301276','SZ300534','SZ300086','SZ300665','SH603383','SZ301398','SH600301','SZ300125','SZ002213','SH603416','SZ002443','SZ002393','SZ002546','SZ300149','SH603989','SZ000622','SH603829','SH600097','SH601686','SZ000530','SH603630','SH603860','SZ002448','SH603368','SH601956','SZ300300','SH600217','SZ000404','SZ002369','SZ002264','SH603197','SZ301233','SZ000888','SH601163','SH603838','SH603602','SZ300353','SZ000025','SH603071','SZ002672','SZ300329','SZ002513','SH603727','SZ003029','SZ301113','SZ301109','SZ300563','SH600488','SZ300564','SH603595','SZ301049','SZ002083','SZ002296','SZ002631','SZ002358','SZ301285','SZ301065','SZ001230','SZ002076','SZ002576','SH603788','SH603700','SH600653','SZ301085','SZ002822','SZ002014','SZ002896','SZ002775','SZ300888','SZ300479','SH600984','SZ002870','SH600883','SZ002750','SH600613','SZ300619','SH600235','SH603680','SH603069','SZ301077','SH603949','SH603982','SZ002357','SZ300938','SZ002612','SZ300706','SZ002553','SH603612','SZ003026','SZ301379','SH600054','SZ301218','SH601113','SZ002452','SZ003031','SH603238','SZ300066','SH603227','SZ000836','SZ300472','SZ000553','SZ002117','SZ002866','SZ000852','SZ002969','SZ300426','SH603879','SZ301117','SH603515','SZ002909','SH600075','SH603798','SH603699','SH603187','SH603500','SZ000411','SH600992','SZ000757','SZ002722','SZ300698','SH600469','SH603112','SZ301257','SH600391','SH603567','SZ300532','SZ002621','SH603110','SZ300005','SZ300565','SH600593','SZ002161','SZ301087','SH603976','SZ300747','SH603316','SH603018','SH603279','SZ002629','SZ002233','SH603950','SZ000881','SZ002615','SH600512','SH601700','SZ002628','SH600818','SZ002752','SZ002842','SZ002741','SZ300990','SZ001296','SH603878','SZ002778','SZ300547','SZ301207','SH600251','SZ001333','SZ301290','SZ300797','SH603511','SZ301121','SZ002641','SH603908','SZ300021','SZ002073','SH603277','SZ002820','SZ000551','SZ300821','SZ002913','SZ301003','SZ300019','SH600828','SH603757','SZ001338','SH603758','SZ300162','SH603100','SZ300238','SH603199','SZ300046','SH600866','SZ003013','SZ301162','SH603922','SZ300818','SZ002991','SZ002534','SZ000061','SZ003012','SZ300701','SZ300936','SZ301183','SZ300109','SZ300384','SH603131','SZ003036','SZ300140','SZ000751','SZ000826','SZ301200','SZ300137','SZ301186','SZ300441','SZ300061','SZ300391','SZ002148','SH603050','SZ002833','SZ301339','SH603931','SZ002782','SH603995','SZ002303','SZ000045','SH605158','SZ002170','SH605155','SZ301008','SZ300419','SH600351','SZ300497','SZ300538','SZ002283','SZ300385','SH600337','SZ300843','SZ301229','SZ300453','SZ000905','SH601086','SZ000534','SZ301234','SZ300854','SH600379','SZ000930','SZ002158','SH605287','SZ300965','SZ300135','SZ002191','SZ300242','SZ300345','SH603718','SZ300632','SZ002254','SZ300687','SZ300907','SH603256','SZ300537','SH605598','SZ000571','SZ002869','SZ000968','SZ300050','SZ300236','SH600854','SH600099','SZ301235','SZ300477','SZ300292','SZ002900','SZ002382','SH600323','SZ002554','SZ300480','SH603858','SZ300091','SZ002455','SZ300164']
   
    # 获取数据
    df = data_loader.get_chg_wind(specific_codes)
    
    # 转换数据
    df['pct_chg'] = pd.to_numeric(df['pct_chg'], errors='coerce')
    
    # 计算协方差矩阵
    pivot_df = df.pivot_table(index='date', columns='code', values='pct_chg')
    cov_matrix = pivot_df.cov()
    
    # 计算协方差矩阵的逆矩阵
    Z_inv = np.linalg.inv(cov_matrix)
    
    # 创建全为1的向量
    ones = np.ones((len(cov_matrix), 1))
    
    # 计算 Z_inv * ones
    Z_inv_ones = Z_inv.dot(ones)
    
    # 使用公式2.12计算lambda值
    lambda_value = 1 / np.dot(ones.T, Z_inv_ones)
    
    # 使用公式2.13计算最小方差投资组合权重
    W_min = (lambda_value * Z_inv_ones).flatten()
    
    # 使用公式2.14计算最小方差sigma_min^2
    sigma_min_sq = lambda_value.item()
    
    # 打印最小方差投资组合权重及对应的方差
    print("最小方差投资组合权重 W_min:", W_min)
    print("最小方差 sigma_min^2:", sigma_min_sq)
    
    # 创建一个新的DataFrame，包含股票代码和W_min
    codes = pivot_df.columns
    W_min_df = pd.DataFrame({'code': codes, 'W_min': W_min})
    
    # 添加一行 sigma_min_sq
    W_min_df.loc[len(W_min_df)] = ['sigma_min_sq', sigma_min_sq]
    
    # 将结果写入CSV文件
    W_min_df.to_csv('portfolio_weights_min_variance.csv', index=False)
    
    print("最小方差投资组合权重和sigma_min_sq已写入 'portfolio_weights_min_variance.csv'")

# %%
