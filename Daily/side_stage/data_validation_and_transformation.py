import os
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.offline import plot
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Daily.Utils.db_client import get_client_U 
from urllib.parse import quote_plus

client_u = get_client_U(m='r')  

class LimitPriceChecker:
    """检查股票涨跌停状态的类"""

    def get_limit_status(self, date, codes):
        """
        获取指定日期的股票涨跌停状态
        
        Args:
            date: 日期，可以是字符串或datetime对象
            codes: 股票代码列表
            
        Returns:
            dict: 股票代码到涨跌停状态的映射，1表示涨跌停，0表示非涨跌停
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        date_str = date.strftime('%Y-%m-%d')
        
        # 将NumPy数组转换为list
        codes_list = codes.tolist() if isinstance(codes, np.ndarray) else codes
        
        # 从MongoDB获取数据
        t_limit = client_u.basic_jq.jq_daily_price_none
        use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
        
        df_limit = pd.DataFrame(t_limit.find({"date": date_str, "code": {"$in": codes_list}}, use_cols, batch_size=3000000))
        
        limit_status = {code: 0 for code in codes}  # 默认为非涨跌停状态
        
        if not df_limit.empty:
            # 使用numpy向量化操作替代apply
            closes = df_limit['close'].values
            high_limits = df_limit['high_limit'].values
            low_limits = df_limit['low_limit'].values
            
            # 向量化计算涨跌停状态
            limit_array = np.zeros(len(df_limit))
            limit_array = np.where(closes == high_limits, 1, limit_array)
            limit_array = np.where(closes == low_limits, -1, limit_array)
            
            df_limit['limit'] = limit_array.astype('int')
            
            # 更新涨跌停状态
            limit_dict = dict(zip(df_limit['code'], df_limit['limit']))
            limit_status.update(limit_dict)
                
        return limit_status
        
    def get_trade_status(self, date, codes):
        """
        获取指定日期的股票交易状态
        
        Args:
            date: 日期，可以是字符串或datetime对象
            codes: 股票代码列表
            
        Returns:
            dict: 股票代码到交易状态的映射，1表示可交易，0表示停牌
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        date_str = date.strftime('%Y-%m-%d')
        
        # 将NumPy数组转换为list
        codes_list = codes.tolist() if isinstance(codes, np.ndarray) else codes
        
        # 从MongoDB获取数据
        t_info = client_u.basic_wind.w_basic_info
        use_cols = {"_id": 0, "date": 1, "code": 1, "trade_status": 1}
        
        df_info = pd.DataFrame(t_info.find(
            {"date": date_str, "code": {"$in": codes_list}}, 
            use_cols,
            batch_size=3000000
        ))
        
        trade_status = {code: 1 for code in codes}  # 默认可交易
        
        if not df_info.empty:
            for _, row in df_info.iterrows():
                trade_status[row['code']] = row['trade_status']
                
        return trade_status
    
    def can_adjust_weight(self, code, weight_change, limit_status, trade_status):
        """
        判断是否可以调整权重
        
        Args:
            code: 股票代码
            weight_change: 权重变化值（正值表示增加权重，负值表示减少权重）
            limit_status: 涨跌停状态字典（1表示涨停，-1表示跌停，0表示非涨跌停）
            trade_status: 交易状态字典（1表示可交易，0表示停牌）
            
        Returns:
            bool: 是否可以调整权重
        """
        # 检查交易状态
        if trade_status.get(code, 1) == 0:  # 停牌状态
            return False
            
        # 检查涨跌停状态
        status = limit_status.get(code, 0)
        
        # 涨停时不能增加权重
        if status == 1 and weight_change > 0:
            return False
            
        # 跌停时不能减少权重
        if status == -1 and weight_change < 0:
            return False
            
        return True

class PortfolioWeightAdjuster:
    def __init__(self, weights_array, dates, codes, change_limit=0.05):
        """
        初始化调整器
        
        Args:
            weights_array: shape为(n_dates, n_codes)的numpy数组，表示每日每个股票的权重
            dates: 日期数组，长度为n_dates
            codes: 股票代码数组，长度为n_codes
            change_limit: 单日调整上限
        """
        self._start_time = time.time()  
        self.weights = weights_array
        self.dates = pd.to_datetime(dates)
        self.codes = np.array(codes)
        self.change_limit = change_limit
        self.limit_checker = LimitPriceChecker()
        print(f"初始化耗时: {time.time() - self._start_time:.2f}秒")

    def validate_weights_sum(self) -> bool:
        """验证权重和"""
        _start = time.time()  
        try:
            # 直接使用numpy计算每日权重和
            daily_sums = np.sum(self.weights, axis=1)
            valid_sums = np.logical_and(daily_sums >= 0.999, daily_sums <= 1.001)
            
            if not np.all(valid_sums):
                invalid_indices = np.where(~valid_sums)[0]
                print("数据验证失败：以下日期的权重和不为1:")
                for idx in invalid_indices:
                    print(f"{self.dates[idx]}: {daily_sums[idx]}")
                return False
                
            print("所有日期的权重和验证通过")
            print(f"权重验证耗时: {time.time() - _start:.2f}秒")
            return True
            
        except Exception as e:
            print(f"权重验证出错：{e}")
            print(f"权重验证耗时: {time.time() - _start:.2f}秒")
            return False

    def adjust_weights_over_days(self):
        """调整权重"""
        _start = time.time()  
        
        # 初始化结果数组
        n_dates, n_codes = self.weights.shape
        adjusted_weights = np.zeros_like(self.weights)
        current_weights = self.weights[0].copy()  # 使用第一天的权重作为初始权重
        
        for day in range(n_dates):
            _loop_start = time.time()
            
            # 获取市场状态
            limit_status = self.limit_checker.get_limit_status(self.dates[day], self.codes)
            trade_status = self.limit_checker.get_trade_status(self.dates[day], self.codes)
            
            # 计算目标权重
            target_weights = self.weights[day]
            
            # 计算权重变化
            weight_changes = target_weights - current_weights
            
            # 创建可调整的掩码数组
            can_adjust_mask = np.zeros(n_codes, dtype=bool)
            for i, code in enumerate(self.codes):
                can_adjust_mask[i] = self.limit_checker.can_adjust_weight(
                    code, weight_changes[i], limit_status, trade_status)
            
            # 向量化处理权重调整
            weight_changes_limited = np.clip(weight_changes, -self.change_limit, self.change_limit)
            current_weights[can_adjust_mask] += weight_changes_limited[can_adjust_mask]
            
            # 保存当日调整后的权重
            adjusted_weights[day] = current_weights.copy()
            
            if day % 50 == 0:
                print(f"处理第 {day+1}/{n_dates} 天, "
                      f"单次耗时: {time.time() - _loop_start:.2f}秒")
        
        print(f"\n权重调整总耗时: {time.time() - _start:.2f}秒")
        print(f"平均每天耗时: {(time.time() - _start)/n_dates:.2f}秒")
        
        return adjusted_weights

    def plot_adjusted_weight_sums(self, adjusted_weights):
        """绘制权重和变化图"""
        _start = time.time()
        try:
            # 计算每日权重和
            adjusted_sums = np.sum(adjusted_weights, axis=1)
            
            # 创建图形
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.dates,
                    y=adjusted_sums,
                    mode='lines+markers',
                    name='实际权重和'
                )
            )
            
            # 添加目标权重和的参考线
            fig.add_hline(
                y=1.0,
                line=dict(color='#E74C3C', dash='dash'),
                opacity=0.5,
                name='目标权重和'
            )
            
            # 更新布局
            fig.update_layout(
                title={
                    'text': '调整后权重和变化',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                xaxis_title='时间',
                yaxis_title='权重和',
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                xaxis=dict(
                    tickangle=30,
                    tickformat='%Y-%m-%d'
                ),
                hovermode='x unified'
            )
            
            # 调整y轴范围
            max_sum = max(adjusted_sums)
            min_sum = min(adjusted_sums)
            margin = (max_sum - min_sum) * 0.1
            fig.update_yaxes(range=[min_sum - margin, max_sum + margin])
            
            # 显示图形
            fig.show()
            
        except Exception as e:
            print(f"绘制图形时出错：{e}")
        finally:
            print(f"绘图耗时: {time.time() - _start:.2f}秒")

    @staticmethod
    def load_data(data_source, source_type='csv'):
        """
        通用数据加载接口
        
        Args:
            data_source: 数据源，可以是CSV文件路径或numpy数组
            source_type: 数据源类型，支持 'csv' 或 'numpy'
            
        Returns:
            tuple: (weights_array, dates, codes)
        """
        if source_type == 'csv':
            return PortfolioWeightAdjuster._from_csv(data_source)
        elif source_type == 'numpy':
            return PortfolioWeightAdjuster._from_numpy(data_source)
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
    
    @staticmethod
    def _from_csv(csv_file_path):
        """从CSV文件加载数据"""
        df = pd.read_csv(csv_file_path)
        
        # 确定时间列名
        time_col = 'datetime' if 'datetime' in df.columns else 'date'
        df[time_col] = pd.to_datetime(df[time_col])
        
        dates = sorted(df[time_col].unique())
        all_codes = sorted(df['code'].unique())
        
        # 创建权重矩阵
        n_dates = len(dates)
        n_codes = len(all_codes)
        weights_array = np.zeros((n_dates, n_codes))
        
        # 创建代码到索引的映射
        code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
        
        # 按日期填充权重矩阵
        for date_idx, date in enumerate(dates):
            daily_data = df[df[time_col] == date]
            for _, row in daily_data.iterrows():
                code_idx = code_to_idx[row['code']]
                weights_array[date_idx, code_idx] = row['weight']
        
        return weights_array, dates, all_codes
    
    @staticmethod
    def _from_numpy(data_dict):
        """
        从numpy数组加载数据
        
        Args:
            data_dict: 字典，包含以下键：
                - weights: shape为(n_dates, n_codes)的numpy数组
                - dates: 日期列表或数组
                - codes: 股票代码列表或数组
        """
        required_keys = ['weights', 'dates', 'codes']
        if not all(key in data_dict for key in required_keys):
            raise ValueError(f"数据字典必须包含以下键: {required_keys}")
            
        weights = np.array(data_dict['weights'])
        dates = pd.to_datetime(data_dict['dates'])
        codes = np.array(data_dict['codes'])
        
        if len(dates) != weights.shape[0] or len(codes) != weights.shape[1]:
            raise ValueError("数组维度不匹配")
            
        return weights, dates, codes

def print_numpy_info(data, name="数组"):
    """
    打印NumPy数组的详细信息
    
    参数:
        data: numpy数组
        name: 数组的名称（默认为"数组"）
    """
    print(f"\n{name}的信息:")
    print(f"数据类型: {data.dtype}")
    print(f"形状: {data.shape}")
    print(f"维度: {data.ndim}")
    print(f"数组大小: {data.size}")
    print(f"数据示例:\n{data[:5]}")  
    print(f"描述性统计:\n{np.nanmean(data)=:.4f}\n{np.nanstd(data)=:.4f}")
    print("-" * 50)

def main():
    from pathlib import Path

    # 交互式参数输入
    source_type = input("请输入数据源类型 [csv/numpy] (默认csv): ") or 'csv'
    change_limit = float(input("请输入单日调整上限 (默认0.05): ") or 0.05)
    
    if source_type == 'csv':
        csv_path = input("请输入CSV文件路径 (默认csv/test_daily_weight.csv): ") or 'csv/test_daily_weight.csv'
        data_source = str(Path(__file__).parent.parent / csv_path)
    else:
        numpy_path = input("请输入numpy文件路径 (默认adjusted_weights.npy): ") or 'adjusted_weights.npy'
        loaded_data = np.load(numpy_path, allow_pickle=True)
        
        # 兼容新旧两种数据格式
        if isinstance(loaded_data, np.ndarray):
            print("检测到旧版numpy格式，自动转换...")
            # 从原始数据加载dates和codes
            csv_backup = input("请提供原始CSV文件路径以获取日期和代码信息: ") or 'csv/test_daily_weight.csv'
            _, backup_dates, backup_codes = PortfolioWeightAdjuster._from_csv(
                str(Path(__file__).parent.parent / csv_backup)
            )
            
            data_source = {
                'weights': loaded_data,
                'dates': backup_dates,
                'codes': backup_codes
            }
        elif isinstance(loaded_data, dict):
            data_source = loaded_data
        else:
            raise ValueError("numpy文件格式无效，必须包含字典或权重数组")

    # 自动执行全流程
    weights_array, dates, codes = PortfolioWeightAdjuster.load_data(data_source, source_type)
    adjuster = PortfolioWeightAdjuster(weights_array, dates, codes, change_limit)
    
    if adjuster.validate_weights_sum():
        adjusted = adjuster.adjust_weights_over_days()  
        adjuster.plot_adjusted_weight_sums(adjusted)    

        output_data = {
            'weights': adjusted,  
            'dates': dates,
            'codes': codes
        }
        np.save('adjusted_weights.npy', output_data)
        print(f"调整后的权重已保存到: adjusted_weights.npy")

# 新增编程式调用接口
def adjust_weights(source_type='csv', data_source=None, change_limit=0.05, output_path='adjusted_weights.npy'):
    """
    编程式调用接口
    
    Args:
        source_type: 数据源类型 ['csv'/'numpy']
        data_source: 数据源路径（CSV文件路径或numpy字典路径）
        change_limit: 单日调整上限
        output_path: 结果保存路径
    
    Returns:
        dict: 包含调整后的权重、日期、代码的字典
    """
    from pathlib import Path
    
    # 加载数据
    if source_type == 'csv':
        abs_path = str(Path(__file__).parent.parent / data_source)
        weights, dates, codes = PortfolioWeightAdjuster.load_data(abs_path, 'csv')
    else:
        loaded = np.load(data_source, allow_pickle=True)
        if isinstance(loaded, np.ndarray):
            # 旧格式需要原始CSV路径
            csv_path = str(Path(__file__).parent.parent / 'csv/test_daily_weight.csv')
            _, dates, codes = PortfolioWeightAdjuster._from_csv(csv_path)
            data_dict = {'weights': loaded, 'dates': dates, 'codes': codes}
        else:
            data_dict = loaded.item() if isinstance(loaded, np.ndarray) else loaded
        weights, dates, codes = PortfolioWeightAdjuster.load_data(data_dict, 'numpy')
    
    # 执行调整
    adjuster = PortfolioWeightAdjuster(weights, dates, codes, change_limit)
    if adjuster.validate_weights_sum():
        adjusted = adjuster.adjust_weights_over_days()
        output_data = {
            'weights': adjusted,
            'dates': dates,
            'codes': codes
        }
        np.save(output_path, output_data)
        return output_data
    return None

if __name__ == "__main__":
    main()
