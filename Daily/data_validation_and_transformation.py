import pandas as pd
import os
import plotly.graph_objects as go
from plotly.offline import plot
from pymongo import MongoClient
from db_client import get_client_U

client_u = get_client_U(m='r')  # 在文件开头初始化 client_u

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
        
        # 从MongoDB获取数据
        t_limit = client_u.basic_jq.jq_daily_price_none
        use_cols = {"_id": 0, "date": 1, "code": 1, "close": 1, "high_limit": 1, "low_limit": 1}
        
        df_limit = pd.DataFrame(t_limit.find({"date": date_str, "code": {"$in": codes}}, use_cols,batch_size=3000000))
        
        limit_status = {code: 0 for code in codes}  # 默认为非涨跌停状态
        
        if not df_limit.empty:
            # 1表示涨停，-1表示跌停，0表示非涨跌停
            df_limit['limit'] = df_limit.apply(
                lambda x: 1 if x["close"] == x["high_limit"] else (-1 if x["close"] == x["low_limit"] else 0), axis=1)
            df_limit['limit'] = df_limit['limit'].astype('int')
            
            # 更新涨跌停状态
            for _, row in df_limit.iterrows():
                limit_status[row['code']] = row['limit']
                
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
        
        # 从MongoDB获取数据
        t_info = client_u.basic_wind.w_basic_info
        use_cols = {"_id": 0, "date": 1, "code": 1, "trade_status": 1}
        
        df_info = pd.DataFrame(t_info.find(
            {"date": date_str, "code": {"$in": codes}}, 
            use_cols,
            batch_size=3000000
        ))
        
        trade_status = {code: 1 for code in codes}  # 默认为可交易状态
        
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
    def __init__(self, csv_file_path, change_limit=0.05):
        self.csv_file_path = csv_file_path
        self.change_limit = change_limit
        self.df = pd.read_csv(csv_file_path)
        
        # 检查是否有 'datetime' 列，并将其转换为日期时间格式
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.time_column = 'datetime'
        else:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.time_column = 'date'
        
        # 获取所有代码的集合并排序
        self.all_codes = sorted(set(self.df['code']))
        # 获取时间列的唯一值
        self.time_values = self.df[self.time_column].unique()
        self.limit_checker = LimitPriceChecker()

    def validate_weights_sum(self) -> bool:
        """验证CSV文件中每个时间点的权重和是否为1"""
        try:
            # 按时间分组并计算每组的权重和
            grouped = self.df.groupby(self.time_column)['weight'].sum()
            for time_value, weight_sum in grouped.items():
                # 检查每个时间点的权重和是否在允许的范围内
                if not (0.999 <= weight_sum <= 1.001):
                    print(f"数据验证失败：{self.csv_file_path} 中 {time_value} 的权重和不为1 (当前和为 {weight_sum})")
                    return False
            print(f"{self.csv_file_path} 的每个时间点的权重和验证通过")
            return True
        except Exception as e:
            print(f"读取 {self.csv_file_path} 时出错：{e}")
            return False

    def get_target_weights_from_csv(self):
        """从CSV文件中提取目标权重"""
        target_weights_list = []
        codes_list = []
        # 按时间分组提取每组的权重和代码
        for time_value, group in self.df.groupby(self.time_column):
            target_weights = group['weight'].tolist()
            codes = group['code'].tolist()
            target_weights_list.append(target_weights)
            codes_list.append(codes)
        return target_weights_list, codes_list

    def get_initial_weights(self):
        """从CSV文件中提取初始权重"""
        # 获取第一个时间点的权重
        first_time_value = self.df[self.time_column].iloc[0]
        initial_weights = self.df[self.df[self.time_column] == first_time_value]['weight'].tolist()
        return initial_weights

    def adjust_weights_over_days(self, current_weights, target_weights_list, codes_list):
        """
        调整当前权重向多个目标权重靠近，同时考虑涨跌停和交易限制。
        
        Args:
            current_weights (list): 当前持仓的权重列表
            target_weights_list (list): 目标权重的列表，每个元素对应一个时间点的目标权重
            codes_list (list): 股票代码的列表，与target_weights_list对应
            
        Returns:
            list: 调整后的权重列表，包含每个时间点的调整后权重
            
        工作流程:
            1. 对每个时间点进行遍历
            2. 获取当天的涨跌停和交易状态
            3. 对每只股票计算权重调整值
            4. 应用调整限制（涨跌停、交易状态、变化幅度）
            5. 更新权重并进入下一个时间点
        """
        adjusted_weights_list = []
        
        # 遍历每个时间点
        for time_idx, (target_weights, codes) in enumerate(zip(target_weights_list, codes_list)):
            current_time = pd.to_datetime(self.time_values[time_idx])
            adjusted_weights = []
            
            # 获取当天的市场状态
            limit_status = self.limit_checker.get_limit_status(current_time, self.all_codes)
            trade_status = self.limit_checker.get_trade_status(current_time, self.all_codes)
            
            # 对每只股票进行权重调整
            for code in self.all_codes:
                # 获取目标权重（如果股票不在当前组合中则为0）
                target_weight = target_weights[codes.index(code)] if code in codes else 0
                
                # 获取当前权重
                current_index = self.all_codes.index(code)
                current_weight = current_weights[current_index] if current_index < len(current_weights) else 0
                
                # 计算需要调整的权重变化值
                weight_change = target_weight - current_weight
                
                # 根据市场状态决定是否可以调整权重
                if not self.limit_checker.can_adjust_weight(code, weight_change, limit_status, trade_status):
                    # 如果不能调整（涨停不能买入，跌停不能卖出，停牌不能调整），保持原权重
                    adjusted_weight = current_weight
                else:
                    # 正常调整权重，但需要考虑变化限制
                    if abs(weight_change) > self.change_limit:
                        # 如果变化超过限制，则按照限制值调整
                        weight_change = (self.change_limit if weight_change > 0 
                                       else -self.change_limit)
                    adjusted_weight = current_weight + weight_change
                
                adjusted_weights.append(adjusted_weight)
            
            # 更新当前权重并保存调整结果
            current_weights = adjusted_weights
            adjusted_weights_list.append(adjusted_weights)
        
        return adjusted_weights_list

    def save_adjusted_weights_to_csv(self, adjusted_weights_list, output_file):
        """保存调整后的权重到CSV"""
        with open(output_file, 'w') as f:
            f.write(f'{self.time_column},code,adjusted_weight\n')
            # 写入每个时间点的调整后权重
            for time_value, weights in zip(self.time_values, adjusted_weights_list):
                for code, weight in zip(self.all_codes, weights):
                    if weight != 0:
                        f.write(f'{time_value},{code},{weight}\n')

    def plot_adjusted_weight_sums(self, adjusted_weights_list):
        """使用plotly绘制调整后的权重和随时间的变化图"""
        try:
            # 计算每个时间点调整后的权重和
            adjusted_sums = [sum(weights) for weights in adjusted_weights_list]
            
            # 创建图形
            fig = go.Figure()
            
            # 添加实际权重和的线
            fig.add_trace(
                go.Scatter(
                    x=self.time_values,
                    y=adjusted_sums,
                    mode='lines+markers',
                    name='实际权重和',
                    line=dict(color='#2E86C1', width=2),
                    marker=dict(
                        size=8,
                        color='white',
                        line=dict(color='#2E86C1', width=2)
                    )
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
            print(f"绘制调整后权重和图时出错：{e}")

if __name__ == "__main__":
    # 定义输入和输出文件路径
    csv_file_path = 'csv/test_daily_weight.csv' 
    output_file = 'csv/adjusted_weights.csv'
    
    # 创建PortfolioWeightAdjuster对象
    adjuster = PortfolioWeightAdjuster(csv_file_path)

    # 验证权重和并进行调整和保存
    if adjuster.validate_weights_sum():
        target_weights_list, codes_list = adjuster.get_target_weights_from_csv()
        current_weights = adjuster.get_initial_weights()
        new_weights_over_days = adjuster.adjust_weights_over_days(current_weights, target_weights_list, codes_list)
        adjuster.save_adjusted_weights_to_csv(new_weights_over_days, output_file)
        adjuster.plot_adjusted_weight_sums(new_weights_over_days)
