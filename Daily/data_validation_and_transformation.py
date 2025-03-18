import pandas as pd
import os
import plotly.graph_objects as go
from plotly.offline import plot

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
        """调整当前权重向多个目标权重靠近，具有变化限制。"""
        adjusted_weights_list = []
        # 遍历每个目标权重列表
        for target_weights, codes in zip(target_weights_list, codes_list):
            adjusted_weights = []
            # 遍历所有代码
            for code in self.all_codes:
                if code in codes:
                    target_index = codes.index(code)
                    target_weight = target_weights[target_index]
                else:
                    target_weight = 0

                current_index = self.all_codes.index(code)
                current_weight = current_weights[current_index] if current_index < len(current_weights) else 0

                # 计算权重变化并应用限制
                weight_change = target_weight - current_weight
                if abs(weight_change) > self.change_limit:
                    weight_change = self.change_limit if weight_change > 0 else -self.change_limit

                if target_weight == 0 and current_weight > self.change_limit:
                    weight_change = -self.change_limit

                adjusted_weight = current_weight + weight_change
                adjusted_weights.append(adjusted_weight)

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
