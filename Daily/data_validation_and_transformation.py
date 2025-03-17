import pandas as pd
import os

class PortfolioWeightAdjuster:
    def __init__(self, csv_file_path, change_limit=0.05):
        self.csv_file_path = csv_file_path
        self.change_limit = change_limit
        self.df = pd.read_csv(csv_file_path)
        
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.time_column = 'datetime'
        else:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.time_column = 'date'
        
        self.all_codes = sorted(set(self.df['code']))
        self.time_values = self.df[self.time_column].unique()

    def validate_weights_sum(self) -> bool:
        """验证CSV文件中每个时间点的权重和是否为1"""
        try:
            grouped = self.df.groupby(self.time_column)['weight'].sum()
            for time_value, weight_sum in grouped.items():
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
        for time_value, group in self.df.groupby(self.time_column):
            target_weights = group['weight'].tolist()
            codes = group['code'].tolist()
            target_weights_list.append(target_weights)
            codes_list.append(codes)
        return target_weights_list, codes_list

    def get_initial_weights(self):
        """从CSV文件中提取初始权重"""
        first_time_value = self.df[self.time_column].iloc[0]
        initial_weights = self.df[self.df[self.time_column] == first_time_value]['weight'].tolist()
        return initial_weights

    def adjust_weights_over_days(self, current_weights, target_weights_list, codes_list):
        """调整当前权重向多个目标权重靠近，具有变化限制。"""
        adjusted_weights_list = []
        for target_weights, codes in zip(target_weights_list, codes_list):
            adjusted_weights = []
            for code in self.all_codes:
                if code in codes:
                    target_index = codes.index(code)
                    target_weight = target_weights[target_index]
                else:
                    target_weight = 0

                current_index = self.all_codes.index(code)
                current_weight = current_weights[current_index] if current_index < len(current_weights) else 0

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
            for time_value, weights in zip(self.time_values, adjusted_weights_list):
                for code, weight in zip(self.all_codes, weights):
                    if weight != 0:
                        f.write(f'{time_value},{code},{weight}\n')

if __name__ == "__main__":
    csv_file_path = 'csv_folder/test_daily_weight.csv' 
    output_file = 'csv_folder/adjusted_weights.csv'
    adjuster = PortfolioWeightAdjuster(csv_file_path)

    if adjuster.validate_weights_sum():
        target_weights_list, codes_list = adjuster.get_target_weights_from_csv()
        current_weights = adjuster.get_initial_weights()
        new_weights_over_days = adjuster.adjust_weights_over_days(current_weights, target_weights_list, codes_list)
        adjuster.save_adjusted_weights_to_csv(new_weights_over_days, output_file)
