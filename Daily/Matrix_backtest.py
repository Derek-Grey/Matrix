# 导入所需的库
import os
import time
import math
import pandas as pd
from loguru import logger
from Matrix_plotter import StrategyPlotter   # 从另一个文件导入绘图类

class Backtest:
    """
    股票回测策略类，支持固定持仓和动态持仓两种策略
    
    Attributes:
        stocks_matrix: 股票收益率矩阵
        limit_matrix: 涨跌停限制矩阵
        risk_warning_matrix: 风险警示矩阵
        trade_status_matrix: 交易状态矩阵
        score_matrix: 股票评分矩阵
        output_dir: 输出目录
    """
    
    def __init__(self, stocks_matrix, limit_matrix, risk_warning_matrix, 
                 trade_status_matrix, score_matrix, output_dir='output'):
        """
        初始化回测类
        
        Args:
            stocks_matrix: 股票收益率矩阵
            limit_matrix: 涨跌停限制矩阵
            risk_warning_matrix: 风险警示矩阵
            trade_status_matrix: 交易状态矩阵
            score_matrix: 股票评分矩阵
            output_dir: 输出目录路径
        """
        self.stocks_matrix = stocks_matrix
        self.limit_matrix = limit_matrix
        self.risk_warning_matrix = risk_warning_matrix
        self.trade_status_matrix = trade_status_matrix
        self.score_matrix = score_matrix
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成有效性矩阵
        self._generate_validity_matrices()
        self.plotter = StrategyPlotter(output_dir)  # 创建绘图器实例
    
    def _generate_validity_matrices(self):
        """生成有效性矩阵和受限股票矩阵"""
        # 生成有效性矩阵：股票必须同时满足三个条件
        # 1. 不在风险警示板 (risk_warning_matrix == 0)
        # 2. 正常交易状态 (trade_status_matrix == 1)
        # 3. 不是涨跌停状态 (limit_matrix == 0)
        self.risk_warning_validity = (self.risk_warning_matrix == 0).astype(int)
        self.trade_status_validity = (self.trade_status_matrix == 1).astype(int)
        self.limit_validity = (self.limit_matrix == 0).astype(int)
        self.valid_stocks_matrix = (self.risk_warning_validity * 
                                  self.trade_status_validity * 
                                  self.limit_validity)
        # 受限股票矩阵：只考虑交易状态和涨跌停限制
        self.restricted_stocks_matrix = (self.trade_status_validity * self.limit_validity)
    
    def run_fixed_strategy(self, hold_count, rebalance_frequency, strategy_name="fixed"):
        """
        运行固定持仓策略
        
        Args:
            hold_count: 持仓数量
            rebalance_frequency: 再平衡频率（天数）
            strategy_name: 策略名称
            
        Returns:
            results: 包含回测结果的DataFrame
        """
        start_time = time.time()
        
        # 创建持仓矩阵，使用与stocks_matrix相同的索引和列
        position_matrix = pd.DataFrame(0, 
                                     index=self.stocks_matrix.index,
                                     columns=self.stocks_matrix.columns)
        
        # 创建DataFrame保存收益率等信息
        position_history = pd.DataFrame(
            index=self.stocks_matrix.index,
            columns=["daily_return", "turnover_rate", "strategy"]
        )
        position_history["strategy"] = strategy_name

        # 执行回测循环
        for day in range(1, len(self.stocks_matrix)):
            self._update_positions_fixed_matrix(
                position_matrix, position_history, 
                day, hold_count, rebalance_frequency
            )
        
        # 处理结果
        results = self._process_results_matrix(
            position_matrix, position_history, 
            strategy_name, start_time
        )
        return results

    def _update_positions_fixed_matrix(self, position_matrix, position_history, 
                                     day, hold_count, rebalance_frequency):
        """
        更新固定持仓策略的持仓（矩阵版本）
        
        Args:
            position_matrix: 持仓矩阵
            position_history: 持仓历史DataFrame
            day: 当前交易日索引
            hold_count: 持仓数量
            rebalance_frequency: 再平衡频率
        """
        current_date = position_matrix.index[day]
        previous_date = position_matrix.index[day - 1]
        
        # 如果评分矩阵的前一天数据全为NaN，保持前一天的持仓
        if self.score_matrix.iloc[day - 1].isna().all():
            position_matrix.iloc[day] = position_matrix.iloc[day - 1]
            return

        # 获取前一天的持仓
        previous_positions = position_matrix.iloc[day - 1]
        previous_holdings = previous_positions[previous_positions == 1].index.tolist()

        # 计算有效股票和受限股票
        valid_stocks = self.valid_stocks_matrix.iloc[day].astype(bool)
        restricted = self.restricted_stocks_matrix.iloc[day].astype(bool)
        valid_scores = self.score_matrix.loc[previous_date]
        
        # 受限股票
        restricted_stocks = [stock for stock in previous_holdings if not restricted[stock]]

        # 每隔 rebalance_frequency 天重新平衡持仓
        if (day - 1) % rebalance_frequency == 0:
            sorted_stocks = valid_scores.sort_values(ascending=False)
            try:
                top_stocks = sorted_stocks.iloc[:hold_count].index
                retained_stocks = list(set(previous_holdings) & set(top_stocks) | set(restricted_stocks))

                new_positions_needed = hold_count - len(retained_stocks)
                final_positions = set(retained_stocks)

                if new_positions_needed > 0:
                    new_stocks = sorted_stocks[valid_stocks].index
                    new_stocks = [stock for stock in new_stocks if stock not in final_positions]
                    final_positions.update(new_stocks[:new_positions_needed])
            except IndexError:
                logger.warning(f"日期 {current_date}: 可用股票数量不足，使用所有有效股票")
                final_positions = set(sorted_stocks[valid_stocks].index[:hold_count])
        else:
            final_positions = set(previous_holdings)

        # 更新持仓
        position_history.loc[current_date, "hold_positions"] = ','.join(final_positions)

        # 计算每日收益率
        if previous_date in self.stocks_matrix.index:
            daily_returns = self.stocks_matrix.loc[current_date, list(final_positions)].astype(float)
            daily_return = daily_returns.mean()
            position_history.loc[current_date, "daily_return"] = daily_return

        # 计算换手率
        previous_positions_set = previous_positions
        current_positions_set = final_positions
        turnover_rate = len(previous_positions_set - current_positions_set) / max(len(previous_positions_set), 1)
        position_history.at[current_date, "turnover_rate"] = turnover_rate

    def _process_results(self, position_history, strategy_name, start_time):
        """
        处理回测结果
        
        Args:
            position_history: 持仓历史DataFrame
            strategy_name: 策略名称
            start_time: 开始时间
            
        Returns:
            results: 处理后的结果DataFrame
        """
        # 删除没有持仓记录的行
        position_history = position_history.dropna(subset=["hold_positions"])

        # 计算持仓数量
        position_history['hold_count'] = position_history['hold_positions'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
        
        # 保存结果
        results = position_history[['hold_positions', 'hold_count', 'turnover_rate', 'daily_return']]
        results.index.name = 'date'
        
        csv_file = os.path.join(self.output_dir, f'strategy_results_{strategy_name}.csv')
        results.to_csv(csv_file)
        
        # 计算统计指标
        cumulative_return = (1 + results['daily_return']).cumprod().iloc[-1] - 1
        avg_daily_return = results['daily_return'].mean()
        avg_turnover = results['turnover_rate'].mean()
        avg_holdings = results['hold_count'].mean()
        
        # 输出统计信息
        logger.info(f"\n=== {strategy_name}策略统计 ===")
        logger.info(f"累计收益率: {cumulative_return:.2%}")
        logger.info(f"平均日收益率: {avg_daily_return:.2%}")
        logger.info(f"平均换手率: {avg_turnover:.2%}")
        logger.info(f"平均持仓量: {avg_holdings:.1f}")
        logger.info(f"结果已保存到: {csv_file}")
        logger.info(f"策略运行耗时: {time.time() - start_time:.2f} 秒")
        
        return results

    def plot_results(self, results, strategy_type, turn_loss=0.003):
        """
        绘制策略结果图表
        
        Args:
            results: 回测结果DataFrame
            strategy_type: 策略类型
            turn_loss: 换手损失率
        """
        self.plotter.plot_net_value(results, strategy_type, turn_loss)

