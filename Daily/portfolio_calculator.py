import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import plotly.graph_objects as go
import os

class PortfolioCalculator:
    """
    计算投资组合的收益率和换手率
    
    Attributes:
        stocks_matrix: 股票收益率矩阵
        output_dir: 输出目录
    """
    
    def __init__(self, stocks_matrix: pd.DataFrame, output_dir='output'):
        """
        初始化计算器
        
        Args:
            stocks_matrix: 股票日收益率矩阵 (index为日期, columns为股票代码)
            output_dir: 输出目录路径
        """
        self.stocks_matrix = stocks_matrix
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def calculate_portfolio_returns(self, holdings_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        计算投资组合的每日收益率和换手率
        
        Args:
            holdings_matrix: 持仓权重矩阵DataFrame
                - index为日期
                - columns为股票代码
                - 值为持仓权重
        
        Returns:
            包含以下列的DataFrame:
            - date: 日期
            - daily_return: 当日收益率
            - turnover_rate: 换手率
            - net_value: 净值
            - weight_changes: 每只股票的权重变化(字典格式)
        """
        try:
            # 确保日期格式正确
            holdings_matrix.index = pd.to_datetime(holdings_matrix.index)
            
            # 初始化结果DataFrame
            results = pd.DataFrame(index=self.stocks_matrix.index)
            results.index.name = 'date'
            
            # 创建一个单独的DataFrame来存储权重变化
            weight_changes = pd.DataFrame(index=self.stocks_matrix.index, columns=holdings_matrix.columns)
            
            # 计算每日收益率和换手率
            previous_weights = None
            for date in results.index:
                if date not in holdings_matrix.index:
                    continue
                    
                # 获取当日持仓权重
                current_weights = holdings_matrix.loc[date]
                
                # 计算当日收益率
                daily_returns = self.stocks_matrix.loc[date]
                
                # 使用矩阵运算计算组合收益率
                valid_stocks = current_weights.index.intersection(daily_returns.index)
                portfolio_return = (current_weights[valid_stocks] * daily_returns[valid_stocks]).sum()
                
                results.loc[date, 'daily_return'] = portfolio_return
                
                # 计算换手率和权重变化
                if previous_weights is not None:
                    # 计算每只股票的权重变化
                    changes = current_weights - previous_weights
                    weight_changes.loc[date] = changes
                    
                    # 计算换手率
                    turnover = abs(changes).sum() / 2
                    results.loc[date, 'turnover_rate'] = turnover
                else:
                    # 首日记录初始权重
                    weight_changes.loc[date] = current_weights
                    results.loc[date, 'turnover_rate'] = 1.0  # 首日换手率为100%
                
                previous_weights = current_weights
            
            # 计算累计净值
            results['net_value'] = (1 + results['daily_return']).cumprod()
            
            # 计算最大回撤
            results['max_net_value'] = results['net_value'].expanding().max()
            results['drawdown'] = (results['net_value'] - results['max_net_value']) / results['max_net_value']
            
            return results, weight_changes
        
        except Exception as e:
            logger.error(f"计算投资组合收益率失败: {e}")
            raise
    
    def plot_results(self, results: pd.DataFrame, weight_changes: pd.DataFrame, strategy_name: str = "Portfolio"):
        """
        绘制策略结果图表
        
        Args:
            results: 包含回测结果的DataFrame
            weight_changes: 包含权重变化的DataFrame
            strategy_name: 策略名称
        """
        try:
            # 计算统计指标
            total_days = len(results)
            annualized_return = (results['net_value'].iloc[-1] ** (252/total_days) - 1)
            monthly_volatility = results['daily_return'].std() * np.sqrt(21)
            max_drawdown = results['drawdown'].min()
            avg_turnover = results['turnover_rate'].mean()
            
            # 创建净值和回撤的plotly图形
            fig = go.Figure()
            
            # 添加净值曲线
            fig.add_trace(go.Scatter(
                x=results.index,
                y=results['net_value'],
                name='净值',
                line=dict(color='blue')
            ))
            
            # 添加回撤曲线
            fig.add_trace(go.Scatter(
                x=results.index,
                y=results['drawdown'] * 100,
                name='回撤(%)',
                yaxis='y2',
                line=dict(color='red')
            ))
            
            # 更新布局
            fig.update_layout(
                title=f"{strategy_name}策略表现<br>" + 
                      f"年化收益率: {annualized_return:.2%}<br>" +
                      f"月度波动率: {monthly_volatility:.2%}<br>" +
                      f"最大回撤: {max_drawdown:.2%}<br>" +
                      f"平均换手率: {avg_turnover:.2%}",
                yaxis=dict(title='净值'),
                yaxis2=dict(
                    title='回撤(%)',
                    overlaying='y',
                    side='right'
                ),
                height=800
            )
            
            # 显示图表
            fig.show()
            
            # 保存结果
            results_path = os.path.join(self.output_dir, f'{strategy_name}_results.csv')
            results.to_csv(results_path)
            logger.info(f"结果已保存至: {results_path}")
            
            # 保存权重变化明细
            weight_changes_path = os.path.join(self.output_dir, f'{strategy_name}_weight_changes.csv')
            weight_changes.to_csv(weight_changes_path)
            logger.info(f"权重变化明细已保存至: {weight_changes_path}")
            
        except Exception as e:
            logger.error(f"绘制结果图表失败: {e}")
            raise

def example_usage():
    """使用示例"""
    # 创建示例数据
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    
    # 创建随机收益率矩阵
    stocks_matrix = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), len(stocks))),
        index=dates,
        columns=stocks
    )
    
    # 创建持仓权重矩阵
    holdings_matrix = pd.DataFrame(
        index=dates,
        columns=stocks
    )
    
    # 为每一天生成随机权重
    for date in dates:
        weights = np.random.dirichlet(np.ones(len(stocks)))  # 生成和为1的随机权重
        holdings_matrix.loc[date] = weights
    
    # 初始化计算器并计算结果
    calculator = PortfolioCalculator(stocks_matrix)
    results, weight_changes = calculator.calculate_portfolio_returns(holdings_matrix)
    calculator.plot_results(results, weight_changes, "示例策略")

if __name__ == "__main__":
    example_usage() 

    