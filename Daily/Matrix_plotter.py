import os
import plotly.graph_objects as go
from loguru import logger
import pandas as pd
class StrategyPlotter:
    """
    策略结果可视化类
    
    Attributes:
        output_dir: 输出图表目录
    """
    
    def __init__(self, output_dir='output'):
        """
        初始化绘图类
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_net_value(self, df: pd.DataFrame, strategy_name: str, turn_loss: float = 0.003):
        """
        绘制策略的累计净值和回撤曲线
        
        Args:
            df: 包含回测结果的DataFrame
            strategy_name: 策略名称
            turn_loss: 换手损失率
        """
        df = df.copy()  # 创建副本避免修改原始数据
        df.reset_index(inplace=True)
        df.set_index('date', inplace=True)
        start_date = df.index[0]
        
        # 确保必要的列存在
        if 'daily_return' not in df.columns:
            logger.error("DataFrame 不包含 'daily_return' 列。")
            return
        if 'turnover_rate' not in df.columns:
            logger.error("DataFrame 不包含 'turnover_rate' 列。")
            return

        # 计算成本和净值
        self._calculate_costs_and_returns(df, turn_loss)
        
        # 计算回撤
        self._calculate_drawdown(df)
        
        # 计算统计指标
        stats = self._calculate_statistics(df)
        
        # 绘制图表
        self._create_plot(df, strategy_name, start_date, stats)
        
        # 保存图表（可选）
        # self._save_plot(strategy_name)
    
    def _calculate_costs_and_returns(self, df: pd.DataFrame, turn_loss: float):
        """计算成本和收益"""
        # 设置固定成本，并针对特定日期进行调整
        df['loss'] = 0.0013  # 初始固定成本
        df.loc[df.index > '2023-08-31', 'loss'] = 0.0008  # 特定日期后的调整成本
        df['loss'] += float(turn_loss)  # 加上换手损失

        # 计算调整后的变动和累计净值
        df['chg_'] = df['daily_return'] - df['turnover_rate'] * df['loss']
        df['net_value'] = (df['chg_'] + 1).cumprod()
    
    def _calculate_drawdown(self, df: pd.DataFrame):
        """计算最大回撤"""
        # 计算最大净值和回撤
        dates = df.index.unique().tolist()
        for date in dates:
            df.loc[date, 'max_net'] = df.loc[:date].net_value.max()
        df['back_net'] = df['net_value'] / df['max_net'] - 1
    
    def _calculate_statistics(self, df: pd.DataFrame):
        """计算统计指标"""
        s_ = df.iloc[-1]
        return {
            'annualized_return': format(s_.net_value ** (252 / df.shape[0]) - 1, '.2%'),
            'monthly_volatility': format(df.net_value.pct_change().std() * 21 ** 0.5, '.2%'),
            'end_date': s_.name
        }
    
    def _create_plot(self, df: pd.DataFrame, strategy_name: str, start_date, stats: dict):
        """创建图表"""
        # 创建净值和回撤的plotly图形对象
        g1 = go.Scatter(x=df.index.unique().tolist(), y=df['net_value'], name='净值')
        g2 = go.Scatter(x=df.index.unique().tolist(), y=df['back_net'] * 100, name='回撤', xaxis='x', yaxis='y2', mode="none",
                        fill="tozeroy")

        # 配置并显示图表
        fig = go.Figure(
            data=[g1, g2],
            layout={
                'height': 1122,
                "title": f"{strategy_name}策略，<br>净值（左）& 回撤（右），<br>全期：{start_date} ~ {stats['end_date']}，<br>年化收益：{stats['annualized_return']}，月波动：{stats['monthly_volatility']}",
                "font": {"size": 22},
                "yaxis": {"title": "累计净值"},
                "yaxis2": {"title": "最大回撤", "side": "right", "overlaying": "y", "ticksuffix": "%", "showgrid": False},
            }
        )
        fig.show()
    
    def _save_plot(self, strategy_name: str):
        """保存图表到文件（如果需要）"""
        # TODO: 实现图表保存功能
        pass