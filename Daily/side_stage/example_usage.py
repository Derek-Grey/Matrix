from cal_returnV1 import PortfolioMetrics

def example_usage():
    # 定义文件路径和参数
    weight_file = 'D:\\Derek\\Code\\Matrix\\Daily\\csv\\test_daily_weight.csv'
    return_file = None  # 如果有收益率文件，可以指定路径
    use_equal_weights = True
    data_directory = 'D:\\Data'

    # 初始化PortfolioMetrics对象
    portfolio_metrics = PortfolioMetrics(
        weight_file=weight_file,
        return_file=return_file,
        use_equal_weights=use_equal_weights,
        data_directory=data_directory
    )

    # 计算投资组合指标
    returns, turnover = portfolio_metrics.calculate_portfolio_metrics()

if __name__ == "__main__":
    example_usage()