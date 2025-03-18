# Daily Project

## 项目简介

Daily 是一个用于管理和分析日常数据的项目。它包含多个模块，旨在提供高效的数据处理和分析功能。

## 目录结构

- **CSV 数据文件**：
  - `csv/aligned_stocks_matrix.csv` - 对齐的股票矩阵数据。
  - `csv/aligned_trade_status_matrix.csv` - 对齐的交易状态矩阵数据。
  - `csv/raw_limit_info.csv` - 原始限制信息数据。
  - `csv/raw_stocks_info.csv` - 原始股票信息数据。
  - `csv/raw_wind_data.csv` - 原始风数据。
  - `csv/stra_V3_11.csv` - 策略版本 3.11 的数据。
  - `csv/test_daily_return.csv` - 测试日收益数据。
  - `csv/test_daily_weight.csv` - 测试日权重数据。
  - `csv/test_minute_return.csv` - 测试分钟收益数据。
  - `csv/test_minute_weight.csv` - 测试分钟权重数据。

- **Daily**：
  - `Daily/portfolio_metrics_df.py` - 处理投资组合指标数据框的模块。
  - `Daily/portfolio_metrics.py` - 计算投资组合指标的模块。
  - `Daily/data_create.py` - 数据创建和初始化模块。
  - `Daily/Att.py` - 处理和分析 Att 数据的模块。
  - `Daily/db_client.py` - 数据库客户端模块，用于连接和操作数据库。
  - `Daily/load_data.py` - 数据加载模块，从外部源加载数据。
  - `Daily/utils.py` - 实用工具函数模块，提供通用功能。
  - `Daily/settings.py` - 项目配置和设置。

- **old_matrix**：
  - 包含旧版本的矩阵处理脚本，如 `Aold_all.py` 和 `Asus_juzhen1.4.py`。

- **A_workplace_data**：
  - 用于存储工作区相关的数据。

- **Utils**：
  - 实用工具模块，提供通用功能。

## 所需数据

- **CSV 文件**：存储在 `csv` 目录下，用于各种数据分析和处理。
- **数据库连接信息**：在 `Daily/settings.py` 中配置数据库连接字符串。

## 安装

1. 克隆仓库：

   ```bash
   git clone git@github.com:Derek-Grey/Daily.git
   ```

2. 进入项目目录：

   ```bash
   cd Daily
   ```

3. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. 创建数据：

   ```bash
   python Daily/data_create.py
   ```

2. 加载数据：

   ```bash
   python Daily/load_data.py
   ```

3. 运行分析：

   ```bash
   python Daily/portfolio_metrics.py
   ```

## 贡献

欢迎贡献代码！请 fork 本仓库并提交 pull request。

## 许可证

此项目使用 MIT 许可证。详情请参阅 LICENSE 文件。
