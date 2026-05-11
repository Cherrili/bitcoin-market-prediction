# 比特币市场状态预测

基于比特币链上指标，使用机器学习方法预测市场状态（**牛市 / 震荡 / 熊市**三分类）。

## 项目概览

| 项目 | 说明 |
|------|------|
| **数据集** | Bitcoin Network On-Chain Blockchain Data（Kaggle） |
| **时间范围** | 2010年8月 — 2023年9月（约4700条日频数据） |
| **任务类型** | 三分类 |
| **标签定义** | 30日价格涨跌幅：>+15% 牛市，<-15% 熊市，其余震荡 |
| **特征** | 176个工程特征 → RF重要性筛选至 Top 40 |
| **数据划分** | 时间顺序 80/20，不随机打乱，无数据泄露 |

## 模型与结果

| 模型 | Accuracy | F1 (macro) | ROC AUC |
|------|----------|------------|---------|
| Logistic Regression | 0.1977 | 0.1843 | 0.5611 |
| **Random Forest** | **0.5781** | **0.3105** | 0.5229 |
| XGBoost | 0.2200 | 0.1575 | 0.5392 |
| LightGBM | 0.2232 | 0.1216 | 0.5764 |
| SVM | 0.2168 | 0.1218 | 0.3870 |
| **KNN** | 0.5770 | **0.3119** | 0.4974 |

所有模型使用 `GridSearchCV` + `TimeSeriesSplit(n_splits=5)` 进行超参数调优，类别不平衡通过 `class_weight="balanced"`（XGBoost 使用 `sample_weight`）处理。

## 项目结构

```
├── src/
│   ├── main.py                # 程序入口
│   ├── config.py              # 路径与常量配置
│   ├── data_loader.py         # 数据读取与清洗
│   ├── feature_engineering.py # 标签生成与特征工程
│   ├── models.py              # 模型定义与超参数网格
│   ├── training.py            # 数据划分、特征筛选、GridSearchCV训练
│   ├── evaluation.py          # 评估指标、混淆矩阵、ROC曲线
│   └── visualization.py      # 特征重要性图
├── output/                    # 所有图表与结果文件
├── app.py                     # Streamlit 可视化界面
└── requirements.txt           # 依赖列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

**训练模型**（结果自动保存至 `output/`）：
```bash
python -m src.main
```

**启动可视化界面**：
```bash
python -m streamlit run app.py
```

## 输出文件说明

| 文件 | 内容 |
|------|------|
| `eda_label_distribution.png` | 标签分布与比特币价格走势图 |
| `*_confusion_matrix.png` | 各模型混淆矩阵（共6个） |
| `roc_curves_all_models.png` | 所有模型 ROC 曲线（One-vs-Rest） |
| `feature_importance_*.png` | 各模型 Top 20 特征重要性图 |
| `model_summary_table.png` | 模型性能对比表 |
| `model_summary.csv` | 模型评估指标（CSV格式） |
