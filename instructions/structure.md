项目根目录为inverse-iv-surface-pinn/，其下各子目录和文件功能如下：

data/：数据存储目录

raw/：原始数据文件（已从WRDS抓取的SPY期权数据Parquet文件存于此）

processed/：经预处理的中间数据（如提取特征、过滤异常后的数据）

notebooks/：Jupyter Notebook工作目录（用于探索性分析和过程记录）

data_fetch.ipynb：已有的数据获取笔记本，记录从WRDS提取数据的过程

eda.ipynb：探索性数据分析（Exploratory Data Analysis）笔记本，用于可视化原始隐含波动率曲面形状、数据质量检查等

training_experiments.ipynb：模型训练与验证实验笔记本，用于交互式调试模型、观察训练动态和结果可视化

src/：源代码模块目录（核心实现）

data_loader.py：数据加载与处理模块。实现读取Parquet数据为DataFrame，转换必要字段（如计算隐含波动率或moneyness）、划分训练/测试集等功能

preprocessing.py：数据预处理模块。包含清洗异常数据（例如剔除无效价格、无效期限）、按照需要对数据进行过滤（如滤除深度实值/虚值合约）、以及将原始数据整理为模型输入格式的函数

black_scholes.py：Black–Scholes定价模块。提供Black–Scholes公式的实现函数，用于根据标的价格、执行价、到期、利率、股息和波动率$\sigma$计算欧式期权理论价格，也可提供反解隐含波动率的函数（用于验证或可视化）

model.py：神经网络模型定义模块。定义用于逼近$\sigma(K,T)$的神经网络架构（即隐含波动率曲面的近似函数），并包含必要的正则化或特殊层以确保输出的平滑性；如果采用PINN策略，可在此定义利用自动微分计算关于输入$K,T$的梯度的功能（用于约束）

pinn_constraints.py：物理约束/无套利约束模块。实现对Black–Scholes方程或无套利条件的约束计算，例如：Black–Scholes PDE残差计算、隐含波动率曲面无套利条件（时间单调性、凸性条件）的检查或惩罚项计算
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org

train.py：模型训练模块。包含训练循环代码，将数据喂入模型，根据损失函数优化模型参数。损失函数由价格误差（模型输出的$\sigma$通过BS公式计算价格与市场价之差）加上物理约束损失和无套利惩罚项构成
medium.com
。该模块还负责日志记录（如每轮训练误差）、模型保存和加载

evaluate.py：模型评估模块。提供对训练结果的评估函数，例如计算预测价格与市场价格的误差指标，验证无套利条件是否满足（检查凸性和单调性），以及生成评估报告

visualize.py：可视化工具模块。封装绘制隐含波动率曲面$\sigma(K,T)$的函数（3D曲面图或热力图）、不同期限的波动率微笑曲线、价格拟合对比图等，便于直观展示模型效果

utils.py：通用工具函数模块。如设置随机种子保证复现结果、读取配置文件、通用的数学函数（如插值等）

tests/：测试用例目录（可选，用于编写单元测试确保各模块正确性）

README.md：项目说明文档。描述项目背景、运行环境、使用方法（如何从数据到训练到评估）、以及结果展示

requirements.txt或environment.yml：Python依赖列表，确保环境可复现（例如TensorFlow/PyTorch版本、数据分析库等）