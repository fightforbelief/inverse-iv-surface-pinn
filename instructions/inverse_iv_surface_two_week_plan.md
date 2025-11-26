
# inverse-iv-surface-pinn 项目：两周（14 天）完成计划

> 说明：以下用 **D1–D14** 表示第几天。每一天都写明要操作的文件和具体任务。你可以根据实际时间前后微调。

当前项目结构（供参考）：

```text
inverse-iv-surface-pinn/
  notebooks/
    00_data_wrds.ipynb
    10_mvp_smile_fit.ipynb
    20_surface_fit_adam_lbfgs.ipynb
    30_pinn_constraints.ipynb
    99_figures.ipynb
  ivpinn/
    __init__.py
    data.py
    bs.py
    surface.py
    losses.py
    train.py
    eval.py
  configs/
    default.yaml
    pinn_ablation.yaml
  scripts/
    wrds_fetch.py
  tests/
    test_bs.py
  data/
  out/
```

---

## D1：Black–Scholes 基础函数 + 单元测试

**目标：** Black–Scholes 定价函数正确可用，有基础单元测试。

**要动的文件**

- `ivpinn/bs.py`
- `tests/test_bs.py`
- （可选）`configs/default.yaml`

**具体任务**

1. 在 `ivpinn/bs.py` 实现核心函数：

   - `bs_call_price(S, K, T, sigma, r=0.0, q=0.0)`
   - `bs_put_price(S, K, T, sigma, r=0.0, q=0.0)`
   - 至少一个 Greek：`bs_vega(...)`

   要求：

   - 支持 numpy 或 torch 张量（任选一种先实现）。
   - 处理标量和向量输入。

2. 在 `tests/test_bs.py` 写单元测试：

   - 若干简单 case：
     - `sigma=0` 时，`C ≈ max(S e^{-qT} - K e^{-rT}, 0)`。
     - `K=0` 时，call 价格应接近 `S e^{-qT}`。
   - 单调性检查：
     - 增大 `sigma`，call 价格上升。
     - 增大 `T`，call 价格非减（多数情况）。

3. 运行测试：

   ```bash
   cd inverse-iv-surface-pinn
   pytest tests/test_bs.py
   ```

4. 在 `configs/default.yaml` 预留 Black–Scholes 参数块（可先随便填）：

   ```yaml
   bs:
     r: 0.02
     q: 0.0
     S0: 450.0
   ```

---

## D2：WRDS / OptionMetrics 数据获取接口

**目标：** 从 WRDS/OptionMetrics 拉取某日 SPY 期权链并存成干净 CSV。

**要动的文件**

- `scripts/wrds_fetch.py`
- `ivpinn/data.py`
- `notebooks/data_wrds.ipynb`
- `data/` 目录

**具体任务**

1. 在 `ivpinn/data.py` 设计数据接口函数：

   - `fetch_optionmetrics_wrds(conn, trade_date, symbol="SPY") -> DataFrame`
   - `clean_option_chain(df)`：过滤欧式 call、去缺失、去极端 strike。
   - `compute_mid_price(df)`：从 bid/ask 计算 mid。

   先写出函数签名和基本逻辑。

2. 在 `scripts/wrds_fetch.py` 写命令行脚本：

   - 步骤：
     - 建立 WRDS 连接：`wrds.Connection()`
     - 调用 `fetch_optionmetrics_wrds`
     - `clean_option_chain` + `compute_mid_price`
     - 保存到 `data/raw/options_SPY_YYYYMMDD.csv`

3. 在 `notebooks/data_wrds.ipynb`：

   - 导入 `from ivpinn.data import fetch_optionmetrics_wrds, clean_option_chain, compute_mid_price`
   - 对指定交易日跑一遍，展示：
     - 总合约数
     - `K`、`T`、mid price 分布概要（`describe()`）。

4. 在 `ivpinn/data.py` 增加：

   - `load_clean_option_data(path)`：读入 CSV，返回 `(S0, K_array, T_array, C_mkt_array)`。

---

## D3：隐含波动率反推 + MVP 单期限 smile

**目标：** 对单个 maturity 做 IV 反推并画出 smile 散点。

**要动的文件**

- `ivpinn/bs.py`
- `ivpinn/data.py`
- `notebooks/10_mvp_smile_fit.ipynb`
- `data/processed/`（输出含 IV 的 CSV）

**具体任务**

1. 在 `ivpinn/bs.py` 实现 IV 反推：

   ```python
   def implied_vol_call(C_mkt, S, K, T, r=0.0, q=0.0,
                        sigma_lb=1e-4, sigma_ub=5.0):
       """使用 Brent 法求解隐含波动率"""
   ```

   - 用 `scipy.optimize.brentq` 在 `[sigma_lb, sigma_ub]` 上求根。
   - 支持 vector 输入（`np.vectorize` 或循环）。

2. 在 `ivpinn/data.py` 增加：

   - `compute_implied_vol_column(df)`：为 DataFrame 添加 `iv` 列（对每行 call 价格求解）。

3. 在 `notebooks/10_mvp_smile_fit.ipynb`：

   - 读入 `data/raw/options_SPY_YYYYMMDD.csv`
   - 选定一个到期日，过滤该 maturity 的 call。
   - 调用 `compute_implied_vol_column` 得到 `iv`。
   - 画图：`K/S0` vs `iv`（scatter），保存到 `out/figures/smile_SPY_YYYYMMDD_Txxx.png`。
   - 将带 `iv` 的数据另存为 `data/processed/options_iv_SPY_YYYYMMDD.csv`。

---

## D4：`surface.py`：σ(K,T) 简单参数化 + 定价接口

**目标：** 搭建最简单的 σ(K,T) 参数化（例如 2D 网格 + 最近邻/双线性插值），并能输出 BS 价格。

**要动的文件**

- `ivpinn/surface.py`
- `ivpinn/bs.py`
- `notebooks/10_mvp_smile_fit.ipynb`

**具体任务**

1. 在 `ivpinn/surface.py` 定义基础曲面类（示例）：

   ```python
   import torch
   import torch.nn as nn

   class BilinearVolSurface(nn.Module):
       def __init__(self, K_grid, T_grid, init_sigma=0.2):
           super().__init__()
           self.register_buffer("K_grid", torch.tensor(K_grid, dtype=torch.float32))
           self.register_buffer("T_grid", torch.tensor(T_grid, dtype=torch.float32))
           self.log_sigma = nn.Parameter(
               torch.full((len(T_grid), len(K_grid)), float(np.log(init_sigma)))
           )

       def forward(self, K, T):
           # 先实现最近邻插值，之后可升级为双线性
           ...
   ```

2. 在 `surface.py` 增加辅助函数：

   ```python
   def price_from_surface(surface, S0, K, T, r=0.0, q=0.0):
       # 用 surface(K,T) 得到 sigma，再调用 torch 版 bs_call_price
   ```

3. 在 `notebooks/10_mvp_smile_fit.ipynb` 做简单测试：

   - 从 `options_iv_SPY_YYYYMMDD.csv` 读取某个 maturity 的 `K` 与 `iv`。
   - 手动构造一个只在该 maturity 生效的 surface（其他 T 先随便填）。
   - 调 `price_from_surface` 计算价格，与 `C_mkt` 对比，确保链条正确。

---

## D5：`losses.py`：定价误差 + 平滑正则

**目标：** 实现最小可用的 loss 组合：定价误差 + 曲面平滑正则。

**要动的文件**

- `ivpinn/losses.py`
- `ivpinn/surface.py`
- `ivpinn/bs.py`
- `notebooks/10_mvp_smile_fit.ipynb`
- `configs/default.yaml`

**具体任务**

1. 在 `losses.py` 定义：

   - `pricing_mse(pred_prices, mkt_prices)`：普通 MSE。
   - `surface_smoothness_l2(param_grid)`：
     - 对 `log_sigma` 做相邻差分平方和（K 与 T 方向）。
   - `total_loss(pred_prices, mkt_prices, surface, weights)`：
     - `L = mse + lambda_smooth * smoothness`。

2. 在 `configs/default.yaml` 写入 loss 参数：

   ```yaml
   loss:
     lambda_smooth: 1e-3
   ```

3. 在 `notebooks/10_mvp_smile_fit.ipynb` 做单 maturity 训练 demo：

   - 构造 surface、使用 torch Adam：
     - 前向：`sigma = surface(K, T)` → `price = bs_call_price`。
     - Loss：`pricing_mse + lambda_smooth * smoothness`。
   - 跑几十个 step，看 loss 是否下降、拟合是否改善。

---

## D6：`train.py`：通用 2D surface 训练循环（Adam + L-BFGS）

**目标：** 有一个统一的训练入口函数，可通过 config 训练完整 σ(K,T) 曲面。

**要动的文件**

- `ivpinn/train.py`
- `ivpinn/data.py`
- `ivpinn/surface.py`
- `ivpinn/losses.py`
- `configs/default.yaml`
- `notebooks/20_surface_fit_adam_lbfgs.ipynb`

**具体任务**

1. 在 `ivpinn/train.py` 实现主函数：

   ```python
   def train_surface(config_path="configs/default.yaml"):
       # 1. 读取 config
       # 2. 加载数据 (load_clean_option_data)
       # 3. 初始化 surface 模型
       # 4. 使用 Adam 预训练，再用 L-BFGS 精调（可选）
       # 5. 记录 loss，保存 best checkpoint
   ```

2. 在 `configs/default.yaml` 填写训练相关参数：

   ```yaml
   data:
     path: "data/processed/options_iv_SPY_YYYYMMDD.csv"
     S0: 450.0
     r: 0.02
     q: 0.0

   surface:
     type: "bilinear"
     n_K: 30
     n_T: 10
     init_sigma: 0.2

   optim:
     adam_lr: 1e-2
     adam_steps: 500
     use_lbfgs: true
     lbfgs_max_iter: 50

   loss:
     lambda_smooth: 1e-3
   ```

3. 在 `notebooks/20_surface_fit_adam_lbfgs.ipynb`：

   - 调用：

     ```python
     from ivpinn.train import train_surface
     train_surface("configs/default.yaml")
     ```

   - 画出 training loss 曲线（从训练日志或返回值中读取）。

4. 在 `train.py` 中保存 checkpoint 和日志：

   - `out/models/surface_default.pt`
   - `out/logs/default_log.csv`（记录 epoch、loss、data_loss、smooth_loss）。

---

## D7：`eval.py` + `99_figures.ipynb`：误差分析和曲面可视化

**目标：** 能载入训练好的 model，画 surface、smile 截面、误差图。

**要动的文件**

- `ivpinn/eval.py`
- `ivpinn/surface.py`
- `ivpinn/bs.py`
- `notebooks/99_figures.ipynb`
- `out/`

**具体任务**

1. 在 `ivpinn/eval.py` 写评估函数：

   - `load_surface_from_ckpt(ckpt_path, config_path)`：返回模型实例。
   - `compute_pricing_errors(model, data)`：计算整体 MSE、按 maturity/moneyness 分组误差。
   - `sample_surface_grid(model, K_grid, T_grid)`：输出 `sigma_grid`。

2. 在 `notebooks/99_figures.ipynb`：

   - 对 baseline 模型：
     - 3D IV surface 图（`K`–`T`–`σ`）。
     - 几个典型 maturity 的 smile（模型 vs market IV 散点）。
     - 价格残差 heatmap（`K` vs `T`）。
   - 图像保存到：
     - `out/figures/surface_default_3d.png`
     - `out/figures/smile_default_Txx.png`
     - `out/figures/residual_default.png` 等。

3. 在 `eval.py` 加 CLI 入口（可选）：

   ```bash
   python -m ivpinn.eval --config configs/default.yaml --ckpt out/models/surface_default.pt
   ```

   打印几个关键指标供快速查看。

---

## D8：无套利约束 loss（convexity + calendar）

**目标：** 在 `losses.py` 中加入简单静态无套利 penalty 并可计算。

**要动的文件**

- `ivpinn/losses.py`
- `ivpinn/bs.py`
- `ivpinn/surface.py`
- `ivpinn/train.py`
- `notebooks/30_pinn_constraints.ipynb`
- `configs/default.yaml`

**具体任务**

1. 在 `losses.py` 实现无套利损失：

   - Convexity（对 K 的凸性）：

     ```python
     def no_arb_convexity_loss(C_grid, K_grid):
         # 对每个 T 切片做二阶差分，Δ²C < 0 的部分计入 penalty
     ```

   - Calendar（期限不减性）：

     ```python
     def no_arb_calendar_loss(C_grid, T_grid):
         # 对每个 K 切片检查 C(T_longer) >= C(T_shorter)
     ```

   - 封装：

     ```python
     def no_arb_loss(surface, S0, K_grid, T_grid, r, q, weights):
         # 用 surface+bs_price 生成 C_grid，然后计算 convexity+calendar 的 penalty 和
     ```

2. 在 `notebooks/30_pinn_constraints.ipynb` 中测试：

   - 构造一个明显违背凸性的 toy surface，计算 `no_arb_loss`，应偏大。
   - 构造一个平滑、近似凸的 surface，`no_arb_loss` 应明显变小。

3. 在 `configs/default.yaml` 增加参数：

   ```yaml
   loss:
     lambda_smooth: 1e-3
     lambda_no_arb: 0.0  # 先设为 0，确保与当前 baseline 相同
   ```

4. 在 `train.py` 的 `total_loss` 中接入 `no_arb_loss`（但权重现在为 0，不影响结果）。

---

## D9：PINN / PDE 残差约束（简化版）

**目标：** 实现一个简单 PDE 残差 loss（Dupire PDE 的离散近似），作为 PINN-like 正则。

**要动的文件**

- `ivpinn/losses.py`
- `ivpinn/bs.py`
- `ivpinn/surface.py`
- `notebooks/30_pinn_constraints.ipynb`
- `configs/pinn_ablation.yaml`

**具体任务**

1. 在 `losses.py` 实现 PDE-like 残差：

   - 对给定 `K_grid, T_grid`：
     - 用有限差分近似 `∂C/∂T` 和 `∂²C/∂K²`。
     - 根据简化 Dupire 形式，构造 residual：`res = LHS - RHS`。
     - `pde_loss = (res**2).mean()`。

2. 在 `notebooks/30_pinn_constraints.ipynb` 中：

   - 对当前训练好的 surface 计算 `pde_loss`。
   - 手动修改 surface（更平滑）或减弱噪声，看 `pde_loss` 是否有变化。

3. 新建 `configs/pinn_ablation.yaml`（复制 `default.yaml`）：

   - 在其基础上设置：

     ```yaml
     loss:
       lambda_smooth: 1e-3
       lambda_no_arb: 1e-3
       lambda_pde: 1e-4
     ```

   - 为后续对比实验做准备。

---

## D10：带无套利 & PDE 残差的训练（对比实验）

**目标：** 跑两组实验：baseline vs PINN-ish，比较拟合质量和约束情况。

**要动的文件**

- `ivpinn/train.py`
- `configs/default.yaml`
- `configs/pinn_ablation.yaml`
- `notebooks/20_surface_fit_adam_lbfgs.ipynb`
- `notebooks/30_pinn_constraints.ipynb`
- `out/`

**具体任务**

1. 跑 baseline 实验：

   ```python
   from ivpinn.train import train_surface
   train_surface("configs/default.yaml")
   ```

   - 生成：`out/models/surface_default.pt`
   - 记录 baseline 的 data loss、smoothness、no_arb_loss、pde_loss（即使 λ=0，也可以事后算）。

2. 跑 PINN-ish 实验：

   ```python
   train_surface("configs/pinn_ablation.yaml")
   ```

   - 生成：`out/models/surface_pinn.pt`
   - 同样记录各类 loss 指标。

3. 在 `notebooks/30_pinn_constraints.ipynb` 中比较：

   - 对两种模型：
     - 计算 `no_arb_loss`、`pde_loss`。
     - 计算价格 RMSE。
   - 写一个小表格或简单文字对比结论。

4. 将对比结果记入 `out/notes/ablation_pinn_vs_baseline.md`（新建）：

   - 表格列如：model / RMSE / no_arb_loss / pde_loss。
   - 简短文字：PINN-ish 在无套利和 PDE 残差上的改善 vs 拟合上的 trade-off。

---

## D11：评估 & 图像补全（包含 PINN 模型）

**目标：** 所有报告需要的图，都能通过 `99_figures.ipynb` 自动生成（对 baseline 和 PINN 模型）。

**要动的文件**

- `ivpinn/eval.py`
- `notebooks/99_figures.ipynb`
- `out/figures/`

**具体任务**

1. 在 `eval.py` 中扩展评估接口：

   ```python
   def evaluate_model(config_path, ckpt_path) -> dict:
       # 返回 rmse, no_arb_loss, pde_loss 等
   ```

2. 在 `notebooks/99_figures.ipynb` 中：

   - 对 `surface_default.pt` 和 `surface_pinn.pt` 分别生成：
     - 3D IV surface。
     - 几条 maturity 的 smile（模型 vs market IV）。
     - 价格残差 heatmap。
     - 无套利违约 heatmap（例如标记 Δ²C < 0 的区域）。

   - 文件命名示例：
     - `out/figures/surface_default_3d.png`
     - `out/figures/surface_pinn_3d.png`
     - `out/figures/smile_default_T1.png`
     - `out/figures/smile_pinn_T1.png`
     - `out/figures/residual_default.png`
     - `out/figures/residual_pinn.png`

3. 重启 kernel，从头运行 `99_figures.ipynb`，确保一次性成功执行。

---

## D12：代码整理 & README 文档

**目标：** 清理代码、补充注释和 README，使项目对外展示更友好。

**要动的文件**

- `ivpinn/data.py`
- `ivpinn/bs.py`
- `ivpinn/surface.py`
- `ivpinn/losses.py`
- `ivpinn/train.py`
- `ivpinn/eval.py`
- `scripts/wrds_fetch.py`
- `README.md`
- `configs/*.yaml`

**具体任务**

1. 在主要模块顶部增加模块级 docstring，描述：

   - 模块功能。
   - 关键类/函数列表。

2. 为核心函数补充 docstring：

   - 明确所有参数单位（例如 `T` 为年化到期时间）。
   - 输入输出形状（标量 / 向量 / 网格）。

3. 梳理 `README.md`：

   - 项目简介。
   - 环境安装步骤：
     - `conda create ...`
     - `pip install -r requirements.txt`
   - 从 0 到结果的流程说明：
     1. WRDS 拉数（脚本或 notebook）。
     2. 处理数据生成 `data/processed/...`。
     3. 训练 baseline：`python -m ivpinn.train --config configs/default.yaml`
     4. 训练 PINN-ish：同上。
     5. 生成图：运行 `notebooks/99_figures.ipynb`。

4. 清理 `configs/*.yaml`：

   - 保证参数与实际实现一致，没有废弃字段。
   - 在每个配置文件开头加 1–2 行注释说明用途。

---

## D13：报告 / slides 内容撰写（文字）

**目标：** 根据项目实际完成情况，准备课程项目报告或 slides 的文字内容。

**要动的文件**

- 报告或演示文稿（例如 `report/main.tex` 或 `.pptx`）
- `out/notes/*.md`
- `out/figures/*.png`

**具体任务**

1. 按照课程要求大纲，撰写报告结构：

   - 引言：Black–Scholes 逆问题背景、IV surface 的重要性。
   - 方法：数据来源、σ(K,T) 参数化、损失函数（data + smooth + no-arb + PDE）、优化方法。
   - 实验：baseline vs PINN-ish 设置。
   - 结果：关键数值表 + 图（surface、smile、残差）。
   - 讨论：无套利与 PDE 约束带来的收益和 trade-off。

2. 将 `out/notes/ablation_pinn_vs_baseline.md` 内容整理成 1–2 个结果表格（LaTeX 或 slides 表）。

3. 写一小节 “Limitations & Future Work”：

   - 例如：
     - 数据只选单日 / 单标的。
     - PDE 约束采用简化 Dupire、数值近似比较粗糙。
     - 未来可以扩展到 SABR/NN 参数化、跨标的 joint surface 等。

---

## D14：完整 dry run + 最终收尾

**目标：** 按 README 流程从头跑一遍，确保一件不落；然后整理提交版本。

**要动的文件**

- 整个项目（尤其 README、configs、notebooks）

**具体任务**

1. 完整 dry run 流程：

   1. 安装依赖（可在新环境中再跑一次 `pip install -r requirements.txt`）。
   2. 若需要，运行 `scripts/wrds_fetch.py` 或直接使用已有 `data/raw` 文件。
   3. 运行 `00_data_wrds.ipynb` 生成 `data/processed/options_iv_SPY_YYYYMMDD.csv`。
   4. 使用 `python -m ivpinn.train --config configs/default.yaml` 训练 baseline。
   5. 使用 `python -m ivpinn.train --config configs/pinn_ablation.yaml` 训练 PINN-ish。
   6. 运行 `99_figures.ipynb` 生成所有最终图像。
   7. 编译报告（如适用）。

2. 在过程中修复所有小问题：

   - import 路径错误、相对路径问题、缺少目录等。
   - 确保 `.gitignore` 中排除大数据文件（`data/raw` 等）但保留必要的示例或 metadata。

3. Git 提交 & 打标签：

   - 提交信息例：

     ```bash
     git add .
     git commit -m "final: AMCS6045 inverse IV surface PINN project"
     git tag v1.0-amcs6045
     ```

   - push 到远程仓库（若有），方便展示给老师或招聘方。

---

完成以上 14 天计划后，你的 `inverse-iv-surface-pinn` 项目会具备：

- 清晰的 BS/PINN 数学实现；
- 完整的数据 pipeline（WRDS → 清洗 → IV 计算）；
- 可训练的 IV surface（baseline + 带无套利/PDE 的 PINN-ish 模型）；
- 完整的评估与可视化；
- 可直接用于展示/投简历的 Repo 结构和文档。
