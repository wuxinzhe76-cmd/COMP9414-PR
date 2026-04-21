# COMP9417 Project Execution Plan (The HD Blueprint) - Revised

## 1. 实验数据集矩阵 (The 5 Datasets)

| # | 数据集 | UCI ID | 任务类型 | 样本量 | 特征数 | 特征类型 | 选择理由 |
|---|--------|--------|----------|--------|--------|----------|----------|
| 1 | **Dry Bean** | 602 | 多分类(7类) | 13,611 | 16 | 纯数值型 | **主实验 + Task B 可解释性分析**。几何特征物理意义明确，适合对比 AGOP/PCA 等。*注意：评估时必须使用宏平均 AUC (`average='macro', multi_class='ovr'`).* |
| 2 | **AI4I Predictive Maintenance** | 601 | 二分类 | 10,000 | 14 | 混合(数值+类别)| **Mixed-type 要求**。类别特征需要 OneHot 编码，测试模型对混合特征的处理能力。剔除无预测意义的 UID 和 Product ID。 |
| 3 | **Online News Popularity** | 332 | 回归 | 39,644 | 60 | 纯数值型 | **d > 50 高维要求**。剔除 url 和 timedelta 防止数据泄漏。测试模型在高维环境下的表现。 |
| 4 | **Bank Marketing** | 222 | 二分类 | 45,211 | 16 | 混合(数值+类别)| **极度不平衡要求**。正负样本极度失衡，作为 Discussion 章节中分析 xRFM 失效场景（理论探讨）的绝佳标靶。 |
| 5 | **Superconductivity Data** | 464 | 回归 | 21,263 | 81 | 纯数值型 | **n > 10,000 Scaling 实验**。纯静态物理特征，绝对无时间序列泄漏风险。专门用于 Task C 展示核方法 $O(n^2)$ 时间复杂度。 |

**安全检查确认：** 以上 5 个数据集均已与 xRFM 原论文（arxiv:2508.10053）核对，**100% 无重合**，绝对合规。

## 2. 基线模型阵容 (The "All-In" Roster)
- **核心主角:** xRFM
- **树模型阵营:** XGBoost, Random Forest
- **深度学习阵营 (不调参):** MLP, TabNet
- **反面教材 (仅限 Task C 展示复杂度):** SVM / KRR

## 3. 核心任务清单 (Tasks to Implement)

### Task A: 数据管道与解耦训练流
- [x] `src/data_loader.py`: 统一的数据下载、清洗与切分 (60/20/20)。
- [x] `src/train_trees_and_xrfm.py`: 预处理 (Fit 仅在 Train 上) 并保存 `preprocessor.pkl`。训练 xRFM, XGBoost, RF。
- [x] `src/train_deep_learning.py`: 继承预处理规则。训练 MLP, TabNet。
- [ ] `notebooks/01_Main_Results.ipynb`: 汇总 25 组结果。**红线修复：多分类任务 (Dry Bean) 的 AUC 必须明确调用 `roc_auc_score(multi_class='ovr', average='macro')`。**

### Task B: 可解释性对比 (Interpretability)
- [ ] `notebooks/02_Interpretability.ipynb`: **基于叶子节点的特征重要性提取。** 必须遍历每个 xRFM 叶子节点提取 AGOP 对角线，并按叶内样本量进行**加权平均 (Weighted Average)**，绝不能仅仅提取模型全局级别的矩阵。
- [ ] 计算 PCA Loadings, Mutual Information, Permutation Importance。绘制并排柱状图。
- [ ] **深度分析：** 详细论述这 4 种方法产生分歧的原因（重点强调有监督方法 vs 无监督方法对特征敏感度的差异）。

### Task C: 扩展性崩溃实验 (Scaling & Efficiency)
- [ ] `notebooks/03_Scaling_Experiment.ipynb`: 递增截取训练集 (10%, 20%, 40%, 60%, 80%, 100%)。
- [ ] **对比对象全覆盖：** 折线图必须包含所有主实验模型（xRFM, XGBoost, RF, DL），**外加** 传统 SVM/KRR 作为计算崩溃的对比标靶。
- [ ] 绘制两条折线图：`Test Performance vs n` 和 `Training Time vs n`，清晰暴露传统核方法 $O(n^2)$ 的算力瓶颈。

### Task D: Bonus 冲刺 (Residual-weighted AGOP)
- [ ] **(i) 概念证明：** 解释残差加权为何有效。
- [ ] **(ii) 分裂对比：** 在小数据集上对比标准 AGOP 和 残差 AGOP 选择分裂方向的差异。
- [ ] **(iii) 分歧案例：** 找到并解释两者产生不同分裂方向的实例。
- [ ] **(iv) 性能验证 (核心修复项)：** 必须实际将残差 AGOP 应用于 xRFM 的构建中，并在至少一个数据集上证明其预测性能（如降低 RMSE）优于标准 AGOP。

## 4. 报告与提交硬性规范 (Report & Submission Checklist)

### 报告撰写 (Methodology & Discussion)
- [ ] **超参数文档化：** 在实验设置部分，必须逐个数据集列出 xRFM、XGBoost、RF 最终采用的超参数配置。
- [ ] **计算复杂度：** 必须在报告中单开一段，明确探讨 xRFM 的计算复杂性（时间/空间复杂度）。
- [ ] **符号一致性：** 报告中的所有数学符号必须与 xRFM 原始论文 (arxiv:2508.10053) 保持 100% 一致。
- [ ] **理论失败案例分析：** 在 Discussion 中，基于算法底层原理探讨 xRFM 表现不如 XGBoost 的场景（例如 Bank Marketing 数据集的极度不平衡与离散性如何破坏梯度平滑假设）。

### 格式与提交物
- [ ] **README.md：** 必须包含清晰的从零运行代码的环境配置和执行步骤说明。
- [ ] **标题加分标识：** 报告的封面标题后必须追加 `"+bonus"` 字样。
- [ ] **必带参考文献：** 至少包含两篇硬性要求的论文：
      1. xRFM paper (arxiv:2508.10053)
      2. Radhakrishnan et al., 2024, *Science* 383(6690), 1461–1467.