# DSTD-CID

**Dynamic Semantic Textual Dynamics for Early Chinese Cyberbullying Identification**

面向中文网络事件的早期网络欺凌识别方法。项目以事件为基本建模单元，将评论流按时间聚合，通过语义表示、动态文本池和时间序列建模，在尽可能早的时间点判断事件是否包含网络欺凌，并同时评估识别准确性与检测及时性。

> 本仓库包含数据预处理、动态语义文本池构建和 ECTS（Early Cyberbullying Temporal Sequence）模型训练代码。当前 README 按代码中的默认路径和参数整理；实验结果请以实际运行日志为准。

## 1. 方法概览

给定一个由时间戳排序的事件评论流，系统执行以下步骤：

1. 使用 `Qwen/Qwen3-Embedding-0.6B` 对评论及事件先验文本进行语义编码。
2. 在小时粒度上进行候选筛选、鲁棒 K-Means 聚类和代表性文本选择，形成动态文本池。
3. 为每个事件补充类别标签、累计评论数量和累计恶意评论数量等时间特征。
4. 使用冻结的中文 BERT 编码文本序列，并由 Autoformer-lite/Transformer 风格的 ECTS 模型联合完成事件分类与停止时刻预测。
5. 在事件级划分的训练集、验证集和测试集上计算分类及早期检测指标。

## 2. 项目结构

```text
DSTD-CID/
├── cyberbullying_dataset/       # 原始事件数据，按类别组织
│   ├── events/                  # 小时级评论 CSV
│   ├── cyberbullying/           # 网络欺凌事件
│   └── non_cyberbullying/       # 非网络欺凌事件
├── dynamic_text_pool/           # 语义编码、聚类、打分与文本池构建
├── training/                    # 数据集、ECTS 模型、训练与分析工具
├── .github/                     # GitHub 工作流或项目配置
├── add_event_labels.py          # 为文本池结果添加事件标签
├── batch_enhance_all_pools.py   # 添加累计时间序列特征
├── model_download.py            # 下载预训练文本向量模型
├── event_summary.csv            # 事件先验文本映射（name/content）
└── requirements.txt             # Python 依赖
```

## 3. 环境配置

建议使用 Python 3.10，并在具备 NVIDIA GPU 的环境中运行。仓库依赖文件固定了 PyTorch 2.8.0、Transformers 4.56.2、Sentence-Transformers 5.1.1、scikit-learn 1.7.2 等版本；CUDA 版本应与本机驱动及 PyTorch 安装方式匹配。

```bash
python3.10 -m venv .venv
source .venv/bin/activate             # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

文本编码和训练阶段可能需要约 8 GB 以上显存，实际需求取决于事件数量、批大小和最大时间长度。无 GPU 时可以运行部分预处理，但完整实验会明显变慢。

## 4. 数据格式

### 4.1 小时级事件评论

`cyberbullying_dataset/events/` 中的每个 CSV 对应一个事件，至少包含以下列：

| 列名 | 含义 |
| --- | --- |
| `content` | 评论文本 |
| `timestamp` | 评论时间戳，可被 Pandas 解析 |

`event_summary.csv` 用于提供事件先验文本，至少包含：

| 列名 | 含义 |
| --- | --- |
| `name` | 事件文件名或文件 stem |
| `content` | 对应事件的先验文本 |

类别目录中的事件文件名应与 `events/` 中的文件 stem 对应：`cyberbullying/` 标记为 1，`non_cyberbullying/` 标记为 0。数据集和模型权重如涉及版权或隐私，请在公开发布前确认授权范围。

## 5. 复现实验流程

推荐从项目根目录执行以下命令。

### Step 1：下载文本编码模型

```bash
python model_download.py
```

该脚本通过 ModelScope 下载 `Qwen/Qwen3-Embedding-0.6B`。如果使用自定义模型目录，需要同步修改 `dynamic_text_pool/pipeline.py` 中的 `model_path`，以及 `training/train_early.py` 中的 `bert_model_path`。

### Step 2：构建动态文本池

```bash
cd dynamic_text_pool
python pipeline.py
cd ..
```

默认读取 `../cyberbullying_dataset/events` 和 `../event_summary.csv`，并在项目上一级目录生成：

```text
hourly_outputs_0_0_1/<event_name>/all_pools.csv
cluster_plots_0_0_1/<event_name>/...
```

处理已存在 `all_pools.csv` 的事件时，流水线会跳过该事件。若先验文本映射缺失，程序会终止并提示对应事件。

### Step 3：添加标签和累计时间特征

```bash
python add_event_labels.py
python batch_enhance_all_pools.py
```

输出结果会包含 `event_label`，以及供训练使用的 `hourly_total_history` 和 `hourly_malicious_history` 两个累计时间序列特征。

### Step 4：训练与测试 ECTS

```bash
python training/train_early.py
```

训练脚本默认使用随机种子 42、最多 72 个小时、事件级 80/10/10 划分、批大小 32、训练 20 个 epoch，并使用 `models/bert-base-chinese` 作为冻结的中文 BERT 编码器。训练产物默认写入 `training_logs/`、`early_ckpt/best.pt`、`test_predictions.csv` 和 `earliness_accuracy_curve.png`。

## 6. 评价指标

项目同时关注分类质量和检测时效性：

- Accuracy、Precision、Recall、F1：时间步级别的分类指标；
- Event-level Accuracy/Precision/Recall/F1：聚合到事件级别的指标；
- Earliness：模型做出有效判断时所处的相对时间位置；
- Harmonic Mean：综合 F1 与早期检测能力的指标；
- Threshold 分析：通过停止概率和阈值扫描生成准确率—及时性曲线。

训练日志会记录验证集和测试集指标，详细预测结果保存在 `test_predictions.csv` 中。

## 7. 实验注意事项

1. 训练集、验证集和测试集按事件划分，避免同一事件的时间片跨集合泄漏。
2. 预处理输出目录、缓存和 checkpoint 是运行时生成文件，建议加入本地 `.gitignore`，不要将大模型权重或实验产物提交到仓库。
3. `pipeline.py`、`add_event_labels.py` 和 `batch_enhance_all_pools.py` 使用相对路径；修改执行目录后应同步检查脚本中的路径配置。
4. 复现实验时请记录 Python、PyTorch、CUDA、GPU、随机种子、数据版本、模型版本和关键超参数。
5. 当前代码默认从 `models/bert-base-chinese` 读取 BERT 权重；该目录需要提前准备，或在训练脚本中改为可访问的模型路径。

## 8. 局限性

当前实现依赖小时级时间戳、事件先验文本和预训练模型，数据质量、事件划分方式及阈值选择都会影响结果。仓库尚未提供统一的公开基准结果表、完整配置文件和自动化端到端评测脚本，因此论文或报告中的结果应附带数据版本和运行配置。

## 9. 引用

如果本项目用于论文、报告或其他研究工作，请补充正式论文信息后引用：

```bibtex
@software{dstd_cid,
  title  = {DSTD-CID: Dynamic Semantic Textual Dynamics for Early Chinese Cyberbullying Identification},
  author = {Machoman6},
  url    = {https://github.com/Machoman6/DSTD-CID},
  year   = {2026}
}
```

## 10. License

本仓库当前未提供独立的 LICENSE 文件。使用、再分发代码及数据前，请先确认仓库所有者、数据提供方和预训练模型的许可条款。
