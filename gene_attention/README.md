# stDiff gene-attention 使用指引

`MyCode/training.py --method stdiff` 默认使用 gene attention。数据读取、预处理、
gene fold、噪声预测目标和反向扩散采样沿用现有流程。

## 改动内容

- 原实现的 attention 发生在 batch/spot 维度。
- 本实现把每个 gene 作为一个 token，attention logits 的形状为
  `[batch, heads, genes, genes]`。
- 训练把架构写入 `experiment.json` 和每折 checkpoint。推理按 checkpoint 权重
  及记录的架构加载模型，旧版未标记权重可按状态字典识别。
- 原版对照可显式指定 `--architecture batch_attention` 并使用独立的输出目录。

## 1. Gene-attention 版本

从工作区根目录运行以下命令。gene attention 的默认训练配置为
`batch_size=8, hidden_size=256`。

```bash
python MyCode/training.py \
  --method stdiff \
  --dataset dataset5 \
  --data-root /Dataset/yezhong/ai4bio/stDiff_dataset \
  --output-dir ./gene_attention_outputs/data5/ \
  --batch-size 8 \
  --hidden-size 256 \
  --heads 16 \
  --steps 1500 \
  --epochs 900 \
  --learning-rate 0.00016046744893538737 \
  --noise-std 10 \
  --device cuda:0 \
  --seed 42

python MyCode/inference.py \
  --checkpoint-dir ./gene_attention_outputs/data5/checkpoints \
  --output-dir ./gene_attention_outputs/data5/inference \
  --device cuda:0 \
  --seed 42 \
  --batch-size 8
```

## 2. 官方 batch-attention 使用的参数

这里仅记录原始实现参数，不要求额外运行 `batch_size=1` 对照。官方 README 按
gene 数量选择训练配置：

| gene 数量 | batch size | hidden size |
|---:|---:|---:|
| `< 512` | 2048 | 512 |
| `512 < G < 1024` | 512 | 1024 |
| `1024 < G < 2048` | 512 | 2048 |

因此仓库中的两个数据集通常使用：

| 数据集 | gene 数 | 训练 batch size | 推理 batch size | hidden size |
|---|---:|---:|---:|---:|
| dataset2 | 33 | 2048 | 2048 | 512 |
| dataset5 | 915 | 512 | 512 | 1024 |

`--architecture batch_attention` 的默认值是 `batch_size=512, hidden_size=1024`，因此
dataset5 不额外指定时与上表一致；dataset2 若要复现官方配置，需要显式传入
`--batch-size 2048 --hidden-size 512`。

原始 `stDiff_code/test-stDiff.py` 在训练 DataLoader 和推理 DataLoader 中复用同一个
`--batch_size`，并不是分别调两个 batch size。其余固定参数为：

| 参数 | 值 |
|---|---:|
| depth | 6 |
| heads | 16 |
| MLP ratio | 4.0 |
| diffusion steps | 1500 |
| epochs | 900 |
| learning rate | 0.00016046744893538737 |
| augmentation noise std | 10 |
| gene folds | 5 |
| fold random state | 0 |
| per-fold model seed | 1202 |
| diffusion beta schedule | cosine |
| prediction target | noise |

送入 batch-attention 模型的三个参数在两个阶段分别是：

| 阶段 | `x` | `t` | `y`（condition） |
|---|---|---|---|
| 训练 | 加噪后的 scRNA 表达，已知 gene 再用真值回填 | 每个 sc cell 随机采样的 diffusion timestep | mask 后的 scRNA 表达 |
| 推理 | 当前 spatial reverse-diffusion 状态，已知 gene 每步用真值回填 | 当前 reverse timestep | mask 后的 spatial 表达 |

因此原模型训练时 attention 混合的是同一 batch 中的 sc cells，推理时混合的是同一
batch 中的 spatial spots；batch 的大小和分组成员成为模型语义的一部分。单个 batch
内部的置换本身不改变 self-attention，但改变整个数据集的行顺序可能改变 batch
边界和成员组合，从而改变结果。

训练 DataLoader 使用 `shuffle=True`，推理 DataLoader 使用 `shuffle=False`。由于
batch-attention 会让同一 batch 内的样本互相影响，忠实复现时推理必须沿用该模型
训练时的 batch size 和固定 spot 顺序。在迁移后的 `MyCode/inference.py` 中，不传
`--batch-size` 会自动读取 checkpoint 对应 `experiment.json` 中保存的训练 batch
size；也可以显式传入相同数值。

Gene-attention 在每个 spot 内计算，因此训练和推理的 batch size 可以分别按显存选取。

## 3. 输出文件

训练目录包含：

- `checkpoints/fold_*.pt`：各 gene fold 的模型权重；
- `experiment.json`：模型和训练参数；
- `gene_folds.json`：已知/未知 gene 划分；
- `train_history.csv`：训练损失。

推理目录包含：

- `prediction.csv`；
- `prediction.h5ad`；
- `inference.json`。

Gene attention 的显存复杂度随 gene 数量近似按 `O(G^2)` 增长。若完整配置显存
不足，可以先用较小的 `--hidden-size` 和 `--heads` 做连通性测试。正式实验应完整
报告两种架构各自使用的配置；不能仅凭相同 batch size 把两者解释成严格的单变量
attention 消融。

## 4. 结果分析需要提供的文件

若比较已经完成的 batch-attention 与 gene-attention 结果，每个实验目录提供：

- `experiment.json`；
- `data_info.json`；
- `gene_folds.json`；
- `train_history.csv`；
- `inference/prediction.csv`；
- `inference/inference.json`。

另外提供一份共同的空间转录组 ground truth：原始 spatial `.h5ad`，或按对应实验记录的
预处理协议处理后、spot 和 gene 名称完整的 `ground_truth.csv`。
若提供 `prediction.csv`，无需再提供内容重复的
`prediction.h5ad`。

只有在需要分析 attention、参数或具体样本的中间响应时，才需要额外提供
`checkpoints/fold_*.pt`。若要比较运行成本，还应一并提供训练总时长和峰值 GPU
显存；若要做统计检验，则每种模型最好提供多个相同 seed 集合的独立重复实验。
