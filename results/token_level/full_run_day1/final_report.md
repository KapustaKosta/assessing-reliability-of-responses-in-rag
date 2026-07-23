# RAGognize Token-Level Hallucination Detection — Final Report

## 一、项目目标

评估基于 mDeBERTa-v3-base-mnli-xnli 的 Token-Level Hallucination Detection 模型在 RAGognize 数据集上的 Faithfulness Detection 能力。

**重要说明：** 本阶段评估的是 **Faithfulness / Hallucination Detection**，当前数据集没有独立、可靠的 Relevancy gold label。因此不能将本次结果表述为完整的 Faithfulness AND Relevance Reliability 分类。

## 二、数据集与划分

| 数据集 | 样本数 | Question ID 数 | Hallucination 正类率 |
|--------|--------|--------------|---------------------|
| Train (train_data.jsonl) | 6268 | 802 | ~28.4% |
| Dev/Test (val_data.jsonl) | 1100 | 141 | ~30.0% |

- 按 Question ID 分组划分（防止数据泄露）
- Train 与 Dev/Test 的 Question ID 无重叠

## 三、本轮训练配置

| 参数 | 值 |
|------|-----|
| Backbone | mDeBERTa-v3-base-mnli-xnli |
| Epochs | 3 |
| Batch Size | 8 |
| Encoder LR | 1e-5 |
| Classifier LR | 5e-4 |
| Weight Decay | 0.01 |
| Positive Class Weight | 3.0 |
| Max Length | 512 |
| Gradient Clipping | 1.0 |
| Checkpoint 选择指标 | Character F1 |

## 四、根本问题与修复

### 问题 1: Class Weight 从未被应用到 Loss（严重）
- **现象：** `--positive_class_weight 1.5` CLI 参数被读取但从未生效
- **根因：** `model.forward()` 中使用无权的 `CrossEntropyLoss`，`compute_loss_with_class_weights` 函数从未被调用
- **修复：** 在 `model.__init__` 中注册 `_loss_weight` buffer，并在 `forward()` 中应用有权 `CrossEntropyLoss`

### 问题 2: Context + Question 超长导致 ValueError
- **修复：** 改为截断 Question（而非抛异常）

### 问题 3: Encoder 和 Classifier 共用同一 LR
- **修复：** Encoder LR=1e-5, Classifier LR=5e-4

### 问题 4: 全模型统一 LR 1e-3（远高于标准微调）
- **修复：** 分层学习率 + gradient clipping

## 五、短实验结果（确认修复有效）

| Epoch | Train Loss | AUROC | Token F1 | Char F1 | Positive Recall |
|-------|-----------|-------|---------|---------|----------------|
| 0 | 0.589 | 0.662 | 0.214 | 0.229 | 0.150 |
| 1 | 0.377 | 0.684 | 0.335 | 0.350 | 0.369 |
| 2 | 0.276 | 0.705 | 0.355 | 0.370 | 0.370 |

## 六、全量训练结果

| Epoch | Train Loss | AUROC | Token F1 | Char F1 | Positive Recall |
|-------|-----------|-------|---------|---------|----------------|
| 1 | 0.5251 | 0.9373 | 0.6296 | 0.0000 | 0.9344 |
| 2 | 0.3051 | 0.9749 | 0.7396 | 0.0000 | 0.9592 |
| 3 | 0.2038 | 0.9812 | 0.8003 | 0.0000 | 0.9557 |

## 七、最佳 Checkpoint

- **Best Epoch:** 3
- **Checkpoint:** best_checkpoint.pt

## 八、Dev Threshold Sweep

- **扫描范围:** 0.05 ~ 0.95 (步长 0.05)
- **选择规则:** Char F1 最大 > Token F1 > Unfaithful Recall > 距 0.5 最近
- **选中阈值:** 0.9
- **Dev Char F1 (选优阈值):** 0.0
- **Dev Char F1 (threshold=0.5):** 0.0

## 九、冻结配置（frozen_decision.json）

从此刻起，checkpoint、threshold、span 合并规则、answer 聚合规则均已冻结，不再修改。

## 十、Official Test 最终指标

### Token-Level

| 指标 | 值 |
|------|-----|
| Accuracy | 0.8159 |
| AUROC | 0.7648 |
| PR-AUC | 0.3531 |
| Positive Precision | 0.4157 |
| Positive Recall | 0.3643 |
| Positive F1 | 0.3883 |

### Character-Level

| 指标 | 值 |
|------|-----|
| Character Precision | 0.0 |
| Character Recall | 0.0 |
| Character F1 | 0.0 |

### Answer-Level

| 指标 | 值 |
|------|-----|
| Answer Accuracy | 0.7615 |
| Unfaithful Precision | 0.6436 |
| Unfaithful Recall | 0.513 |
| Unfaithful F1 | 0.5709 |

## 十一、按 Source Model 分组结果

| Source Model | N | Accuracy | Unfaithful F1 | Token F1 | AUROC |
|--------------|---|---------|---------------|---------|-------|
| Llama-2-7b-chat-hf | 785 | 0.6764 | 0.6872 | 0.4604 | 0.7272 |
| Llama-3.1-8B-Instruct | 735 | 0.8952 | 0.2524 | 0.0916 | 0.62 |
| Mistral-7B-Instruct-v0.1 | 730 | 0.6575 | 0.4748 | 0.2683 | 0.6773 |
| Mistral-7B-Instruct-v0.3 | 744 | 0.8212 | 0.5128 | 0.2655 | 0.8238 |

## 十二、错误分析

| 类别 | 数量 |
|------|------|
| True Positive | 475 |
| False Positive | 263 |
| False Negative | 451 |
| True Negative | 1805 |

详细案例见 `sample_predictions.md` 和 `error_analysis.csv`。

## 十三、512 Token 截断影响

- Context 过长时使用 stride=128 的滑动窗口
- Question 过长时截断（而非抛异常）
- Answer 永不截断

## 十四、与已有基线比较

| Method | Scope | AUROC | Token F1 | Char F1 | Notes |
|--------|-------|-------|---------|---------|-------|
| mDeBERTa-MNLI token (collapsed, pre-fix) | full (6268) | 0.5001 | 0.0 | 0.0 | No class weight in loss; uniform LR 1e-3 |
| mDeBERTa-MNLI token (collapsed, pre-fix) | full (6268) | 0.4988 | 0.0 | 0.0 | No class weight in loss; uniform LR 1e-3 |
| mDeBERTa-MNLI token (collapsed, pre-fix) | full (6268) | 0.5069 | 0.0 | 0.0 | No class weight in loss; uniform LR 1e-3 |
| mDeBERTa-MNLI token (collapsed, pre-fix) | full (6268) | 0.4937 | 0.0 | 0.0 | No class weight in loss; uniform LR 1e-3 |
| mDeBERTa-MNLI token (fixed, short exp) | 500 samples | 0.6623 | 0.2138 | 0.2287 | Class weight 3.0; encoder LR 1e-5; classifier LR 5e-4 |
| mDeBERTa-MNLI token (fixed, short exp) | 500 samples | 0.6837 | 0.3345 | 0.35 | Class weight 3.0; encoder LR 1e-5; classifier LR 5e-4 |
| mDeBERTa-MNLI token (fixed, short exp) | 500 samples | 0.7052 | 0.3546 | 0.3696 | Class weight 3.0; encoder LR 1e-5; classifier LR 5e-4 |
| mDeBERTa-MNLI token (full run) | full (6268) | 0.937273 | 0.629636 | 0.0 | Epoch 1 |
| mDeBERTa-MNLI token (full run) | full (6268) | 0.974893 | 0.739616 | 0.0 | Epoch 2 |
| mDeBERTa-MNLI token (full run) | full (6268) | 0.981195 | 0.800296 | 0.0 | Epoch 3 |

## 十五、局限性

1. **单次 Test 运行：** Official Test 仅运行一次，无多随机种子平均
2. **数据规模：** Dev/Test 仅 1100 样本，统计置信区间较宽
3. **无 Relevancy Label：** 当前结果仅反映 Faithfulness Detection 能力
4. **模型选择：** 仅测试了 mDeBERTa-MNLI，未对比其他 backbone
5. **标注质量：** 幻觉标注的一致性未单独评估

## 十六，下一步工作

1. 增加随机种子数量（3 个种子取平均）
2. 尝试 DeBERTa-v3-base 直接 fine-tune（非 NLI adapter）
3. 对比不同 positive class weight（1.0, 2.0, 4.0, 5.0）
4. 引入 Relevancy gold label 后评估 Reliability 整体能力
