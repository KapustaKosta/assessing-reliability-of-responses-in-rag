# Overnight Autonomous Training Report

## 1. 执行时间与环境

- **开始时间**: 2026-07-21 23:44 CST
- **结束时间**: 2026-07-22 00:13 CST
- **总耗时**: 约 29 分钟
- **主机**: notebook-cedc4280-d123-4507-b11b-70db7fd27004
- **Python**: 3.11.10
- **NPU**: Ascend 910B4 (NPU 7), 可用
- **当前分支**: chengyi/claim-mil-correctness-fixes
- **Git commit**: ebcaab96

## 2. 开始前状态

- **是否有训练任务运行**: 是（之前的 Tiny Overfit 正在运行）
- **初始测试结果**:
  - Token-classifier 单元测试: 104 passed
  - Claim-MIL 测试: 已通过
- **初始 Git 状态**: dirty=True（有未提交的修改）
- **已知失败**:
  - Token-level Tiny Overfit span_f1=0（代码 bug）
  - 数据只有 answer-level markers，无字符级 span

## 3. 数据监督审计

### 3.1 是否有字符 Span？
**否。** 数据中不存在字符级 span 标注。

### 3.2 Markers 列分析
- 列名：markers
- 非空样本数：89/1450 (6.1%)
- 内容：类别标签列表，如 `reason_hallucinated_fact`、`reason_incomplete_answer` 等

### 3.3 使用了路线 A 还是路线 B？
**路线 B：无字符 span，使用 Answer-level 模型**

### 3.4 选择原因
1. markers 只是类别标签，不是具体位置
2. 没有 start/end 字段
3. 无法进行 Token-level 真实监督
4. Token-level 模型只允许 synthetic Tiny Overfit

### 3.5 是否存在数据泄漏？
无明显泄漏。train/dev/test 按 question_id 分组。

## 4. 修复内容

### 4.1 修复的文件

| 文件 | 修改原因 |
|------|----------|
| `src/token_classifier/train.py` | 修复 span_f1 bug，添加 `compute_sample_level_span_metrics` 函数调用 |
| `src/token_classifier/metrics.py` | 添加 `compute_sample_level_span_metrics` 函数用于聚合 span metrics |
| `src/token_classifier/dataset.py` | 在 `__getitem__` 和 `_collate_fn` 中添加 answer_text、gold_spans、answer_offsets |
| `src/claim_mil/simple_train.py` | 新建脚本用于本地 CSV 数据训练 |

### 4.2 主要 Bug 修复
1. **span_f1 赋值错误**: `span_f1 = eval_metrics.get("token_metrics", {}).get("positive_f1", 0)` → `span_f1 = eval_metrics.get("character_f1", 0)`
2. **span metrics 未计算**: evaluate 函数没有调用 span metrics 计算

## 5. Tiny Overfit 结果

### 5.1 Token-level Tiny Overfit (Synthetic)
- **初始 Loss**: 0.6203
- **最终 Loss**: 0.0086
- **Token F1**: 1.0 ✓
- **Span F1**: 0.984 ✓
- **Positive 预测数**: 90
- **Negative 预测数**: 255
- **Checkpoint**: 无（指标未改进）
- **Reload**: 未测试
- **验收**: ✓ 通过（Synthetic 数据）

### 5.2 说明
Token-level Tiny Overfit 在 synthetic 数据上通过，但**不适用于真实数据**（无字符 span）。

## 6. 实验汇总

| ID | 模型 | 标签 | Train | Dev | Macro F1 | Accuracy | Pos F1 | Neg F1 | 耗时 |
|----|------|------|-------|-----|----------|----------|---------|--------|------|
| exp_001 | SimpleMIL | Faithfulness | 300 | 100 | **0.5175** | 0.58 | 0.6912 | 0.3438 | ~4min |
| exp_002 | SimpleMIL | Relevancy | 400 | 150 | **0.5997** | 0.887 | 0.9386 | 0.2609 | ~5min |
| exp_003 | SimpleMIL | Reliability | 400 | 150 | **0.5816** | 0.707 | 0.8103 | 0.3529 | ~5min |

## 7. 最佳 Dev 模型

### 7.1 Relevancy 模型（最佳）
- **模型路线**: Route B - Answer-level SimpleMILClassifier
- **配置**:
  - encoder: mDeBERTa-v3-base-mnli-xnli (frozen feature extractor)
  - pooling: max pooling
  - dropout: 0.1
  - lr: 2e-5
  - epochs: 5
  - batch_size: 4
- **Primary Metric**: Dev Macro F1 = 0.5997
- **详细指标**:
  - Accuracy: 0.887
  - Positive Precision: 0.909
  - Positive Recall: 0.970
  - Positive F1: 0.939
  - Negative F1: 0.261
  - Confusion Matrix: [[3, 13], [4, 130]]
- **Checkpoint**: `results/overnight_autorun/experiments/claim_mil_relevancy_v1/best_model.pt`

### 7.2 Reliability 模型
- **Primary Metric**: Dev Macro F1 = 0.5816
- **详细指标**:
  - Accuracy: 0.707
  - Positive F1: 0.810
  - Negative F1: 0.353
- **Checkpoint**: `results/overnight_autorun/experiments/claim_mil_reliability_v1/best_model.pt`

### 7.3 Faithfulness 模型
- **Primary Metric**: Dev Macro F1 = 0.5175
- **详细指标**:
  - Accuracy: 0.58
  - Positive F1: 0.691
  - Negative F1: 0.344
- **Checkpoint**: `results/overnight_autorun/experiments/claim_mil_faithfulness_v1/best_model.pt`

## 8. 最终 Test

**未运行。** 根据规则，需要满足以下条件：
1. Tiny Overfit 在真实数据上通过 → 否（无字符 span）
2. 模型配置冻结 → 部分满足
3. threshold 在 dev 冻结 → 是（使用 0.5）
4. checkpoint reload 通过 → 未测试

由于 Token-level 真实监督不可行，Claim-MIL 模型配置尚未完全冻结。

## 9. 与 Baseline 比较

### 9.1 Majority Baseline
| 标签 | Majority Class | Accuracy |
|------|----------------|----------|
| Faithfulness | True (73%) | 0.73 |
| Relevancy | True (87%) | 0.87 |
| Reliability | 1 (72%) | 0.72 |

### 9.2 模型 vs Majority
| 标签 | Majority Acc | 模型 Acc | 提升 |
|------|-------------|----------|------|
| Faithfulness | 0.73 | 0.58 | **-0.15** |
| Relevancy | 0.87 | 0.887 | **+0.02** |
| Reliability | 0.72 | 0.707 | **-0.01** |

### 9.3 分析
- Relevancy 模型略高于 majority baseline
- Faithfulness 和 Reliability 模型低于 majority baseline
- Negative class 识别能力较弱（Negative F1 普遍较低）

## 10. 当前可信指标

### 10.1 Relevancy 模型（最佳）
| 指标 | 值 |
|------|-----|
| Accuracy | 0.887 |
| Macro F1 | 0.5997 |
| Positive F1 | 0.9386 |
| Negative F1 | 0.2609 |
| Positive Precision | 0.9091 |
| Positive Recall | 0.9701 |

### 10.2 Reliability 模型
| 指标 | 值 |
|------|-----|
| Accuracy | 0.707 |
| Macro F1 | 0.5816 |
| Positive F1 | 0.8103 |
| Negative F1 | 0.3529 |

### 10.3 Faithfulness 模型
| 指标 | 值 |
|------|-----|
| Accuracy | 0.58 |
| Macro F1 | 0.5175 |
| Positive F1 | 0.6912 |
| Negative F1 | 0.3438 |

## 11. 尚未解决的问题

### 11.1 技术问题
1. **Negative class 识别能力弱**: 所有模型的 Negative F1 都很低（0.26-0.35）
2. **Checkpoint 未保存**: Token-level Tiny Overfit 因指标未改进未保存 checkpoint
3. **Checkpoint reload 未验证**: 所有模型的 reload 未测试

### 11.2 数据问题
1. **无字符级标注**: 无法进行 Token-level 真实监督
2. **类别不平衡**: Relevancy 标签正类 87%，Faithfulness 正类 73%

### 11.3 计算资源问题
- 无

### 11.4 下一步建议
1. **优化 Negative class 识别**:
   - 调整 class weight
   - 使用 focal loss
   - 尝试 threshold 搜索
2. **扩大训练数据**: 当前只用了 300-400 样本，可扩大到全量
3. **实现完整的 Claim-MIL**: 使用 claim_bags 进行真正的 MIL 训练
4. **尝试融合模型**: Reliability = Faithfulness AND Relevancy

## 12. Git 状态

```
git status --short:
 M src/claim_mil/simple_train.py
 M src/token_classifier/dataset.py
 M src/token_classifier/metrics.py
 M src/token_classifier/train.py
```

**说明**: 没有执行 git commit 和 git push。

## 13. 输出文件

- 实验结果: `results/overnight_autorun/experiment_table.csv`
- 数据审计: `results/overnight_autorun/diagnostics/data_supervision_audit.json`
- Tiny Overfit: `results/overnight_autorun/experiments/token_tiny_v2/overfit_diagnostic.json`
- Faithfulness 模型: `results/overnight_autorun/experiments/claim_mil_faithfulness_v1/`
- Relevancy 模型: `results/overnight_autorun/experiments/claim_mil_relevancy_v1/`
- Reliability 模型: `results/overnight_autorun/experiments/claim_mil_reliability_v1/`
