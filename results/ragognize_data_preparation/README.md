# RAGognize 数据适配说明

## 概述

本模块提供 RAGognize 数据集 (`F4biian/RAGognize`) 的适配器，用于转换为统一的 `UnifiedSample` 格式，支持 Token 级别幻觉检测和 NLI 忠实度评估。

## 数据来源

- **来源**: HuggingFace `F4biian/RAGognize`
- **本地路径**:
  - Train: `data/raw/ragognize/data/train-00000-of-00001.parquet`
  - Test: `data/raw/ragognize/data/test-00000-of-00001.parquet`
- **离线模式**: 必须设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`

## 数据规模

| Split | 原始问题数 | 展开后样本数 |
|-------|-----------|-------------|
| Train (dev) | 802 | 6,268 |
| Validation | 141 | 1,100 |
| Test (official) | 1,416 | 11,124 |
| **总计** | 2,359 | 18,492 |

## 数据结构

### 顶层字段

| 字段 | 类型 | 说明 |
|-----|------|-----|
| `user_prompt_index` | int | 问题唯一索引 (可能重复) |
| `user_prompt` | string | 用户问题 |
| `answerable` | bool | 问题是否可回答 |
| `information_type` | string | 信息类型 |
| `category` | string | 分类 |
| `tags` | list[string] | 标签 |
| `documents` | list[dict] | 检索文档列表 |
| `responses` | dict | 各模型回答 |

### Responses 结构

每个模型的响应包含:

```python
{
    "text": str,  # 模型回答文本
    "hallucinations": [  # 顶层幻觉 span 列表
        {
            "text": str,      # 幻觉文本
            "start": int,      # 起始位置 (包含)
            "end": int,        # 结束位置 (不包含)
            "valid": bool      # 是否有效标注 (全部为 True)
        }
    ],
    "details": {
        "annotations": {
            "original_output": {
                "addressed_user_request": bool,  # 是否回答用户请求
                "cluelessness": bool,            # 是否无理解能力
                "completely_hallucinated": bool,  # 完全幻觉
                "partially_hallucinated": bool,   # 部分幻觉
                "hallucinations": [...],          # 详细幻觉分析
            }
        },
        "answerable": bool  # 模型认为是否可回答
    }
}
```

## Source Models

- `Llama-2-7b-chat-hf`
- `Llama-3.1-8B-Instruct`
- `Mistral-7B-Instruct-v0.1`
- `Mistral-7B-Instruct-v0.3`

**注意**: `golden_answer` 不作为 source model，包含在 responses 中但不会被适配器处理。

## UnifiedSample 字段映射

| UnifiedSample 字段 | 来源 |
|-------------------|-----|
| `case_id` | `{prompt_index}_{model}_{row_index}` 的 MD5 |
| `user_prompt_index` | 顶层 `user_prompt_index` |
| `question` | 顶层 `user_prompt` |
| `answer` | `responses[model]["text"]` |
| `chunks` | `documents[*]["text"]` |
| `chunk_titles` | `documents[*]["title"]` |
| `golden_answer` | `responses["golden_answer"]` |
| `hallucination_spans` | `responses[model]["hallucinations"]` |
| `has_hallucination` | 1=有有效幻觉, 0=无 |
| `faithfulness_label` | 1=忠实, 0=不忠实 |
| `answerable` | 顶层 `answerable` |
| `source_model` | 模型名称 |

## Faithfulness 标签定义

- `faithfulness_label = 1`: 忠实 (faithful) - 无有效幻觉
- `faithfulness_label = 0`: 不忠实 (unfaithful) - 存在有效幻觉

**判断规则**:
1. 优先使用 `hallucination_spans` 中的有效 spans
2. 同时考虑 `completely_hallucinated = True` 的情况
3. 所有 span 的 `valid` 字段均为 `True`

### Faithfulness 分布

| Split | Faithful | Unfaithful | Rate |
|-------|----------|------------|------|
| Train | 4,467 | 1,801 | 71.3% |
| Val | 796 | 304 | 72.4% |
| Test | 7,930 | 3,194 | 71.3% |

## Relevancy 标签

**当前状态**: RAGognize 数据中未发现可靠的 Relevancy 标注。

- `details.answerable`: 模型自身认为是否可回答，与顶层 `answerable` 一致
- `annotations.original_output.addressed_user_request`: 需要进一步验证是否适合作为 Relevancy 标签

**建议**: Relevancy 作为后续独立建模任务，不在当前适配阶段生成。

## Span 验证结果

| 指标 | 值 |
|-----|-----|
| 分析的响应数 | 8,000 |
| 包含幻觉的响应 | 2,314 |
| 总 span 数 | 3,079 |
| 完全匹配率 | 100% |
| 区间类型 | 闭区间 [start, end) |

**重要发现**:
- 所有 span 的 `valid = True`
- `valid` 字段不用于区分验证状态
- 所有 span 的 `answer[start:end]` 与 `span["text"]` 完全匹配

## Validation Split 防泄漏方法

1. **按问题分组**: 同一 `user_prompt_index` 的 4 个模型回答必须在同一 split
2. **随机分层**: 使用 `random_state=42`，验证集比例 15%
3. **无重叠**: train/val/test 的 `user_prompt_index` 集合互不相交
4. **官方 test 保持不变**: 不参与任何划分

## 与现有 NLI 框架的兼容性

`src/nli_faithfulness/` 模块存在以下旧字段依赖:

| 字段 | 位置 | 说明 |
|-----|------|-----|
| `chunk_1` - `chunk_8` | 多处 | 旧俄语数据的固定 chunk 数量 |
| `binary_faithfulness` | `data.py`, `evaluation.py` | 旧标签名 |
| `retrieval_config` | `inference.py` | 旧检索配置 |

**适配建议**:
- `UnifiedSample` 提供了 `chunks` (动态长度) 替代 `chunk_1`-`chunk_8`
- `faithfulness_label` 替代 `binary_faithfulness`
- 适配器可生成符合 NLI 模块输入格式的数据，无需修改 NLI 核心

## 使用方法

```python
from src.ragognize_adapter import (
    load_ragognize_dataset,
    create_prompt_split,
    apply_split,
)

# 1. 加载数据
dataset = load_ragognize_dataset()

# 2. 创建 split
split_info = create_prompt_split(dataset, val_ratio=0.15, seed=42)

# 3. 展开为 UnifiedSamples
expanded = apply_split(dataset, split_info)

# 4. 访问样本
for sample in expanded['train']:
    print(sample.case_id, sample.faithfulness_label)
```

## Smoke Test

运行以下命令验证适配器:

```bash
.venv/bin/python -c "
from src.ragognize_adapter import load_ragognize_dataset, create_prompt_split, apply_split
dataset = load_ragognize_dataset()
split_info = create_prompt_split(dataset)
expanded = apply_split(dataset, split_info)
print(f'Train: {len(expanded[\"train\"])} samples')
print(f'Val: {len(expanded[\"val\"])} samples')
print(f'Test: {len(expanded[\"test\"])} samples')
"
```

## 禁止提交的文件

以下文件/目录不应提交到 Git:
- `data/processed/*.csv` (如需缓存)
- `results/ragognize_data_preparation/` 下的缓存文件
- HuggingFace/模型 cache

## 当前阶段限制

- [ ] 未运行完整模型训练
- [ ] 未运行官方 test 评估
- [ ] 未生成展开后的完整 CSV 文件
- [ ] 未验证 NLI 模块适配

## 报告输出

- `results/ragognize_data_preparation/schema_summary.json`: 数据结构摘要
- `results/ragognize_data_preparation/span_validation_summary.json`: Span 验证结果
- `results/ragognize_data_preparation/split_summary.json`: Split 统计
- `results/ragognize_data_preparation/split_manifest.csv`: Split 清单
