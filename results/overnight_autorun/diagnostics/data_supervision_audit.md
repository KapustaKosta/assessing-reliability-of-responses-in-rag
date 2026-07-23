# 数据监督粒度审计报告

## 审计日期
2026-07-21

## 数据来源
- processed/train.csv (1450 样本)
- processed/val.csv (290 样本)
- processed/test.csv (290 样本)

## 关键发现

### 1. 是否有字符级错误 Span？
**否。** 数据中不存在字符级 span 标注。

### 2. Markers 列分析
- 列名：markers
- 非空样本数：89/1450 (6.1%)
- 内容：类别标签列表，如：
  - reason_hallucinated_fact
  - reason_incomplete_answer
  - reason_false_verification
  - reason_reveals_ai_identity
  - reason_irrelevant_chunk_used
  - 等

### 3. 答案级标签分布
| 标签 | True/1 | False/0 |
|------|--------|---------|
| binary_faithfulness | 1059 | 391 |
| binary_relevancy | 1267 | 183 |
| binary_reliability | 1042 | 408 |

### 4. 是否有数据泄漏？
- train/dev/test 按 question_id 分组
- 无明显泄漏

### 5. 选择的路线
**Route B：无字符 span，使用 Answer-level Claim-MIL**

### 6. 理由
1. markers 只是类别标签，不是具体位置
2. 没有 start/end 字段
3. 无法进行 Token-level 真实监督
4. Token-level 模型只允许 synthetic Tiny Overfit

### 7. 推荐方案
使用 Answer-level Claim-MIL：
- Faithfulness: binary_faithfulness
- Relevancy: binary_relevancy
- Reliability: binary_reliability (或 Faithfulness AND Relevancy)

### 8. Tiny Overfit 结果
Token-level Tiny Overfit 已通过（Synthetic 数据）：
- Token F1: 1.0
- Span F1: 0.984
- Loss: 0.6203 → 0.0086

但这仅验证代码路径，不应用于真实数据。
