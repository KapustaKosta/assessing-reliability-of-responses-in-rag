from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_SEED = 42
CHUNK_COLUMNS = [f"chunk_{i}" for i in range(1, 9)]
LABEL_COLUMNS = ["binary_relevancy", "binary_faithfulness"]


# ============================================================
# Text normalization and stable identifiers
# 文本标准化与稳定标识符
# ============================================================


def normalize_text(value: Any, *, lowercase: bool = False) -> str:
    """
    Normalize Unicode, non-breaking spaces, and repeated whitespace.

    统一 Unicode、不可分割空格以及重复空白字符。
    """

    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def stable_hash(parts: list[str], *, prefix: str, length: int = 20) -> str:
    """
    Create a deterministic SHA-256 identifier from several text fields.

    根据多个文本字段创建确定性的 SHA-256 标识符。
    """

    payload = "\u241e".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return f"{prefix}_{digest}"


# ============================================================
# Label validation
# 标签验证
# ============================================================


def parse_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    """
    Convert common Boolean representations to a strict Boolean dtype.

    将常见的布尔表示形式转换为严格的布尔类型。
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }

    parsed = series.map(mapping)

    if parsed.isna().any():
        unknown_values = sorted(series.loc[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Unknown values in '{column_name}': {unknown_values}. "
            f"/ 列“{column_name}”中存在未知值：{unknown_values}。"
        )

    return parsed.astype(bool)


# ============================================================
# Client-request extraction
# 客户请求提取
# ============================================================


# The expressions below are intentionally conservative. A row is quarantined
# only when every client turn looks like a short request for a human agent.
# 以下表达式有意采用保守策略。只有当所有客户消息都像是简短的转人工请求时，
# 该行才会被隔离。
ROLE_RE = re.compile(
    r"\b(?:"
    r"оператор\w*|"
    r"сотрудник\w*|"
    r"специалист\w*|"
    r"менеджер\w*|"
    r"жив\w*\s+человек\w*|"
    r"человек\w*"
    r")\b",
    re.IGNORECASE,
)

HANDOFF_ACTION_RE = re.compile(
    r"\b(?:"
    r"соедин\w*|соеден\w*|"
    r"связ\w*|свяж\w*|"
    r"переве\w*|перевод\w*|"
    r"переключ\w*|подключ\w*|"
    r"позв\w*|позов\w*|"
    r"приглас\w*|"
    r"выз\w*|зов\w*|"
    r"дайте|дай|"
    r"нуж\w*|"
    r"помощ\w*|"
    r"чат\w*|"
    r"поговор\w*|пообщ\w*|"
    r"ответ\w*|вопрос\w*|"
    r"можно|хочу|есть|подожд\w*|"
    r"заказ\w*\s+звон\w*|звон\w*"
    r")\b",
    re.IGNORECASE,
)

GREETING_RE = re.compile(
    r"^(?:(?:здравствуйте|добрый\s+(?:день|вечер)|привет)[,!\s.\-]*)+",
    re.IGNORECASE,
)

ROLE_ONLY_RE = re.compile(
    r"^(?:(?:срочно|просто|только|мне|пожалуйста|а|ну|бы)\s+)*"
    r"(?:(?:оператор\w*|сотрудник\w*|специалист\w*|менеджер\w*|"
    r"жив\w*\s+человек\w*|человек\w*)\s*)+"
    r"[.!?\s)]*$",
    re.IGNORECASE,
)

CLIENT_SPLIT_RE = re.compile(r"Клиент:\s*", re.IGNORECASE)
NEXT_ROLE_RE = re.compile(r"(?:Ассистент|Оператор):", re.IGNORECASE)


def extract_client_turns(dialog: Any) -> list[str]:
    """
    Extract all client turns from a role-labelled dialogue.

    从带角色标签的对话中提取所有客户消息。
    """

    if not isinstance(dialog, str):
        return []

    turns: list[str] = []

    for part in CLIENT_SPLIT_RE.split(dialog)[1:]:
        turn = NEXT_ROLE_RE.split(part, maxsplit=1)[0].strip()
        if turn:
            turns.append(turn)

    return turns


def is_handoff_only(text: str) -> bool:
    """
    Detect a short message whose only intent is transfer to a human agent.

    检测意图仅为转接人工客服的简短消息。
    """

    normalized = normalize_text(text, lowercase=True)
    normalized = GREETING_RE.sub("", normalized).strip(" ,.!?—-)")

    if not normalized:
        return False

    if ROLE_ONLY_RE.fullmatch(normalized):
        return True

    # Longer messages are treated as substantive to avoid removing real
    # banking questions that merely mention an operator or manager.
    # 较长消息被视为有实际内容，以避免删除仅仅提到客服或经理的真实银行问题。
    if len(normalized.split()) > 16:
        return False

    return bool(ROLE_RE.search(normalized) and HANDOFF_ACTION_RE.search(normalized))


def extract_last_substantive_request(dialog: Any) -> str | None:
    """
    Return the last client turn that is not a handoff-only request.

    返回最后一条并非仅用于转人工的客户消息。
    """

    for turn in reversed(extract_client_turns(dialog)):
        if not is_handoff_only(turn):
            return turn

    return None


# ============================================================
# Split optimization helpers
# 数据划分优化辅助函数
# ============================================================


def normalized_distribution(series: pd.Series, categories: list[Any]) -> np.ndarray:
    """Return a normalized frequency vector over fixed categories. / 返回固定类别上的归一化频率向量。"""

    return (
        series.value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
        .to_numpy(dtype=float)
    )


def choose_validation_and_test_folds(df: pd.DataFrame) -> tuple[int, int, float]:
    """
    Select two of seven folds as validation and test folds while balancing
    dataset size, joint labels, and retrieval configuration.

    从七个折中选择两个作为验证集和测试集，同时平衡数据规模、联合标签以及检索配置。
    """

    target_sizes = {"train": 0.70, "val": 0.15, "test": 0.15}
    label_categories = sorted(df["joint_label"].unique().tolist())
    chunk_categories = sorted(df["chunk_count"].unique().tolist())

    global_label_dist = normalized_distribution(df["joint_label"], label_categories)
    global_chunk_dist = normalized_distribution(df["chunk_count"], chunk_categories)

    best_pair: tuple[int, int] | None = None
    best_score = float("inf")

    fold_values = sorted(df["fold"].unique().tolist())

    for val_fold, test_fold in permutations(fold_values, 2):
        candidate_split = np.where(
            df["fold"].eq(val_fold),
            "val",
            np.where(df["fold"].eq(test_fold), "test", "train"),
        )

        score = 0.0

        for split_name in ("train", "val", "test"):
            mask = candidate_split == split_name
            part = df.loc[mask]

            size_error = (len(part) / len(df) - target_sizes[split_name]) ** 2
            label_error = np.square(
                normalized_distribution(part["joint_label"], label_categories)
                - global_label_dist
            ).sum()
            chunk_error = np.square(
                normalized_distribution(part["chunk_count"], chunk_categories)
                - global_chunk_dist
            ).sum()

            # Joint-label balance is most important. Chunk-count balance is
            # included because the dataset contains top-5 and top-8 retrieval runs.
            # 联合标签平衡最重要。由于数据包含 top-5 和 top-8 两种检索配置，
            # 因此也考虑文本块数量的平衡。
            score += 8.0 * size_error + 4.0 * label_error + 1.5 * chunk_error

        if score < best_score:
            best_score = score
            best_pair = (int(val_fold), int(test_fold))

    if best_pair is None:
        raise RuntimeError(
            "Unable to choose validation and test folds. "
            "/ 无法选择验证折和测试折。"
        )

    return best_pair[0], best_pair[1], best_score


# ============================================================
# Main preparation pipeline
# 主数据准备流程
# ============================================================


def prepare_dataset(input_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    """Run the complete first-stage data preparation pipeline. / 运行完整的第一阶段数据准备流程。"""

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path).reset_index().rename(columns={"index": "source_row"})
    df["source_csv_row"] = df["source_row"] + 2

    required_columns = {
        "full_dialog",
        "answer",
        "markers",
        *CHUNK_COLUMNS,
        *LABEL_COLUMNS,
    }
    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"/ 缺少必需列：{missing_columns}。"
        )

    initial_rows = len(df)

    for column in LABEL_COLUMNS:
        df[column] = parse_boolean_series(df[column], column)

    # Reliability is the conjunction of relevancy and faithfulness.
    # 可靠性是相关性与忠实性的逻辑与。
    df["binary_reliability"] = (
        df["binary_relevancy"] & df["binary_faithfulness"]
    ).astype(int)

    df["chunk_count"] = df[CHUNK_COLUMNS].notna().sum(axis=1).astype(int)
    df["retrieval_config"] = df["chunk_count"].map(lambda n: f"top_{n}")
    df["has_marker"] = df["markers"].fillna("").str.strip().ne("")
    df["mask_count"] = df[["full_dialog", "answer", *CHUNK_COLUMNS]].fillna("").apply(
        lambda column: column.str.count(r"\[[A-ZА-Я][A-ZА-Я0-9_]*\]")
    ).sum(axis=1).astype(int)

    df["question"] = df["full_dialog"].map(extract_last_substantive_request)
    df["has_question"] = df["question"].notna() & df["question"].fillna("").str.strip().ne("")

    # Build normalized hashes for duplicate detection and grouped splitting.
    # 构建标准化哈希，用于重复检测和分组划分。
    input_hashes: list[str] = []
    dialog_hashes: list[str] = []

    for row in df.itertuples(index=False):
        full_input_parts = [
            normalize_text(getattr(row, "full_dialog")),
            normalize_text(getattr(row, "answer")),
            *[normalize_text(getattr(row, column)) for column in CHUNK_COLUMNS],
        ]
        input_hashes.append(stable_hash(full_input_parts, prefix="input"))
        dialog_hashes.append(
            stable_hash(
                [normalize_text(getattr(row, "full_dialog"), lowercase=True)],
                prefix="dialog",
            )
        )

    df["input_hash"] = input_hashes
    df["dialog_group_id"] = dialog_hashes

    # Prefer a duplicate row with a more informative marker.
    # 对于重复行，优先保留错误标记信息更丰富的记录。
    df["_marker_length"] = df["markers"].fillna("").str.len()
    df = df.sort_values(["_marker_length", "source_row"], ascending=[False, True])

    exact_duplicate_mask = df.duplicated(
        subset=["input_hash", *LABEL_COLUMNS],
        keep="first",
    )
    exact_duplicates = df.loc[exact_duplicate_mask].copy()
    df = df.loc[~exact_duplicate_mask].copy()

    # After exact duplicates are removed, repeated input hashes indicate
    # contradictory binary annotations. All conflicting rows are quarantined.
    # 删除完全重复项后，重复的输入哈希表示二元标注相互矛盾。
    # 所有冲突行都会被隔离。
    conflict_mask = df.duplicated(subset=["input_hash"], keep=False)
    conflicts = df.loc[conflict_mask].copy()
    df = df.loc[~conflict_mask].copy()

    df = df.sort_values("source_row").drop(columns=["_marker_length"]).reset_index(drop=True)
    exact_duplicates = exact_duplicates.drop(columns=["_marker_length"], errors="ignore")
    conflicts = conflicts.drop(columns=["_marker_length"], errors="ignore")

    # Rows without a substantive request remain available in quarantine but are
    # excluded from the main relevance/reliability benchmark.
    # 缺少实质性请求的行会保留在隔离文件中，但不进入主要相关性/可靠性基准。
    no_question = df.loc[~df["has_question"]].copy()
    main_df = df.loc[df["has_question"]].copy().reset_index(drop=True)

    main_df["case_id"] = main_df["input_hash"].str.replace("input_", "case_", regex=False)

    if main_df["case_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate case identifiers remain after cleaning. "
            "/ 清洗后仍存在重复的样本标识符。"
        )

    main_df["joint_label"] = (
        main_df["binary_relevancy"].astype(int).astype(str)
        + "_"
        + main_df["binary_faithfulness"].astype(int).astype(str)
    )

    # Seven grouped folds produce an approximately 71/14/14 split. We then
    # choose validation and test folds that best match the requested 70/15/15
    # proportions and preserve label/configuration distributions.
    # 七个分组折得到约 71/14/14 的划分。随后选择最接近 70/15/15 且能保持
    # 标签与检索配置分布的验证折和测试折。
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    fold_ids = np.full(len(main_df), -1, dtype=int)

    for fold_id, (_, held_out_indices) in enumerate(
        splitter.split(
            X=main_df,
            y=main_df["joint_label"],
            groups=main_df["dialog_group_id"],
        )
    ):
        fold_ids[held_out_indices] = fold_id

    if (fold_ids < 0).any():
        raise RuntimeError(
            "Some rows were not assigned to a fold. "
            "/ 部分行未被分配到任何折。"
        )

    main_df["fold"] = fold_ids
    val_fold, test_fold, split_score = choose_validation_and_test_folds(main_df)
    main_df["split"] = np.where(
        main_df["fold"].eq(val_fold),
        "val",
        np.where(main_df["fold"].eq(test_fold), "test", "train"),
    )

    # Leakage checks: every dialogue group must belong to exactly one split.
    # 泄漏检查：每个对话组必须只属于一个数据划分。
    group_split_counts = main_df.groupby("dialog_group_id")["split"].nunique()
    if not group_split_counts.eq(1).all():
        raise RuntimeError(
            "Dialogue-group leakage was detected across splits. "
            "/ 检测到对话组在不同数据划分之间发生泄漏。"
        )

    if main_df.groupby("case_id")["split"].nunique().max() != 1:
        raise RuntimeError(
            "Case leakage was detected across splits. "
            "/ 检测到样本在不同数据划分之间发生泄漏。"
        )

    # Save processed datasets.
    # 保存处理后的数据集。
    ordered_splits = ["train", "val", "test"]
    for split_name in ordered_splits:
        main_df.loc[main_df["split"].eq(split_name)].to_csv(
            output_dir / f"{split_name}.csv",
            index=False,
        )

    manifest_columns = [
        "case_id",
        "source_row",
        "source_csv_row",
        "dialog_group_id",
        "split",
        "fold",
        "binary_relevancy",
        "binary_faithfulness",
        "binary_reliability",
        "chunk_count",
        "retrieval_config",
        "has_marker",
        "mask_count",
    ]
    main_df[manifest_columns].to_csv(output_dir / "split_manifest.csv", index=False)

    no_question.to_csv(output_dir / "quarantine_no_question.csv", index=False)
    exact_duplicates.to_csv(output_dir / "removed_exact_duplicates.csv", index=False)
    conflicts.to_csv(output_dir / "quarantine_label_conflicts.csv", index=False)

    # Build a compact split summary for review.
    # 构建简洁的数据划分摘要以便审查。
    summary_rows: list[dict[str, Any]] = []

    for split_name in ["all", *ordered_splits]:
        part = main_df if split_name == "all" else main_df.loc[main_df["split"].eq(split_name)]
        combo_counts = part["joint_label"].value_counts().to_dict()
        chunk_counts = part["chunk_count"].value_counts().to_dict()

        summary_rows.append(
            {
                "split": split_name,
                "rows": len(part),
                "proportion": len(part) / len(main_df),
                "dialog_groups": part["dialog_group_id"].nunique(),
                "relevancy_positive_rate": float(part["binary_relevancy"].mean()),
                "faithfulness_positive_rate": float(part["binary_faithfulness"].mean()),
                "reliability_positive_rate": float(part["binary_reliability"].mean()),
                "label_0_0": int(combo_counts.get("0_0", 0)),
                "label_0_1": int(combo_counts.get("0_1", 0)),
                "label_1_0": int(combo_counts.get("1_0", 0)),
                "label_1_1": int(combo_counts.get("1_1", 0)),
                "top_5_rows": int(chunk_counts.get(5, 0)),
                "top_8_rows": int(chunk_counts.get(8, 0)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "split_summary.csv", index=False)

    report: dict[str, Any] = {
        "input_file": str(input_path.resolve()),
        "seed": seed,
        "initial_rows": initial_rows,
        "exact_duplicates_removed": int(len(exact_duplicates)),
        "conflicting_annotation_rows_quarantined": int(len(conflicts)),
        "rows_without_substantive_question_quarantined": int(len(no_question)),
        "main_rows": int(len(main_df)),
        "dialog_groups": int(main_df["dialog_group_id"].nunique()),
        "selected_validation_fold": int(val_fold),
        "selected_test_fold": int(test_fold),
        "split_optimization_score": float(split_score),
        "split_counts": {
            name: int(main_df["split"].eq(name).sum()) for name in ordered_splits
        },
        "group_leakage_detected": False,
        "notes": [
            "Different answers for the same full_dialog are retained.",
            "All rows sharing the same normalized full_dialog are assigned to one split.",
            "Rows without a substantive client request are quarantined rather than permanently discarded.",
            "Markers are not used as model inputs and are retained only as metadata/partial supervision.",
        ],
        "中文说明": [
            "同一 full_dialog 下的不同回答会被保留。",
            "具有相同标准化 full_dialog 的所有行会被分配到同一个数据划分。",
            "缺少实质性客户请求的行会被隔离，而不是永久删除。",
            "markers 不作为模型输入，仅作为元数据或部分辅助监督保留。",
        ],
    }

    with (output_dir / "data_quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(
        f"Initial rows / 原始行数: {initial_rows}\n"
        f"Exact duplicates removed / 删除的完全重复行数: {len(exact_duplicates)}\n"
        f"Conflicting rows quarantined / 隔离的冲突标注行数: {len(conflicts)}\n"
        f"No-question rows quarantined / 隔离的无实质问题行数: {len(no_question)}\n"
        f"Main benchmark rows / 主基准行数: {len(main_df)}\n"
    )
    print(summary_df.to_string(index=False))
    print(
        f"\nSaved outputs to / 输出已保存至: {output_dir.resolve()}"
    )

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed grouped train/validation/test splits for the RAG "
            "reliability dataset. / 为 RAG 可靠性数据集准备固定的分组训练、验证和测试划分。"
        )
    )
    parser.add_argument("--input", type=Path, default=Path("data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prepare_dataset(arguments.input, arguments.output_dir, arguments.seed)
