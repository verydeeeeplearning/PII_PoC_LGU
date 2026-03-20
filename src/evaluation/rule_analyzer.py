"""Rule 기여도 분석 함수.

RuleLabeler.label_batch() 출력을 분석하여 각 룰의 히트율과 정밀도를 계산.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rule_contribution(
    rule_labels_df: pd.DataFrame,
    y_true,
    tp_label: str = "TP",
) -> pd.DataFrame:
    """룰별 히트 통계 및 정밀도 계산.

    Args:
        rule_labels_df : RuleLabeler.label_batch() 출력 DataFrame.
                         필수 컬럼: pk_event, rule_matched, rule_id, rule_primary_class
        y_true         : 실제 레이블 array (rule_labels_df와 동일 순서)
        tp_label       : TP 레이블 문자열

    Returns:
        DataFrame{rule_id, hit_count, hit_rate, precision, dominant_class}
        hit_count 내림차순 정렬. rule_matched=True인 행만 포함.
    """
    if rule_labels_df.empty or len(y_true) == 0:
        return pd.DataFrame(columns=["rule_id", "hit_count", "hit_rate", "precision", "dominant_class"])

    y_true_arr = np.asarray(y_true)
    total_rows = len(rule_labels_df)

    # rule_matched=True인 행만
    df = rule_labels_df.copy().reset_index(drop=True)
    df["_y_true"] = y_true_arr

    matched = df[df["rule_matched"] == True].copy()

    if matched.empty:
        return pd.DataFrame(columns=["rule_id", "hit_count", "hit_rate", "precision", "dominant_class"])

    records = []
    for rule_id, grp in matched.groupby("rule_id", sort=False):
        hit_count = len(grp)
        hit_rate = hit_count / total_rows

        # precision: 그룹 내 실제 FP(비TP) 비율
        n_fp = int((grp["_y_true"] != tp_label).sum())
        precision = n_fp / hit_count if hit_count > 0 else 0.0

        # dominant_class: 가장 많이 예측된 rule_primary_class
        if "rule_primary_class" in grp.columns:
            dominant_class = grp["rule_primary_class"].mode().iloc[0] if len(grp) > 0 else None
        else:
            dominant_class = None

        records.append({
            "rule_id": rule_id,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "precision": precision,
            "dominant_class": dominant_class,
        })

    result = pd.DataFrame(records).sort_values("hit_count", ascending=False).reset_index(drop=True)
    return result


def compute_rule_vs_ml_coverage(
    rule_labels_df: pd.DataFrame,
    y_pred_ml: np.ndarray,
    y_true: np.ndarray,
    tp_label: str = "TP",
) -> dict:
    """Rule 단독 Coverage vs ML 총 Coverage 비교.

    Coverage 정의: 실제 FP 중 자동으로 FP 판정된 비율.

    Args:
        rule_labels_df : RuleLabeler.label_batch() 출력 DataFrame
        y_pred_ml      : ML 예측 레이블 array (rule_labels_df와 동일 순서)
        y_true         : 실제 레이블 array
        tp_label       : TP 레이블 문자열

    Returns:
        {rule_only_coverage, ml_total_coverage,
         ml_additional_coverage, overlap_count, overlap_rate}
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred_ml)
    n_actual_fp = int((y_true_arr != tp_label).sum())

    if n_actual_fp == 0:
        return {
            "rule_only_coverage": 0.0,
            "ml_total_coverage": 0.0,
            "ml_additional_coverage": 0.0,
            "overlap_count": 0,
            "overlap_rate": 0.0,
        }

    # Rule이 FP로 판정한 인덱스
    if rule_labels_df.empty or "rule_matched" not in rule_labels_df.columns:
        rule_fp_idx = set()
    else:
        rule_fp_idx = set(
            rule_labels_df.index[rule_labels_df["rule_matched"] == True].tolist()
        )

    # ML이 FP(비TP)로 판정한 인덱스
    ml_fp_idx = set(np.where(y_pred_arr != tp_label)[0].tolist())

    # 실제 FP 인덱스
    actual_fp_idx = set(np.where(y_true_arr != tp_label)[0].tolist())

    rule_correct = rule_fp_idx & actual_fp_idx
    ml_correct = ml_fp_idx & actual_fp_idx
    overlap = rule_correct & ml_correct
    union_correct = rule_correct | ml_correct

    rule_only_coverage = len(rule_correct) / n_actual_fp
    ml_total_coverage = len(union_correct) / n_actual_fp
    ml_additional_coverage = len(ml_correct - rule_correct) / n_actual_fp
    overlap_count = len(overlap)
    overlap_rate = overlap_count / len(rule_correct) if len(rule_correct) > 0 else 0.0

    return {
        "rule_only_coverage": round(rule_only_coverage, 4),
        "ml_total_coverage": round(ml_total_coverage, 4),
        "ml_additional_coverage": round(ml_additional_coverage, 4),
        "overlap_count": overlap_count,
        "overlap_rate": round(overlap_rate, 4),
    }


def compute_class_rule_contribution(
    rule_labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """클래스별 룰 기여도 집계.

    Args:
        rule_labels_df : RuleLabeler.label_batch() 출력 DataFrame.
                         필수 컬럼: rule_matched, rule_id, rule_primary_class

    Returns:
        DataFrame{class_name, rule_id_count, total_hits}
        total_hits 내림차순 정렬.
    """
    if rule_labels_df.empty:
        return pd.DataFrame(columns=["class_name", "rule_id_count", "total_hits"])

    matched = rule_labels_df[rule_labels_df["rule_matched"] == True].copy()

    if matched.empty or "rule_primary_class" not in matched.columns:
        return pd.DataFrame(columns=["class_name", "rule_id_count", "total_hits"])

    records = []
    for class_name, grp in matched.groupby("rule_primary_class", sort=False):
        rule_id_count = grp["rule_id"].nunique() if "rule_id" in grp.columns else 0
        total_hits = len(grp)
        records.append({
            "class_name": class_name,
            "rule_id_count": rule_id_count,
            "total_hits": total_hits,
        })

    result = pd.DataFrame(records).sort_values("total_hits", ascending=False).reset_index(drop=True)
    return result
