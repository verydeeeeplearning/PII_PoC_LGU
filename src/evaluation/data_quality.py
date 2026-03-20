"""Phase 0 데이터 품질 분석 함수 모음.

Phase 0-B 5종 사전 계산:
1. 라벨 충돌률 (동일 피처 벡터에 TP+FP 혼재 비율)
2. Bayes Error 하한 (피처 공간 기준)
3. 조직 간 라벨링 일관성
4. fp_description unique 목록 + 빈도 분석
5. Go/No-Go 게이트 판정

외부 의존성 없음 - pandas, numpy만 사용 (sklearn 불필요).
"""
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 라벨 충돌률
# ─────────────────────────────────────────────────────────────────────────────

def compute_label_conflict_rate(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label_raw",
    tp_value: str = "TP",
    fp_value: str = "FP",
) -> dict:
    """동일 피처 벡터에 TP+FP가 혼재하는 비율(충돌률)을 계산한다.

    피처 공간을 feature_cols 기준으로 그룹화하고,
    동일 피처 벡터 그룹 내에 TP와 FP 레이블이 모두 존재하면 '충돌'로 판정한다.

    Args:
        df:           분석 대상 DataFrame
        feature_cols: 그룹화에 사용할 피처 컬럼 목록
        label_col:    레이블 컬럼명 (기본: "label_raw")
        tp_value:     정탐 레이블 값 (기본: "TP")
        fp_value:     오탐 레이블 값 (기본: "FP")

    Returns:
        {
            "conflict_rate": float,      # 충돌 그룹 / 전체 그룹
            "conflicted_groups": int,    # 충돌 그룹 수
            "total_groups": int,         # 전체 그룹 수
        }
    """
    if df.empty or label_col not in df.columns:
        return {"conflict_rate": 0.0, "conflicted_groups": 0, "total_groups": 0}

    # feature_cols 중 실제로 존재하는 컬럼만 사용
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        logger.warning("compute_label_conflict_rate: feature_cols 없음 - 빈 결과 반환")
        return {"conflict_rate": 0.0, "conflicted_groups": 0, "total_groups": 0}

    # 그룹별 TP/FP 존재 여부 집계
    def _has_tp_and_fp(labels: pd.Series) -> bool:
        unique_labels = set(labels.unique())
        return tp_value in unique_labels and fp_value in unique_labels

    grouped = df.groupby(available_cols, sort=False)[label_col]
    conflict_flags = grouped.apply(_has_tp_and_fp)

    total_groups = len(conflict_flags)
    conflicted_groups = int(conflict_flags.sum())
    conflict_rate = conflicted_groups / total_groups if total_groups > 0 else 0.0

    logger.info(
        "라벨 충돌률: %.4f (%d / %d 그룹)",
        conflict_rate, conflicted_groups, total_groups,
    )
    return {
        "conflict_rate": conflict_rate,
        "conflicted_groups": conflicted_groups,
        "total_groups": total_groups,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bayes Error 하한
# ─────────────────────────────────────────────────────────────────────────────

def compute_bayes_error_lower_bound(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label_raw",
    tp_value: str = "TP",
    fp_value: str = "FP",
) -> dict:
    """피처 공간 기준 Bayes Error 하한(lower bound)을 계산한다.

    각 피처 그룹 내에서 min(p_tp, p_fp)를 구하고,
    그룹 크기 가중 평균을 최종 Bayes Error 하한으로 반환한다.

    수식:
        p_tp_g = n_tp_g / n_g
        p_fp_g = n_fp_g / n_g
        bayes_g = min(p_tp_g, p_fp_g)
        bayes_error_lb = Σ_g (n_g / N) * bayes_g

    Args:
        df:           분석 대상 DataFrame
        feature_cols: 그룹화에 사용할 피처 컬럼 목록
        label_col:    레이블 컬럼명 (기본: "label_raw")
        tp_value:     정탐 레이블 값 (기본: "TP")
        fp_value:     오탐 레이블 값 (기본: "FP")

    Returns:
        {"bayes_error_lb": float}  # 범위: [0.0, 0.5]
    """
    if df.empty or label_col not in df.columns:
        return {"bayes_error_lb": 0.0}

    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        logger.warning("compute_bayes_error_lower_bound: feature_cols 없음 - bayes_error_lb = 0.0")
        return {"bayes_error_lb": 0.0}

    total_n = len(df)
    if total_n == 0:
        return {"bayes_error_lb": 0.0}

    bayes_error_weighted = 0.0

    for _, group in df.groupby(available_cols, sort=False):
        n_g = len(group)
        labels = group[label_col]
        n_tp = int((labels == tp_value).sum())
        n_fp = int((labels == fp_value).sum())
        n_labeled = n_tp + n_fp

        if n_labeled == 0:
            continue

        p_tp = n_tp / n_labeled
        p_fp = n_fp / n_labeled
        bayes_g = min(p_tp, p_fp)

        bayes_error_weighted += (n_g / total_n) * bayes_g

    logger.info("Bayes Error 하한: %.4f", bayes_error_weighted)
    return {"bayes_error_lb": bayes_error_weighted}


# ─────────────────────────────────────────────────────────────────────────────
# 3. 조직 간 라벨링 일관성
# ─────────────────────────────────────────────────────────────────────────────

def compute_org_consistency(
    df: pd.DataFrame,
    pattern_cols: List[str],
    org_col: str = "organization",
    label_col: str = "label_raw",
    fp_value: str = "FP",
) -> pd.DataFrame:
    """조직별 × 패턴 그룹별 FP 비율을 계산하여 조직 간 일관성을 분석한다.

    동일 패턴 그룹에서 조직 간 FP 비율 차이가 크면 조직별 판단 기준이
    불일치함을 의미한다.

    Args:
        df:           분석 대상 DataFrame
        pattern_cols: 패턴 피처 컬럼 목록 (그룹화 기준)
        org_col:      조직 컬럼명 (기본: "organization")
        label_col:    레이블 컬럼명 (기본: "label_raw")
        fp_value:     오탐 레이블 값 (기본: "FP")

    Returns:
        DataFrame with columns: [org_col, "feature_group", "fp_rate", "n_samples"]
        pattern_cols가 df에 없으면 빈 DataFrame 반환
    """
    # 필요한 컬럼 존재 여부 확인
    available_pattern_cols = [c for c in pattern_cols if c in df.columns]
    if not available_pattern_cols:
        logger.warning("compute_org_consistency: pattern_cols 없음 - 빈 DataFrame 반환")
        return pd.DataFrame()

    if org_col not in df.columns or label_col not in df.columns:
        logger.warning("compute_org_consistency: org_col 또는 label_col 없음 - 빈 DataFrame 반환")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    records = []
    group_cols = available_pattern_cols + [org_col]

    for keys, group in df.groupby(group_cols, sort=False):
        # keys는 tuple (pattern_vals..., org_val) or 단일 값
        if isinstance(keys, tuple):
            org_val = keys[-1]
            pattern_vals = keys[:-1]
        else:
            org_val = keys
            pattern_vals = (keys,)

        n_samples = len(group)
        n_fp = int((group[label_col] == fp_value).sum())
        fp_rate = n_fp / n_samples if n_samples > 0 else 0.0

        # feature_group: 피처 값 튜플을 문자열로
        feature_group = str(tuple(str(v) for v in pattern_vals))

        records.append({
            org_col: org_val,
            "feature_group": feature_group,
            "fp_rate": fp_rate,
            "n_samples": n_samples,
        })

    if not records:
        return pd.DataFrame(columns=[org_col, "feature_group", "fp_rate", "n_samples"])

    result = pd.DataFrame(records)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. fp_description 분석
# ─────────────────────────────────────────────────────────────────────────────

def analyze_fp_description(
    df: pd.DataFrame,
    fp_desc_col: str = "fp_description",
    org_col: str = "organization",
    label_col: str = "label_raw",
    fp_value: str = "FP",
) -> pd.DataFrame:
    """FP 행의 fp_description unique 목록과 빈도를 분석한다.

    fp_description 그룹화는 수동 작업이므로 여기서는 수행하지 않는다.
    unique 목록, 빈도, 조직별 출현 분포만 출력한다.

    Args:
        df:          분석 대상 DataFrame
        fp_desc_col: fp_description 컬럼명 (기본: "fp_description")
        org_col:     조직 컬럼명 (기본: "organization")
        label_col:   레이블 컬럼명 (기본: "label_raw")
        fp_value:    오탐 레이블 값 (기본: "FP")

    Returns:
        DataFrame with columns: ["fp_description", "count", "orgs"]
        FP 행이 없거나 fp_desc_col 컬럼이 없으면 빈 DataFrame 반환
    """
    if fp_desc_col not in df.columns:
        logger.warning("analyze_fp_description: '%s' 컬럼 없음 - 빈 DataFrame 반환", fp_desc_col)
        return pd.DataFrame()

    if label_col not in df.columns:
        logger.warning("analyze_fp_description: '%s' 컬럼 없음 - 빈 DataFrame 반환", label_col)
        return pd.DataFrame()

    fp_df = df[df[label_col] == fp_value].copy()
    if fp_df.empty:
        logger.info("analyze_fp_description: FP 행 없음 - 빈 DataFrame 반환")
        return pd.DataFrame()

    # fp_description별 빈도 집계
    count_series = fp_df[fp_desc_col].value_counts(dropna=False)

    records = []
    for desc_val, count in count_series.items():
        # 해당 description이 출현한 조직 목록 (org_col이 있을 때만)
        if org_col in fp_df.columns:
            mask = fp_df[fp_desc_col] == desc_val
            orgs_list = sorted(fp_df.loc[mask, org_col].dropna().unique().tolist())
            orgs_str = ", ".join(str(o) for o in orgs_list)
        else:
            orgs_str = ""

        records.append({
            "fp_description": desc_val,
            "count": count,
            "orgs": orgs_str,
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    # count 내림차순 정렬 (value_counts가 이미 정렬하지만 명시적으로 보장)
    result = result.sort_values("count", ascending=False).reset_index(drop=True)
    logger.info("fp_description 분석 완료: %d개 unique 값", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Go/No-Go 판정
# ─────────────────────────────────────────────────────────────────────────────

def make_go_no_go_decision(conflict_rate: float, bayes_error_lb: float) -> dict:
    """충돌률과 Bayes Error 하한을 기반으로 Go/No-Go 판정을 내린다.

    판정 기준:
        conflict_rate > 0.15         -> "JOIN 우선"
        conflict_rate < 0.05
          AND bayes_error_lb < 0.10  -> "유망"
        그 외                         -> "주의 필요"

    Args:
        conflict_rate:   라벨 충돌률 (0.0 ~ 1.0)
        bayes_error_lb:  Bayes Error 하한 (0.0 ~ 0.5)

    Returns:
        {
            "verdict":       str,   # "유망" / "JOIN 우선" / "주의 필요"
            "reason":        str,   # 판정 근거 설명
            "conflict_rate": float,
            "bayes_error_lb": float,
        }
    """
    if conflict_rate > 0.15:
        verdict = "JOIN 우선"
        reason = (
            f"라벨 충돌률 {conflict_rate:.1%} > 15% - "
            "Sumologic JOIN 선행 필요"
        )
    elif conflict_rate < 0.05 and bayes_error_lb < 0.10:
        verdict = "유망"
        reason = "충돌률/Bayes Error 양호 - ML 학습 즉시 진행 가능"
    else:
        reason_parts = []
        if conflict_rate >= 0.05:
            reason_parts.append(f"충돌률 {conflict_rate:.1%} (5% ~ 15% 구간)")
        if bayes_error_lb >= 0.10:
            reason_parts.append(f"Bayes Error {bayes_error_lb:.4f} ≥ 0.10")
        verdict = "주의 필요"
        reason = " / ".join(reason_parts) if reason_parts else "충돌률 또는 Bayes Error 임계값 경계"

    logger.info("Go/No-Go 판정: %s - %s", verdict, reason)
    return {
        "verdict": verdict,
        "reason": reason,
        "conflict_rate": conflict_rate,
        "bayes_error_lb": bayes_error_lb,
    }
