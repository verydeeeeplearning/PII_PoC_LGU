"""PoC 결과 지표 계산 함수 모음.

Architecture §19 기반 PoC 판정용 통계:
- Binary TP/FP 분포 통계
- 클래스 불균형 분포
- Coverage-Precision 커브 (τ 스윕)
- 3종 Split 비교표

외부 의존성: pandas, numpy, sklearn(f1_score)만 사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_binary_stats(
    df: pd.DataFrame,
    label_col: str = "label_raw",
    month_col: str = "label_work_month",
    tp_value: str = "TP",
    fp_value: str = "FP",
) -> dict:
    """TP/FP 이진 분포 통계.

    Args:
        df         : 레이블 DataFrame
        label_col  : 레이블 컬럼명
        month_col  : 월 컬럼명 (월별 breakdown용)
        tp_value   : TP 레이블 값
        fp_value   : FP 레이블 값

    Returns:
        {
            "total": {"tp": int, "fp": int, "total": int,
                      "tp_ratio": float, "fp_ratio": float},
            "by_month": DataFrame{month, tp_count, fp_count, total, tp_ratio}
        }
    """
    if df.empty or label_col not in df.columns:
        return {
            "total": {"tp": 0, "fp": 0, "total": 0, "tp_ratio": 0.0, "fp_ratio": 0.0},
            "by_month": pd.DataFrame(
                columns=["month", "tp_count", "fp_count", "total", "tp_ratio"]
            ),
        }

    n_tp = int((df[label_col] == tp_value).sum())
    n_fp = int((df[label_col] == fp_value).sum())
    n_total = len(df)

    total_stats = {
        "tp": n_tp,
        "fp": n_fp,
        "total": n_total,
        "tp_ratio": n_tp / n_total if n_total > 0 else 0.0,
        "fp_ratio": n_fp / n_total if n_total > 0 else 0.0,
    }

    # 월별 breakdown
    if month_col in df.columns:
        records = []
        for month, grp in df.groupby(month_col, sort=True):
            m_tp = int((grp[label_col] == tp_value).sum())
            m_fp = int((grp[label_col] == fp_value).sum())
            m_total = len(grp)
            records.append({
                "month": month,
                "tp_count": m_tp,
                "fp_count": m_fp,
                "total": m_total,
                "tp_ratio": m_tp / m_total if m_total > 0 else 0.0,
            })
        by_month = pd.DataFrame(records)
    else:
        by_month = pd.DataFrame(
            columns=["month", "tp_count", "fp_count", "total", "tp_ratio"]
        )

    return {"total": total_stats, "by_month": by_month}


def compute_class_imbalance(
    df: pd.DataFrame,
    label_col: str = "label_raw",
) -> pd.DataFrame:
    """클래스별 건수 및 비율.

    Args:
        df        : 레이블 DataFrame
        label_col : 레이블 컬럼명

    Returns:
        DataFrame{class_name, count, ratio}  - ratio 합 = 1.0
    """
    if df.empty or label_col not in df.columns:
        return pd.DataFrame(columns=["class_name", "count", "ratio"])

    counts = df[label_col].value_counts(dropna=False)
    total = counts.sum()
    records = [
        {"class_name": cls, "count": int(cnt), "ratio": cnt / total if total > 0 else 0.0}
        for cls, cnt in counts.items()
    ]
    return pd.DataFrame(records)


def compute_coverage_precision_curve(
    y_true,
    y_pred_proba,
    tau_range: tuple[float, float, float] = (0.5, 1.0, 0.05),
    tp_label: str = "TP",
    precision_target: float = 0.95,
) -> dict:
    """τ 스윕 -> Coverage/Precision 커브.

    자동FP: FP로 예측(즉, 예측 != tp_label) + 최대 확률 >= tau
    Coverage  = 자동FP 수 / 전체 실제FP 수
    Precision = 자동FP 중 실제FP 수 / 자동FP 수
    TP Safety = 자동FP 중 실제TP 비율

    Args:
        y_true          : 실제 레이블 array (문자열)
        y_pred_proba    : 최대 예측 확률 array (shape [N,] 또는 [N, C])
        tau_range       : (start, stop, step) - np.arange(start, stop, step)
        tp_label        : TP 클래스 레이블 문자열
        precision_target: 권장 τ 결정 기준 Precision

    Returns:
        {
            "curve": DataFrame{tau, coverage, precision, tp_safety_rate, auto_fp_count},
            "recommended_tau": float | None,
        }
    """
    y_true_arr = np.asarray(y_true)
    # y_pred_proba가 2D면 max prob 취득
    proba_arr = np.asarray(y_pred_proba)
    if proba_arr.ndim == 2:
        proba_arr = proba_arr.max(axis=1)

    n_actual_fp = int((y_true_arr != tp_label).sum())

    start, stop, step = tau_range
    taus = np.arange(start, stop + step / 2, step)

    records = []
    for tau in taus:
        # 자동FP: 예측이 비TP이고 proba >= tau
        # 여기서 "예측이 비TP"는 y_pred_proba가 확률값이므로, proba >= tau를 FP로 자동 분류
        auto_fp_mask = proba_arr >= tau
        auto_fp_count = int(auto_fp_mask.sum())

        if auto_fp_count == 0:
            coverage = 0.0
            precision_val = 1.0
            tp_safety = 0.0
        else:
            # auto_fp 중 실제 FP
            actual_fp_in_auto = int((y_true_arr[auto_fp_mask] != tp_label).sum())
            # auto_fp 중 실제 TP (오분류 TP -> FP)
            actual_tp_in_auto = int((y_true_arr[auto_fp_mask] == tp_label).sum())

            coverage = actual_fp_in_auto / n_actual_fp if n_actual_fp > 0 else 0.0
            precision_val = actual_fp_in_auto / auto_fp_count
            tp_safety = actual_tp_in_auto / auto_fp_count

        records.append({
            "tau": round(float(tau), 4),
            "coverage": coverage,
            "precision": precision_val,
            "tp_safety_rate": tp_safety,
            "auto_fp_count": auto_fp_count,
        })

    curve_df = pd.DataFrame(records)

    # 권장 τ: precision >= target를 처음 달성하는 τ
    above_target = curve_df[curve_df["precision"] >= precision_target]
    recommended_tau = float(above_target["tau"].iloc[0]) if len(above_target) > 0 else None

    return {"curve": curve_df, "recommended_tau": recommended_tau}


def compute_class_metrics(
    y_true,
    y_pred,
    tp_label: str = "TP",
) -> pd.DataFrame:
    """클래스별 Precision/Recall/F1/Support.

    sklearn classification_report(output_dict=True) 래핑.

    Returns:
        DataFrame{class_name, precision, recall, f1_score, support}
        support 내림차순 정렬.
    """
    from sklearn.metrics import classification_report

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        return pd.DataFrame(columns=["class_name", "precision", "recall", "f1_score", "support"])

    report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
    skip = {"accuracy", "macro avg", "weighted avg"}
    records = [
        {
            "class_name": cls,
            "precision": round(float(v["precision"]), 4),
            "recall": round(float(v["recall"]), 4),
            "f1_score": round(float(v["f1-score"]), 4),
            "support": int(v["support"]),
        }
        for cls, v in report.items()
        if cls not in skip
    ]
    return (
        pd.DataFrame(records)
        .sort_values("support", ascending=False)
        .reset_index(drop=True)
    )


def compute_org_stats(
    df: pd.DataFrame,
    label_col: str = "label_raw",
    org_col: str = "organization",
    source_file_col: str = "_source_file",
    tp_value: str = "TP",
) -> pd.DataFrame:
    """조직별 TP/FP 분포.

    org_col이 있으면 사용, 없으면 source_file_col 파일명에서 조직명 추출.

    Returns:
        DataFrame{organization, tp_count, fp_count, total, tp_ratio}
    """
    if df.empty or label_col not in df.columns:
        return pd.DataFrame(columns=["organization", "tp_count", "fp_count", "total", "tp_ratio"])

    # 조직 컬럼 결정
    if org_col in df.columns:
        org_series = df[org_col]
    elif source_file_col in df.columns:
        from src.evaluation.split_strategies import _extract_org_from_filename
        org_series = df[source_file_col].apply(
            lambda x: _extract_org_from_filename(str(x)) if pd.notna(x) else None
        )
    else:
        return pd.DataFrame(columns=["organization", "tp_count", "fp_count", "total", "tp_ratio"])

    tmp = df[[label_col]].copy()
    tmp["_org"] = org_series.values

    records = []
    for org_name, grp in tmp.groupby("_org", sort=True):
        if org_name is None:
            continue
        tp_cnt = int((grp[label_col] == tp_value).sum())
        fp_cnt = int((grp[label_col] != tp_value).sum())
        total = len(grp)
        records.append({
            "organization": org_name,
            "tp_count": tp_cnt,
            "fp_count": fp_cnt,
            "total": total,
            "tp_ratio": tp_cnt / total if total > 0 else 0.0,
        })

    return pd.DataFrame(records)


def compute_confidence_distribution(
    ml_proba,
    bins: list | None = None,
) -> pd.DataFrame:
    """ML top1_proba 구간별 분포.

    Args:
        ml_proba : 확률 array (shape [N,])
        bins     : 구간 경계값 리스트 (기본: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    Returns:
        DataFrame{proba_range, count, ratio, cumulative_ratio}
    """
    proba_arr = np.asarray(ml_proba, dtype=float)

    if len(proba_arr) == 0:
        return pd.DataFrame(columns=["proba_range", "count", "ratio", "cumulative_ratio"])

    if bins is None:
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    total = len(proba_arr)
    records = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == bins[-1]:
            mask = (proba_arr >= lo) & (proba_arr <= hi)
        else:
            mask = (proba_arr >= lo) & (proba_arr < hi)
        cnt = int(mask.sum())
        records.append({
            "proba_range": f"{lo:.1f}~{hi:.1f}",
            "count": cnt,
            "ratio": round(cnt / total, 4) if total > 0 else 0.0,
        })

    result_df = pd.DataFrame(records)
    result_df["cumulative_ratio"] = result_df["ratio"].cumsum().round(4)
    # 마지막 누적 비율이 부동소수점 오차로 1.0을 넘지 않도록 클리핑
    result_df["cumulative_ratio"] = result_df["cumulative_ratio"].clip(upper=1.0)
    return result_df


def compute_split_comparison(
    split_results: list[dict],
) -> pd.DataFrame:
    """여러 Split의 평가 결과를 비교 테이블로 정리.

    Args:
        split_results: 각 split 결과 dict 목록.
            각 dict에는 다음 키가 있어야 함:
            - split_name: str
            - train_n: int
            - test_n: int
            - y_true: array
            - y_pred: array
            - tp_label: str
            - coverage_at_target: float (optional, default 0.0)

    Returns:
        DataFrame{split_name, train_n, test_n, f1_macro, tp_recall,
                  fp_precision, auto_fp_coverage_at_95, poc_verdict}
    """
    from src.evaluation.evaluator import check_poc_criteria

    records = []
    for res in split_results:
        name = res.get("split_name", "unknown")
        train_n = res.get("train_n", 0)
        test_n = res.get("test_n", 0)
        y_true = np.asarray(res.get("y_true", []))
        y_pred = np.asarray(res.get("y_pred", []))
        tp_label = res.get("tp_label", "TP")
        coverage = res.get("coverage_at_target", 0.0)

        if len(y_true) == 0:
            records.append({
                "split_name": name,
                "train_n": train_n,
                "test_n": test_n,
                "f1_macro": 0.0,
                "tp_recall": 0.0,
                "fp_precision": 0.0,
                "auto_fp_coverage_at_95": coverage,
                "poc_verdict": "SKIP",
            })
            continue

        poc = check_poc_criteria(y_true, y_pred, tp_label=tp_label)
        verdict = "PASS" if poc.get("passes", False) else "FAIL"

        records.append({
            "split_name": name,
            "train_n": train_n,
            "test_n": test_n,
            "f1_macro": round(poc.get("f1_macro", 0.0), 4),
            "tp_recall": round(poc.get("tp_recall", 0.0), 4),
            "fp_precision": round(poc.get("fp_precision", 0.0), 4),
            "auto_fp_coverage_at_95": coverage,
            "poc_verdict": verdict,
        })

    return pd.DataFrame(records)
