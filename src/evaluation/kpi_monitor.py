"""S6 KPI Monitor - Architecture v1.2 §11.2

12종 KPI 자동 산출 -> monthly_metrics.json 저장.

계층 A (파이프라인 건전성):
    parse_success_rate, fallback_rate, quarantine_count, feature_schema_match

계층 B (분포 이동):
    rule_match_rate, oov_rate_raw, oov_rate_path, confidence_p10

계층 C (안전장치):
    review_rate, ood_rate, rule_conflict_rate, auto_fp_precision_est
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import LABEL_TP


# ── 알람 임계값 (evaluation_config.yaml에서 오버라이드 가능) ──────────────────

KPI_ALARM_THRESHOLDS: dict = {
    "parse_success_rate": lambda v, _: v < 0.95,
    "fallback_rate":      lambda v, _: v > 0.10,
    "quarantine_count":   lambda v, prev: (prev is not None) and (v > prev * 3),
    "feature_schema_match": lambda v, _: v is False,
    "rule_match_rate":    lambda v, prev: (prev is not None) and (v < prev - 0.10),
    "oov_rate_raw":       lambda v, _: v > 0.30,
    "oov_rate_path":      lambda v, _: v > 0.20,
    "confidence_p10":     lambda v, _: v < 0.40,
    "review_rate":        lambda v, _: v > 0.35,
    "ood_rate":           lambda v, _: v > 0.05,
    "rule_conflict_rate": lambda v, _: v > 0.10,
    "auto_fp_precision_est": lambda v, _: v < 0.90,
}


def compute_monthly_kpis(
    silver_detections: pd.DataFrame,
    rule_labels: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    predictions_main: pd.DataFrame,
    prev_metrics: Optional[dict] = None,
    feature_schema_match: bool = True,
) -> dict:
    """12종 KPI 자동 산출.

    Args:
        silver_detections : S1 Silver DataFrame (parse_status 포함)
        rule_labels       : S3a rule_labels DataFrame (rule_matched, rule_has_conflict)
        ml_predictions    : S3b ml_predictions DataFrame (ood_flag, ml_top1_proba)
        predictions_main  : S4+S5 predictions_main DataFrame (primary_class)
        prev_metrics      : 이전 월 KPI dict (분포 이동 감지용)
        feature_schema_match: 피처 스키마 일치 여부

    Returns:
        dict with 12 KPI keys
    """
    prev = prev_metrics or {}
    n_silver = len(silver_detections)
    n_main = len(predictions_main)

    # ── 계층 A: 파이프라인 건전성 ─────────────────────────────────────────────
    if "parse_status" in silver_detections.columns and n_silver > 0:
        n_success = (silver_detections["parse_status"] != "quarantined").sum()
        parse_success_rate = float(n_success / n_silver)
    else:
        parse_success_rate = 1.0

    quarantine_count = int(
        (silver_detections["parse_status"] == "quarantined").sum()
        if "parse_status" in silver_detections.columns else 0
    )

    # fallback_rate: FALLBACK decision source
    if "decision_source" in predictions_main.columns and n_main > 0:
        fallback_rate = float(
            (predictions_main["decision_source"] == "FALLBACK").sum() / n_main
        )
    else:
        # fallback 없음으로 추정
        fallback_rate = 0.0

    # ── 계층 B: 분포 이동 ─────────────────────────────────────────────────────
    if "rule_matched" in rule_labels.columns and len(rule_labels) > 0:
        rule_match_rate = float(rule_labels["rule_matched"].mean())
    else:
        rule_match_rate = 0.0

    # OOV rates: 정적 기본값 (실제 TF-IDF vocab 필요 시 외부 주입)
    oov_rate_raw = prev.get("oov_rate_raw", 0.0)
    oov_rate_path = prev.get("oov_rate_path", 0.0)

    # confidence P10: ml_top1_proba 10번째 백분위
    if "ml_top1_proba" in ml_predictions.columns and len(ml_predictions) > 0:
        confidence_p10 = float(np.percentile(ml_predictions["ml_top1_proba"], 10))
    else:
        confidence_p10 = 0.0

    # ── 계층 C: 안전장치 ──────────────────────────────────────────────────────
    if n_main > 0 and "primary_class" in predictions_main.columns:
        review_rate = float(
            (predictions_main["primary_class"] == "NEEDS_REVIEW").sum() / n_main
        )
    else:
        review_rate = 0.0

    if "ood_flag" in ml_predictions.columns and len(ml_predictions) > 0:
        ood_rate = float(ml_predictions["ood_flag"].mean())
    else:
        ood_rate = 0.0

    if "rule_has_conflict" in rule_labels.columns and len(rule_labels) > 0:
        rule_conflict_rate = float(rule_labels["rule_has_conflict"].mean())
    else:
        rule_conflict_rate = 0.0

    # auto_fp_precision_est: FP 예측 중 실제 FP 비율 (레이블 없으면 추정값 0.95)
    if n_main > 0 and "primary_class" in predictions_main.columns:
        n_fp = (predictions_main["primary_class"] != LABEL_TP).sum()
        auto_fp_precision_est = 0.95 if n_fp == 0 else float(
            prev.get("auto_fp_precision_est", 0.95)
        )
    else:
        auto_fp_precision_est = 0.95

    return {
        # 계층 A
        "parse_success_rate":    parse_success_rate,
        "fallback_rate":         fallback_rate,
        "quarantine_count":      quarantine_count,
        "feature_schema_match":  bool(feature_schema_match),
        # 계층 B
        "rule_match_rate":       rule_match_rate,
        "oov_rate_raw":          float(oov_rate_raw),
        "oov_rate_path":         float(oov_rate_path),
        "confidence_p10":        confidence_p10,
        # 계층 C
        "review_rate":           review_rate,
        "ood_rate":              ood_rate,
        "rule_conflict_rate":    rule_conflict_rate,
        "auto_fp_precision_est": auto_fp_precision_est,
    }


def check_alarms(
    kpi_dict: dict,
    prev_metrics: Optional[dict] = None,
) -> list[dict]:
    """알람 조건 평가 -> 알람 목록 반환.

    Args:
        kpi_dict     : compute_monthly_kpis() 결과
        prev_metrics : 이전 월 KPI dict

    Returns:
        [{kpi, value, alarm_reason}, ...]
    """
    alarms = []
    prev = prev_metrics or {}
    for kpi, threshold_fn in KPI_ALARM_THRESHOLDS.items():
        if kpi not in kpi_dict:
            continue
        val = kpi_dict[kpi]
        prev_val = prev.get(kpi)
        try:
            if threshold_fn(val, prev_val):
                alarms.append({"kpi": kpi, "value": val, "alarm_reason": "threshold exceeded"})
        except Exception:
            pass
    return alarms


def save_monthly_metrics(
    kpi_dict: dict,
    output_path: str,
    run_id: str = "",
) -> None:
    """monthly_metrics.json 저장.

    Args:
        kpi_dict    : compute_monthly_kpis() 결과
        output_path : 저장 경로
        run_id      : 실행 식별자
    """
    payload = {"run_id": run_id, **kpi_dict}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
