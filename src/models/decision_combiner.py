"""S4 Decision Combiner - Architecture v1.2 §8.11

RULE + ML 결과를 결합하여 최종 판정을 생성한다.

Case 0: OOD 또는 고 엔트로피 + 룰 없음 -> UNKNOWN
Case 1: RULE 고확신 (≥0.85) -> RULE 라벨
        단, ML tp_proba ≥ 0.60이면 TP 안전 override
Case 2: ML 고확신 (conf≥0.70, margin≥0.20) -> ML 라벨
        ML 애매 + tp_proba ≥ 0.40 -> TP override
Case 3: Fallback -> TP (보수적 결정)
"""

from __future__ import annotations

from src.utils.constants import LABEL_TP


_DEFAULT_THRESHOLDS: dict = {
    "rule_conf":             0.85,  # Case 1 RULE 고확신 임계값
    "ml_conf":               0.70,  # Case 2 ML 고확신 임계값
    "ml_margin":             0.20,  # Case 2 ML margin 임계값
    "ml_tp_proba_override":  0.60,  # RULE FP + ML tp_proba -> TP override
    "ml_tp_proba_ambiguous": 0.40,  # ML 애매 + tp_proba -> TP override
    "entropy_unknown":       2.5,   # 고 엔트로피 + 룰 없음 -> UNKNOWN
}


def combine_decisions(
    rule_result: dict,
    ml_result: dict,
    thresholds: dict | None = None,
) -> dict:
    """RULE + ML 결과 결합 -> 최종 판정 dict.

    Args:
        rule_result : RuleLabeler.label() 결과 dict
        ml_result   : predict_with_uncertainty() 단일 row dict
        thresholds  : 임계값 오버라이드 dict (부분 오버라이드 가능)

    Returns:
        {primary_class, reason_code, confidence, decision_source, risk_flag}
    """
    thr = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}

    ood_flag          = ml_result.get("ood_flag", False)
    entropy           = ml_result.get("ml_entropy", 0.0)
    rule_matched      = rule_result.get("rule_matched", False)
    rule_conf         = rule_result.get("rule_confidence_lb", 0.0)
    rule_class        = rule_result.get("rule_primary_class")
    rule_reason       = rule_result.get("rule_reason_code", "RULE")
    rule_has_conflict = rule_result.get("rule_has_conflict", False)
    ml_top1_cls       = ml_result.get("ml_top1_class_name", LABEL_TP)
    ml_top1_proba     = ml_result.get("ml_top1_proba", 0.0)
    ml_margin         = ml_result.get("ml_margin", 0.0)
    ml_tp_proba       = ml_result.get("ml_tp_proba", 0.0)

    # 룰 충돌 시 confidence 하향 (여러 클래스가 동시 매칭 -> 판단 불확실)
    # 오분류 리스크 최소화: 충돌 상황에서는 룰 신뢰도를 낮춰 ML/TP fallback 우선
    if rule_has_conflict:
        rule_conf = rule_conf * 0.7

    # ── Case 0: OOD ───────────────────────────────────────────────────────
    if ood_flag:
        return _make_result("UNKNOWN", "OOD", 0.0, "OOD", False)

    # ── Case 0: 고 엔트로피 + 룰 없음 ────────────────────────────────────
    if entropy >= thr["entropy_unknown"] and not rule_matched:
        return _make_result("UNKNOWN", "HIGH_ENTROPY", 0.0, "OOD", False)

    # ── Case 1: RULE 고확신 ───────────────────────────────────────────────
    if rule_matched and rule_conf >= thr["rule_conf"]:
        # TP 안전 override: RULE이 FP라고 해도 ML이 TP 강하게 주장하면 TP
        if rule_class != LABEL_TP and ml_tp_proba >= thr["ml_tp_proba_override"]:
            return _make_result(LABEL_TP, "TP_SAFETY", ml_tp_proba, "ML_OVERRIDE", True)
        return _make_result(
            rule_class, rule_reason, rule_conf, "RULE", rule_class == LABEL_TP
        )

    # ── Case 2: ML 고확신 ─────────────────────────────────────────────────
    if ml_top1_proba >= thr["ml_conf"] and ml_margin >= thr["ml_margin"]:
        is_tp = ml_top1_cls == LABEL_TP
        return _make_result(ml_top1_cls, "ML_CONFIDENT", ml_top1_proba, "ML", is_tp)

    # ── Case 2: 애매한 ML + TP 신호 ──────────────────────────────────────
    if ml_tp_proba >= thr["ml_tp_proba_ambiguous"]:
        return _make_result(LABEL_TP, "TP_SAFETY", ml_tp_proba, "ML_TP_OVERRIDE", True)

    # ── Case 3: Fallback -> TP (보수적 결정) ──────────────────────────────
    return _make_result(LABEL_TP, "TP_FALLBACK", 0.5, "FALLBACK", True)


def _make_result(
    primary_class: str,
    reason_code: str,
    confidence: float,
    decision_source: str,
    risk_flag: bool,
) -> dict:
    return {
        "primary_class":   primary_class,
        "reason_code":     reason_code,
        "confidence":      confidence,
        "decision_source": decision_source,
        "risk_flag":       risk_flag,
    }
