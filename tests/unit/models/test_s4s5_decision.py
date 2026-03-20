"""Phase 5 S4+S5: Decision Combiner & Output Writer 단위 테스트

Tests:
    5.1  Case 0 OOD → UNKNOWN
    5.2  Case 1 RULE 고확신 → RULE 라벨
    5.3  Case 1 ML TP 충돌 → TP override
    5.4  Case 2 애매한 ML → TP override
    5.5  Case 3 Fallback → TP
    5.6  output_writer predictions_main 스키마
    5.7  output_writer evidence long-format
    5.8  AutoAdjudicator 파일 합의 판정
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np

from src.utils.constants import LABEL_TP


# ── helpers ──────────────────────────────────────────────────────────────────

def _rule(matched=True, conf=0.90, cls="FP-내부도메인", reason="INT_DOMAIN"):
    if not matched:
        return {"rule_matched": False, "rule_confidence_lb": 0.0,
                "rule_primary_class": None, "rule_reason_code": None,
                "rule_confidence_type": "PRIOR"}
    return {"rule_matched": True, "rule_confidence_lb": conf,
            "rule_primary_class": cls, "rule_reason_code": reason,
            "rule_confidence_type": "BAYESIAN_LB"}


def _ml(top1_cls=LABEL_TP, top1_proba=0.80, margin=0.30,
        entropy=0.5, tp_proba=0.80, ood=False):
    return {
        "ml_top1_class_name": top1_cls,
        "ml_top1_proba": top1_proba,
        "ml_margin": margin,
        "ml_entropy": entropy,
        "ml_tp_proba": tp_proba,
        "ood_flag": ood,
    }


# ── Case 0~3 ─────────────────────────────────────────────────────────────────

class TestCombineDecisions:

    def test_case0_ood_returns_unknown(self):
        """Test 5.1: OOD → UNKNOWN"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(_rule(matched=False), _ml(ood=True))
        assert result["primary_class"] == "UNKNOWN"

    def test_case0_high_entropy_no_rule_returns_unknown(self):
        """고 엔트로피 + 룰 없음 → UNKNOWN"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(
            _rule(matched=False),
            _ml(entropy=3.0, ood=False)
        )
        assert result["primary_class"] == "UNKNOWN"

    def test_case1_high_conf_rule_wins(self):
        """Test 5.2: RULE conf≥0.85 → RULE 라벨"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(
            _rule(matched=True, conf=0.90, cls="FP-내부도메인"),
            _ml(top1_cls="FP-내부도메인", tp_proba=0.05)
        )
        assert result["primary_class"] == "FP-내부도메인"
        assert result["decision_source"] == "RULE"

    def test_case1_ml_tp_conflict_overrides_rule(self):
        """Test 5.3: RULE FP + ml_tp_proba≥0.60 → TP override"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(
            _rule(matched=True, conf=0.90, cls="FP-내부도메인"),
            _ml(top1_cls="FP-내부도메인", tp_proba=0.75)
        )
        assert result["primary_class"] == LABEL_TP

    def test_case2_confident_ml_wins(self):
        """ML conf≥0.70 + margin≥0.20 → ML 라벨"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(
            _rule(matched=False),
            _ml(top1_cls="FP-bytes크기", top1_proba=0.80, margin=0.35,
                entropy=0.5, tp_proba=0.10)
        )
        assert result["primary_class"] == "FP-bytes크기"
        assert result["decision_source"] == "ML"

    def test_case2_ambiguous_ml_tp_override(self):
        """Test 5.4: ML conf<0.70 + tp_proba≥0.40 → TP override"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(
            _rule(matched=False),
            _ml(top1_cls="FP-bytes크기", top1_proba=0.55, margin=0.10,
                entropy=0.8, tp_proba=0.45)
        )
        assert result["primary_class"] == LABEL_TP

    def test_case3_fallback_tp(self):
        """Test 5.5: 둘 다 없음 → TP"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(
            _rule(matched=False),
            _ml(top1_proba=0.40, margin=0.05, entropy=2.0, tp_proba=0.10, ood=False)
        )
        assert result["primary_class"] == LABEL_TP
        assert result["decision_source"] == "FALLBACK"

    def test_result_has_required_keys(self):
        """결과 dict에 필수 키 포함"""
        from src.models.decision_combiner import combine_decisions
        result = combine_decisions(_rule(), _ml())
        required = {"primary_class", "reason_code", "confidence",
                    "decision_source", "risk_flag"}
        assert required.issubset(set(result.keys()))

    def test_custom_thresholds(self):
        """커스텀 임계값 적용 확인"""
        from src.models.decision_combiner import combine_decisions
        # rule_conf=0.99 → 0.90 rule won't trigger Case 1
        result = combine_decisions(
            _rule(matched=True, conf=0.90),
            _ml(top1_cls="FP-내부도메인", top1_proba=0.80, margin=0.30,
                tp_proba=0.05, entropy=0.3),
            thresholds={"rule_conf": 0.99},
        )
        # rule_conf < 0.99 → falls through to ML (Case 2)
        assert result["decision_source"] in ("ML", "FALLBACK")


# ── Output Writer ─────────────────────────────────────────────────────────────

class TestOutputWriter:

    def _make_decisions_df(self, n=5):
        rows = []
        for i in range(n):
            rows.append({
                "pk_event": f"evt_{i:03d}",
                "pk_file": f"file_{i:03d}",
                "primary_class": LABEL_TP if i == 0 else "FP-내부도메인",
                "reason_code": "TP_SAFETY" if i == 0 else "INT_DOMAIN",
                "confidence": 0.9,
                "decision_source": "RULE",
                "risk_flag": i == 0,
            })
        return pd.DataFrame(rows)

    def _make_silver_df(self, n=5):
        return pd.DataFrame({
            "pk_event": [f"evt_{i:03d}" for i in range(n)],
            "pk_file": [f"file_{i:03d}" for i in range(n)],
            "server_name": ["svr01"] * n,
            "agent_ip": ["10.0.0.1"] * n,
            "file_path": [f"/var/log/file{i}.txt" for i in range(n)],
            "file_name": [f"file{i}.txt" for i in range(n)],
            "pii_type_inferred": ["email"] * n,
            "detection_time": pd.to_datetime("2026-01-01"),
        })

    def test_predictions_main_schema(self):
        """Test 5.6: predictions_main 스키마 확인"""
        from src.models.output_writer import build_predictions_main
        df_dec = self._make_decisions_df()
        df_silver = self._make_silver_df()
        result = build_predictions_main(df_dec, df_silver, run_id="run001")

        required_cols = {
            "pk_event", "pk_file", "server_name", "file_path",
            "primary_class", "reason_code", "decision_source",
            "confidence", "risk_flag", "run_id", "run_date",
        }
        missing = required_cols - set(result.columns)
        assert not missing, f"누락 컬럼: {missing}"
        assert len(result) == len(df_dec)

    def test_prediction_evidence_long_format(self):
        """Test 5.7: prediction_evidence long-format"""
        from src.models.output_writer import build_prediction_evidence
        rule_ev = pd.DataFrame([
            {"pk_event": "evt_000", "evidence_rank": 0, "evidence_type": "RULE_MATCH",
             "source": "RULE", "rule_id": "L1_001", "matched_value": "@lguplus.co.kr",
             "matched_span_start": 0, "matched_span_end": 14, "snippet": "@lguplus.co.kr"},
        ])
        ml_ev = pd.DataFrame([
            {"pk_event": "evt_001", "evidence_type": "KEYWORD_FOUND",
             "feature_name": "has_byte_kw", "description": "bytes 발견",
             "weight_or_contribution": 1.0},
        ])
        result = build_prediction_evidence(rule_ev, ml_ev)
        assert "pk_event" in result.columns
        assert len(result) == len(rule_ev) + len(ml_ev)


# ── AutoAdjudicator ───────────────────────────────────────────────────────────

class TestAutoAdjudicator:

    def test_auto_adjudicator_importable(self):
        """Test 5.8: AutoAdjudicator import 가능"""
        from src.models.auto_adjudicator import AutoAdjudicator
        adj = AutoAdjudicator()
        assert adj is not None

    def test_file_consensus_majority_wins(self):
        """동일 pk_file 내 다수결"""
        from src.models.auto_adjudicator import AutoAdjudicator
        adj = AutoAdjudicator()
        # 3 FP-내부도메인, 1 TP-실제개인정보 → FP wins
        file_decisions = [
            {"pk_event": "e1", "primary_class": "FP-내부도메인"},
            {"pk_event": "e2", "primary_class": "FP-내부도메인"},
            {"pk_event": "e3", "primary_class": "FP-내부도메인"},
            {"pk_event": "e4", "primary_class": LABEL_TP},
        ]
        review = {"pk_event": "e4", "pk_file": "f001", "primary_class": "NEEDS_REVIEW"}
        result = adj.adjudicate(review, context={"file_decisions": file_decisions})
        assert result["adjudicated_class"] == "FP-내부도메인"

    def test_tp_conservative_fallback(self):
        """동의 없음 → TP 보수적 결정"""
        from src.models.auto_adjudicator import AutoAdjudicator
        adj = AutoAdjudicator()
        review = {"pk_event": "e1", "pk_file": "f001", "primary_class": "NEEDS_REVIEW"}
        result = adj.adjudicate(review, context={"file_decisions": []})
        assert result["adjudicated_class"] == LABEL_TP
