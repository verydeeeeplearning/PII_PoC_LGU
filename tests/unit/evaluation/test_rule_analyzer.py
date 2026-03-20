"""Phase 2 테스트: Rule 기여도 분석 (TDD — RED 먼저)"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_rule_labels_df():
    """rule_labeler 출력 형식 모의 DataFrame"""
    rows = [
        # rule_id=R001, matched=True, primary_class=FP
        {"pk_event": "evt_001", "rule_matched": True,  "rule_id": "R001",
         "rule_primary_class": "FP-숫자나열/코드", "rule_reason_code": "numeric"},
        {"pk_event": "evt_002", "rule_matched": True,  "rule_id": "R001",
         "rule_primary_class": "FP-숫자나열/코드", "rule_reason_code": "numeric"},
        {"pk_event": "evt_003", "rule_matched": True,  "rule_id": "R002",
         "rule_primary_class": "FP-타임스탬프",    "rule_reason_code": "timestamp"},
        {"pk_event": "evt_004", "rule_matched": False, "rule_id": None,
         "rule_primary_class": None,               "rule_reason_code": None},
        {"pk_event": "evt_005", "rule_matched": True,  "rule_id": "R001",
         "rule_primary_class": "FP-숫자나열/코드", "rule_reason_code": "numeric"},
    ]
    return pd.DataFrame(rows)


def _make_y_true():
    """pk_event 정렬 순서와 동일한 실제 레이블"""
    return np.array(["FP", "FP", "FP", "TP", "TP"])


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeRuleContribution
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRuleContribution:
    def test_hit_count_per_rule_correct(self):
        from src.evaluation.rule_analyzer import compute_rule_contribution

        rule_df = _make_rule_labels_df()
        y_true = _make_y_true()
        result = compute_rule_contribution(rule_df, y_true, tp_label="TP")

        r001 = result[result["rule_id"] == "R001"].iloc[0]
        assert r001["hit_count"] == 3  # evt_001, evt_002, evt_005

    def test_hit_rate_fraction_of_total(self):
        from src.evaluation.rule_analyzer import compute_rule_contribution

        rule_df = _make_rule_labels_df()
        y_true = _make_y_true()
        result = compute_rule_contribution(rule_df, y_true, tp_label="TP")

        total_rows = len(rule_df)
        r001 = result[result["rule_id"] == "R001"].iloc[0]
        assert abs(r001["hit_rate"] - 3 / total_rows) < 1e-6

    def test_precision_fp_among_matched(self):
        from src.evaluation.rule_analyzer import compute_rule_contribution

        rule_df = _make_rule_labels_df()
        y_true = _make_y_true()
        result = compute_rule_contribution(rule_df, y_true, tp_label="TP")

        # R001: evt_001(FP), evt_002(FP), evt_005(TP) → precision = 2/3
        r001 = result[result["rule_id"] == "R001"].iloc[0]
        assert abs(r001["precision"] - 2/3) < 1e-6

    def test_empty_input_returns_empty_dataframe(self):
        from src.evaluation.rule_analyzer import compute_rule_contribution

        empty_df = pd.DataFrame(columns=[
            "pk_event", "rule_matched", "rule_id",
            "rule_primary_class", "rule_reason_code"
        ])
        result = compute_rule_contribution(empty_df, np.array([]), tp_label="TP")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_unmatched_rows_excluded(self):
        from src.evaluation.rule_analyzer import compute_rule_contribution

        rule_df = _make_rule_labels_df()
        y_true = _make_y_true()
        result = compute_rule_contribution(rule_df, y_true, tp_label="TP")

        # rule_matched=False인 evt_004는 제외 → None rule_id 없어야 함
        assert None not in result["rule_id"].values

    def test_sorted_by_hit_count_desc(self):
        from src.evaluation.rule_analyzer import compute_rule_contribution

        rule_df = _make_rule_labels_df()
        y_true = _make_y_true()
        result = compute_rule_contribution(rule_df, y_true, tp_label="TP")

        hit_counts = result["hit_count"].tolist()
        assert hit_counts == sorted(hit_counts, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeClassRuleContribution
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeClassRuleContribution:
    def test_class_aggregation_correct(self):
        from src.evaluation.rule_analyzer import compute_class_rule_contribution

        rule_df = _make_rule_labels_df()
        result = compute_class_rule_contribution(rule_df)

        assert "class_name" in result.columns
        assert "rule_id_count" in result.columns
        assert "total_hits" in result.columns

        fp_numeric = result[result["class_name"] == "FP-숫자나열/코드"].iloc[0]
        # R001이 3번 hit → total_hits = 3
        assert fp_numeric["total_hits"] == 3

    def test_sorted_by_total_hits_desc(self):
        from src.evaluation.rule_analyzer import compute_class_rule_contribution

        rule_df = _make_rule_labels_df()
        result = compute_class_rule_contribution(rule_df)

        total_hits = result["total_hits"].tolist()
        assert total_hits == sorted(total_hits, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeRuleVsMlCoverage
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRuleVsMlCoverage:
    def test_returns_required_keys(self):
        from src.evaluation.rule_analyzer import compute_rule_vs_ml_coverage

        rule_df = _make_rule_labels_df()
        # All FP predictions from ML
        y_pred_ml = np.array(["FP", "FP", "FP", "FP", "FP"])
        y_true = _make_y_true()

        result = compute_rule_vs_ml_coverage(rule_df, y_pred_ml, y_true)

        assert isinstance(result, dict)
        for key in ["rule_only_coverage", "ml_total_coverage",
                    "ml_additional_coverage", "overlap_count", "overlap_rate"]:
            assert key in result

    def test_rule_only_coverage_le_ml_total(self):
        from src.evaluation.rule_analyzer import compute_rule_vs_ml_coverage

        rule_df = _make_rule_labels_df()
        y_pred_ml = np.array(["FP", "FP", "FP", "TP", "FP"])
        y_true = _make_y_true()

        result = compute_rule_vs_ml_coverage(rule_df, y_pred_ml, y_true)

        # rule coverage ≤ ml total coverage (ML >= rule by union)
        assert result["rule_only_coverage"] <= result["ml_total_coverage"] + 1e-9

    def test_empty_rule_df_returns_zeros(self):
        from src.evaluation.rule_analyzer import compute_rule_vs_ml_coverage

        empty_df = pd.DataFrame(columns=[
            "pk_event", "rule_matched", "rule_id",
            "rule_primary_class", "rule_reason_code",
        ])
        y_pred_ml = np.array(["FP", "TP"])
        y_true = np.array(["FP", "TP"])

        result = compute_rule_vs_ml_coverage(empty_df, y_pred_ml, y_true)
        assert result["rule_only_coverage"] == 0.0
