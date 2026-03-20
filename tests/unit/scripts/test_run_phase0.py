"""Unit tests for Phase 0-B: data_quality.py 함수 및 run_phase0_validation.py.

TDD RED phase: 구현 전 작성 → 실패 확인 후 GREEN 구현.

테스트 대상:
  - src/evaluation/data_quality.compute_label_conflict_rate
  - src/evaluation/data_quality.compute_bayes_error_lower_bound
  - src/evaluation/data_quality.compute_org_consistency
  - src/evaluation/data_quality.analyze_fp_description
  - src/evaluation/data_quality.make_go_no_go_decision
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data_quality import (
    analyze_fp_description,
    compute_bayes_error_lower_bound,
    compute_label_conflict_rate,
    compute_org_consistency,
    make_go_no_go_decision,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["pattern_count", "ssn_count", "phone_count", "email_count"]


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """테스트용 최소 DataFrame 생성 헬퍼."""
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeLabelConflictRate
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeLabelConflictRate:
    """compute_label_conflict_rate 함수 단위 테스트."""

    def test_empty_dataframe_returns_zero_conflict_rate(self):
        """빈 DataFrame → conflict_rate = 0.0."""
        df = pd.DataFrame(
            columns=FEATURE_COLS + ["label_raw"]
        )
        result = compute_label_conflict_rate(df, FEATURE_COLS)
        assert result["conflict_rate"] == 0.0
        assert result["conflicted_groups"] == 0
        assert result["total_groups"] == 0

    def test_all_tp_labels_no_conflict(self):
        """모든 행이 TP → 충돌 없음 → conflict_rate = 0.0."""
        df = _make_df([
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 2, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
        ])
        result = compute_label_conflict_rate(df, FEATURE_COLS)
        assert result["conflict_rate"] == 0.0
        assert result["conflicted_groups"] == 0

    def test_different_pk_events_no_shared_feature_vector_no_conflict(self):
        """서로 다른 피처 벡터를 가진 TP/FP 행 → 충돌 없음."""
        df = _make_df([
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 2, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "FP"},
        ])
        result = compute_label_conflict_rate(df, FEATURE_COLS)
        assert result["conflict_rate"] == 0.0
        assert result["conflicted_groups"] == 0

    def test_same_feature_vector_with_tp_and_fp_is_conflict(self):
        """동일 피처 벡터에 TP + FP 혼재 → 충돌 탐지."""
        df = _make_df([
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 1, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 1, "email_count": 0, "label_raw": "FP"},
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 1, "email_count": 0, "label_raw": "FP"},
        ])
        result = compute_label_conflict_rate(df, FEATURE_COLS)
        assert result["conflict_rate"] > 0.0
        assert result["conflicted_groups"] >= 1

    def test_multiple_conflicted_groups_correct_rate(self):
        """충돌 그룹 2개, 비충돌 그룹 2개 → conflict_rate = 2/4 = 0.5."""
        df = _make_df([
            # 그룹 A: TP+FP → 충돌
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "FP"},
            # 그룹 B: TP+FP → 충돌
            {"pattern_count": 2, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 2, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "FP"},
            # 그룹 C: TP만 → 비충돌
            {"pattern_count": 3, "ssn_count": 0, "phone_count": 1, "email_count": 0, "label_raw": "TP"},
            # 그룹 D: FP만 → 비충돌
            {"pattern_count": 4, "ssn_count": 0, "phone_count": 0, "email_count": 1, "label_raw": "FP"},
        ])
        result = compute_label_conflict_rate(df, FEATURE_COLS)
        assert result["total_groups"] == 4
        assert result["conflicted_groups"] == 2
        assert abs(result["conflict_rate"] - 0.5) < 1e-9

    def test_result_dict_has_required_keys(self):
        """결과 dict에 conflict_rate, conflicted_groups, total_groups 키 포함."""
        df = _make_df([
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
        ])
        result = compute_label_conflict_rate(df, FEATURE_COLS)
        assert "conflict_rate" in result
        assert "conflicted_groups" in result
        assert "total_groups" in result


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeBayesErrorLowerBound
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBayesErrorLowerBound:
    """compute_bayes_error_lower_bound 함수 단위 테스트."""

    def test_fully_separable_case_bayes_error_zero(self):
        """TP와 FP의 피처 벡터가 완전 분리 → bayes_error = 0.0."""
        df = _make_df([
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 2, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "FP"},
            {"pattern_count": 2, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "FP"},
        ])
        result = compute_bayes_error_lower_bound(df, FEATURE_COLS)
        assert result["bayes_error_lb"] == 0.0

    def test_perfectly_mixed_fifty_fifty_bayes_error_half(self):
        """동일 피처 벡터에 TP 1건 + FP 1건 → bayes_error = 0.5."""
        df = _make_df([
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 1, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 1, "email_count": 0, "label_raw": "FP"},
        ])
        result = compute_bayes_error_lower_bound(df, FEATURE_COLS)
        assert abs(result["bayes_error_lb"] - 0.5) < 1e-9

    def test_asymmetric_mixing_one_tp_nine_fp(self):
        """동일 피처 벡터에 TP 1건 + FP 9건 → bayes_error ≈ 0.1."""
        rows = [
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 1, "email_count": 0, "label_raw": "TP"}
        ] + [
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 1, "email_count": 0, "label_raw": "FP"}
            for _ in range(9)
        ]
        df = _make_df(rows)
        result = compute_bayes_error_lower_bound(df, FEATURE_COLS)
        assert abs(result["bayes_error_lb"] - 0.1) < 1e-9

    def test_multiple_groups_weighted_average(self):
        """다중 그룹의 가중 평균 Bayes Error 검증.

        그룹 A (4행): TP 2건 + FP 2건 → group_bayes = 0.5, weight = 4/6
        그룹 B (2행): TP 2건만 → group_bayes = 0.0, weight = 2/6
        기대 bayes_error = 0.5 * (4/6) + 0.0 * (2/6) = 2/6 ≈ 0.3333
        """
        df = _make_df([
            # 그룹 A
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "FP"},
            {"pattern_count": 5, "ssn_count": 1, "phone_count": 0, "email_count": 0, "label_raw": "FP"},
            # 그룹 B
            {"pattern_count": 9, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
            {"pattern_count": 9, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
        ])
        result = compute_bayes_error_lower_bound(df, FEATURE_COLS)
        expected = 0.5 * (4 / 6) + 0.0 * (2 / 6)
        assert abs(result["bayes_error_lb"] - expected) < 1e-9

    def test_result_dict_has_bayes_error_lb_key(self):
        """결과 dict에 bayes_error_lb 키 포함, 값 범위 [0, 1]."""
        df = _make_df([
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0, "label_raw": "TP"},
        ])
        result = compute_bayes_error_lower_bound(df, FEATURE_COLS)
        assert "bayes_error_lb" in result
        assert 0.0 <= result["bayes_error_lb"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeOrgConsistency
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeOrgConsistency:
    """compute_org_consistency 함수 단위 테스트."""

    def test_all_orgs_same_fp_rate_high_consistency(self):
        """동일 패턴 그룹에서 모든 조직이 동일 FP 비율 → 반환 DataFrame에 fp_rate 컬럼 존재."""
        df = _make_df([
            # CTO: 패턴 그룹 (1,0,0,0) → FP 1건, TP 1건 → fp_rate 0.5
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "CTO", "label_raw": "FP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "CTO", "label_raw": "TP"},
            # NW: 패턴 그룹 (1,0,0,0) → FP 1건, TP 1건 → fp_rate 0.5
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "NW", "label_raw": "FP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "NW", "label_raw": "TP"},
        ])
        result = compute_org_consistency(df, FEATURE_COLS)
        assert isinstance(result, pd.DataFrame)
        assert "fp_rate" in result.columns

    def test_orgs_with_different_fp_rates_consistency_less_than_one(self):
        """조직별 FP 비율 차이가 있으면 결과 DataFrame에 차이가 반영됨."""
        df = _make_df([
            # CTO: FP만 → fp_rate = 1.0
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "CTO", "label_raw": "FP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "CTO", "label_raw": "FP"},
            # NW: TP만 → fp_rate = 0.0
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "NW", "label_raw": "TP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "NW", "label_raw": "TP"},
        ])
        result = compute_org_consistency(df, FEATURE_COLS)
        assert isinstance(result, pd.DataFrame)
        # fp_rate 값이 서로 다른 행이 존재해야 함
        assert len(result["fp_rate"].unique()) > 1

    def test_result_has_required_columns(self):
        """결과 DataFrame에 organization, feature_group, fp_rate, n_samples 컬럼 포함."""
        df = _make_df([
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "CTO", "label_raw": "FP"},
            {"pattern_count": 1, "ssn_count": 0, "phone_count": 0, "email_count": 0,
             "organization": "NW", "label_raw": "TP"},
        ])
        result = compute_org_consistency(df, FEATURE_COLS)
        required_cols = {"organization", "feature_group", "fp_rate", "n_samples"}
        assert required_cols.issubset(set(result.columns))

    def test_missing_pattern_cols_returns_empty_dataframe(self):
        """pattern_cols가 DataFrame에 없으면 빈 DataFrame 반환."""
        df = _make_df([
            {"server_name": "srv1", "organization": "CTO", "label_raw": "FP"},
        ])
        result = compute_org_consistency(df, ["nonexistent_col"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# TestAnalyzeFpDescription
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeFpDescription:
    """analyze_fp_description 함수 단위 테스트."""

    def test_returns_dataframe_with_unique_fp_descriptions(self):
        """FP 행의 fp_description unique 값 + 빈도 반환."""
        df = _make_df([
            {"label_raw": "FP", "fp_description": "숫자 나열", "organization": "CTO"},
            {"label_raw": "FP", "fp_description": "숫자 나열", "organization": "NW"},
            {"label_raw": "FP", "fp_description": "서비스 로그", "organization": "CTO"},
            {"label_raw": "TP", "fp_description": None, "organization": "CTO"},
        ])
        result = analyze_fp_description(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # 2 unique descriptions

    def test_result_includes_org_distribution_column(self):
        """결과에 orgs 컬럼(조직 분포) 포함."""
        df = _make_df([
            {"label_raw": "FP", "fp_description": "라이선스", "organization": "CTO"},
        ])
        result = analyze_fp_description(df)
        assert "orgs" in result.columns

    def test_only_fp_rows_included(self):
        """TP 행은 분석에서 제외."""
        df = _make_df([
            {"label_raw": "TP", "fp_description": "정탐근거", "organization": "CTO"},
            {"label_raw": "FP", "fp_description": "숫자 나열", "organization": "NW"},
        ])
        result = analyze_fp_description(df)
        # FP 1건만 → 결과 1행
        assert len(result) == 1
        if not result.empty:
            assert "정탐근거" not in result["fp_description"].values

    def test_empty_fp_rows_returns_empty_dataframe(self):
        """FP 행이 없으면 빈 DataFrame 반환."""
        df = _make_df([
            {"label_raw": "TP", "fp_description": "근거", "organization": "CTO"},
        ])
        result = analyze_fp_description(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_fp_description_column_returns_empty_dataframe(self):
        """fp_description 컬럼 없으면 빈 DataFrame 반환."""
        df = _make_df([
            {"label_raw": "FP", "organization": "CTO"},
        ])
        result = analyze_fp_description(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_result_sorted_by_count_descending(self):
        """결과는 count 내림차순 정렬."""
        df = _make_df([
            {"label_raw": "FP", "fp_description": "A", "organization": "CTO"},
            {"label_raw": "FP", "fp_description": "B", "organization": "CTO"},
            {"label_raw": "FP", "fp_description": "B", "organization": "NW"},
            {"label_raw": "FP", "fp_description": "B", "organization": "CTO"},
        ])
        result = analyze_fp_description(df)
        assert not result.empty
        counts = result["count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_result_has_required_columns(self):
        """결과 DataFrame에 fp_description, count, orgs 컬럼 포함."""
        df = _make_df([
            {"label_raw": "FP", "fp_description": "숫자 나열", "organization": "CTO"},
        ])
        result = analyze_fp_description(df)
        assert "fp_description" in result.columns
        assert "count" in result.columns
        assert "orgs" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# TestGoNoGoDecision
# ─────────────────────────────────────────────────────────────────────────────

class TestGoNoGoDecision:
    """make_go_no_go_decision 함수 단위 테스트."""

    def test_low_conflict_low_bayes_error_verdict_promising(self):
        """conflict_rate < 0.05 AND bayes_error_lb < 0.10 → verdict = '유망'."""
        result = make_go_no_go_decision(conflict_rate=0.03, bayes_error_lb=0.05)
        assert result["verdict"] == "유망"

    def test_high_conflict_rate_verdict_join_first(self):
        """conflict_rate > 0.15 → verdict = 'JOIN 우선'."""
        result = make_go_no_go_decision(conflict_rate=0.20, bayes_error_lb=0.05)
        assert result["verdict"] == "JOIN 우선"

    def test_conflict_rate_exactly_at_boundary_fifteen_percent(self):
        """conflict_rate = 0.16 > 0.15 → verdict = 'JOIN 우선'."""
        result = make_go_no_go_decision(conflict_rate=0.16, bayes_error_lb=0.05)
        assert result["verdict"] == "JOIN 우선"

    def test_moderate_conflict_rate_verdict_caution(self):
        """0.05 ≤ conflict_rate ≤ 0.15 → verdict = '주의 필요'."""
        result = make_go_no_go_decision(conflict_rate=0.10, bayes_error_lb=0.05)
        assert result["verdict"] == "주의 필요"

    def test_low_conflict_but_high_bayes_error_verdict_caution(self):
        """conflict_rate < 0.05이지만 bayes_error_lb ≥ 0.10 → verdict = '주의 필요'."""
        result = make_go_no_go_decision(conflict_rate=0.03, bayes_error_lb=0.15)
        assert result["verdict"] == "주의 필요"

    def test_result_dict_has_required_keys(self):
        """결과 dict에 verdict, reason, conflict_rate, bayes_error_lb 키 포함."""
        result = make_go_no_go_decision(conflict_rate=0.05, bayes_error_lb=0.08)
        assert "verdict" in result
        assert "reason" in result
        assert "conflict_rate" in result
        assert "bayes_error_lb" in result

    def test_result_contains_input_values(self):
        """결과 dict의 conflict_rate, bayes_error_lb 값이 입력값과 일치."""
        cr, be = 0.07, 0.12
        result = make_go_no_go_decision(conflict_rate=cr, bayes_error_lb=be)
        assert result["conflict_rate"] == cr
        assert result["bayes_error_lb"] == be

    def test_reason_is_non_empty_string(self):
        """reason 값은 비어 있지 않은 문자열."""
        result = make_go_no_go_decision(conflict_rate=0.03, bayes_error_lb=0.05)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0
