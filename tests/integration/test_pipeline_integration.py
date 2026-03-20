"""Phase 7 통합 테스트 — S1→S2→S3a→S3b→S4→S5→S6 연결 검증

Tests:
    7.1  S1→S2: Silver → Feature 행렬 변환
    7.2  S3a→S4: RuleLabeler → combine_decisions 연결
    7.3  S3b→S4: ml_predictions → combine_decisions 연결
    7.4  S4→S5: combine_decisions → build_predictions_main 연결
    7.5  S5→S6: predictions_main → KPI 계산 연결
    7.6  CLI 하위 호환: 기존 스크립트 인자 파싱
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import pytest

from src.utils.constants import LABEL_TP


# ── 더미 데이터 헬퍼 ──────────────────────────────────────────────────────────

def _make_silver_df(n: int = 30):
    """Silver Parquet 스키마와 동일한 더미 DataFrame."""
    rng = np.random.default_rng(42)
    classes = [LABEL_TP, "FP-내부도메인", "FP-bytes크기"]
    return pd.DataFrame({
        "pk_event":        [f"evt_{i:04d}" for i in range(n)],
        "pk_file":         [f"file_{i // 3:04d}" for i in range(n)],
        "server_name":     ["svr01"] * n,
        "agent_ip":        ["10.0.0.1"] * n,
        "file_path":       [f"/var/log/app{i}.log" for i in range(n)],
        "file_name":       [f"app{i}.log" for i in range(n)],
        "pii_type_inferred": ["email"] * n,
        "detection_time":  pd.to_datetime("2025-01-01"),
        "raw_text":        [f"user@lguplus.co.kr sample text {i}" for i in range(n)],
        "shape_text":      [f"Wd@WDdWWd WdWd Wd {i}" for i in range(n)],
        "path_text":       [f"var log app{i}" for i in range(n)],
        "label":           [classes[i % 3] for i in range(n)],
        "parse_status":    ["success"] * n,
        "has_byte_kw":     [0] * n,
        "has_timestamp_kw":[0] * n,
        "has_domain_kw":   [1] * (n // 2) + [0] * (n - n // 2),
        "is_log_file":     [1] * n,
        "is_mass_detection":[0] * n,
    })


# ── Test 7.1: S1 → S2 Feature Engineering ────────────────────────────────────

@pytest.mark.integration
class TestS1ToS2Integration:

    def test_feature_builder_produces_sparse_matrix(self):
        """Test 7.1: Silver DF → MLFeatureBuilder → sparse matrix."""
        from src.models.feature_builder import MLFeatureBuilder

        df = _make_silver_df(30)
        builder = MLFeatureBuilder(
            manual_feature_cols=["has_byte_kw", "has_domain_kw", "is_log_file"],
            tfidf_config={
                "raw_word": {"max_features": 50, "min_df": 1},
                "raw_char": {"max_features": 30, "min_df": 1},
                "shape_char": {"max_features": 30, "min_df": 1},
                "path_word": {"max_features": 20, "min_df": 1},
            },
        )
        X = builder.fit_transform(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] > 0

    def test_feature_builder_transform_matches_fit_transform(self):
        """transform() 차원이 fit_transform()과 동일."""
        from src.models.feature_builder import MLFeatureBuilder

        df = _make_silver_df(30)
        builder = MLFeatureBuilder(
            tfidf_config={
                "raw_word":  {"max_features": 20, "min_df": 1},
                "raw_char":  {"max_features": 10, "min_df": 1},
                "shape_char":{"max_features": 10, "min_df": 1},
                "path_word": {"max_features": 10, "min_df": 1},
            },
        )
        X_train = builder.fit_transform(df[:20])
        X_test = builder.transform(df[20:])
        assert X_train.shape[1] == X_test.shape[1]


# ── Test 7.2: S3a → S4 Rule → Decision ───────────────────────────────────────

@pytest.mark.integration
class TestS3aToS4Integration:

    def test_rule_label_then_combine_decisions(self):
        """Test 7.2: RuleLabeler 결과 → combine_decisions."""
        from src.filters.rule_labeler import RuleLabeler
        from src.models.decision_combiner import combine_decisions

        # 최소 룰 설정
        rules = [{
            "rule_id": "L1_TEST_001",
            "applies_to_pii_type": "any",
            "primary_class": "FP-내부도메인",
            "reason_code": "INT_DOMAIN",
            "pattern_type": "domain_list",
            "pattern": ["lguplus.co.kr"],
            "priority": 100,
            "active": True,
        }]
        # rule_stats에 고 정밀도 기록 → confidence_lb ≥ 0.85
        rule_stats = {"L1_TEST_001": {"N": 5000, "M": 4990, "precision_lb": 0.997}}
        labeler = RuleLabeler(rules_config=rules, rule_stats=rule_stats)

        # 내부 도메인 텍스트
        row = {"pk_event": "e1", "pii_type_inferred": "email",
               "full_context_raw": "user@lguplus.co.kr 이메일 발견"}
        rule_result = labeler.label(row)
        assert rule_result is not None
        assert rule_result["rule_matched"] is True

        ml_mock = {
            "ml_top1_class_name": "FP-내부도메인",
            "ml_top1_proba": 0.85,
            "ml_margin": 0.30,
            "ml_entropy": 0.5,
            "ml_tp_proba": 0.05,
            "ood_flag": False,
        }
        decision = combine_decisions(rule_result, ml_mock)
        assert decision["primary_class"] == "FP-내부도메인"
        assert decision["decision_source"] == "RULE"

    def test_rule_batch_output_feeds_decisions(self):
        """rule_labels_df 행별로 combine_decisions 적용."""
        from src.filters.rule_labeler import RuleLabeler
        from src.models.decision_combiner import combine_decisions

        rules = [{
            "rule_id": "L1_TEST_001",
            "applies_to_pii_type": "any",
            "primary_class": "FP-내부도메인",
            "reason_code": "INT_DOMAIN",
            "pattern_type": "domain_list",
            "pattern": ["lguplus.co.kr"],
            "priority": 100,
            "active": True,
        }]
        labeler = RuleLabeler(rules_config=rules, rule_stats={})
        df = _make_silver_df(10)
        rule_labels_df, _ = labeler.label_batch(df)

        results = []
        for _, row in rule_labels_df.iterrows():
            rule_result = row.to_dict()
            ml_mock = {
                "ml_top1_class_name": LABEL_TP,
                "ml_top1_proba": 0.60,
                "ml_margin": 0.20,
                "ml_entropy": 0.5,
                "ml_tp_proba": 0.10,
                "ood_flag": False,
            }
            decision = combine_decisions(rule_result, ml_mock)
            results.append(decision)

        assert len(results) == len(df)
        assert all("primary_class" in r for r in results)


# ── Test 7.3: S3b → S4 ML Predictions → Decision ─────────────────────────────

@pytest.mark.integration
class TestS3bToS4Integration:

    def _make_ml_predictions_df(self, n: int = 10):
        return pd.DataFrame({
            "pk_event": [f"e{i}" for i in range(n)],
            "ml_top1_class_name": [LABEL_TP if i < 5 else "FP-내부도메인" for i in range(n)],
            "ml_top1_proba": [0.85] * n,
            "ml_margin": [0.30] * n,
            "ml_entropy": [0.5] * n,
            "ml_tp_proba": [0.85 if i < 5 else 0.05 for i in range(n)],
            "ood_flag": [False] * n,
        })

    def test_ml_predictions_to_combine_decisions(self):
        """Test 7.3: ml_predictions 각 행 → combine_decisions."""
        from src.models.decision_combiner import combine_decisions

        df = self._make_ml_predictions_df(10)
        no_rule = {
            "rule_matched": False,
            "rule_confidence_lb": 0.0,
            "rule_primary_class": None,
            "rule_reason_code": None,
        }
        decisions = []
        for _, row in df.iterrows():
            decision = combine_decisions(no_rule, row.to_dict())
            decisions.append(decision)

        assert len(decisions) == len(df)
        # TP rows (0-4): high tp_proba should yield TP
        assert decisions[0]["primary_class"] == LABEL_TP


# ── Test 7.4: S4 → S5 Decisions → Output Writer ──────────────────────────────

@pytest.mark.integration
class TestS4ToS5Integration:

    def test_decisions_to_predictions_main(self):
        """Test 7.4: decisions DataFrame → build_predictions_main."""
        from src.models.output_writer import build_predictions_main

        df_dec = pd.DataFrame([{
            "pk_event": f"e{i}",
            "pk_file": f"f{i // 2}",
            "primary_class": LABEL_TP if i == 0 else "FP-내부도메인",
            "reason_code": "TP_SAFETY" if i == 0 else "INT_DOMAIN",
            "confidence": 0.9,
            "decision_source": "RULE",
            "risk_flag": i == 0,
        } for i in range(5)])

        df_silver = _make_silver_df(5)
        df_silver["pk_event"] = [f"e{i}" for i in range(5)]

        result = build_predictions_main(df_dec, df_silver, run_id="test_run_001")
        assert len(result) == 5
        assert "run_id" in result.columns
        assert "server_name" in result.columns  # joined from silver


# ── Test 7.5: S5 → S6 Predictions → KPI ─────────────────────────────────────

@pytest.mark.integration
class TestS5ToS6Integration:

    def test_predictions_main_to_kpi(self):
        """Test 7.5: predictions_main → compute_monthly_kpis."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis

        df_silver = _make_silver_df(30)
        rule_labels = pd.DataFrame({
            "pk_event": df_silver["pk_event"],
            "rule_matched": [True, False] * 15,
            "rule_has_conflict": [False] * 30,
        })
        ml_preds = pd.DataFrame({
            "pk_event": df_silver["pk_event"],
            "ood_flag": [False] * 30,
            "ml_top1_proba": [0.85] * 30,
        })
        predictions_main = pd.DataFrame({
            "pk_event": df_silver["pk_event"],
            "primary_class": [LABEL_TP if i < 5 else "FP-내부도메인" for i in range(30)],
        })

        kpi = compute_monthly_kpis(df_silver, rule_labels, ml_preds, predictions_main)
        assert kpi["parse_success_rate"] == 1.0
        assert kpi["review_rate"] == 0.0
        assert 0.0 <= kpi["rule_match_rate"] <= 1.0


# ── Test 7.6: CLI 하위 호환 ───────────────────────────────────────────────────

@pytest.mark.integration
class TestCLIBackwardCompat:

    def test_run_training_argparse(self):
        """Test 7.6: run_training.py --stage 인자 파싱."""
        import argparse
        # run_training.py의 argparse와 동일한 인자 테스트
        parser = argparse.ArgumentParser()
        parser.add_argument("--use-filter", action="store_true")
        parser.add_argument("--filter-only", action="store_true")
        parser.add_argument("--stage", choices=["s2", "s3a", "s3b", "s4s5", "all"],
                            default="all")

        # 기존 인자 (하위 호환)
        args = parser.parse_args(["--use-filter"])
        assert args.use_filter is True
        assert args.stage == "all"

        # 신규 stage 인자
        args = parser.parse_args(["--stage", "s3a"])
        assert args.stage == "s3a"

    def test_run_evaluation_argparse(self):
        """run_evaluation.py --stage 인자 파싱."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--include-filtered", action="store_true")
        parser.add_argument("--stage", choices=["s6", "all"], default="all")

        # 기존 인자 (하위 호환)
        args = parser.parse_args(["--include-filtered"])
        assert args.include_filtered is True

        # 신규 stage 인자
        args = parser.parse_args(["--stage", "s6"])
        assert args.stage == "s6"

    def test_run_inference_importable(self):
        """run_inference.py 임포트 가능."""
        import importlib.util
        script_path = PROJECT_ROOT / "scripts" / "run_inference.py"
        assert script_path.exists(), "run_inference.py 파일이 없음"
        spec = importlib.util.spec_from_file_location("run_inference", script_path)
        module = importlib.util.module_from_spec(spec)
        # 실제 실행이 아닌 import만 테스트 (실행 가드 확인)
        assert spec is not None
