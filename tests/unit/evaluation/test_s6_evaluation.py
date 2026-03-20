"""Phase 6 S6: 평가 체계 & 모니터링 단위 테스트

Tests:
    6.1  Group+Time Split — pk_file 누수 없음
    6.2  Group+Time Split — test가 train보다 최신
    6.3  KPI parse_success_rate — 정상/이상 케이스
    6.4  KPI review_rate — NEEDS_REVIEW 비율
    6.5  ECE 계산 — 완벽 보정 시 ≈0, 과신 시 >0
    6.6  ConfidentLearning — 의도적 라벨 오류 탐지
    6.7  monthly_metrics JSON 스키마 — 12종 KPI 포함
    6.8  PoC 기준 게이트 — F1≥0.70, TP Recall≥0.75
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.utils.constants import LABEL_TP


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_time_df(n_old_files: int = 8, n_new_files: int = 2, n_events: int = 3):
    """pk_file + detection_time DataFrame 생성 (신규 파일은 30일 후)."""
    rows = []
    base_old = datetime(2025, 1, 1)
    base_new = datetime(2025, 2, 1)

    for i in range(n_old_files):
        for j in range(n_events):
            rows.append({
                "pk_file": f"file_{i:03d}",
                "pk_event": f"evt_{i:03d}_{j}",
                "detection_time": base_old + timedelta(hours=j),
                "parse_status": "success",
            })
    for i in range(n_new_files):
        fi = n_old_files + i
        for j in range(n_events):
            rows.append({
                "pk_file": f"file_{fi:03d}",
                "pk_event": f"evt_{fi:03d}_{j}",
                "detection_time": base_new + timedelta(hours=j),
                "parse_status": "success",
            })
    return pd.DataFrame(rows)


def _make_predictions_main(n: int = 20):
    """minimal predictions_main DataFrame."""
    classes = [LABEL_TP if i < 5 else "FP-내부도메인" for i in range(n)]
    return pd.DataFrame({
        "pk_event": [f"e{i}" for i in range(n)],
        "primary_class": classes,
    })


def _make_rule_labels(n: int = 20):
    """minimal rule_labels DataFrame."""
    return pd.DataFrame({
        "pk_event": [f"e{i}" for i in range(n)],
        "rule_matched": [i % 2 == 0 for i in range(n)],
        "rule_has_conflict": [False] * n,
    })


def _make_ml_predictions(n: int = 20):
    """minimal ml_predictions DataFrame."""
    return pd.DataFrame({
        "pk_event": [f"e{i}" for i in range(n)],
        "ood_flag": [False] * n,
        "ml_top1_proba": [0.80] * n,
        "ml_tp_proba": [0.10] * n,
    })


# ── 6.1 + 6.2: Group+Time Split ───────────────────────────────────────────────

class TestGroupTimeSplit:

    def test_no_file_leakage(self):
        """Test 6.1: train / test pk_file 겹침 없음."""
        from src.evaluation.split_strategies import group_time_split
        df = _make_time_df()
        train_idx, test_idx = group_time_split(df, group_col="pk_file",
                                               time_col="detection_time", test_months=1)
        train_files = set(df.iloc[train_idx]["pk_file"].unique())
        test_files = set(df.iloc[test_idx]["pk_file"].unique())
        assert not train_files & test_files, "train/test 파일 중복"

    def test_time_order(self):
        """Test 6.2: test 데이터가 train보다 최신."""
        from src.evaluation.split_strategies import group_time_split
        df = _make_time_df()
        train_idx, test_idx = group_time_split(df, group_col="pk_file",
                                               time_col="detection_time", test_months=1)
        if train_idx and test_idx:
            max_train = df.iloc[list(train_idx)]["detection_time"].max()
            min_test = df.iloc[list(test_idx)]["detection_time"].min()
            assert min_test >= max_train, "test 시간이 train보다 이전"

    def test_train_larger_than_test(self):
        """train이 test보다 큼."""
        from src.evaluation.split_strategies import group_time_split
        df = _make_time_df()
        train_idx, test_idx = group_time_split(df, group_col="pk_file",
                                               time_col="detection_time", test_months=1)
        assert len(train_idx) > len(test_idx)

    def test_all_rows_covered(self):
        """train + test = 전체 행."""
        from src.evaluation.split_strategies import group_time_split
        df = _make_time_df()
        train_idx, test_idx = group_time_split(df, group_col="pk_file",
                                               time_col="detection_time", test_months=1)
        assert len(train_idx) + len(test_idx) == len(df)

    def test_server_group_split_no_leakage(self):
        """서버 그룹 Split — 서버 중복 없음."""
        from src.evaluation.split_strategies import server_group_split
        df = pd.DataFrame({
            "server_name": ["svr01"] * 10 + ["svr02"] * 10 + ["svr03"] * 5,
            "pk_event": [f"e{i}" for i in range(25)],
        })
        train_idx, test_idx = server_group_split(df, server_col="server_name")
        train_servers = set(df.iloc[train_idx]["server_name"].unique())
        test_servers = set(df.iloc[test_idx]["server_name"].unique())
        assert not train_servers & test_servers


# ── 6.3 + 6.4: KPI Monitor ────────────────────────────────────────────────────

class TestKPIMonitor:

    def test_parse_success_rate_all_success(self):
        """Test 6.3: 전체 parse_status=success → 1.0."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis
        df_silver = _make_time_df()
        result = compute_monthly_kpis(
            silver_detections=df_silver,
            rule_labels=_make_rule_labels(),
            ml_predictions=_make_ml_predictions(),
            predictions_main=_make_predictions_main(),
        )
        assert result["parse_success_rate"] == pytest.approx(1.0, abs=1e-6)

    def test_parse_success_rate_with_quarantine(self):
        """Test 6.3b: quarantine 존재 시 parse_success_rate < 1.0."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis
        df_silver = _make_time_df()
        df_silver.loc[:4, "parse_status"] = "quarantined"   # 15 out of 30
        result = compute_monthly_kpis(
            silver_detections=df_silver,
            rule_labels=_make_rule_labels(n=len(df_silver)),
            ml_predictions=_make_ml_predictions(n=len(df_silver)),
            predictions_main=_make_predictions_main(n=len(df_silver)),
        )
        assert result["parse_success_rate"] < 1.0

    def test_review_rate_no_review(self):
        """Test 6.4: NEEDS_REVIEW 없음 → review_rate = 0."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis
        result = compute_monthly_kpis(
            silver_detections=_make_time_df(),
            rule_labels=_make_rule_labels(),
            ml_predictions=_make_ml_predictions(),
            predictions_main=_make_predictions_main(),
        )
        assert result["review_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_review_rate_with_review(self):
        """NEEDS_REVIEW 존재 시 review_rate > 0."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis
        pm = _make_predictions_main(20)
        pm.loc[0:3, "primary_class"] = "NEEDS_REVIEW"  # 4 out of 20 = 0.2
        result = compute_monthly_kpis(
            silver_detections=_make_time_df(),
            rule_labels=_make_rule_labels(),
            ml_predictions=_make_ml_predictions(),
            predictions_main=pm,
        )
        assert result["review_rate"] > 0.0

    def test_ood_rate_no_ood(self):
        """ood_flag=False만 있으면 ood_rate=0."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis
        result = compute_monthly_kpis(
            silver_detections=_make_time_df(),
            rule_labels=_make_rule_labels(),
            ml_predictions=_make_ml_predictions(),
            predictions_main=_make_predictions_main(),
        )
        assert result["ood_rate"] == pytest.approx(0.0, abs=1e-6)


# ── 6.5: ECE 계산 ─────────────────────────────────────────────────────────────

class TestCalibrationEval:

    def test_ece_perfect_calibration(self):
        """Test 6.5: 완벽 보정 → ECE ≈ 0."""
        from src.evaluation.calibration_eval import compute_ece
        rng = np.random.default_rng(42)
        # y_proba bins 중심값으로 설정 → 완벽 보정
        n = 1000
        y_proba = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)
        ece = compute_ece(y_true, y_proba, n_bins=10)
        assert ece < 0.15, f"완벽 보정 ECE가 너무 큼: {ece:.4f}"

    def test_ece_overconfident(self):
        """Test 6.5b: 항상 1.0 예측 + 절반 틀림 → ECE > 0."""
        from src.evaluation.calibration_eval import compute_ece
        n = 100
        y_proba = np.ones(n)
        y_true = np.array([1, 0] * (n // 2))
        ece = compute_ece(y_true, y_proba, n_bins=10)
        assert ece > 0.3, f"과신 ECE가 너무 작음: {ece:.4f}"

    def test_mce_positive(self):
        """MCE > 0 (과신)."""
        from src.evaluation.calibration_eval import compute_mce
        n = 100
        y_proba = np.ones(n)
        y_true = np.array([1, 0] * (n // 2))
        mce = compute_mce(y_true, y_proba, n_bins=10)
        assert mce > 0.0


# ── 6.6: ConfidentLearning ────────────────────────────────────────────────────

class TestConfidentLearning:

    def _make_clean_data(self, n: int = 200):
        """두 클래스 분리 가능한 데이터."""
        rng = np.random.default_rng(0)
        X_0 = rng.normal(0, 1, (n // 2, 2))
        X_1 = rng.normal(4, 1, (n // 2, 2))
        X = np.vstack([X_0, X_1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return X, y

    def test_audit_returns_dict(self):
        """Test 6.6: audit() dict 반환."""
        from src.evaluation.confident_learning import ConfidentLearningAuditor
        from sklearn.linear_model import LogisticRegression
        X, y = self._make_clean_data()
        auditor = ConfidentLearningAuditor()
        result = auditor.audit(X, y, LogisticRegression, n_splits=3)
        assert isinstance(result, dict)
        assert "noise_indices" in result
        assert "noise_rate" in result

    def test_noise_detection(self):
        """Test 6.6b: 의도적 라벨 오류 탐지."""
        from src.evaluation.confident_learning import ConfidentLearningAuditor
        from sklearn.linear_model import LogisticRegression
        X, y = self._make_clean_data(200)
        # 5개 라벨 반전 (명백한 오류)
        y_noisy = y.copy()
        y_noisy[[0, 2, 4, 6, 8]] = 1 - y_noisy[[0, 2, 4, 6, 8]]
        auditor = ConfidentLearningAuditor()
        result = auditor.audit(X, y_noisy, LogisticRegression, n_splits=3)
        # 최소한 일부 노이즈 인덱스를 탐지해야 함
        assert len(result["noise_indices"]) > 0

    def test_clean_labels_reduces_size(self):
        """clean_labels()가 의심 샘플을 제거."""
        from src.evaluation.confident_learning import ConfidentLearningAuditor
        from sklearn.linear_model import LogisticRegression
        X, y = self._make_clean_data(200)
        y_noisy = y.copy()
        y_noisy[[0, 2, 4, 6, 8]] = 1 - y_noisy[[0, 2, 4, 6, 8]]
        auditor = ConfidentLearningAuditor()
        result = auditor.audit(X, y_noisy, LogisticRegression, n_splits=3)
        X_clean, y_clean = auditor.clean_labels(X, y_noisy, result)
        assert len(X_clean) < len(X)


# ── 6.7: monthly_metrics JSON 스키마 ──────────────────────────────────────────

class TestMonthlyMetrics:

    _REQUIRED_KPIS = {
        "parse_success_rate", "fallback_rate", "quarantine_count",
        "feature_schema_match", "rule_match_rate", "oov_rate_raw",
        "oov_rate_path", "confidence_p10", "review_rate",
        "ood_rate", "rule_conflict_rate", "auto_fp_precision_est",
    }

    def test_kpi_schema_all_keys(self):
        """Test 6.7: 12종 KPI 모두 포함."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis
        result = compute_monthly_kpis(
            silver_detections=_make_time_df(),
            rule_labels=_make_rule_labels(),
            ml_predictions=_make_ml_predictions(),
            predictions_main=_make_predictions_main(),
        )
        missing = self._REQUIRED_KPIS - set(result.keys())
        assert not missing, f"누락 KPI: {missing}"

    def test_kpi_values_are_numeric(self):
        """모든 KPI 값이 수치형."""
        from src.evaluation.kpi_monitor import compute_monthly_kpis
        result = compute_monthly_kpis(
            silver_detections=_make_time_df(),
            rule_labels=_make_rule_labels(),
            ml_predictions=_make_ml_predictions(),
            predictions_main=_make_predictions_main(),
        )
        for k, v in result.items():
            if k in self._REQUIRED_KPIS:
                assert isinstance(v, (int, float, bool)), f"{k} 값이 수치형 아님: {v!r}"


# ── 6.8: PoC 기준 게이트 ──────────────────────────────────────────────────────

class TestPoCCriteriaGate:

    def test_passing_criteria(self):
        """Test 6.8: F1≥0.70, TP recall≥0.75 → 통과."""
        from src.evaluation.evaluator import check_poc_criteria
        # Mock high-quality predictions
        n = 100
        y_true = [LABEL_TP] * 30 + ["FP-내부도메인"] * 70
        y_pred = y_true.copy()   # perfect predictions
        result = check_poc_criteria(y_true, y_pred, tp_label=LABEL_TP)
        assert result["passes"] is True

    def test_failing_f1(self):
        """F1 낮으면 실패."""
        from src.evaluation.evaluator import check_poc_criteria
        y_true = [LABEL_TP] * 50 + ["FP-내부도메인"] * 50
        y_pred = ["FP-내부도메인"] * 100   # 전부 FP 예측 → recall=0
        result = check_poc_criteria(y_true, y_pred, tp_label=LABEL_TP)
        assert result["passes"] is False

    def test_failing_tp_recall(self):
        """TP recall 낮으면 실패."""
        from src.evaluation.evaluator import check_poc_criteria
        y_true = [LABEL_TP] * 10 + ["FP-내부도메인"] * 90
        # TP 절반만 맞게 예측
        y_pred = ["FP-내부도메인"] * 10 + ["FP-내부도메인"] * 90
        result = check_poc_criteria(y_true, y_pred, tp_label=LABEL_TP)
        assert result["passes"] is False

    def test_result_has_required_keys(self):
        """결과 dict에 필수 키 포함."""
        from src.evaluation.evaluator import check_poc_criteria
        y_true = [LABEL_TP] * 5 + ["FP-내부도메인"] * 5
        y_pred = y_true.copy()
        result = check_poc_criteria(y_true, y_pred, tp_label=LABEL_TP)
        assert {"passes", "f1_macro", "tp_recall", "fp_precision"}.issubset(result.keys())
