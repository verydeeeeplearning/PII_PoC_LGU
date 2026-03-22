"""Phase 4 S3b: ML Feature Builder & Labeler 단위 테스트

Tests:
    4.1  MLFeatureBuilder fit_transform → sparse matrix
    4.2  MLFeatureBuilder feature_schema 검증
    4.3  ClasswiseCalibrator predict_proba 형태 확인
    4.4  OODDetector 학습 데이터 → 낮은 OOD 점수
    4.5  OODDetector 이상 데이터  → 높은 OOD 점수
    4.6  predict_with_uncertainty → ml_predictions 스키마 확인
    4.7  generate_lightweight_evidence — has_byte_kw=1 → evidence
    4.8  MLFeatureBuilder 새 범주에서 에러 없음
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ── fixtures ──────────────────────────────────────────────────────────────

MANUAL_COLS = [
    "has_timestamp_kw", "has_byte_kw", "has_code_kw",
    "is_docker_overlay", "is_log_file",
    "inspect_count_log1p", "is_mass_detection",
]

_TFIDF_CFG = {
    "raw_word":   {"max_features": 80, "min_df": 1},
    "raw_char":   {"max_features": 40, "min_df": 1, "ngram_range": [4, 6]},
    "shape_char": {"max_features": 80, "min_df": 1, "ngram_range": [3, 5]},
    "path_word":  {"max_features": 40, "min_df": 1},
}


def _make_df(n: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "raw_text":   f"sample text {i} for testing bytes version",
            "shape_text": f"aaaaaa aaaa 000000 {i}",
            "path_text":  f"var log system file {i}",
            **{c: int(rng.integers(0, 2)) for c in MANUAL_COLS},
        })
    df = pd.DataFrame(rows)
    df["inspect_count_log1p"] = np.log1p(rng.integers(1, 1000, n))
    return df


# ── Phase 4.1: MLFeatureBuilder fit_transform ────────────────────────────
# Wave 6: MLFeatureBuilder 삭제됨 → FeatureBuilderSnapshot으로 대체
# 관련 테스트는 tests/test_feature_parity.py로 이동


# ── Phase 4.3: ClasswiseCalibrator ───────────────────────────────────────

class TestClasswiseCalibrator:

    def _train_simple_model(self, n=100, n_classes=3):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 10))
        y = rng.integers(0, n_classes, n)
        model = LogisticRegression(max_iter=300, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_calibrator_predict_proba_shape(self):
        """Test 4.3: predict_proba → shape (n, n_classes)"""
        from src.models.calibrator import ClasswiseCalibrator
        model, X, y = self._train_simple_model()
        cal = ClasswiseCalibrator()
        cal.fit(model, X, y)
        proba = cal.predict_proba(X)
        assert proba.shape == (len(X), len(np.unique(y)))

    def test_calibrator_proba_sums_to_one(self):
        """predict_proba 각 행의 합 ≈ 1.0"""
        from src.models.calibrator import ClasswiseCalibrator
        model, X, y = self._train_simple_model()
        cal = ClasswiseCalibrator()
        cal.fit(model, X, y)
        proba = cal.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_calibrator_small_sample_uses_sigmoid(self):
        """소용량 샘플 → method='sigmoid'"""
        from src.models.calibrator import ClasswiseCalibrator
        model, X, y = self._train_simple_model(n=50)
        cal = ClasswiseCalibrator(minority_threshold=100)
        cal.fit(model, X, y)
        assert cal.method == "sigmoid"


# ── Phase 4.4, 4.5: OODDetector ──────────────────────────────────────────

class TestOODDetector:

    def test_in_distribution_low_score(self):
        """Test 4.4: 학습 데이터 → 낮은 Mahalanobis 점수"""
        from src.models.ood_detector import OODDetector
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5))
        ood = OODDetector()
        ood.fit(X_train)
        scores_in = ood.score(X_train[:10])
        assert scores_in.mean() < ood.threshold_

    def test_out_distribution_high_score(self):
        """Test 4.5: 완전 다른 입력 → 높은 OOD 점수"""
        from src.models.ood_detector import OODDetector
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5))
        ood = OODDetector()
        ood.fit(X_train)
        X_out = np.ones((10, 5)) * 100  # 극단적 이상치
        scores_out = ood.score(X_out)
        assert scores_out.mean() > ood.threshold_

    def test_predict_returns_bool_array(self):
        """predict → boolean ndarray"""
        from src.models.ood_detector import OODDetector
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 5))
        ood = OODDetector()
        ood.fit(X)
        flags = ood.predict(X)
        assert flags.dtype == bool
        assert len(flags) == len(X)

    def test_fit_required_before_score(self):
        """미 학습 상태에서 score → RuntimeError"""
        from src.models.ood_detector import OODDetector
        ood = OODDetector()
        with pytest.raises(RuntimeError):
            ood.score(np.ones((5, 5)))


# ── Phase 4.6: predict_with_uncertainty ──────────────────────────────────

class TestPredictWithUncertainty:

    @pytest.fixture
    def trained_model_and_data(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 8))
        y = rng.integers(0, 4, 60)
        model = LogisticRegression(max_iter=300, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_predict_uncertainty_schema(self, trained_model_and_data):
        """Test 4.6: ml_predictions 스키마 확인"""
        from src.models.trainer import predict_with_uncertainty
        model, X, _ = trained_model_and_data
        import scipy.sparse as sp
        df = predict_with_uncertainty(model=model, X=X)

        required_cols = {
            "ml_top1_class", "ml_top1_class_name", "ml_top1_proba",
            "ml_top2_class", "ml_top2_class_name", "ml_top2_proba",
            "ml_margin", "ml_entropy", "ml_tp_proba",
            "ood_mahalanobis", "ood_leaf_support", "ood_flag",
        }
        missing = required_cols - set(df.columns)
        assert not missing, f"누락 컬럼: {missing}"
        assert len(df) == len(X)

    def test_predict_uncertainty_margin_nonneg(self, trained_model_and_data):
        """ml_margin >= 0"""
        from src.models.trainer import predict_with_uncertainty
        model, X, _ = trained_model_and_data
        df = predict_with_uncertainty(model=model, X=X)
        assert (df["ml_margin"] >= 0).all()

    def test_predict_uncertainty_entropy_nonneg(self, trained_model_and_data):
        """ml_entropy >= 0"""
        from src.models.trainer import predict_with_uncertainty
        model, X, _ = trained_model_and_data
        df = predict_with_uncertainty(model=model, X=X)
        assert (df["ml_entropy"] >= 0).all()

    def test_predict_with_pk_events(self, trained_model_and_data):
        """pk_events 전달 시 pk_event 컬럼 포함"""
        from src.models.trainer import predict_with_uncertainty
        model, X, _ = trained_model_and_data
        pk = [f"evt_{i}" for i in range(len(X))]
        df = predict_with_uncertainty(model=model, X=X, pk_events=pk)
        assert "pk_event" in df.columns
        assert list(df["pk_event"]) == pk


# ── Phase 4.7: generate_lightweight_evidence ─────────────────────────────

class TestLightweightEvidence:

    def test_byte_kw_generates_evidence(self):
        """Test 4.7: has_byte_kw=1 → evidence 생성"""
        from src.models.evidence_generator import generate_lightweight_evidence
        row = {
            "has_byte_kw": 1,
            "has_timestamp_kw": 0,
            "is_log_file": 0,
        }
        evidence = generate_lightweight_evidence(row)
        assert len(evidence) >= 1
        ev_features = [e["feature_name"] for e in evidence]
        assert "has_byte_kw" in ev_features

    def test_no_flag_generates_no_evidence(self):
        """플래그 모두 0 → evidence 없음"""
        from src.models.evidence_generator import generate_lightweight_evidence
        row = {k: 0 for k in ["has_byte_kw", "has_timestamp_kw", "is_log_file"]}
        evidence = generate_lightweight_evidence(row)
        assert evidence == []

    def test_evidence_has_required_keys(self):
        """evidence dict에 필수 키 포함"""
        from src.models.evidence_generator import generate_lightweight_evidence
        row = {"has_timestamp_kw": 1, "has_byte_kw": 0}
        evidence = generate_lightweight_evidence(row)
        for ev in evidence:
            assert "evidence_type" in ev
            assert "feature_name" in ev
            assert "description" in ev
