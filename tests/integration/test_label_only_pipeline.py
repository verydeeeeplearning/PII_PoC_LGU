"""Integration test: Phase 1 레이블 단독 파이프라인 전체 흐름 검증

silver_label.parquet → 메타/경로 피처 → RuleLabeler → dense feature matrix
→ LightGBM 학습 → 출력 스키마 검증
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 공통 Fixture
# ─────────────────────────────────────────────────────────────────────────────

RULES_CONFIG_FC = [
    {
        "rule_id": "PATH_LICENSE_001",
        "applies_to_pii_type": "any",
        "primary_class": "FP-OS저작권",
        "reason_code": "LICENSE_PATH",
        "pattern_type": "feature_condition",
        "conditions": [{"field": "has_license_path", "op": "eq", "value": 1}],
        "logic": "and",
        "priority": 92,
        "evidence_template": "라이선스/저작권 경로",
        "active": True,
    },
    {
        "rule_id": "PATH_LOG_MASS_001",
        "applies_to_pii_type": "any",
        "primary_class": "FP-패턴맥락",
        "reason_code": "LOG_MASS_DETECTION",
        "pattern_type": "feature_condition",
        "conditions": [
            {"field": "is_log_file", "op": "eq", "value": 1},
            {"field": "pattern_count", "op": "gt", "value": 10000},
        ],
        "logic": "and",
        "priority": 88,
        "evidence_template": "로그 파일 + 대량 검출",
        "active": True,
    },
]

RULE_STATS_EMPTY = {}


def _make_label_df(n: int = 60) -> pd.DataFrame:
    """더미 silver_label DataFrame 생성"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "pk_event":   [f"evt{i:04d}" for i in range(n)],
        "pk_file":    [f"pkf{i // 5}" for i in range(n)],
        "file_path":  ["/var/log/app.log" if i % 3 == 0 else "/home/user/data.csv"
                       for i in range(n)],
        "file_name":  ["app.log" if i % 3 == 0 else "data.csv" for i in range(n)],
        "label_raw":  (["TP-실제개인정보", "FP-패턴맥락"] * (n // 2 + 1))[:n],
        "pattern_count": rng.integers(1, 500, n).tolist(),
        "ssn_count":  rng.integers(0, 5, n).tolist(),
        "phone_count": rng.integers(0, 5, n).tolist(),
        "email_count": rng.integers(0, 5, n).tolist(),
        "file_created_at": pd.date_range("2025-01-01", periods=n, freq="h"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelOnlyPipelineIntegration:
    """Phase 1 레이블 단독 파이프라인 통합 검증"""

    @pytest.fixture
    def label_df(self):
        return _make_label_df(n=60)

    def test_meta_features_applied(self, label_df):
        """build_meta_features()가 올바른 피처 컬럼을 추가"""
        from src.features.meta_features import build_meta_features
        result = build_meta_features(label_df)

        expected_cols = [
            "fname_has_date", "fname_has_hash", "fname_has_rotation_num",
            # is_log_file은 path_features.py 단일 소스 관리 (meta_features에서 제거됨)
            "pattern_count_log1p", "pattern_count_bin",
            "is_mass_detection", "is_extreme_detection",
            "created_hour", "created_weekday", "is_weekend", "created_month",
        ]
        for col in expected_cols:
            assert col in result.columns, f"메타 피처 컬럼 누락: {col}"

    def test_path_features_applied(self, label_df):
        """extract_path_features()가 올바른 피처를 추출"""
        from src.features.path_features import extract_path_features
        path_feats = label_df["file_path"].apply(extract_path_features)
        path_df = pd.DataFrame(list(path_feats), index=label_df.index)

        expected = ["path_depth", "is_log_file", "is_docker_overlay",
                    "has_license_path", "is_temp_or_dev"]
        for col in expected:
            assert col in path_df.columns, f"경로 피처 컬럼 누락: {col}"

    def test_rule_labeler_feature_condition_batch(self, label_df):
        """RuleLabeler.label_batch()가 feature_condition 룰로 정상 동작"""
        from src.features.meta_features import build_meta_features
        from src.features.path_features import extract_path_features
        from src.filters.rule_labeler import RuleLabeler

        df = build_meta_features(label_df)
        path_df = pd.DataFrame(list(df["file_path"].apply(extract_path_features)),
                               index=df.index)
        for col in path_df.columns:
            if col not in df.columns:
                df[col] = path_df[col]

        labeler = RuleLabeler(rules_config=RULES_CONFIG_FC, rule_stats=RULE_STATS_EMPTY)
        rule_labels_df, rule_evidence_df = labeler.label_batch(df)

        assert len(rule_labels_df) == len(df), "rule_labels_df 행 수 불일치"
        assert "rule_matched" in rule_labels_df.columns
        assert "pk_event" in rule_labels_df.columns

    def test_build_features_dense_only(self, label_df):
        """build_features(use_multiview_tfidf=False)가 올바른 feature matrix 생성"""
        from src.features.meta_features import build_meta_features
        from src.features.pipeline import build_features

        df = build_meta_features(label_df)
        df["label_binary"] = df["label_raw"]

        result = build_features(
            df,
            label_column="label_binary",
            use_multiview_tfidf=False,
            use_synthetic_expansion=False,
            use_group_split=True,
        )

        assert "X_train" in result
        assert "X_test" in result
        assert result["X_train"].shape[0] > 0
        assert result["X_train"].shape[1] == result["X_test"].shape[1]
        assert result["tfidf_vectorizers"] == {}

    def test_full_pipeline_train_model(self, label_df):
        """전체 파이프라인: meta features → build_features → LightGBM 학습"""
        from src.features.meta_features import build_meta_features
        from src.features.pipeline import build_features
        from src.models.trainer import encode_labels, train_lightgbm

        df = build_meta_features(label_df)
        df["label_binary"] = df["label_raw"]

        result = build_features(
            df,
            label_column="label_binary",
            use_multiview_tfidf=False,
            use_synthetic_expansion=False,
            use_group_split=True,
        )

        y_train_enc, y_test_enc, le = encode_labels(result["y_train"], result["y_test"])

        # LightGBM requires float dtype — cast regardless of sparse/dense
        import numpy as np
        X_tr = result["X_train"].astype(np.float32)
        X_te = result["X_test"].astype(np.float32)

        model, f1, report = train_lightgbm(
            X_tr, y_train_enc,
            X_te, y_test_enc,
            le, use_class_weight=True,
        )

        assert model is not None
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0
        assert isinstance(report, str)
        assert len(le.classes_) >= 2

    def test_parquet_roundtrip(self, label_df, tmp_path):
        """silver_label.parquet 저장/로드 후 파이프라인 동작 (pyarrow 없으면 CSV fallback)"""
        from src.features.meta_features import build_meta_features
        from src.features.pipeline import build_features

        # pyarrow/fastparquet 가용 여부에 따라 저장/로드 방식 선택
        try:
            import pyarrow  # noqa: F401
            parquet_path = tmp_path / "silver_label.parquet"
            label_df.to_parquet(parquet_path, index=False)
            df = pd.read_parquet(parquet_path)
        except ImportError:
            try:
                import fastparquet  # noqa: F401
                parquet_path = tmp_path / "silver_label.parquet"
                label_df.to_parquet(parquet_path, index=False)
                df = pd.read_parquet(parquet_path)
            except ImportError:
                # parquet 엔진 없음 — CSV 기반 roundtrip으로 대체
                csv_path = tmp_path / "silver_label.csv"
                label_df.to_csv(csv_path, index=False)
                df = pd.read_csv(csv_path, parse_dates=["file_created_at"])

        df = build_meta_features(df)
        df["label_binary"] = df["label_raw"]

        result = build_features(
            df,
            label_column="label_binary",
            use_multiview_tfidf=False,
            use_synthetic_expansion=False,
            use_group_split=True,
        )

        assert result["X_train"].shape[0] > 0
        assert result["X_test"].shape[0] > 0
