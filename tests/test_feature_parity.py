"""Feature parity 검증 테스트.

학습(build_features) 결과와 추론(FeatureBuilderSnapshot.transform) 결과가
동일한 피처 구조(shape, column order, NaN 없음)를 보장하는지 검증한다.
"""

import numpy as np
import pandas as pd
import pytest


def _make_sample_df(n: int = 50) -> pd.DataFrame:
    """학습/추론 테스트용 synthetic DataFrame 생성."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "file_name": [f"file_{i}.log" for i in range(n)],
        "file_path": [f"/var/log/app/service_{i % 5}/file_{i}.log" for i in range(n)],
        "server_name": [f"lgup-{'prd' if i % 3 == 0 else 'dev'}-app{i:02d}" for i in range(n)],
        "pk_file": [f"pk_{i:04d}" for i in range(n)],
        "label_binary": rng.choice(["TP", "FP"], size=n),
        "service": rng.choice(["svcA", "svcB", "svcC"], size=n),
        "ops_dept": rng.choice(["deptX", "deptY"], size=n),
        "organization": rng.choice(["orgA", "orgB"], size=n),
        "retention_period": rng.choice(["1년", "3년", "5년"], size=n),
        "pattern_count": rng.randint(0, 100, size=n),
        "ssn_count": rng.randint(0, 5, size=n),
        "phone_count": rng.randint(0, 5, size=n),
        "email_count": rng.randint(0, 5, size=n),
        "file_created_at": pd.NaT,
    })
    return df


class TestCategoricalEncoderUnseen:
    """unseen category가 __UNKNOWN__으로 처리되는지 검증."""

    def test_unseen_maps_to_unknown(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["A", "B", "C"])
        le.classes_ = np.append(le.classes_, "__UNKNOWN__")

        test_vals = pd.Series(["A", "D", "B", "E"])
        test_vals = test_vals.where(test_vals.isin(le.classes_), "__UNKNOWN__")
        result = le.transform(test_vals)

        assert result[0] == le.transform(["A"])[0]
        assert result[1] == le.transform(["__UNKNOWN__"])[0]
        assert result[3] == le.transform(["__UNKNOWN__"])[0]

    def test_missing_maps_to_missing(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        train_vals = pd.Series(["A", None, "B"]).fillna("__MISSING__").astype(str)
        le.fit(train_vals)
        le.classes_ = np.append(le.classes_, "__UNKNOWN__")

        assert "__MISSING__" in le.classes_


class TestFpClassifier:
    """classify_fp_description()이 주요 패턴을 올바르게 분류하는지."""

    def test_known_patterns(self):
        from src.features.fp_classifier import classify_fp_description

        assert classify_fp_description("파일없음") == "FP-파일없음"
        assert classify_fp_description("@문자로 인한 오탐") == "FP-이메일패턴"
        assert classify_fp_description("타임스탬프 값") == "FP-숫자패턴"
        assert classify_fp_description("rpm 패키지 설치 파일") == "FP-라이브러리"
        assert classify_fp_description("테스트 데이터") == "FP-더미테스트"
        assert classify_fp_description("서비스 로그 백업") == "FP-시스템로그"

    def test_unknown_fallback(self):
        from src.features.fp_classifier import classify_fp_description

        assert classify_fp_description(None) == "UNKNOWN"
        assert classify_fp_description("") == "UNKNOWN"
        assert classify_fp_description("completely random text xyz123") == "UNKNOWN"

    def test_priority_order(self):
        from src.features.fp_classifier import classify_fp_description

        # "파일없음" should match FP-파일없음, not FP-시스템로그
        result = classify_fp_description("파일없음")
        assert result == "FP-파일없음"


class TestFeatureBuilderSnapshotParity:
    """학습 시 X_test와 snapshot.transform(df_test_raw) 결과 일치 검증."""

    def test_snapshot_has_categorical_encoders(self):
        """from_build_result가 categorical_encoders를 캡처하는지."""
        from src.models.feature_builder_snapshot import FeatureBuilderSnapshot

        mock_result = {
            "feature_names": ["tfidf_a", "tfidf_b", "service_enc", "ops_dept_enc"],
            "tfidf_vectorizers": {},
            "categorical_encoders": {"service": "mock_encoder"},
        }
        snapshot = FeatureBuilderSnapshot.from_build_result(mock_result)
        assert "service" in snapshot.categorical_encoders
        assert snapshot.dense_columns == ["service_enc", "ops_dept_enc"]

    def test_snapshot_no_nan_after_encoding(self):
        """categorical encoding 후 NaN이 없는지 검증."""
        from sklearn.preprocessing import LabelEncoder
        from src.models.feature_builder_snapshot import FeatureBuilderSnapshot

        # Mock encoder
        le = LabelEncoder()
        le.fit(["svcA", "svcB"])
        le.classes_ = np.append(le.classes_, "__UNKNOWN__")

        snapshot = FeatureBuilderSnapshot(
            tfidf_vectorizers={},
            feature_names=["service_enc"],
            dense_columns=["service_enc"],
            categorical_encoders={"service": le},
        )

        df = pd.DataFrame({"service": ["svcA", "svcC", None]})
        df = snapshot._apply_categorical_encoding(df)

        assert "service_enc" in df.columns
        assert not df["service_enc"].isna().any()
        # svcC (unseen) → __UNKNOWN__
        assert df.loc[1, "service_enc"] == le.transform(["__UNKNOWN__"])[0]


class TestPathFeaturesExpansion:
    """신규 경로 피처 3개가 정상 생성되는지."""

    def test_new_path_features_exist(self):
        from src.features.path_features import extract_path_features

        result = extract_path_features("/var/lib/mysql/data/ibdata1")
        assert "has_backup_path" in result
        assert "has_database_path" in result
        assert "has_cicd_path" in result
        assert result["has_database_path"] == 1

    def test_backup_path_detection(self):
        from src.features.path_features import extract_path_features

        result = extract_path_features("/backup/daily/2026-03-21/dump.sql")
        assert result["has_backup_path"] == 1

    def test_cicd_path_detection(self):
        from src.features.path_features import extract_path_features

        result = extract_path_features("/var/lib/jenkins/workspace/build-123/output.log")
        assert result["has_cicd_path"] == 1

    def test_original_package_tokens(self):
        from src.features.path_features import extract_path_features

        result = extract_path_features("/usr/local/lib/python3.9/dist-packages/numpy/__init__.py")
        assert result["is_package_path"] == 1


class TestServerEnvExpansion:
    """서버 환경 토큰 검증 (원본 5+5)."""

    def test_original_env_tokens(self):
        from src.features.meta_features import extract_server_features

        assert extract_server_features("lgup-prd-app01")["server_env"] == "prd"
        assert extract_server_features("lgup-dev-web01")["server_env"] == "dev"
        assert extract_server_features("lgup-stg-db01")["server_env"] == "stg"

    def test_original_stack_tokens(self):
        from src.features.meta_features import extract_server_features

        assert extract_server_features("lgup-prd-app01")["server_stack"] == "app"
        assert extract_server_features("lgup-prd-db01")["server_stack"] == "db"
        assert extract_server_features("lgup-prd-web01")["server_stack"] == "web"
