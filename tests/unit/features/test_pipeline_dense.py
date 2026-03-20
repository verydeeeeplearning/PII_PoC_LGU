"""Dense-only pipeline 단위 테스트 (Phase 2: use_multiview_tfidf=False)"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


def _make_sample_df(n: int = 50, with_text: bool = False) -> pd.DataFrame:
    """최소 테스트용 DataFrame 생성"""
    rng = np.random.default_rng(42)
    data = {
        "file_path": [f"/var/log/app/file_{i}.log" for i in range(n)],
        "pk_file": [f"pkfile_{i // 5}" for i in range(n)],  # 10 unique files
        "label": (["TP-실제개인정보", "FP-패턴맥락"] * (n // 2))[:n],
    }
    if with_text:
        data["full_context_raw"] = [f"sample text {i}" for i in range(n)]
    return pd.DataFrame(data)


class TestBuildFeaturesDenseOnly:
    """use_multiview_tfidf=False (dense-only) 모드 테스트"""

    def test_build_features_dense_only_no_tfidf(self):
        """TF-IDF 없이 정상 실행되며 tfidf_vectorizers가 빈 dict"""
        from src.features.pipeline import build_features
        df = _make_sample_df(n=50)
        result = build_features(
            df,
            use_multiview_tfidf=False,
            use_synthetic_expansion=False,
            use_group_split=True,
        )
        assert "X_train" in result
        assert "X_test" in result
        assert result["tfidf_vectorizers"] == {}
        assert result.get("tfidf_vectorizer") is None

    def test_build_features_dense_feature_names_correct(self):
        """Dense-only 모드에서 feature_names에 tfidf_ 피처 없음"""
        from src.features.pipeline import build_features
        df = _make_sample_df(n=50)
        result = build_features(
            df,
            use_multiview_tfidf=False,
            use_synthetic_expansion=False,
            use_group_split=True,
        )
        feature_names = result["feature_names"]
        assert isinstance(feature_names, list)
        # No TF-IDF features in dense-only mode
        tfidf_names = [n for n in feature_names if n.startswith("tfidf_")]
        assert len(tfidf_names) == 0, f"TF-IDF features found in dense-only mode: {tfidf_names[:5]}"

    def test_build_features_dense_shape_consistent(self):
        """Train/test 행렬의 컬럼 수가 일치"""
        from src.features.pipeline import build_features
        df = _make_sample_df(n=60)
        result = build_features(
            df,
            use_multiview_tfidf=False,
            use_synthetic_expansion=False,
            use_group_split=True,
        )
        X_train = result["X_train"]
        X_test = result["X_test"]
        assert X_train.shape[1] == X_test.shape[1], (
            f"Train cols={X_train.shape[1]}, Test cols={X_test.shape[1]}"
        )
        assert X_train.shape[1] == len(result["feature_names"]), (
            "feature_names length mismatch"
        )
