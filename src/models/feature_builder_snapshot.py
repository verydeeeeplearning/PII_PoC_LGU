"""FeatureBuilderSnapshot - pipeline.py build_features() 학습 상태 저장/복원.

run_inference.py가 기대하는 .transform(df) 인터페이스를 구현한다.
학습 시 사용한 fitted TF-IDF 벡터라이저와 dense 피처 컬럼명을 저장하여,
새로운 데이터에 동일한 피처 변환을 적용할 수 있게 한다.

사용법:
    # 학습 후 저장 (run_training.py)
    snapshot = FeatureBuilderSnapshot.from_build_result(result)
    snapshot.save("models/final/feature_builder.joblib")

    # 추론 시 로드 (run_inference.py)
    builder = FeatureBuilderSnapshot.load("models/final/feature_builder.joblib")
    X = builder.transform(df_silver)
"""
from __future__ import annotations

import re
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ── 텍스트 뷰 변환 헬퍼 (pipeline.py와 동일 로직) ─────────────────────────────

def _to_shape_text(text: str) -> str:
    """문자를 유형 기호로 추상화 (U/l/D/S)."""
    buf = []
    for c in text:
        if c.isupper():
            buf.append("U")
        elif c.islower():
            buf.append("l")
        elif c.isdigit():
            buf.append("D")
        elif c in " \t\n\r":
            buf.append(" ")
        elif c in "@.-_":
            buf.append(c)
        else:
            buf.append("S")
    return "".join(buf)


def _to_path_text(text: str) -> str:
    """파일 경로 구분자 기준 토큰화 후 소문자 변환."""
    tokens = re.split(r"[/\\._\-\s]+", text or "")
    return " ".join(t.lower() for t in tokens if t)


# TF-IDF view name -> (DataFrame column명, text 변환 함수 or None)
_TFIDF_COLUMN_MAP: dict[str, tuple[str, Optional[Callable]]] = {
    "raw":          ("full_context_raw", None),
    "shape":        ("full_context_raw", _to_shape_text),
    "path":         ("file_path",        _to_path_text),
    "phase1_fname": ("file_name",        None),
    "phase1_path":  ("file_path",        _to_path_text),
}


class FeatureBuilderSnapshot:
    """build_features() 결과를 감싸는 피처 빌더 스냅샷.

    run_inference.py의 ``feature_builder.transform(df)`` 인터페이스를 구현한다.

    Args:
        tfidf_vectorizers: view name -> fitted TfidfVectorizer 딕셔너리
        feature_names:     학습 시 생성된 전체 피처명 목록 (TF-IDF + dense 순서)
        dense_columns:     dense 피처에 해당하는 DataFrame 컬럼명 목록
    """

    def __init__(
        self,
        tfidf_vectorizers: dict,
        feature_names: list[str],
        dense_columns: list[str],
    ) -> None:
        self.tfidf_vectorizers = tfidf_vectorizers
        self.feature_names = feature_names
        self.dense_columns = dense_columns
        self.n_features_ = len(feature_names)

    # ── public API ──────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """새 데이터를 학습 시와 동일한 피처 행렬로 변환.

        Args:
            df: S1 파싱 + 메타/경로 피처 추출이 완료된 DataFrame.
                (file_name, file_path, full_context_raw, server_freq 등 포함 필요)

        Returns:
            shape (n_samples, n_features) sparse matrix
        """
        parts: list[sp.csr_matrix] = []

        # TF-IDF 피처
        for view_name, vec in self.tfidf_vectorizers.items():
            col, transform_fn = _TFIDF_COLUMN_MAP.get(view_name, (view_name, None))
            if col in df.columns:
                text = df[col].fillna("").astype(str)
                if transform_fn is not None:
                    text = text.apply(transform_fn)
            else:
                text = pd.Series([""] * len(df), index=df.index)
            parts.append(vec.transform(text))

        # Dense 피처
        if self.dense_columns:
            dense_arr = (
                df.reindex(columns=self.dense_columns)
                .apply(pd.to_numeric, errors="coerce")
                .values.astype(np.float64)
            )
            parts.append(sp.csr_matrix(dense_arr))

        if not parts:
            return sp.csr_matrix((len(df), self.n_features_))

        return sp.hstack(parts, format="csr")

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FeatureBuilderSnapshot":
        return joblib.load(path)

    # ── factory ─────────────────────────────────────────────────────────────

    @classmethod
    def from_build_result(cls, result: dict) -> "FeatureBuilderSnapshot":
        """build_features() 반환 딕셔너리로부터 스냅샷을 생성한다.

        dense 컬럼은 'tfidf_' 접두사가 없는 피처명으로 자동 추론한다.
        """
        feature_names = result["feature_names"]
        tfidf_vectorizers = result.get("tfidf_vectorizers", {})
        dense_columns = [n for n in feature_names if not n.startswith("tfidf_")]
        return cls(
            tfidf_vectorizers=tfidf_vectorizers,
            feature_names=feature_names,
            dense_columns=dense_columns,
        )
