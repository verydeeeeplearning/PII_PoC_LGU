"""FeatureBuilderSnapshot - pipeline.py build_features() 학습 상태 저장/복원.

run_inference.py가 기대하는 .transform(df) 인터페이스를 구현한다.
학습 시 사용한 fitted TF-IDF 벡터라이저, dense 피처 컬럼명, categorical
LabelEncoder를 저장하여, 새로운 데이터에 동일한 피처 변환을 적용할 수 있게 한다.

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
    "phase1_fname_shape": ("file_name",  _to_shape_text),
    "phase1_path":  ("file_path",        _to_path_text),
}


class FeatureBuilderSnapshot:
    """build_features() 결과를 감싸는 피처 빌더 스냅샷.

    run_inference.py의 ``feature_builder.transform(df)`` 인터페이스를 구현한다.

    Args:
        tfidf_vectorizers: view name -> fitted TfidfVectorizer 딕셔너리
        feature_names:     학습 시 생성된 전체 피처명 목록 (TF-IDF + dense 순서)
        dense_columns:     dense 피처에 해당하는 DataFrame 컬럼명 목록
        categorical_encoders: col_name -> fitted LabelEncoder 딕셔너리
    """

    def __init__(
        self,
        tfidf_vectorizers: dict,
        feature_names: list[str],
        dense_columns: list[str],
        categorical_encoders: Optional[dict] = None,
    ) -> None:
        self.tfidf_vectorizers = tfidf_vectorizers
        self.feature_names = feature_names
        self.dense_columns = dense_columns
        self.categorical_encoders = categorical_encoders or {}
        self.n_features_ = len(feature_names)

    # ── public API ──────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """새 데이터를 학습 시와 동일한 피처 행렬로 변환.

        내부에서 prepare_phase1_features()를 호출하여 meta/path/rule 피처를
        자동 생성하고, 저장된 categorical encoder로 범주형 인코딩을 수행한다.

        Args:
            df: S1 파싱 완료된 DataFrame (silver_detections 등).
                최소 필요 컬럼: file_name, file_path

        Returns:
            shape (n_samples, n_features) sparse matrix
        """
        # 1. 전처리: meta/path/rule 피처 생성 (누락 컬럼 보완)
        df = self._ensure_preprocessing(df)

        # 2. Categorical encoding (저장된 encoder 사용)
        df = self._apply_categorical_encoding(df)

        # 3. TF-IDF 피처
        parts: list[sp.csr_matrix] = []
        for view_name, vec in self.tfidf_vectorizers.items():
            col, transform_fn = _TFIDF_COLUMN_MAP.get(view_name, (view_name, None))
            if col in df.columns:
                text = df[col].fillna("").astype(str)
                if transform_fn is not None:
                    text = text.apply(transform_fn)
            else:
                text = pd.Series([""] * len(df), index=df.index)
            parts.append(vec.transform(text))

        # 4. Dense 피처
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

    def _ensure_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """meta/path/rule 피처가 누락되어 있으면 생성."""
        # 이미 피처가 있으면 건너뜀 (training checkpoint 등)
        _sample_cols = ["fname_has_date", "is_log_file", "server_env"]
        if all(c in df.columns for c in _sample_cols):
            return df

        from src.features.feature_preparer import prepare_phase1_features
        return prepare_phase1_features(df)

    def _apply_categorical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """저장된 LabelEncoder로 범주형 컬럼을 인코딩."""
        if not self.categorical_encoders:
            return df

        df = df.copy()
        for col, encoder in self.categorical_encoders.items():
            enc_col = col + "_enc"
            if enc_col in df.columns:
                continue
            if col in df.columns:
                vals = df[col].fillna("__MISSING__").astype(str)
                vals = vals.where(vals.isin(encoder.classes_), "__UNKNOWN__")
                df[enc_col] = encoder.transform(vals)
            else:
                if "__UNKNOWN__" in encoder.classes_:
                    _unk_idx = int(np.where(encoder.classes_ == "__UNKNOWN__")[0][0])
                    df[enc_col] = _unk_idx
                else:
                    df[enc_col] = 0
        return df

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
        categorical_encoders가 result에 포함되어 있으면 함께 저장한다.
        """
        feature_names = result["feature_names"]
        tfidf_vectorizers = result.get("tfidf_vectorizers", {})
        dense_columns = [n for n in feature_names if not n.startswith("tfidf_")]
        categorical_encoders = result.get("categorical_encoders", {})
        return cls(
            tfidf_vectorizers=tfidf_vectorizers,
            feature_names=feature_names,
            dense_columns=dense_columns,
            categorical_encoders=categorical_encoders,
        )
