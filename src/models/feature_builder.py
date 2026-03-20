"""S3b ML Feature Builder - Architecture v1.2 §8.3

TF-IDF 4채널 + 수동 피처 결합 -> scipy sparse matrix

채널:
    raw_word  : raw_text  Word n-gram   (커스텀 토크나이저)
    raw_char  : raw_text  Char n-gram   (4~6)
    shape_char: shape_text Char n-gram  (3~5)
    path_word : path_text  Word         (공백 기준)
    manual    : keyword_flags + path_features + tabular

출력: scipy.sparse.csr_matrix
"""

from __future__ import annotations

import re
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler


def _custom_tokenizer(text: str) -> list[str]:
    """= : , ; 기준 분리 후 공백 분리 (Architecture §8.3)"""
    text = re.sub(r"[=:,;]", " ", text)
    return text.split()


# 기본 TF-IDF 설정
_DEFAULT_TFIDF_CONFIG: dict = {
    "raw_word": {
        "max_features": 500,
        "min_df": 5,
        "max_df": 0.95,
        "sublinear_tf": True,
    },
    "raw_char": {
        "max_features": 200,
        "min_df": 5,
        "ngram_range": [4, 6],
        "sublinear_tf": True,
    },
    "shape_char": {
        "max_features": 500,
        "min_df": 5,
        "ngram_range": [3, 5],
        "sublinear_tf": True,
    },
    "path_word": {
        "max_features": 200,
        "min_df": 5,
        "sublinear_tf": True,
    },
}


class MLFeatureBuilder:
    """TF-IDF 4채널 + 수동 피처 결합.

    Args:
        manual_feature_cols: 수동 피처 컬럼명 목록
        tfidf_config       : TF-IDF 설정 오버라이드 (dict)
        scale_manual       : 수동 피처 MaxAbsScaler 적용 여부
    """

    def __init__(
        self,
        manual_feature_cols: Optional[list[str]] = None,
        tfidf_config: Optional[dict] = None,
        scale_manual: bool = False,
    ) -> None:
        self.manual_feature_cols = manual_feature_cols or []
        self._cfg = {**_DEFAULT_TFIDF_CONFIG, **(tfidf_config or {})}
        self.scale_manual = scale_manual

        self._fitted = False
        self.n_features_: int = 0
        self.feature_names_: list[str] = []

        self._tfidf_rw = self._make_word_tfidf(self._cfg.get("raw_word", {}))
        self._tfidf_rc = self._make_char_tfidf(self._cfg.get("raw_char", {}))
        self._tfidf_sc = self._make_char_tfidf(self._cfg.get("shape_char", {}))
        self._tfidf_pw = self._make_word_tfidf(self._cfg.get("path_word", {}))
        self._scaler: Optional[MaxAbsScaler] = MaxAbsScaler() if scale_manual else None

    # ── public API ─────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """학습 + 변환 (training split)"""
        parts = []
        names: list[str] = []

        # TF-IDF 4채널
        raw = df["raw_text"].fillna("").astype(str)
        shape = df["shape_text"].fillna("").astype(str)
        path = df["path_text"].fillna("").astype(str)

        Xrw = self._tfidf_rw.fit_transform(raw)
        Xrc = self._tfidf_rc.fit_transform(raw)
        Xsc = self._tfidf_sc.fit_transform(shape)
        Xpw = self._tfidf_pw.fit_transform(path)

        parts.extend([Xrw, Xrc, Xsc, Xpw])
        names += [f"raw_word_{i}" for i in range(Xrw.shape[1])]
        names += [f"raw_char_{i}" for i in range(Xrc.shape[1])]
        names += [f"shape_char_{i}" for i in range(Xsc.shape[1])]
        names += [f"path_word_{i}" for i in range(Xpw.shape[1])]

        # 수동 피처
        X_man = self._extract_manual(df, fit=True)
        if X_man is not None:
            parts.append(X_man)
            names += self._manual_names_

        X = sp.hstack(parts, format="csr")
        self.n_features_ = X.shape[1]
        self.feature_names_ = names
        self._fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """변환 (inference split)"""
        if not self._fitted:
            raise RuntimeError("MLFeatureBuilder must be fit_transform() before transform()")

        raw = df["raw_text"].fillna("").astype(str)
        shape = df["shape_text"].fillna("").astype(str)
        path = df["path_text"].fillna("").astype(str)

        parts = [
            self._tfidf_rw.transform(raw),
            self._tfidf_rc.transform(raw),
            self._tfidf_sc.transform(shape),
            self._tfidf_pw.transform(path),
        ]
        X_man = self._extract_manual(df, fit=False)
        if X_man is not None:
            parts.append(X_man)

        return sp.hstack(parts, format="csr")

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MLFeatureBuilder":
        return joblib.load(path)

    # ── internal helpers ────────────────────────────────────────────────────

    def _extract_manual(self, df: pd.DataFrame, fit: bool) -> Optional[sp.csr_matrix]:
        if not self.manual_feature_cols:
            return None

        # 누락 컬럼은 0으로 채움
        cols: list[str] = []
        for col in self.manual_feature_cols:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0.0
            cols.append(col)

        X_man = df[cols].fillna(0.0).values.astype(np.float32)

        if self.scale_manual and self._scaler is not None:
            if fit:
                X_man = self._scaler.fit_transform(X_man)
            else:
                X_man = self._scaler.transform(X_man)

        if fit:
            self._manual_names_ = cols

        return sp.csr_matrix(X_man)

    @staticmethod
    def _make_word_tfidf(cfg: dict) -> TfidfVectorizer:
        return TfidfVectorizer(
            tokenizer=_custom_tokenizer,
            token_pattern=None,
            max_features=cfg.get("max_features", 500),
            min_df=cfg.get("min_df", 5),
            max_df=cfg.get("max_df", 0.95),
            sublinear_tf=cfg.get("sublinear_tf", True),
        )

    @staticmethod
    def _make_char_tfidf(cfg: dict) -> TfidfVectorizer:
        ngram = tuple(cfg.get("ngram_range", [4, 6]))
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram,
            max_features=cfg.get("max_features", 200),
            min_df=cfg.get("min_df", 5),
            sublinear_tf=cfg.get("sublinear_tf", True),
        )
