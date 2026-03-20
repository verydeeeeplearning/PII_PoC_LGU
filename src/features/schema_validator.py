"""
Stage S2: Feature Schema Validator - Architecture.md §6

feature_schema.json 생성 및 검증.
학습 시 피처 스키마를 저장하고,
추론 시 동일 스키마인지 검증하여 차원 불일치 오류를 조기 탐지한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np


def save_feature_schema(
    X: Union[np.ndarray, "scipy.sparse.spmatrix"],
    feature_names: list[str],
    output_path: str,
) -> dict:
    """
    피처 스키마를 JSON 파일로 저장.

    Parameters
    ----------
    X              : 피처 행렬 (numpy array 또는 scipy sparse)
    feature_names  : 피처명 리스트
    output_path    : 저장 경로 (.json)

    Returns
    -------
    저장된 schema dict:
        n_features    : 피처 수
        feature_names : 피처명 리스트
    """
    n_features = X.shape[1] if hasattr(X, "shape") else len(feature_names)

    schema = {
        "n_features": n_features,
        "feature_names": list(feature_names),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    return schema


def validate_feature_schema(
    X: Union[np.ndarray, "scipy.sparse.spmatrix"],
    schema: Optional[dict],
) -> bool:
    """
    피처 행렬이 저장된 스키마와 일치하는지 검증.

    Parameters
    ----------
    X      : 검증할 피처 행렬
    schema : save_feature_schema 반환값 또는 JSON 로드 dict

    Returns
    -------
    True  : 스키마 일치
    False : 불일치 (차원 불일치, schema=None 등)
    """
    if schema is None:
        return False

    try:
        n_features = X.shape[1] if hasattr(X, "shape") else -1
        return n_features == schema.get("n_features", -1)
    except Exception:
        return False
