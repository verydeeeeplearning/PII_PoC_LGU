"""S3b Evidence Generator - Architecture v1.2 §8.10

ML 예측 결과에서 사람이 이해할 수 있는 evidence를 생성한다.

모드:
    경량 evidence: 바이너리 플래그 기반 (항상 사용)
    SHAP evidence: SHAP top-k 피처 기반 (shap 패키지 설치 시)

출력 (long-format per event):
    evidence_type, feature_name, description, weight_or_contribution
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 피처 -> evidence 매핑 (Architecture §8.10)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_EVIDENCE_MAP: dict[str, tuple[str, str]] = {
    # keyword flags
    "has_byte_kw":           ("KEYWORD_FOUND",  "bytes 키워드 발견"),
    "has_timestamp_kw":      ("KEYWORD_FOUND",  "타임스탬프 키워드 발견"),
    "has_code_kw":           ("KEYWORD_FOUND",  "버전/코드 키워드 발견"),
    "has_domain_kw":         ("KEYWORD_FOUND",  "내부 도메인 키워드 발견"),
    "has_os_copyright_kw":   ("KEYWORD_FOUND",  "OS/오픈소스 저작권 키워드 발견"),
    "has_dev_kw":            ("KEYWORD_FOUND",  "개발/테스트 키워드 발견"),
    "has_json_structure_kw": ("KEYWORD_FOUND",  "JSON 구조 패턴 발견"),
    "has_license_kw":        ("KEYWORD_FOUND",  "라이선스 키워드 발견"),
    "has_kerberos_kw":       ("KEYWORD_FOUND",  "Kerberos 토큰 키워드 발견"),
    "has_hadoop_kw":         ("KEYWORD_FOUND",  "Hadoop/빅데이터 키워드 발견"),
    # path flags
    "is_log_file":           ("PATH_FLAG",      "로그 파일 경로"),
    "is_docker_overlay":     ("PATH_FLAG",      "Docker overlay 경로"),
    "is_temp_or_dev":        ("PATH_FLAG",      "임시/개발 경로"),
    "has_license_path":      ("PATH_FLAG",      "라이선스 관련 경로"),
    "is_system_path":        ("PATH_FLAG",      "시스템 경로"),
    "has_business_token":    ("PATH_FLAG",      "업무 시스템 경로"),
    "has_date_in_path":      ("PATH_FLAG",      "경로에 날짜 포함"),
    # tabular flags
    "is_mass_detection":     ("TABULAR_FLAG",   "대량 검출 (1,000건 초과)"),
    "is_extreme_detection":  ("TABULAR_FLAG",   "극단적 대량 검출 (10,000건 초과)"),
}


def generate_lightweight_evidence(
    row: dict,
    feature_names: Optional[list[str]] = None,
) -> list[dict]:
    """바이너리 플래그 기반 경량 evidence 생성.

    Args:
        row          : 피처 값 딕셔너리 {feature_name: value}
        feature_names: 검사할 피처 목록 (None이면 FEATURE_EVIDENCE_MAP 전체)

    Returns:
        evidence list (active flags only)
    """
    keys = feature_names or list(FEATURE_EVIDENCE_MAP.keys())
    evidence = []
    for feat in keys:
        if feat not in FEATURE_EVIDENCE_MAP:
            continue
        if row.get(feat, 0) == 1:
            ev_type, desc = FEATURE_EVIDENCE_MAP[feat]
            evidence.append({
                "evidence_type":          ev_type,
                "feature_name":           feat,
                "description":            desc,
                "weight_or_contribution": 1.0,
            })
    return evidence


def generate_shap_evidence(
    model,
    X_single,
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict]:
    """SHAP 기반 evidence 생성 (shap 미설치 시 빈 목록 반환).

    Args:
        model        : 학습된 모델 (XGBoost / LightGBM 등)
        X_single     : 단일 샘플 (1, n_features)
        feature_names: 피처 이름 목록
        top_k        : 상위 N개 SHAP 피처

    Returns:
        evidence list
    """
    try:
        import shap  # type: ignore[import]
    except ImportError:
        return []

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)

        # shap_values: (n_classes, n_features) or (n_features,)
        if isinstance(shap_values, list):
            shap_arr = np.abs(np.array(shap_values)).mean(axis=0).flatten()
        else:
            shap_arr = np.abs(shap_values).flatten()

        top_indices = np.argsort(shap_arr)[::-1][:top_k]
        evidence = []
        for rank, idx in enumerate(top_indices):
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            evidence.append({
                "evidence_type":          "SHAP",
                "feature_name":           feat_name,
                "description":            f"SHAP contribution rank {rank + 1}",
                "weight_or_contribution": float(shap_arr[idx]),
            })
        return evidence

    except Exception:
        return []
