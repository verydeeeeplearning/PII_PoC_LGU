"""
Stage S2-5: Synthetic Features (합성변수 3-Tier) - Architecture.md §6.5

기본값: Tier 0 (합성변수 OFF)
CLI: --synth-tier safe | aggressive

원칙:
  Boosting 모델의 내재적 상호작용 학습과 중복 방지.
  도메인 지식 기반 교차만 허용 (Tier 1 SAFE).
  누수 차단: min_support 필터 강제 (df ≥ 50).
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 SAFE 합성변수 정의 (Architecture.md §6.5 표)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Architecture.md §6.5 Tier 1 SAFE 합성변수 10개.

    각 합성변수는 도메인 지식 기반 교차이며,
    Boosting 모델이 단독으로 학습하기 어려운 패턴을 보완한다.
    """
    def col(name: str, default=0) -> pd.Series:
        if name in df.columns:
            return df[name].fillna(0)
        return pd.Series(default, index=df.index)

    def is_email(df: pd.DataFrame) -> pd.Series:
        if "pii_type_inferred" in df.columns:
            return (df["pii_type_inferred"] == "email").astype(int)
        return pd.Series(0, index=df.index)

    features = {
        # 1) 로그 파일 + bytes 키워드 -> FP-bytes 강한 신호
        "log_file_AND_byte_kw": col("is_log_file") * col("has_byte_kw"),

        # 2) Docker + 대량 검출 -> 거의 확정 FP
        "docker_AND_mass": col("is_docker_overlay") * col("is_mass_detection"),

        # 3) 이메일 + 내부 도메인 키워드
        "email_AND_internal": is_email(df) * col("has_domain_kw"),

        # 4) 대량 검출 + 시스템 경로
        "mass_AND_system_path": col("is_mass_detection") * col("has_system_token"),

        # 5) 타임스탬프 키워드 + 높은 숫자 비율
        "timestamp_AND_digit_heavy": (
            col("has_timestamp_kw") * (col("digit_ratio") > 0.6).astype(int)
        ),

        # 6) 라이선스 경로 + OS 저작권 키워드
        "license_path_AND_os_kw": col("has_license_path") * col("has_os_copyright_kw"),

        # 7) 개발 경로 + 개발 키워드
        "temp_path_AND_dev_kw": col("is_temp_or_dev") * col("has_dev_kw"),

        # 8) 극단적 검출(10만+) + 로그 파일 -> 거의 확정 FP
        "extreme_AND_log": col("is_extreme_detection") * col("is_log_file"),

        # 9) 숫자 vs 문자 비율 (비율 관계)
        "digit_alpha_ratio": col("digit_ratio") / (col("digit_ratio").apply(
            lambda x: max(1 - x, 0.01)
        )),

        # 10) 이메일 + 업무 경로 -> TP 방향 강화
        "email_AND_business_path": is_email(df) * col("has_business_token"),
    }

    return features


def _aggressive_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Tier 2 AGGRESSIVE - Tier 1 포함 + 추가 교차.
    운영 배포 금지, 연구/오프라인 분석 전용.
    """
    features = _safe_features(df)

    def col(name: str, default=0) -> pd.Series:
        if name in df.columns:
            return df[name].fillna(0)
        return pd.Series(default, index=df.index)

    # 추가 합성변수 (실험 목적)
    extra = {
        "docker_AND_mass_AND_log": (
            col("is_docker_overlay") * col("is_mass_detection") * col("is_log_file")
        ),
        "kerberos_AND_domain": col("has_kerberos_kw") * col("has_domain_kw"),
        "byte_kw_AND_digit_heavy": (
            col("has_byte_kw") * (col("digit_ratio") > 0.5).astype(int)
        ),
    }
    features.update(extra)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

_VALID_TIERS = {"off", "safe", "aggressive"}


def build_synthetic_features(df: pd.DataFrame, tier: str = "off") -> pd.DataFrame:
    """
    합성변수 DataFrame 생성 (Architecture.md §6.5 3-Tier 정책).

    Parameters
    ----------
    df   : 기본 피처 DataFrame (is_log_file, has_byte_kw 등 포함)
    tier : "off" (기본) | "safe" | "aggressive"

    Returns
    -------
    합성변수 컬럼들만 담은 DataFrame.
    Tier 0("off")는 빈 DataFrame (컬럼 0개).
    """
    if tier not in _VALID_TIERS:
        raise ValueError(
            f"Invalid tier: {tier!r}. Must be one of {_VALID_TIERS}"
        )

    if tier == "off":
        return pd.DataFrame(index=df.index)

    if tier == "safe":
        features = _safe_features(df)
    else:  # aggressive
        features = _aggressive_features(df)

    return pd.DataFrame(features, index=df.index)
