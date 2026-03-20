"""메타 피처 모듈 - Phase 1 피처 추출 (레이블 데이터 전용)

label_loader.py 출력 DataFrame에 피처를 추가하는 독립 모듈.

구현 피처 (~20개):
  파일명 패턴:   fname_has_date, fname_has_hash, fname_has_rotation_num
  검출 통계:     pattern_count_log1p, pattern_count_bin, is_mass_detection,
                 is_extreme_detection, pii_type_ratio
  시간 피처:     created_hour, created_weekday, is_weekend, created_month
  집계 피처:     file_event_count, file_pii_diversity (train fold only)

Note:
  extension / is_log_file은 path_features.py가 단일 소스로 관리 -> 중복 생성 금지.
  [2026-03-15] is_log_file을 extract_fname_features에서 제거.
               path_features.py가 .gz/.bz2 + 로테이션 패턴까지 통합 처리.
  tabular_features.py / file_agg.py는 Sumologic 컬럼 기반이라 레이블 데이터에 사용 불가.
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────

# 파일명 날짜 패턴 (YYYYMMDD / YYYY-MM-DD / YYYYMM)
_RE_DATE = re.compile(r"\d{8}|\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|\d{6}(?!\d)")

# 16자 이상 연속 16진수 문자열 (시스템 생성 파일 해시)
_RE_HASH = re.compile(r"[0-9a-f]{16,}", re.IGNORECASE)

# 로그 로테이션 숫자 패턴 (.1 / -001 / -2025 등)
_RE_ROTATION = re.compile(r"\.\d+$|-\d{3,}$|-\d{4}$", re.IGNORECASE)

# 로그 확장자
_LOG_EXTENSIONS = {".log", ".gz", ".bz2", ".zip"}

# pattern_count 구간 경계 (EDA §4-1)
_COUNT_BINS = [0, 5, 20, 100, 1_000]

# 대량 검출 임계값
_MASS_THRESHOLD = 10_000
_EXTREME_THRESHOLD = 100_000

# PII 컬럼명
_PII_COLS = ["ssn_count", "phone_count", "email_count"]


# ─────────────────────────────────────────────────────────────────────────────
# extract_fname_features
# ─────────────────────────────────────────────────────────────────────────────

def extract_fname_features(file_name: Optional[str]) -> dict:
    """파일명 패턴 피처 추출.

    Parameters
    ----------
    file_name : 파일명 문자열 (None 허용)

    Returns
    -------
    dict:
        fname_has_date         0/1 : YYYYMMDD / YYYY-MM-DD / YYYYMM 패턴
        fname_has_hash         0/1 : 16자+ 16진수 문자열
        fname_has_rotation_num 0/1 : .1 / -001 / -2025 등 로테이션 번호

    Note:
        is_log_file는 path_features.py가 단일 소스로 관리 (중복 제거).
        path_features.extract_path_features()가 fname_has_rotation_num 패턴까지 포함.
    """
    name = (file_name or "").strip()

    fname_has_date = int(bool(_RE_DATE.search(name)))
    fname_has_hash = int(bool(_RE_HASH.search(name)))
    fname_has_rotation_num = int(bool(_RE_ROTATION.search(name)))

    return {
        "fname_has_date": fname_has_date,
        "fname_has_hash": fname_has_hash,
        "fname_has_rotation_num": fname_has_rotation_num,
    }


# ─────────────────────────────────────────────────────────────────────────────
# extract_detection_features
# ─────────────────────────────────────────────────────────────────────────────

def extract_detection_features(row: pd.Series) -> dict:
    """검출 통계 피처 추출 (pattern_count 기반).

    Parameters
    ----------
    row : pd.Series - pattern_count, ssn_count, phone_count, email_count 포함

    Returns
    -------
    dict:
        pattern_count_log1p   float : log1p(pattern_count)
        pattern_count_bin     int   : 구간 0~4 (0-5/5-20/20-100/100-1k/1k+)
        is_mass_detection     0/1   : pattern_count > 10,000
        is_extreme_detection  0/1   : pattern_count > 100,000
        pii_type_ratio        float : ssn / (ssn+phone+email+1)
    """
    import pandas as _pd
    _cnt_raw = _pd.to_numeric(row.get("pattern_count", 0), errors="coerce")
    cnt = float(_cnt_raw) if not _pd.isna(_cnt_raw) else 0.0
    _ssn_raw = _pd.to_numeric(row.get("ssn_count", 0), errors="coerce")
    ssn = float(_ssn_raw) if not _pd.isna(_ssn_raw) else 0.0
    _phone_raw = _pd.to_numeric(row.get("phone_count", 0), errors="coerce")
    phone = float(_phone_raw) if not _pd.isna(_phone_raw) else 0.0
    _email_raw = _pd.to_numeric(row.get("email_count", 0), errors="coerce")
    email = float(_email_raw) if not _pd.isna(_email_raw) else 0.0

    # log1p 변환
    pattern_count_log1p = math.log1p(cnt)

    # 구간 인덱스 (0~4)
    pattern_count_bin = 4
    for i, boundary in enumerate(_COUNT_BINS):
        if cnt < boundary:
            pattern_count_bin = i - 1
            break
    # _COUNT_BINS[0]=0이므로 cnt<0 은 없음; cnt>=1000 이면 bin=4
    if cnt < _COUNT_BINS[0]:
        pattern_count_bin = 0
    elif cnt < _COUNT_BINS[1]:
        pattern_count_bin = 0
    elif cnt < _COUNT_BINS[2]:
        pattern_count_bin = 1
    elif cnt < _COUNT_BINS[3]:
        pattern_count_bin = 2
    elif cnt < _COUNT_BINS[4]:
        pattern_count_bin = 3
    else:
        pattern_count_bin = 4

    is_mass_detection = int(cnt > _MASS_THRESHOLD)
    is_extreme_detection = int(cnt > _EXTREME_THRESHOLD)
    pii_type_ratio = ssn / (ssn + phone + email + 1)

    return {
        "pattern_count_log1p": pattern_count_log1p,
        "pattern_count_bin": pattern_count_bin,
        "is_mass_detection": is_mass_detection,
        "is_extreme_detection": is_extreme_detection,
        "pii_type_ratio": pii_type_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# extract_datetime_features
# ─────────────────────────────────────────────────────────────────────────────

def extract_datetime_features(ts) -> dict:
    """시간 피처 추출 (file_created_at 기반).

    Parameters
    ----------
    ts : pd.Timestamp, datetime, NaT, or None

    Returns
    -------
    dict:
        created_hour    int : 0~23, NaT/None -> -1
        created_weekday int : 0=월 ~ 6=일, NaT/None -> -1
        is_weekend      int : weekday >= 5 -> 1, NaT/None -> -1
        created_month   int : 1~12, NaT/None -> -1
    """
    try:
        if ts is None or pd.isna(ts):
            raise ValueError("NaT/None")
        t = pd.Timestamp(ts)
        return {
            "created_hour": t.hour,
            "created_weekday": t.weekday(),
            "is_weekend": int(t.weekday() >= 5),
            "created_month": t.month,
        }
    except Exception:
        return {
            "created_hour": -1,
            "created_weekday": -1,
            "is_weekend": -1,
            "created_month": -1,
        }


# ─────────────────────────────────────────────────────────────────────────────
# extract_server_features
# ─────────────────────────────────────────────────────────────────────────────

_ENV_TOKENS = [
    ("prd", ["prd", "prod"]),
    ("dev", ["dev", "develop"]),
    ("stg", ["stg", "stage", "staging"]),
    ("sbx", ["sbx", "sandbox"]),
    ("test", ["test"]),
]

_STACK_TOKENS = [
    ("app", ["app"]),
    ("mms", ["mms"]),
    ("db", ["db"]),
    ("web", ["web"]),
    ("batch", ["batch"]),
]


def extract_server_features(server_name: str) -> dict:
    """서버명에서 환경·스택 의미 토큰 피처 추출.

    Parameters
    ----------
    server_name : 서버명 문자열 (None 허용)

    Returns
    -------
    dict:
        server_env      str : prd/dev/stg/sbx/test/unknown
        server_is_prod  int : 1 if server_env == "prd" else 0
        server_stack    str : app/mms/db/web/batch/etc
    """
    name = (server_name or "").strip().lower()

    # 환경 토큰 매칭 (첫 번째 매칭 우선)
    server_env = "unknown"
    for env_name, tokens in _ENV_TOKENS:
        if any(tok in name for tok in tokens):
            server_env = env_name
            break

    server_is_prod = 1 if server_env == "prd" else 0

    # 스택 토큰 매칭 (첫 번째 매칭 우선)
    server_stack = "etc"
    for stack_name, tokens in _STACK_TOKENS:
        if any(tok in name for tok in tokens):
            server_stack = stack_name
            break

    return {
        "server_env": server_env,
        "server_is_prod": server_is_prod,
        "server_stack": server_stack,
    }


# ─────────────────────────────────────────────────────────────────────────────
# build_meta_features
# ─────────────────────────────────────────────────────────────────────────────

def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """위 3개 함수 통합 적용 - 피처 컬럼이 추가된 DataFrame 반환.

    원본 df는 수정하지 않음 (copy 후 작업).
    컬럼 부재 시 graceful 처리 (결측값 -> 기본값).

    Parameters
    ----------
    df : label_loader.load_all() 출력 DataFrame

    Returns
    -------
    피처 컬럼이 추가된 DataFrame (원본 컬럼 + 메타 피처)
    """
    result = df.copy()

    # ── (1) 파일명 패턴 피처 (벡터화) ──────────────────────────────────────────
    _name = (
        result["file_name"].fillna("")
        if "file_name" in result.columns
        else pd.Series([""] * len(result), index=result.index)
    )
    result["fname_has_date"] = _name.str.contains(
        r"\d{8}|\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|\d{6}(?!\d)", regex=True
    ).astype(int)
    result["fname_has_hash"] = _name.str.contains(
        r"[0-9a-fA-F]{16,}", regex=True
    ).astype(int)
    result["fname_has_rotation_num"] = _name.str.contains(
        r"\.\d+$|-\d{3,}$|-\d{4}$", regex=True
    ).astype(int)

    # ── (2) 검출 통계 피처 (벡터화) ──────────────────────────────────────────
    _zero = pd.Series(0, index=result.index, dtype=float)
    _cnt   = pd.to_numeric(result.get("pattern_count", _zero), errors="coerce").fillna(0.0)
    _ssn   = pd.to_numeric(result.get("ssn_count",     _zero), errors="coerce").fillna(0.0)
    _phone = pd.to_numeric(result.get("phone_count",   _zero), errors="coerce").fillna(0.0)
    _email = pd.to_numeric(result.get("email_count",   _zero), errors="coerce").fillna(0.0)

    result["pattern_count_log1p"]  = np.log1p(_cnt)
    result["pattern_count_bin"]    = pd.cut(
        _cnt.clip(lower=0),
        bins=[0, 5, 20, 100, 1_000, np.inf],
        labels=[0, 1, 2, 3, 4],
        right=False,
        include_lowest=True,
    ).astype(float).fillna(0).astype(int)
    result["is_mass_detection"]    = (_cnt > _MASS_THRESHOLD).astype(int)
    result["is_extreme_detection"] = (_cnt > _EXTREME_THRESHOLD).astype(int)
    result["pii_type_ratio"]       = _ssn / (_ssn + _phone + _email + 1)

    # ── (3) 시간 피처 (벡터화) ────────────────────────────────────────────────
    _ts_raw = (
        result["file_created_at"]
        if "file_created_at" in result.columns
        else pd.Series([None] * len(result), index=result.index)
    )
    _ts = pd.to_datetime(_ts_raw, errors="coerce")
    _is_nat = _ts.isna()

    result["created_hour"]    = _ts.dt.hour.fillna(-1).astype(int)
    result["created_weekday"] = _ts.dt.weekday.fillna(-1).astype(int)
    result["created_month"]   = _ts.dt.month.fillna(-1).astype(int)
    result["is_weekend"]      = np.where(
        _is_nat, -1, (_ts.dt.weekday.fillna(0) >= 5).astype(int)
    )

    # ── (4) 서버 의미 토큰 피처 (벡터화) ─── [Tier 2 B7]
    _srv = (
        result["server_name"].fillna("")
        if "server_name" in result.columns
        else pd.Series([""] * len(result), index=result.index)
    ).str.lower()

    _ENV_PATTERNS = [
        ("prd", ["prd", "prod"]),
        ("dev", ["dev", "develop"]),
        ("stg", ["stg", "stage", "staging"]),
        ("sbx", ["sbx", "sandbox"]),
        ("test", ["test"]),
    ]
    _server_env = pd.Series("unknown", index=result.index)
    for _env_name, _tokens in _ENV_PATTERNS:
        _mask = _srv.str.contains("|".join(_tokens), regex=True, na=False)
        _server_env = _server_env.where(~_mask, _env_name)
    # 먼저 매칭된 것이 우선 (위에서 아래로 덮어씀) → 역순으로 적용
    _server_env_final = pd.Series("unknown", index=result.index)
    for _env_name, _tokens in reversed(_ENV_PATTERNS):
        _mask = _srv.str.contains("|".join(_tokens), regex=True, na=False)
        _server_env_final = _server_env_final.where(~_mask, _env_name)

    result["server_env"] = _server_env_final
    result["server_is_prod"] = (result["server_env"] == "prd").astype(int)

    _STACK_PATTERNS = [
        ("app", ["app"]),
        ("mms", ["mms"]),
        ("db", ["db"]),
        ("web", ["web"]),
        ("batch", ["batch"]),
    ]
    _server_stack = pd.Series("etc", index=result.index)
    for _stack_name, _tokens in reversed(_STACK_PATTERNS):
        _mask = _srv.str.contains("|".join(_tokens), regex=True, na=False)
        _server_stack = _server_stack.where(~_mask, _stack_name)

    result["server_stack"] = _server_stack

    return result


# ─────────────────────────────────────────────────────────────────────────────
# pk_file 집계 피처 (train fold only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_aggregates_label(df: pd.DataFrame) -> pd.DataFrame:
    """pk_file 단위 집계 피처 계산 (train fold 전용).

    누수 방지: 반드시 train fold DataFrame만 입력할 것.
    결과를 test fold에 merge_file_aggregates_label()로 left join.

    Parameters
    ----------
    df : pk_file 컬럼 포함 DataFrame

    Returns
    -------
    pk_file 기준 집계 DataFrame:
        pk_file             str
        file_event_count    int  : pk_file당 행 수
        file_pii_diversity  int  : ssn/phone/email 검출된 PII 유형 수 (0~3)
    """
    if "pk_file" not in df.columns:
        raise KeyError("pk_file 컬럼이 없습니다.")

    # 행 수 집계
    event_count = df.groupby("pk_file").size().rename("file_event_count")

    # PII 다양성 집계 - 각 유형이 하나라도 검출됐는지 (>0) 확인 후 합산
    available_pii = [c for c in _PII_COLS if c in df.columns]
    if available_pii:
        pii_numeric = df[available_pii].apply(
            lambda col: pd.to_numeric(col, errors="coerce").fillna(0)
        )
        pii_detected = (pii_numeric > 0).groupby(df["pk_file"]).any()
        pii_diversity = pii_detected.sum(axis=1).rename("file_pii_diversity")
    else:
        pii_diversity = event_count * 0  # 0으로 채움
        pii_diversity.name = "file_pii_diversity"

    agg = pd.concat([event_count, pii_diversity], axis=1).reset_index()
    agg["file_event_count"] = agg["file_event_count"].astype(int)
    agg["file_pii_diversity"] = agg["file_pii_diversity"].astype(int)

    return agg


def merge_file_aggregates_label(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """집계 결과를 df에 left join.

    Parameters
    ----------
    df  : 원본 DataFrame (pk_file 컬럼 포함)
    agg : compute_file_aggregates_label() 출력

    Returns
    -------
    agg 컬럼이 추가된 DataFrame (left join; 불일치 행 -> NaN)
    """
    return df.merge(agg, on="pk_file", how="left")
