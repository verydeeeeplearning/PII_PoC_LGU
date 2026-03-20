"""
Stage S1: Normalize & Parse - Architecture.md §5

핵심 역할:
    1행 = 1검출 이벤트 단위로 분해 (3단 폴백 파싱)
    PK 생성 (pk_file, pk_event)
    PII 유형 재추론 (pii_type_inferred)
    Schema Registry 적용 + quarantine
    parse_success_rate KPI 산출

폴백 순서:
    1차: *{3,} 마스킹 패턴 기반 파싱  -> parse_status = "OK"
    2차: masked_hit 앵커 기반 위치 탐색 -> parse_status = "FALLBACK_ANCHOR"
    3차: row 전체를 단일 이벤트로 생성  -> parse_status = "FALLBACK_SINGLE_EVENT"
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# S1ParserConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class S1ParserConfig:
    """S1 파서 파라미터 설정."""
    window_size: int = 60
    masking_pattern: str = r'(.{0,15})(\S*\*{3,}\S*)(.{0,15})'
    pattern_truncation: int = 50
    context_truncation: int = 200

    @classmethod
    def defaults(cls) -> "S1ParserConfig":
        return cls()

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None) -> "S1ParserConfig":
        """preprocessing_config.yaml의 parsing.s1_parser 섹션 로드. 실패 시 defaults."""
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "preprocessing_config.yaml"
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            s1_cfg = cfg.get("parsing", {}).get("s1_parser", {})
            if not s1_cfg:
                return cls()
            return cls(
                window_size=s1_cfg.get("window_size", 60),
                masking_pattern=s1_cfg.get("masking_pattern", r'(.{0,15})(\S*\*{3,}\S*)(.{0,15})'),
                pattern_truncation=s1_cfg.get("pattern_truncation", 50),
                context_truncation=s1_cfg.get("context_truncation", 200),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("S1ParserConfig YAML 로드 실패 -> defaults 사용: %s", e)
            return cls()


# ─────────────────────────────────────────────────────────────────────────────
# PK 생성
# ─────────────────────────────────────────────────────────────────────────────

def make_pk_file(server_name: str, agent_ip: str, file_path: str) -> str:
    """pk_file = SHA256(server_name|agent_ip|file_path)"""
    raw = f"{server_name}|{agent_ip}|{file_path}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_pk_event(pk_file: str, event_index: int) -> str:
    """pk_event = SHA256(pk_file|event_index)"""
    raw = f"{pk_file}_{event_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 3단 폴백 파서
# ─────────────────────────────────────────────────────────────────────────────

def parse_context_field(
    raw_text: Optional[str],
    pk_file: str,
    masked_hit: Optional[str] = None,
    window: int = 60,
    config: Optional["S1ParserConfig"] = None,
) -> list[dict]:
    """
    하나의 컨텍스트 셀에서 개별 검출 건을 분리.

    Parameters
    ----------
    raw_text:   full_context_raw 필드 원문
    pk_file:    파일 단위 PK (make_pk_file로 생성)
    masked_hit: 마스킹된 검출값 (2차 폴백 앵커)
    window:     앵커 기반 컨텍스트 윈도우 (chars) - config 없을 때 사용
    config:     S1ParserConfig 인스턴스 (None이면 defaults 사용)

    Returns
    -------
    list of event dicts, 각 dict 포함 키:
        pk_event, pk_file, event_index,
        left_ctx, masked_pattern, right_ctx,
        full_context, parse_status
    """
    if not raw_text or (isinstance(raw_text, float) and pd.isna(raw_text)):
        return []

    _cfg = config or S1ParserConfig.defaults()
    _window = window if config is None else _cfg.window_size

    # 1차: *{3,} 마스킹 패턴 기반
    results = _parse_by_masking_pattern(raw_text, pk_file, config=_cfg)
    if results:
        for r in results:
            r["parse_status"] = "OK"
        return results

    # 2차: masked_hit 앵커 기반
    if masked_hit:
        results = _parse_by_anchor(raw_text, pk_file, masked_hit, window=_window, config=_cfg)
        if results:
            for r in results:
                r["parse_status"] = "FALLBACK_ANCHOR"
            return results

    # 3차: 전체를 단일 이벤트로
    pk_event = make_pk_event(pk_file, 0)
    return [{
        "pk_event": pk_event,
        "pk_file": pk_file,
        "event_index": 0,
        "left_ctx": "",
        "masked_pattern": raw_text[:_cfg.pattern_truncation],
        "right_ctx": "",
        "full_context": raw_text[:_cfg.context_truncation],
        "parse_status": "FALLBACK_SINGLE_EVENT",
    }]


def _parse_by_masking_pattern(
    raw_text: str,
    pk_file: str,
    config: Optional["S1ParserConfig"] = None,
) -> list[dict]:
    """1차: *{3,} 패턴 기반 파싱"""
    _cfg = config or S1ParserConfig.defaults()
    pattern = _cfg.masking_pattern
    results = []
    event_index = 0

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        for m in re.finditer(pattern, line):
            left_ctx = m.group(1).strip()
            masked_pattern = m.group(2)
            right_ctx = m.group(3).strip()
            full_context = f"{left_ctx} {masked_pattern} {right_ctx}".strip()

            pk_event = make_pk_event(pk_file, event_index)
            results.append({
                "pk_event": pk_event,
                "pk_file": pk_file,
                "event_index": event_index,
                "left_ctx": left_ctx,
                "masked_pattern": masked_pattern,
                "right_ctx": right_ctx,
                "full_context": full_context,
            })
            event_index += 1

    return results


def _parse_by_anchor(
    raw_text: str,
    pk_file: str,
    masked_hit: str,
    window: int = 60,
    config: Optional["S1ParserConfig"] = None,
) -> list[dict]:
    """2차: masked_hit 앵커 기반 위치 탐색"""
    _cfg = config or S1ParserConfig.defaults()
    _window = window if config is None else _cfg.window_size

    idx = raw_text.find(masked_hit)
    if idx == -1:
        return []

    start = max(0, idx - _window)
    end = min(len(raw_text), idx + len(masked_hit) + _window)
    full_context = raw_text[start:end]

    pk_event = make_pk_event(pk_file, 0)
    return [{
        "pk_event": pk_event,
        "pk_file": pk_file,
        "event_index": 0,
        "left_ctx": raw_text[start:idx].strip(),
        "masked_pattern": masked_hit,
        "right_ctx": raw_text[idx + len(masked_hit):end].strip(),
        "full_context": full_context,
    }]


# ─────────────────────────────────────────────────────────────────────────────
# 이메일 도메인 추출 (L1 RULE Labeler 전처리)
# ─────────────────────────────────────────────────────────────────────────────

_RE_MASKED_DOMAIN = re.compile(r"@([\w.\-]+)", re.IGNORECASE)


def extract_email_domain(masked_content: Optional[str]) -> Optional[str]:
    """마스킹된 이메일에서 도메인 파트 추출.

    Server-i 마스킹 형식: 'park.js***@lguplus.co.kr' (로컬파트만 마스킹)
    도메인은 노출되므로 @ 이후 문자열을 추출한다.

    Parameters
    ----------
    masked_content : dfile_inspectmaskedcontent 원문

    Returns
    -------
    도메인 문자열 (소문자) 또는 None (이메일 아닌 경우)

    Examples
    --------
    >>> extract_email_domain('park.js***@lguplus.co.kr')
    'lguplus.co.kr'
    >>> extract_email_domain('leejw***@redhat.com')
    'redhat.com'
    >>> extract_email_domain('810215-1*****')
    None
    """
    if not masked_content:
        return None
    m = _RE_MASKED_DOMAIN.search(masked_content)
    return m.group(1).lower() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# PII 유형 재추론
# ─────────────────────────────────────────────────────────────────────────────

def infer_pii_type(masked_pattern: str, full_context: str) -> str:
    """
    패턴 종류 필드의 신뢰도 이슈 대응.
    검출 내역의 형태적 특성으로 PII 유형을 재추론.

    우선순위: email > phone > rrn > unknown
    """
    if "@" in full_context:
        return "email"
    if re.search(r"01[016789][\-\*]", full_context):
        return "phone"
    if re.search(r"\d{6}[\-\*]\d?\*", masked_pattern):
        return "rrn"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Schema Registry 적용
# ─────────────────────────────────────────────────────────────────────────────

def apply_schema_registry(
    df: pd.DataFrame,
    schema: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Schema Registry 정책을 DataFrame에 적용.

    1. rename_map: 원본 컬럼명 -> 정규 컬럼명
    2. required_columns 누락 행 -> quarantine_df
    3. 정상 행 -> result_df

    Parameters
    ----------
    df:     입력 DataFrame (원본 컬럼명 그대로)
    schema: schema_registry 딕셔너리
        {
            "rename_map": {원본: 정규, ...},
            "required_columns": [정규 컬럼명, ...],
            "on_missing_required": "quarantine",
        }

    Returns
    -------
    (result_df, quarantine_df)
    """
    rename_map: dict = schema.get("rename_map", {})
    required_columns: list = schema.get("required_columns", [])

    # 1단계: rename_map 적용 (존재하는 컬럼만)
    applicable_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=applicable_rename)

    # 2단계: 필수 컬럼 누락 행 격리
    missing_mask = pd.Series(False, index=df.index)
    for col in required_columns:
        if col not in df.columns:
            # 컬럼 자체가 없으면 전체 행이 missing
            missing_mask = pd.Series(True, index=df.index)
            break
        col_missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        missing_mask = missing_mask | col_missing

    quarantine_df = df[missing_mask].copy()
    quarantine_df["quarantine_reason"] = "MISSING_REQUIRED_FIELD"

    result_df = df[~missing_mask].copy()

    return result_df, quarantine_df


# ─────────────────────────────────────────────────────────────────────────────
# 컨텍스트 정규화 (비파괴 - 원문 보존 + norm 파생)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_context(raw_text: str) -> str:
    """
    full_context_norm 생성 (Architecture.md §5.1 #6).
    원문(full_context_raw)을 변환하지 않고 파생 컬럼으로만 사용.

    처리 순서:
        1. NFKC 유니코드 정규화
        2. Windows/Unix 줄바꿈 통일 (\\r\\n -> \\n)
        3. 탭 -> 공백
        4. 과도한 공백 축소 (2+ spaces -> 1)
    """
    if not raw_text:
        return raw_text
    normalized = unicodedata.normalize("NFKC", raw_text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized.strip()


# ─────────────────────────────────────────────────────────────────────────────
# KPI: parse_success_rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_parse_success_rate(
    events_df: pd.DataFrame,
    original_row_count: int,
) -> float:
    """
    parse_success_rate = #생성 pk_event / #원본 row

    Architecture.md §5.4:
        parse_success_rate가 95% 미만으로 떨어지면 경보 발생.
    """
    if original_row_count == 0:
        return 0.0
    return float(len(events_df)) / float(original_row_count)
