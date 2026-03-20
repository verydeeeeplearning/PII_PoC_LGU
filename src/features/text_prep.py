"""
Stage S2-1: Text Preprocessing - Architecture.md §6.2, §6.3

raw_text:  소문자 + 고엔트로피 placeholder 치환
shape_text: 숫자->0, 영문->a, 한글->가, 특수문자 유지
path_text:  파일 경로 토큰화

원칙 E (멀티뷰 텍스트):
  - raw_text: 의미 신호 (키워드/도메인)
  - shape_text: 구조 신호 (숫자열/마스킹 패턴)
  - path_text: 경로 컨텍스트 신호
"""

from __future__ import annotations

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder 치환 규칙 (순서 중요: 긴 패턴 먼저)
# ─────────────────────────────────────────────────────────────────────────────

# (regex, placeholder) 순서대로 적용
# 긴 패턴을 먼저 처리해야 부분 치환으로 인한 오작동 방지
_PLACEHOLDERS: list[tuple[str, str]] = [
    (r"\*{3,}", "<MASK>"),                      # ***+ -> <MASK> (마스킹 패턴)
    (r"[0-9a-f]{32,}", "<HASH>"),               # MD5/SHA 해시 (소문자 hex 32자+)
    (r"0x[0-9a-f]{6,}", "<HEX>"),               # 0x로 시작하는 hex 주소
    (r"\b\d{13}\b", "<NUM13>"),                 # 13자리 (Unix timestamp ms / 주민번호)
    (r"\b\d{10}\b", "<NUM10>"),                 # 10자리 (Unix timestamp)
    (r"\b\d{8}\b", "<DATE8>"),                  # 8자리 (날짜 YYYYMMDD)
    (r"\b\d{6,9}\b", "<NUMLONG>"),              # 6~9자리 숫자열
]

_COMPILED_PLACEHOLDERS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _PLACEHOLDERS
]

# 경로 구분자 패턴
_PATH_SEP = re.compile(r"[/\\_.\-]+")


# ─────────────────────────────────────────────────────────────────────────────
# make_raw_text
# ─────────────────────────────────────────────────────────────────────────────

def make_raw_text(text: Optional[str]) -> str:
    """
    raw_text 생성 - 소문자 + 고엔트로피 토큰 placeholder 치환.

    Architecture.md §6.2:
      - 키워드/도메인 등 판별 신호는 유지
      - UUID/해시/긴 숫자열 등 고엔트로피 값만 placeholder로 치환
      - placeholder에 길이 정보 포함 (<NUM10>, <NUM13>) -> 판별력 보존

    Parameters
    ----------
    text : 원문 (full_context_raw 또는 local_context_raw)

    Returns
    -------
    소문자화 + placeholder 치환된 문자열
    """
    if not text:
        return ""
    result = text.lower()
    for pattern, placeholder in _COMPILED_PLACEHOLDERS:
        result = pattern.sub(placeholder, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# make_shape_text
# ─────────────────────────────────────────────────────────────────────────────

def make_shape_text(text: Optional[str]) -> str:
    """
    shape_text 생성 - 문자의 '형태(shape)'를 추출.

    Architecture.md §6.2:
      - 숫자 -> '0'
      - 영문(ASCII) -> 'a'
      - 한글 -> '가'
      - 특수문자/구분자(@, -, ., *, =, / 등) -> 유지

    마스킹 환경에서 내용이 사라진 뒤에도 구조적 패턴이 핵심 신호가 된다.

    Examples
    --------
    "xpiryDate=170603*****" -> "aaaaaaaaa=000000*****"
    "****@bdp.lguplus.co.kr" -> "****@aaa.aaaaaaa.aa.aa"
    """
    if not text:
        return ""
    result = []
    for ch in text:
        if ch.isdigit():
            result.append("0")
        elif ch.isascii() and ch.isalpha():
            result.append("a")
        elif "\uAC00" <= ch <= "\uD7A3":  # 한글 완성형
            result.append("가")
        else:
            result.append(ch)  # 특수문자, *, @, -, . 등 유지
    return "".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# make_path_text
# ─────────────────────────────────────────────────────────────────────────────

def make_path_text(file_path: Optional[str]) -> str:
    """
    path_text 생성 - 파일 경로를 토큰화된 텍스트로 변환.

    Architecture.md §6.3:
      /var/log/hadoop/abc -> "var log hadoop abc"

    구분자: /, \\, _, ., -
    빈 토큰 제거, 소문자화.

    Parameters
    ----------
    file_path : 파일 경로 문자열

    Returns
    -------
    공백으로 연결된 소문자 토큰 문자열
    """
    if not file_path:
        return ""
    fp = file_path.lower()
    tokens = _PATH_SEP.split(fp)
    tokens = [t for t in tokens if t]
    return " ".join(tokens)
