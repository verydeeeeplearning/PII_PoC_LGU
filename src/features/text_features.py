"""텍스트 Feature 생성 모듈

회의록 2026-01 반영:
- 내부 도메인 Feature
- OS 저작권 도메인 Feature
- Timestamp 패턴 Feature
- Bytes 패턴 Feature
"""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Tuple, Dict
import joblib

from src.utils.constants import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF, TFIDF_MAX_DF, KEYWORD_GROUPS,
)


# ============================================================
# 회의록 확정 도메인/패턴 리스트
# ============================================================

# 내부 도메인 리스트 (LG U+)
INTERNAL_DOMAINS = [
    "@lguplus.co.kr",
    "@bdp.lguplus.co.kr",
    "@lge.lgt.co.kr",
    "@lgcns.com",
    "@lgelectronics.com",
    # 대문자 변형 (Kerberos 토큰 등)
    "@LGUPLUS.CO.KR",
    "@BDP.LGUPLUS.CO.KR",
    "@BPSS61.LGUPLUS.CO.KR",
]

# OS/오픈소스 저작권 도메인
OS_COPYRIGHT_DOMAINS = [
    "@redhat.com",
    "@fedora-project",
    "@fedoraproject.org",
    "@apache.org",
    "@gnu.org",
    "@fsf.org",
    "@kernel.org",
    "@ubuntu.com",
    "@debian.org",
    "@centos.org",
    # 벤더 도메인
    "@paloaltonetworks.com",
    "@cisco.com",
    "@oracle.com",
    "@microsoft.com",
    "@sun.com",
    "@ibm.com",
]

# 더미 도메인
DUMMY_DOMAINS = [
    "@example.com",
    "@example.org",
    "@test.com",
    "@localhost",
    "@127.0.0.1",
    "@entry.sc",
    "@cherry.email",
]

# Timestamp 패턴 정규식
TIMESTAMP_PATTERNS = [
    r"\b\d{10}\b",                         # Unix timestamp (초)
    r"\b\d{13}\b",                         # Unix timestamp (밀리초)
    r"\d{4}[-/]\d{2}[-/]\d{2}",            # YYYY-MM-DD
    r"\d{2}[-/]\d{2}[-/]\d{4}",            # DD-MM-YYYY
    r"(?:timestamp|time|date)[:=]\s*\d+",  # timestamp 키워드
    r"xpiry[Dd]ate=\d+",                   # Kerberos expiry
    r"duration:\s*\d+",                    # duration 값
]

# Bytes 크기 패턴 정규식
BYTES_PATTERNS = [
    r"\d+\s*bytes?",                       # N bytes
    r"\d+\s*[KMGT]B",                      # 파일 크기 (KB, MB 등)
    r"size[:=]\s*\d+",                     # size 키워드
]

# 버전번호 패턴 정규식
VERSION_PATTERNS = [
    r"\d+\.\d+\.\d+[-.]?\d*",              # 1.2.3.4
    r"v\d+\.\d+",                          # v1.2
    r"JGNORE\s*=\s*\d+",                   # JGNORE 코드
]


# ============================================================
# 기존 Feature 함수
# ============================================================

def create_tfidf_features(
    train_texts: pd.Series,
    test_texts: Optional[pd.Series] = None,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
    save_path: Optional[str] = None,
) -> Tuple:
    """
    TF-IDF Feature 생성

    Args:
        train_texts: 학습 텍스트 Series
        test_texts: 테스트 텍스트 Series (None이면 학습 전용)
        max_features: 최대 Feature 수
        ngram_range: n-gram 범위
        min_df: 최소 문서 빈도
        max_df: 최대 문서 빈도 비율
        save_path: TfidfVectorizer 저장 경로

    Returns:
        test_texts 있음: (train_tfidf, test_tfidf, vectorizer)
        test_texts 없음: (train_tfidf, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )

    train_filled = train_texts.fillna("")
    tfidf_train = vectorizer.fit_transform(train_filled)

    print(f"[TF-IDF 결과]")
    print(f"  학습 행렬: {tfidf_train.shape}")
    print(f"  Vocabulary 크기: {len(vectorizer.vocabulary_):,}")

    if save_path is not None:
        joblib.dump(vectorizer, save_path)
        print(f"  [저장] {save_path}")

    if test_texts is not None:
        test_filled = test_texts.fillna("")
        tfidf_test = vectorizer.transform(test_filled)
        print(f"  테스트 행렬: {tfidf_test.shape}")
        return tfidf_train, tfidf_test, vectorizer

    return tfidf_train, vectorizer


def create_keyword_features(
    texts: pd.Series,
    keyword_groups: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    키워드 존재 여부 Feature 생성

    Args:
        texts: 텍스트 Series
        keyword_groups: {feature명: [키워드, ...]} (기본값: constants.KEYWORD_GROUPS)

    Returns:
        키워드 Feature DataFrame (각 컬럼 0/1)
    """
    if keyword_groups is None:
        keyword_groups = KEYWORD_GROUPS

    features = {}
    lower_texts = texts.fillna("").str.lower()

    for group_name, keywords in keyword_groups.items():
        pattern = "|".join(re.escape(kw) for kw in keywords)
        features[group_name] = lower_texts.str.contains(
            pattern, regex=True
        ).astype(int)

    df_kw = pd.DataFrame(features)
    print(f"[키워드 Feature] {df_kw.shape[1]}개 생성")
    return df_kw


def create_text_stat_features(texts: pd.Series) -> pd.DataFrame:
    """
    텍스트 통계 Feature 생성

    생성 Feature (8개):
        - text_length: 문자열 길이
        - word_count: 단어 수
        - digit_count: 숫자 문자 수
        - digit_ratio: 숫자 비율
        - special_char_ratio: 특수문자 비율
        - uppercase_ratio: 대문자 비율
        - has_email_pattern: 이메일 패턴 존재 여부 (0/1)
        - has_phone_pattern: 전화번호 패턴 존재 여부 (0/1)

    Args:
        texts: 텍스트 Series

    Returns:
        통계 Feature DataFrame (8개 컬럼)
    """
    texts_filled = texts.fillna("")

    features = {
        "text_length": texts_filled.str.len(),
        "word_count": texts_filled.str.split().str.len().fillna(0).astype(int),
        "digit_count": texts_filled.str.count(r'\d'),
        "digit_ratio": texts_filled.apply(
            lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)
        ),
        "special_char_ratio": texts_filled.apply(
            lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / max(len(x), 1)
        ),
        "uppercase_ratio": texts_filled.apply(
            lambda x: sum(c.isupper() for c in x) / max(len(x), 1)
        ),
        "has_email_pattern": texts_filled.str.contains(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', regex=True
        ).astype(int),
        "has_phone_pattern": texts_filled.str.contains(
            r'\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4}', regex=True
        ).astype(int),
    }

    df_stats = pd.DataFrame(features)
    print(f"[텍스트 통계 Feature] {df_stats.shape[1]}개 생성")
    return df_stats


# ============================================================
# 회의록 반영 신규 Feature 함수
# ============================================================

def create_domain_features(
    texts: pd.Series,
    internal_domains: Optional[List[str]] = None,
    os_copyright_domains: Optional[List[str]] = None,
    dummy_domains: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    도메인 관련 Feature 생성

    회의록 확정:
    - 내부 도메인 (@lguplus.co.kr 등)
    - OS 저작권 도메인 (@redhat.com 등)
    - 더미 도메인 (@example.com 등)

    생성 Feature (3개):
        - has_internal_domain: 내부 도메인 포함 여부 (0/1)
        - has_os_copyright_domain: OS 저작권 도메인 포함 여부 (0/1)
        - has_dummy_domain: 더미 도메인 포함 여부 (0/1)

    Args:
        texts: 텍스트 Series
        internal_domains: 내부 도메인 리스트 (기본값: INTERNAL_DOMAINS)
        os_copyright_domains: OS 저작권 도메인 리스트 (기본값: OS_COPYRIGHT_DOMAINS)
        dummy_domains: 더미 도메인 리스트 (기본값: DUMMY_DOMAINS)

    Returns:
        도메인 Feature DataFrame (3개 컬럼)
    """
    if internal_domains is None:
        internal_domains = INTERNAL_DOMAINS
    if os_copyright_domains is None:
        os_copyright_domains = OS_COPYRIGHT_DOMAINS
    if dummy_domains is None:
        dummy_domains = DUMMY_DOMAINS

    texts_lower = texts.fillna("").str.lower()

    def check_domains(text: str, domains: List[str]) -> bool:
        text_lower = text.lower()
        return any(d.lower() in text_lower for d in domains)

    features = {
        "has_internal_domain": texts_lower.apply(
            lambda x: int(check_domains(x, internal_domains))
        ),
        "has_os_copyright_domain": texts_lower.apply(
            lambda x: int(check_domains(x, os_copyright_domains))
        ),
        "has_dummy_domain": texts_lower.apply(
            lambda x: int(check_domains(x, dummy_domains))
        ),
    }

    df_domain = pd.DataFrame(features)
    print(f"[도메인 Feature] {df_domain.shape[1]}개 생성")
    print(f"  내부 도메인: {features['has_internal_domain'].sum():,}건")
    print(f"  OS 저작권:  {features['has_os_copyright_domain'].sum():,}건")
    print(f"  더미:       {features['has_dummy_domain'].sum():,}건")

    return df_domain


def create_pattern_features(
    texts: pd.Series,
    timestamp_patterns: Optional[List[str]] = None,
    bytes_patterns: Optional[List[str]] = None,
    version_patterns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    특정 패턴 존재 여부 Feature 생성

    회의록 확정:
    - Timestamp 패턴 (Unix timestamp, 날짜 형식 등)
    - Bytes 크기 패턴 (N bytes, KB/MB 등)
    - 버전번호 패턴 (1.2.3.4, v1.2 등)

    생성 Feature (3개):
        - has_timestamp_pattern: 타임스탬프 패턴 포함 여부 (0/1)
        - has_bytes_pattern: 바이트 크기 패턴 포함 여부 (0/1)
        - has_version_pattern: 버전번호 패턴 포함 여부 (0/1)

    Args:
        texts: 텍스트 Series
        timestamp_patterns: 타임스탬프 정규식 리스트
        bytes_patterns: 바이트 정규식 리스트
        version_patterns: 버전번호 정규식 리스트

    Returns:
        패턴 Feature DataFrame (3개 컬럼)
    """
    if timestamp_patterns is None:
        timestamp_patterns = TIMESTAMP_PATTERNS
    if bytes_patterns is None:
        bytes_patterns = BYTES_PATTERNS
    if version_patterns is None:
        version_patterns = VERSION_PATTERNS

    texts_filled = texts.fillna("")

    def check_patterns(text: str, patterns: List[str]) -> bool:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    features = {
        "has_timestamp_pattern": texts_filled.apply(
            lambda x: int(check_patterns(x, timestamp_patterns))
        ),
        "has_bytes_pattern": texts_filled.apply(
            lambda x: int(check_patterns(x, bytes_patterns))
        ),
        "has_version_pattern": texts_filled.apply(
            lambda x: int(check_patterns(x, version_patterns))
        ),
    }

    df_pattern = pd.DataFrame(features)
    print(f"[패턴 Feature] {df_pattern.shape[1]}개 생성")
    print(f"  Timestamp: {features['has_timestamp_pattern'].sum():,}건")
    print(f"  Bytes:     {features['has_bytes_pattern'].sum():,}건")
    print(f"  Version:   {features['has_version_pattern'].sum():,}건")

    return df_pattern


def create_masking_features(texts: pd.Series) -> pd.DataFrame:
    """
    마스킹 관련 Feature 생성

    생성 Feature (3개):
        - masking_count: 마스킹 패턴 (*****) 출현 횟수
        - masking_ratio: 마스킹 문자 비율
        - context_length: 마스킹 제외 컨텍스트 길이

    Args:
        texts: 텍스트 Series

    Returns:
        마스킹 Feature DataFrame (3개 컬럼)
    """
    texts_filled = texts.fillna("")

    # 마스킹 패턴 카운트
    masking_count = texts_filled.str.count(r'\*{3,}')

    # 마스킹 문자 비율
    def calc_mask_ratio(text: str) -> float:
        if not text:
            return 0.0
        mask_chars = text.count('*')
        return mask_chars / len(text)

    masking_ratio = texts_filled.apply(calc_mask_ratio)

    # 컨텍스트 길이 (마스킹 제거 후)
    context_length = texts_filled.str.replace(r'\*+', '', regex=True).str.len()

    features = {
        "masking_count": masking_count,
        "masking_ratio": masking_ratio,
        "context_length": context_length,
    }

    df_mask = pd.DataFrame(features)
    print(f"[마스킹 Feature] {df_mask.shape[1]}개 생성")

    return df_mask


def create_all_text_features(
    texts: pd.Series,
    include_domain: bool = True,
    include_pattern: bool = True,
    include_masking: bool = True,
) -> pd.DataFrame:
    """
    모든 텍스트 Feature를 통합 생성

    Args:
        texts: 텍스트 Series
        include_domain: 도메인 Feature 포함 여부
        include_pattern: 패턴 Feature 포함 여부
        include_masking: 마스킹 Feature 포함 여부

    Returns:
        통합 Feature DataFrame
    """
    print("\n[텍스트 Feature 통합 생성]")

    feature_dfs = []

    # 1. 키워드 Feature (7개)
    df_kw = create_keyword_features(texts)
    feature_dfs.append(df_kw)

    # 2. 텍스트 통계 Feature (8개)
    df_stat = create_text_stat_features(texts)
    feature_dfs.append(df_stat)

    # 3. 도메인 Feature (3개) - 회의록 반영
    if include_domain:
        df_domain = create_domain_features(texts)
        feature_dfs.append(df_domain)

    # 4. 패턴 Feature (3개) - 회의록 반영
    if include_pattern:
        df_pattern = create_pattern_features(texts)
        feature_dfs.append(df_pattern)

    # 5. 마스킹 Feature (3개)
    if include_masking:
        df_mask = create_masking_features(texts)
        feature_dfs.append(df_mask)

    # 통합
    df_all = pd.concat(feature_dfs, axis=1)

    print(f"\n[통합 결과] 총 {df_all.shape[1]}개 Feature 생성")
    return df_all
