"""Pytest 설정 및 공통 Fixture

테스트 실행:
    pytest tests/ -v
    pytest tests/test_filters.py -v
    pytest tests/ -v --tb=short
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_detection_df():
    """샘플 검출 데이터 DataFrame"""
    return pd.DataFrame({
        "detection_id": [f"det_{i}" for i in range(100)],
        "server_name": [f"server{i % 5}" for i in range(100)],
        "agent_ip": [f"10.0.0.{i % 256}" for i in range(100)],
        "file_path": [f"/app/data/file_{i}.txt" for i in range(100)],
        "detected_text_with_context": [
            f"Sample text {i} with context" for i in range(100)
        ],
        "pattern_type": np.random.choice(
            ["이메일", "휴대폰", "주민번호", "기타"], 100
        ),
    })


@pytest.fixture
def sample_label_df():
    """샘플 레이블 DataFrame"""
    labels = [
        "TP-실제개인정보",
        "FP-숫자나열/코드",
        "FP-더미데이터",
        "FP-타임스탬프",
        "FP-내부도메인",
        "FP-bytes크기",
        "FP-OS저작권",
        "FP-패턴맥락",
    ]

    return pd.DataFrame({
        "detection_id": [f"det_{i}" for i in range(80)],  # 일부만 레이블 있음
        "label": np.random.choice(labels, 80),
    })


@pytest.fixture
def sample_text_series():
    """샘플 텍스트 Series"""
    texts = [
        "user@lguplus.co.kr",
        "admin@bdp.lguplus.co.kr",
        "dev@redhat.com",
        "test@example.com",
        "timestamp: 1704067200",
        "size: 45 bytes 141022",
        "version: 1.3.3.32-2087-1512",
        "/var/lib/docker/overlay2/abc123/file.txt",
        "hadoop-cmf-hdfs-DATANODE-server01.log",
        "고객명: 홍길동, 연락처: ***-****-5678",
        "일반 텍스트입니다",
        "Another normal text without patterns",
    ]
    return pd.Series(texts)


@pytest.fixture
def masked_text_series():
    """마스킹된 텍스트 Series"""
    return pd.Series([
        "이메일: ***@***.com",
        "연락처: 010-****-5678",
        "주민번호: ******-*******",
        "고객명: 홍*동",
        "plain text without masking",
    ])


@pytest.fixture
def pk_config():
    """PK 설정 딕셔너리"""
    return {
        "primary": ["server_name", "agent_ip", "file_path"],
        "fallback": ["detection_id"],
    }


@pytest.fixture
def filter_config():
    """필터 설정 딕셔너리"""
    return {
        "keyword_filter": {
            "internal_domains": [
                "@lguplus.co.kr",
                "@bdp.lguplus.co.kr",
            ],
            "os_copyright_domains": [
                "@redhat.com",
                "@apache.org",
            ],
            "dummy_domains": [
                "@example.com",
                "@test.com",
            ],
        },
        "rule_filter": {
            "timestamp_patterns": [
                r"\b\d{10}\b",
                r"\b\d{13}\b",
            ],
            "bytes_patterns": [
                r"\d+\s*bytes?\b",
            ],
        },
    }


# pytest 설정
def pytest_configure(config):
    """Pytest 설정"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
