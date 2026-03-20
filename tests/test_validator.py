"""데이터 검증 모듈 테스트

회의록 2026-01 반영:
- 마스킹 검증
- 패턴 타입 검증
- PII 노출 검사
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np

from src.data.validator import (
    validate_data,
    validate_masking,
    validate_pattern_type,
    auto_correct_pattern_type,
    full_validation,
    MaskingValidationResult,
    PatternTypeValidationResult,
)


class TestValidateData:
    """기본 데이터 검증 테스트"""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "text": ["hello", "world", None, "test"],
            "label": ["TP-실제개인정보", "FP-더미데이터", "FP-더미데이터", "TP-실제개인정보"],
            "value": [1, 2, 3, 4],
        })

    def test_basic_validation(self, sample_df):
        """기본 검증 테스트"""
        result = validate_data(sample_df, label_column="label")

        assert isinstance(result, dict)
        assert result["n_rows"] == 4
        assert result["n_cols"] == 3
        assert len(result["label_distribution"]) == 2

    def test_duplicate_detection(self):
        """중복 행 감지"""
        df = pd.DataFrame({
            "text": ["a", "a", "b"],
            "label": ["TP", "TP", "FP"],
        })

        result = validate_data(df, label_column="label")
        assert result["n_duplicates"] == 1


class TestMaskingValidation:
    """마스킹 검증 테스트"""

    @pytest.fixture
    def masked_df(self):
        return pd.DataFrame({
            "text": [
                "고객명: 홍***",
                "이메일: ***@gmail.com",
                "연락처: 010-****-5678",
                "plain text without masking",
            ]
        })

    def test_masking_detection(self, masked_df):
        """마스킹 패턴 감지"""
        result = validate_masking(masked_df, "text")

        assert isinstance(result, MaskingValidationResult)
        assert result.total_rows == 4
        assert result.masked_rows == 3  # 3개 마스킹됨
        assert result.masking_rate > 0.5

    def test_context_length_calculation(self, masked_df):
        """컨텍스트 길이 계산"""
        result = validate_masking(masked_df, "text")

        # 컨텍스트 길이 통계 확인
        assert "mean" in result.context_length_stats
        assert "max" in result.context_length_stats
        assert result.context_length_stats["mean"] > 0

    def test_pii_exposure_detection(self):
        """PII 노출 감지"""
        df = pd.DataFrame({
            "text": [
                "마스킹됨: ***@***.com",
                "노출됨: 010-1234-5678",  # 마스킹 안 됨
                "주민번호: 901231-1234567",  # 마스킹 안 됨
            ]
        })

        result = validate_masking(df, "text")

        # 노출이 감지되어야 함
        assert result.has_exposure is True
        assert result.exposed_phone > 0 or result.exposed_jumin > 0

    def test_empty_column(self):
        """빈 컬럼 처리"""
        df = pd.DataFrame({
            "text": [None, None, ""]
        })

        result = validate_masking(df, "text")
        assert result.total_rows == 3
        assert result.masked_rows == 0
        assert result.masking_rate == 0


class TestPatternTypeValidation:
    """패턴 타입 검증 테스트"""

    @pytest.fixture
    def pattern_df(self):
        return pd.DataFrame({
            "content": [
                "user@example.com",      # 이메일
                "010-1234-5678",          # 휴대폰
                "901231-1234567",         # 주민번호
                "user@lguplus.co.kr",    # 이메일
            ],
            "pattern_type": [
                "이메일",       # 정확
                "휴대폰",       # 정확
                "이메일",       # 불일치! 주민번호인데 이메일로 표기
                "주민번호",     # 불일치! 이메일인데 주민번호로 표기
            ]
        })

    def test_pattern_mismatch_detection(self, pattern_df):
        """패턴 불일치 감지"""
        result = validate_pattern_type(
            pattern_df, "content", "pattern_type"
        )

        assert isinstance(result, PatternTypeValidationResult)
        assert result.total_rows == 4
        # 불일치가 있어야 함
        assert result.mismatched_rows > 0

    def test_email_pattern_validation(self):
        """이메일 패턴 검증"""
        df = pd.DataFrame({
            "content": ["user@domain.com", "another@test.org"],
            "pattern_type": ["주민번호", "휴대폰"]  # 둘 다 불일치
        })

        result = validate_pattern_type(df, "content", "pattern_type")
        assert result.mismatch_rate == 1.0  # 100% 불일치

    def test_phone_pattern_validation(self):
        """전화번호 패턴 검증"""
        # 패턴 탐지기는 01x로 시작하고 5자리 이상의 숫자/* 조합을 기대
        df = pd.DataFrame({
            "content": ["01012345678", "010****5678"],
            "pattern_type": ["휴대폰번호", "휴대폰"]  # 정규화 후 일치
        })

        result = validate_pattern_type(df, "content", "pattern_type")
        # 정규화 후 일치해야 함
        assert result.matched_rows >= 1


class TestAutoCorrectPatternType:
    """패턴 타입 자동 보정 테스트"""

    def test_auto_correct(self):
        """자동 보정 테스트"""
        df = pd.DataFrame({
            "content": [
                "user@example.com",
                "010-1234-5678",
            ],
            "pattern_type": [
                "주민번호",  # 잘못됨
                "이메일",    # 잘못됨
            ]
        })

        corrected_df = auto_correct_pattern_type(
            df, "content", "pattern_type"
        )

        # 보정된 컬럼이 추가되어야 함
        assert "pattern_type_corrected" in corrected_df.columns
        # 이메일로 보정되어야 함
        assert corrected_df["pattern_type_corrected"].iloc[0] == "이메일"

    def test_no_correction_needed(self):
        """보정 불필요한 경우"""
        df = pd.DataFrame({
            "content": ["user@example.com"],
            "pattern_type": ["이메일"]  # 정확함
        })

        corrected_df = auto_correct_pattern_type(
            df, "content", "pattern_type"
        )

        # 원본과 동일해야 함
        assert corrected_df["pattern_type_corrected"].iloc[0] == "이메일"


class TestFullValidation:
    """통합 검증 테스트"""

    def test_full_validation_report(self):
        """전체 검증 리포트"""
        df = pd.DataFrame({
            "text": [
                "마스킹: ***@***.com",
                "user@lguplus.co.kr",
            ],
            "label": ["TP-실제개인정보", "FP-내부도메인"],
            "pattern_type": ["이메일", "이메일"],
        })

        report = full_validation(
            df,
            text_column="text",
            label_column="label",
            pattern_type_column="pattern_type"
        )

        assert "basic" in report
        assert "masking" in report
        # pattern_type 검증도 수행됨
        assert "pattern_type" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
