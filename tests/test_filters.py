"""3-Layer Filter 모듈 테스트

회의록 2026-01 반영:
- 키워드 필터 (내부 도메인, OS 저작권)
- 룰 필터 (타임스탬프, bytes, 버전)
- 파이프라인 통합 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd

from src.filters import KeywordFilter, RuleFilter, FilterPipeline
from src.filters.base_filter import FilterResult


class TestKeywordFilter:
    """키워드 필터 테스트"""

    @pytest.fixture
    def keyword_filter(self):
        """내부 도메인 등이 설정된 키워드 필터"""
        config = {
            "internal_domains": [
                "@lguplus.co.kr",
                "@bdp.lguplus.co.kr",
            ],
            "os_copyright_domains": [
                "@redhat.com",
                "@apache.org",
                "@fedora-project",
            ],
            "dummy_domains": [
                "@example.com",
                "@test.com",
                "@localhost",
            ],
            "case_insensitive": True,
        }
        return KeywordFilter(config=config)

    def test_internal_domain_detection(self, keyword_filter):
        """내부 도메인 이메일 감지"""
        df = pd.DataFrame({"text": ["user@lguplus.co.kr"]})
        result = keyword_filter.apply(df, "text")

        assert isinstance(result, FilterResult)
        assert result.total_filtered == 1
        assert result.total_passed == 0
        assert "FP-내부도메인" in result.label_counts

    def test_bdp_internal_domain(self, keyword_filter):
        """BDP 내부 도메인 감지"""
        df = pd.DataFrame({"text": ["admin@bdp.lguplus.co.kr"]})
        result = keyword_filter.apply(df, "text")

        assert result.total_filtered == 1
        assert "FP-내부도메인" in result.label_counts

    def test_os_copyright_domain(self, keyword_filter):
        """OS 저작권 도메인 감지"""
        df = pd.DataFrame({
            "text": [
                "Copyright by user@redhat.com",
                "Contact: dev@apache.org",
                "Maintainer: author@fedora-project",
            ]
        })

        result = keyword_filter.apply(df, "text")
        assert result.total_filtered == 3
        assert "FP-OS저작권" in result.label_counts

    def test_dummy_domain(self, keyword_filter):
        """더미/테스트 도메인 감지"""
        df = pd.DataFrame({
            "text": [
                "test@example.com",
                "user@test.com",
                "sample@localhost",
            ]
        })

        result = keyword_filter.apply(df, "text")
        assert result.total_filtered == 3
        assert "FP-더미데이터" in result.label_counts

    def test_real_email_not_filtered(self, keyword_filter):
        """실제 외부 이메일은 필터링되지 않음"""
        df = pd.DataFrame({
            "text": [
                "john.doe@gmail.com",
                "user@naver.com",
                "customer@company.co.kr",
            ]
        })

        result = keyword_filter.apply(df, "text")
        assert result.total_filtered == 0
        assert result.total_passed == 3

    def test_case_insensitive(self, keyword_filter):
        """대소문자 무시 매칭"""
        df = pd.DataFrame({"text": ["USER@LGUPLUS.CO.KR"]})
        result = keyword_filter.apply(df, "text")

        assert result.total_filtered == 1


class TestRuleFilter:
    """룰 필터 테스트"""

    @pytest.fixture
    def rule_filter(self):
        """룰 필터 설정"""
        config = {
            "timestamp_patterns": [
                {"pattern": r"\b\d{10}\b", "label": "FP-타임스탬프", "description": "Unix timestamp 10자리"},
                {"pattern": r"\b\d{13}\b", "label": "FP-타임스탬프", "description": "Unix timestamp 13자리 (ms)"},
            ],
            "bytes_patterns": [
                {"pattern": r"\d+\s*bytes?\b", "label": "FP-Bytes크기", "description": "bytes 크기 패턴"},
            ],
            "version_patterns": [
                {"pattern": r"\d+\.\d+\.\d+[-.\d]*", "label": "FP-숫자코드", "description": "버전번호 패턴"},
            ],
            "path_rules": [
                {"pattern": r"/docker/|/overlay/|/overlay2/", "label": "FP-컨텍스트", "description": "Docker 경로"},
                {"pattern": r"hadoop-cmf-hdfs|DATANODE|NAMENODE", "label": "FP-컨텍스트", "description": "Hadoop 경로"},
            ],
        }
        return RuleFilter(config=config)

    def test_timestamp_unix_10digit(self, rule_filter):
        """Unix timestamp (10자리) 감지"""
        df = pd.DataFrame({"text": ["created: 1704067200"]})
        result = rule_filter.apply(df, "text")

        assert result.total_filtered == 1
        assert any("타임스탬프" in label for label in result.label_counts.keys())

    def test_timestamp_unix_13digit(self, rule_filter):
        """Unix timestamp (13자리, ms) 감지"""
        df = pd.DataFrame({"text": ["timestamp: 1704067200000"]})
        result = rule_filter.apply(df, "text")

        assert result.total_filtered == 1

    def test_bytes_pattern(self, rule_filter):
        """bytes 크기 패턴 감지"""
        df = pd.DataFrame({
            "text": [
                "size: 45 bytes 141022",
                "file_size: 1024 bytes",
                "content-length: 256bytes",
            ]
        })

        result = rule_filter.apply(df, "text")
        # 최소 일부는 매칭되어야 함
        assert result.total_filtered >= 1

    def test_version_pattern(self, rule_filter):
        """버전 번호 패턴 감지"""
        df = pd.DataFrame({
            "text": [
                "version: 1.3.3.32-2087-1512",
                "v2.0.1",
                "release-3.14.159",
            ]
        })

        result = rule_filter.apply(df, "text")
        # 버전 패턴이 매칭되어야 함
        assert result.total_filtered >= 1

    def test_docker_path(self, rule_filter):
        """Docker 경로 감지 (file_path_column 사용)"""
        df = pd.DataFrame({
            "text": ["some content"],
            "file_path": ["/var/lib/docker/overlay2/abc123/merged/app/data.txt"]
        })
        result = rule_filter.apply(df, "text", file_path_column="file_path")

        # Docker 경로 패턴이 매칭되어야 함
        assert result.total_filtered == 1

    def test_hadoop_path(self, rule_filter):
        """Hadoop 로그 경로 감지 (file_path_column 사용)"""
        df = pd.DataFrame({
            "text": ["some content"],
            "file_path": ["hadoop-cmf-hdfs-DATANODE-server01.log"]
        })
        result = rule_filter.apply(df, "text", file_path_column="file_path")

        assert result.total_filtered == 1

    def test_real_pii_not_filtered(self, rule_filter):
        """일반 텍스트는 필터링되지 않음"""
        df = pd.DataFrame({
            "text": [
                "홍길동의 이메일",
                "연락처 정보",
                "주소: 서울시 강남구",
            ]
        })

        result = rule_filter.apply(df, "text")
        # 이런 텍스트는 룰 필터에서 걸리지 않아야 함
        assert result.total_filtered == 0


class TestFilterPipeline:
    """필터 파이프라인 통합 테스트"""

    @pytest.fixture
    def pipeline(self):
        config = {
            "keyword_filter": {
                "internal_domains": ["@lguplus.co.kr"],
                "os_copyright_domains": ["@redhat.com"],
                "dummy_domains": ["@example.com"],
            },
            "rule_filter": {
                "timestamp_patterns": [
                    {"pattern": r"\b\d{10}\b", "label": "FP-타임스탬프", "description": "Unix timestamp"}
                ],
                "bytes_patterns": [
                    {"pattern": r"\d+\s*bytes?\b", "label": "FP-Bytes크기", "description": "bytes 크기"}
                ],
            },
        }
        return FilterPipeline(config=config)

    def test_layer1_priority(self, pipeline):
        """Layer 1 (키워드)가 먼저 적용됨"""
        df = pd.DataFrame({
            "text": ["1704067200 user@lguplus.co.kr"]
        })
        result = pipeline.apply(df, "text")

        # 키워드 필터가 먼저 적용되어야 함
        assert result.total_filtered >= 1

    def test_layer2_fallback(self, pipeline):
        """Layer 1에서 안 걸리면 Layer 2로 넘어감"""
        df = pd.DataFrame({"text": ["timestamp: 1704067200"]})
        result = pipeline.apply(df, "text")

        assert result.total_filtered == 1

    def test_no_filter_for_real_pii(self, pipeline):
        """실제 개인정보 형태는 필터 통과 후 ML로"""
        df = pd.DataFrame({
            "text": ["고객명: 홍길동, 연락처: ***-****-5678"]
        })
        result = pipeline.apply(df, "text")

        assert result.total_ml_input == 1
        assert result.total_filtered == 0

    def test_batch_processing(self, pipeline):
        """배치 처리 테스트"""
        df = pd.DataFrame({
            "text": [
                "user@lguplus.co.kr",       # Layer 1
                "timestamp: 1704067200",    # Layer 2
                "고객 정보",                  # ML
            ]
        })

        result = pipeline.apply(df, "text")
        assert result.total_filtered == 2
        assert result.total_ml_input == 1


class TestFilterResult:
    """FilterResult 데이터클래스 테스트"""

    def test_empty_result(self):
        """빈 결과"""
        result = FilterResult()

        assert result.total_filtered == 0
        assert result.total_passed == 0
        assert result.filter_rate == 0.0

    def test_full_result(self):
        """완전한 결과"""
        filtered_df = pd.DataFrame({"text": ["a", "b"]})
        passed_df = pd.DataFrame({"text": ["c"]})

        result = FilterResult(
            filtered_df=filtered_df,
            passed_df=passed_df,
            label_counts={"FP-내부도메인": 2},
            filter_name="KeywordFilter",
        )

        assert result.total_filtered == 2
        assert result.total_passed == 1
        assert result.filter_rate == 2 / 3

    def test_summary(self):
        """요약 문자열 생성"""
        result = FilterResult(
            filtered_df=pd.DataFrame({"text": ["a"]}),
            passed_df=pd.DataFrame({"text": ["b", "c"]}),
            label_counts={"FP-테스트": 1},
            filter_name="TestFilter",
        )

        summary = result.summary()
        assert "TestFilter" in summary
        assert "1" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
