"""Layer 1: Keyword 기반 필터

명확한 오탐 패턴을 키워드 매칭으로 사전 제거합니다.
- 내부 도메인 (@lguplus.co.kr 등)
- OS/오픈소스 저작권 도메인 (@redhat.com 등)
- 더미 도메인 (@example.com 등)
"""

import pandas as pd
from typing import Optional, Dict, Any, List

from src.filters.base_filter import BaseFilter, FilterResult
from src.utils.constants import (
    LABEL_FP_INTERNAL_DOMAIN,
    LABEL_FP_OS_COPYRIGHT,
    LABEL_FP_DUMMY_DATA,
)


class KeywordFilter(BaseFilter):
    """키워드 기반 오탐 필터 (Layer 1)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 필터 설정 (filter_config.yaml의 keyword_filter 섹션)
        """
        super().__init__(name="KeywordFilter", config=config)

        # 설정에서 키워드 리스트 로드
        self.internal_domains = self.config.get("internal_domains", [])
        self.os_copyright_domains = self.config.get("os_copyright_domains", [])
        self.dummy_domains = self.config.get("dummy_domains", [])
        self.case_insensitive = self.config.get("case_insensitive", True)

        # 키워드를 소문자로 정규화 (대소문자 무시 시)
        if self.case_insensitive:
            self.internal_domains = [d.lower() for d in self.internal_domains]
            self.os_copyright_domains = [d.lower() for d in self.os_copyright_domains]
            self.dummy_domains = [d.lower() for d in self.dummy_domains]

    def _check_keywords(
        self, text: str, keywords: List[str]
    ) -> bool:
        """텍스트에 키워드가 포함되어 있는지 확인"""
        if pd.isna(text):
            return False

        check_text = text.lower() if self.case_insensitive else text

        for keyword in keywords:
            if keyword in check_text:
                return True
        return False

    def _classify_row(
        self, text: str, file_path: Optional[str] = None
    ) -> Optional[str]:
        """
        단일 행을 분류합니다.

        Returns:
            분류 레이블 또는 None (필터 통과)
        """
        if pd.isna(text):
            return None

        check_text = text.lower() if self.case_insensitive else text

        # 1. 내부 도메인 체크
        for domain in self.internal_domains:
            if domain in check_text:
                return LABEL_FP_INTERNAL_DOMAIN

        # 2. OS 저작권 도메인 체크
        for domain in self.os_copyright_domains:
            if domain in check_text:
                return LABEL_FP_OS_COPYRIGHT

        # 3. 더미 도메인 체크
        for domain in self.dummy_domains:
            if domain in check_text:
                return LABEL_FP_DUMMY_DATA

        return None  # 필터 통과

    def apply(
        self,
        df: pd.DataFrame,
        text_column: str,
        file_path_column: Optional[str] = None,
    ) -> FilterResult:
        """
        키워드 필터를 적용합니다.

        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            file_path_column: 파일 경로 컬럼명 (현재 미사용)

        Returns:
            FilterResult
        """
        if not self.enabled:
            return FilterResult(
                filtered_df=pd.DataFrame(),
                passed_df=df.copy(),
                label_counts={},
                filter_name=self.name,
                metadata={"status": "disabled"},
            )

        if text_column not in df.columns:
            raise ValueError(f"텍스트 컬럼 '{text_column}'이 DataFrame에 없습니다.")

        # 각 행에 대해 분류 수행
        labels = df[text_column].apply(self._classify_row)

        # 분류된 행 (None이 아닌 경우)
        mask = labels.notna()

        return self._create_result(df, mask, labels)

    def get_keyword_stats(self) -> Dict[str, int]:
        """설정된 키워드 통계 반환"""
        return {
            "internal_domains": len(self.internal_domains),
            "os_copyright_domains": len(self.os_copyright_domains),
            "dummy_domains": len(self.dummy_domains),
            "total": (
                len(self.internal_domains)
                + len(self.os_copyright_domains)
                + len(self.dummy_domains)
            ),
        }

    def __repr__(self) -> str:
        stats = self.get_keyword_stats()
        return (
            f"KeywordFilter(enabled={self.enabled}, "
            f"internal={stats['internal_domains']}, "
            f"os_copyright={stats['os_copyright_domains']}, "
            f"dummy={stats['dummy_domains']})"
        )
