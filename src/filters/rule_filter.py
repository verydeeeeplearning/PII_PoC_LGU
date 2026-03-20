"""Layer 2: Rule 기반 필터

정규식과 비즈니스 로직을 사용하여 오탐을 분류합니다.
- Timestamp 패턴
- Bytes 크기 패턴
- 버전번호/코드 패턴
- 파일 경로 기반 규칙
"""

import re
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from src.filters.base_filter import BaseFilter, FilterResult
from src.utils.constants import (
    LABEL_FP_TIMESTAMP,
    LABEL_FP_BYTES,
    LABEL_FP_NUMERIC_CODE,
    LABEL_FP_CONTEXT,
    LABEL_FP_DUMMY_DATA,
)


@dataclass
class RulePattern:
    """규칙 패턴 정의"""
    pattern: str
    label: str
    description: str
    compiled: Optional[re.Pattern] = None

    def __post_init__(self):
        """패턴 컴파일"""
        if self.compiled is None:
            self.compiled = re.compile(self.pattern, re.IGNORECASE)


class RuleFilter(BaseFilter):
    """규칙 기반 오탐 필터 (Layer 2)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 필터 설정 (filter_config.yaml의 rule_filter 섹션)
        """
        super().__init__(name="RuleFilter", config=config)

        self.case_insensitive = self.config.get("case_insensitive", True)
        self.re_flags = re.IGNORECASE if self.case_insensitive else 0

        # 패턴 로드 및 컴파일
        self.timestamp_patterns = self._load_patterns(
            self.config.get("timestamp_patterns", []),
            default_label=LABEL_FP_TIMESTAMP,
        )
        self.bytes_patterns = self._load_patterns(
            self.config.get("bytes_patterns", []),
            default_label=LABEL_FP_BYTES,
        )
        self.version_patterns = self._load_patterns(
            self.config.get("version_patterns", []),
            default_label=LABEL_FP_NUMERIC_CODE,
        )
        self.path_rules = self._load_patterns(
            self.config.get("path_rules", []),
            default_label=LABEL_FP_CONTEXT,
        )

        # 모든 텍스트 패턴 통합 (우선순위 순)
        self.text_patterns: List[RulePattern] = (
            self.timestamp_patterns
            + self.bytes_patterns
            + self.version_patterns
        )

    def _load_patterns(
        self,
        patterns_config: List[Dict[str, str]],
        default_label: str,
    ) -> List[RulePattern]:
        """설정에서 패턴 로드 및 컴파일"""
        patterns = []

        for p in patterns_config:
            pattern_str = p.get("pattern", "")
            if not pattern_str:
                continue

            try:
                compiled = re.compile(pattern_str, self.re_flags)
                patterns.append(
                    RulePattern(
                        pattern=pattern_str,
                        label=p.get("label", default_label),
                        description=p.get("description", ""),
                        compiled=compiled,
                    )
                )
            except re.error as e:
                print(f"  [경고] 정규식 컴파일 실패: {pattern_str} - {e}")

        return patterns

    def _classify_text(self, text: str) -> Optional[str]:
        """
        텍스트에 규칙 패턴을 적용하여 분류합니다.

        Returns:
            분류 레이블 또는 None
        """
        if pd.isna(text):
            return None

        for rule in self.text_patterns:
            if rule.compiled and rule.compiled.search(text):
                return rule.label

        return None

    def _classify_path(self, file_path: str) -> Optional[str]:
        """
        파일 경로에 규칙 패턴을 적용하여 분류합니다.

        Returns:
            분류 레이블 또는 None
        """
        if pd.isna(file_path):
            return None

        for rule in self.path_rules:
            if rule.compiled and rule.compiled.search(file_path):
                return rule.label

        return None

    def _classify_row(
        self, text: str, file_path: Optional[str] = None
    ) -> Optional[str]:
        """
        단일 행을 분류합니다.

        텍스트 패턴을 먼저 검사하고, 일치하지 않으면 경로 패턴을 검사합니다.

        Returns:
            분류 레이블 또는 None (필터 통과)
        """
        # 텍스트 패턴 검사 (우선)
        label = self._classify_text(text)
        if label:
            return label

        # 경로 패턴 검사
        if file_path:
            label = self._classify_path(file_path)
            if label:
                return label

        return None

    def apply(
        self,
        df: pd.DataFrame,
        text_column: str,
        file_path_column: Optional[str] = None,
    ) -> FilterResult:
        """
        규칙 필터를 적용합니다.

        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명
            file_path_column: 파일 경로 컬럼명

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

        # 파일 경로 컬럼 확인
        has_path = file_path_column and file_path_column in df.columns

        # 각 행에 대해 분류 수행
        if has_path:
            labels = df.apply(
                lambda row: self._classify_row(
                    row[text_column],
                    row[file_path_column] if has_path else None,
                ),
                axis=1,
            )
        else:
            labels = df[text_column].apply(
                lambda text: self._classify_row(text, None)
            )

        # 분류된 행 (None이 아닌 경우)
        mask = labels.notna()

        result = self._create_result(df, mask, labels)
        result.metadata["patterns_applied"] = {
            "timestamp": len(self.timestamp_patterns),
            "bytes": len(self.bytes_patterns),
            "version": len(self.version_patterns),
            "path": len(self.path_rules),
        }

        return result

    def get_pattern_stats(self) -> Dict[str, int]:
        """설정된 패턴 통계 반환"""
        return {
            "timestamp_patterns": len(self.timestamp_patterns),
            "bytes_patterns": len(self.bytes_patterns),
            "version_patterns": len(self.version_patterns),
            "path_rules": len(self.path_rules),
            "total": (
                len(self.timestamp_patterns)
                + len(self.bytes_patterns)
                + len(self.version_patterns)
                + len(self.path_rules)
            ),
        }

    def test_pattern(self, text: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        단일 텍스트/경로에 대해 패턴 테스트 (디버깅용)

        Returns:
            매칭 결과 딕셔너리
        """
        result = {
            "text": text,
            "file_path": file_path,
            "matched_label": None,
            "matched_patterns": [],
        }

        # 텍스트 패턴 매칭
        for rule in self.text_patterns:
            if rule.compiled and rule.compiled.search(text or ""):
                result["matched_patterns"].append({
                    "type": "text",
                    "pattern": rule.pattern,
                    "label": rule.label,
                    "description": rule.description,
                })
                if not result["matched_label"]:
                    result["matched_label"] = rule.label

        # 경로 패턴 매칭
        if file_path:
            for rule in self.path_rules:
                if rule.compiled and rule.compiled.search(file_path):
                    result["matched_patterns"].append({
                        "type": "path",
                        "pattern": rule.pattern,
                        "label": rule.label,
                        "description": rule.description,
                    })
                    if not result["matched_label"]:
                        result["matched_label"] = rule.label

        return result

    def __repr__(self) -> str:
        stats = self.get_pattern_stats()
        return (
            f"RuleFilter(enabled={self.enabled}, "
            f"timestamp={stats['timestamp_patterns']}, "
            f"bytes={stats['bytes_patterns']}, "
            f"version={stats['version_patterns']}, "
            f"path={stats['path_rules']})"
        )
