"""필터 기반 클래스"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd


@dataclass
class FilterResult:
    """필터 적용 결과를 담는 데이터 클래스"""

    # 필터에 의해 분류된 데이터
    filtered_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 필터를 통과한 데이터 (다음 Layer로 전달)
    passed_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 분류 통계 {레이블: 건수}
    label_counts: Dict[str, int] = field(default_factory=dict)

    # 필터 이름
    filter_name: str = ""

    # 추가 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_filtered(self) -> int:
        """필터에 의해 분류된 총 건수"""
        return len(self.filtered_df)

    @property
    def total_passed(self) -> int:
        """필터를 통과한 총 건수"""
        return len(self.passed_df)

    @property
    def filter_rate(self) -> float:
        """필터링 비율 (0~1)"""
        total = self.total_filtered + self.total_passed
        if total == 0:
            return 0.0
        return self.total_filtered / total

    def summary(self) -> str:
        """결과 요약 문자열"""
        lines = [
            f"[{self.filter_name}] 결과:",
            f"  - 분류됨: {self.total_filtered:,}건 ({self.filter_rate * 100:.1f}%)",
            f"  - 통과:   {self.total_passed:,}건",
        ]

        if self.label_counts:
            lines.append("  - 레이블별 분류:")
            for label, count in sorted(
                self.label_counts.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"      {label}: {count:,}건")

        return "\n".join(lines)


class BaseFilter(ABC):
    """필터 추상 기반 클래스

    모든 필터는 이 클래스를 상속받아 구현합니다.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            name: 필터 이름 (로깅/리포트용)
            config: 필터 설정 딕셔너리
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    @abstractmethod
    def apply(
        self,
        df: pd.DataFrame,
        text_column: str,
        file_path_column: Optional[str] = None,
    ) -> FilterResult:
        """
        필터를 적용합니다.

        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명 (검출 내역)
            file_path_column: 파일 경로 컬럼명 (선택)

        Returns:
            FilterResult: 필터 적용 결과
        """
        pass

    def _create_result(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        labels: pd.Series,
    ) -> FilterResult:
        """
        마스크와 레이블로부터 FilterResult를 생성합니다.

        Args:
            df: 원본 DataFrame
            mask: 필터링된 행을 나타내는 Boolean Series
            labels: 각 행의 분류 레이블 Series (필터링된 행만 값이 있음)

        Returns:
            FilterResult
        """
        filtered_df = df[mask].copy()
        passed_df = df[~mask].copy()

        # 분류된 데이터에 레이블 추가
        if len(filtered_df) > 0:
            filtered_df["filter_label"] = labels[mask]

        # 레이블별 카운트
        label_counts = {}
        if len(filtered_df) > 0:
            label_counts = filtered_df["filter_label"].value_counts().to_dict()

        return FilterResult(
            filtered_df=filtered_df,
            passed_df=passed_df,
            label_counts=label_counts,
            filter_name=self.name,
        )

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
