"""3-Layer Filter 통합 파이프라인

Layer 1 (Keyword) -> Layer 2 (Rule) -> Layer 3 (ML) 순으로 적용하여
각 Layer에서 분류된 샘플을 분리하고, 잔여 샘플만 다음 Layer로 전달합니다.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import yaml

from src.filters.base_filter import FilterResult
from src.filters.keyword_filter import KeywordFilter
from src.filters.rule_filter import RuleFilter
from src.utils.constants import PROJECT_ROOT


@dataclass
class PipelineResult:
    """파이프라인 전체 결과"""

    # Layer별 결과
    layer1_result: Optional[FilterResult] = None  # Keyword Filter
    layer2_result: Optional[FilterResult] = None  # Rule Filter

    # 최종 분류된 데이터 (Layer 1 + Layer 2)
    filtered_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ML 모델로 전달할 데이터 (Layer 1, 2를 통과한 데이터)
    ml_input_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 전체 통계
    total_input: int = 0
    total_filtered: int = 0
    total_ml_input: int = 0

    def summary(self) -> str:
        """결과 요약"""
        lines = [
            "=" * 60,
            "[3-Layer Filter Pipeline 결과]",
            "=" * 60,
            f"총 입력:     {self.total_input:,}건",
            f"Layer 1-2 분류: {self.total_filtered:,}건 ({self.total_filtered / max(self.total_input, 1) * 100:.1f}%)",
            f"ML 모델 입력: {self.total_ml_input:,}건 ({self.total_ml_input / max(self.total_input, 1) * 100:.1f}%)",
            "",
        ]

        if self.layer1_result:
            lines.append(self.layer1_result.summary())
            lines.append("")

        if self.layer2_result:
            lines.append(self.layer2_result.summary())
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_label_distribution(self) -> Dict[str, int]:
        """전체 레이블 분포 반환"""
        distribution = {}

        if self.layer1_result:
            for label, count in self.layer1_result.label_counts.items():
                distribution[label] = distribution.get(label, 0) + count

        if self.layer2_result:
            for label, count in self.layer2_result.label_counts.items():
                distribution[label] = distribution.get(label, 0) + count

        return distribution


class FilterPipeline:
    """3-Layer Filter 파이프라인"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            config_path: filter_config.yaml 경로
            config: 직접 전달하는 설정 딕셔너리 (config_path보다 우선)
        """
        # 설정 로드
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            # 기본 경로
            default_path = PROJECT_ROOT / "config" / "filter_config.yaml"
            self.config = self._load_config(default_path)

        # 필터 초기화
        self.keyword_filter = KeywordFilter(
            config=self.config.get("keyword_filter", {})
        )
        self.rule_filter = RuleFilter(
            config=self.config.get("rule_filter", {})
        )

        # 로깅 설정
        logging_config = self.config.get("logging", {})
        self.show_layer_stats = logging_config.get("show_layer_stats", True)
        self.save_filtered_samples = logging_config.get("save_filtered_samples", False)
        self.output_dir = Path(logging_config.get("output_dir", "outputs/filter_results/"))

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        path = Path(config_path)
        if not path.exists():
            print(f"[경고] 설정 파일 없음: {config_path}, 기본값 사용")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def apply(
        self,
        df: pd.DataFrame,
        text_column: str,
        file_path_column: Optional[str] = None,
    ) -> PipelineResult:
        """
        3-Layer 필터 파이프라인을 적용합니다.

        Args:
            df: 입력 DataFrame
            text_column: 텍스트 컬럼명 (검출 내역)
            file_path_column: 파일 경로 컬럼명

        Returns:
            PipelineResult: 파이프라인 결과
        """
        print("\n" + "=" * 60)
        print("[3-Layer Filter Pipeline 시작]")
        print("=" * 60)
        print(f"입력 데이터: {len(df):,}건")

        result = PipelineResult(total_input=len(df))
        current_df = df.copy()
        all_filtered = []

        # === Layer 1: Keyword Filter ===
        print("\n[Layer 1: Keyword Filter]")
        layer1_result = self.keyword_filter.apply(
            current_df, text_column, file_path_column
        )
        result.layer1_result = layer1_result

        if self.show_layer_stats:
            print(layer1_result.summary())

        if len(layer1_result.filtered_df) > 0:
            all_filtered.append(layer1_result.filtered_df)

        current_df = layer1_result.passed_df

        # === Layer 2: Rule Filter ===
        print("\n[Layer 2: Rule Filter]")
        layer2_result = self.rule_filter.apply(
            current_df, text_column, file_path_column
        )
        result.layer2_result = layer2_result

        if self.show_layer_stats:
            print(layer2_result.summary())

        if len(layer2_result.filtered_df) > 0:
            all_filtered.append(layer2_result.filtered_df)

        current_df = layer2_result.passed_df

        # === 결과 집계 ===
        if all_filtered:
            result.filtered_df = pd.concat(all_filtered, ignore_index=True)
        else:
            result.filtered_df = pd.DataFrame()

        result.ml_input_df = current_df
        result.total_filtered = len(result.filtered_df)
        result.total_ml_input = len(result.ml_input_df)

        # 최종 요약
        print("\n" + result.summary())

        # 결과 저장 (선택)
        if self.save_filtered_samples:
            self._save_results(result)

        return result

    def _save_results(self, result: PipelineResult) -> None:
        """결과를 파일로 저장"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Layer 1 분류 결과
        if result.layer1_result and len(result.layer1_result.filtered_df) > 0:
            path = self.output_dir / "layer1_filtered.csv"
            result.layer1_result.filtered_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  Layer 1 결과 저장: {path}")

        # Layer 2 분류 결과
        if result.layer2_result and len(result.layer2_result.filtered_df) > 0:
            path = self.output_dir / "layer2_filtered.csv"
            result.layer2_result.filtered_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  Layer 2 결과 저장: {path}")

        # ML 입력 데이터
        if len(result.ml_input_df) > 0:
            path = self.output_dir / "ml_input.csv"
            result.ml_input_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  ML 입력 데이터 저장: {path}")

    def get_filter_summary(self) -> Dict[str, Any]:
        """필터 설정 요약"""
        return {
            "keyword_filter": {
                "enabled": self.keyword_filter.enabled,
                "stats": self.keyword_filter.get_keyword_stats(),
            },
            "rule_filter": {
                "enabled": self.rule_filter.enabled,
                "stats": self.rule_filter.get_pattern_stats(),
            },
        }

    def test_single(
        self,
        text: str,
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        단일 텍스트에 대해 필터 테스트 (디버깅용)

        Returns:
            테스트 결과 딕셔너리
        """
        result = {
            "text": text,
            "file_path": file_path,
            "layer1_match": None,
            "layer2_match": None,
            "final_label": None,
            "goes_to_ml": True,
        }

        # Layer 1 테스트
        df_test = pd.DataFrame({"text": [text], "file_path": [file_path]})
        layer1_result = self.keyword_filter.apply(df_test, "text", "file_path")

        if layer1_result.total_filtered > 0:
            result["layer1_match"] = layer1_result.label_counts
            result["final_label"] = list(layer1_result.label_counts.keys())[0]
            result["goes_to_ml"] = False
            return result

        # Layer 2 테스트
        layer2_detail = self.rule_filter.test_pattern(text, file_path)
        if layer2_detail["matched_label"]:
            result["layer2_match"] = layer2_detail
            result["final_label"] = layer2_detail["matched_label"]
            result["goes_to_ml"] = False
            return result

        return result

    def __repr__(self) -> str:
        summary = self.get_filter_summary()
        return (
            f"FilterPipeline(\n"
            f"  keyword_filter={summary['keyword_filter']},\n"
            f"  rule_filter={summary['rule_filter']}\n"
            f")"
        )
