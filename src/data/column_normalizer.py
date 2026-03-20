"""한글 컬럼명 -> 영문 snake_case 정규화 모듈

EDA 인사이트 리포트 §2-1 기반.
레이블 Excel의 한글 컬럼명(서버이름, 에이전트IP 등)을
영문 snake_case(server_name, agent_ip 등)로 변환한다.

주의: 조직/월별로 한글 표기가 미세하게 다를 수 있음.
새로운 표기 발견 시 config/column_name_mapping.yaml의
aliases에 추가할 것.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[2] / "config" / "column_name_mapping.yaml"


class ColumnNormalizer:
    """한글 컬럼명 -> 영문 snake_case 변환기.

    config/column_name_mapping.yaml의 mappings 섹션을 읽어
    primary + aliases 표기를 모두 처리한다.
    drop=True인 컬럼(file_size 등)은 변환 후 자동 제거한다.
    """

    def __init__(self, mapping_path: Optional[Path] = None, strict: bool = False):
        self._mapping_path = mapping_path or _DEFAULT_MAPPING_PATH
        self._strict = strict
        self._raw_to_std: Dict[str, str] = {}  # {한글표기: 영문표준명}
        self._std_to_drop: set = set()          # drop=True인 영문표준명
        self._raw_to_drop: set = set()          # drop=True인 한글원본명(primary+aliases)
        self._load_mapping()

    # ── public API ────────────────────────────────────────────────────────────

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼명을 영문으로 정규화하고 drop 컬럼을 제거한다.

        Args:
            df: 원본 DataFrame (수정되지 않음)

        Returns:
            컬럼명이 정규화된 새 DataFrame
        """
        if df.empty and len(df.columns) == 0:
            return df.copy()

        df = df.copy()

        # 앞뒤 공백/탭 제거 (Excel에서 발생하는 minor 표기 차이 대응)
        ws_strip = {col: str(col).strip() for col in df.columns if isinstance(col, str) and col != col.strip()}
        if ws_strip:
            logger.debug("컬럼명 앞뒤 공백/탭 제거: %s", list(ws_strip.keys()))
            df = df.rename(columns=ws_strip)

        rename_map: Dict[str, str] = {}
        unrecognized: List[str] = []

        for col in df.columns:
            if col in self._raw_to_std:
                rename_map[col] = self._raw_to_std[col]
            elif col not in self._raw_to_std.values():
                # 이미 영문 표준명이면 조용히 통과, 그 외 한글이면 경고
                if any(ord(c) > 127 for c in str(col)):
                    unrecognized.append(col)

        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug("컬럼 변환: %d개", len(rename_map))

        if unrecognized:
            if self._strict:
                raise ValueError(
                    f"strict mode: 인식되지 않은 한글 컬럼명 발견 - "
                    f"config/column_name_mapping.yaml에 추가 필요: {unrecognized}"
                )
            logger.warning(
                "인식되지 않은 한글 컬럼명 발견 (config/column_name_mapping.yaml aliases에 추가 필요): %s",
                unrecognized,
            )

        # drop 컬럼 제거 (변환 후 기준: 영문 표준명 + 미변환 한글 원본명)
        drop_candidates = self._std_to_drop | self._raw_to_drop
        drop_present = [c for c in drop_candidates if c in df.columns]
        if drop_present:
            df = df.drop(columns=drop_present)
            logger.debug("드롭 컬럼 제거: %s", drop_present)

        return df

    def get_rename_map(self, columns: List[str]) -> Dict[str, str]:
        """주어진 컬럼 목록에 대한 변환 매핑 반환."""
        return {col: self._raw_to_std[col] for col in columns if col in self._raw_to_std}

    # ── private ───────────────────────────────────────────────────────────────

    def _load_mapping(self) -> None:
        if not self._mapping_path.exists():
            logger.warning("컬럼 매핑 파일 없음: %s", self._mapping_path)
            return

        with open(self._mapping_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        for std_name, cfg in config.get("mappings", {}).items():
            if not isinstance(cfg, dict):
                continue

            is_drop = cfg.get("drop", False)
            primary = cfg.get("primary", "")
            aliases = cfg.get("aliases", []) or []

            if primary:
                self._raw_to_std[primary] = std_name
                if is_drop:
                    self._raw_to_drop.add(primary)

            for alias in aliases:
                self._raw_to_std[alias] = std_name
                if is_drop:
                    self._raw_to_drop.add(alias)

            if is_drop:
                self._std_to_drop.add(std_name)
