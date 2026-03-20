"""데이터 소스 레지스트리 - preprocessing_config.yaml data_sources 섹션 관리.

새 검출 데이터 소스를 YAML에만 추가하면 코드 변경 없이 즉시 인식.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "preprocessing_config.yaml"


class DataSourceRegistry:
    """preprocessing_config.yaml data_sources 섹션 관리.

    사용 예시:
        registry = DataSourceRegistry()
        col_map = registry.get_column_map("sumologic_server_i")
        pk_fields = registry.get_pk_fields("sumologic_server_i")
        files = registry.find_files("sumologic_server_i")
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._sources: dict = {}
        self._load()

    def _load(self) -> None:
        if not self._config_path.exists():
            logger.warning("preprocessing_config.yaml 없음: %s", self._config_path)
            return
        with open(self._config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self._sources = cfg.get("data_sources", {})

    def _get_source(self, source_name: str) -> dict:
        if source_name not in self._sources:
            raise KeyError(
                f"알 수 없는 데이터 소스: '{source_name}'. "
                f"등록된 소스: {list(self._sources.keys())}"
            )
        return self._sources[source_name]

    def get_column_map(self, source_name: str) -> dict[str, str]:
        """source -> standard 컬럼명 매핑 딕셔너리 반환."""
        return dict(self._get_source(source_name).get("column_map", {}))

    def get_pk_fields(self, source_name: str) -> list[str]:
        """pk_file 생성에 사용할 필드 이름 목록 반환."""
        return list(self._get_source(source_name).get("pk_file_fields", []))

    def find_files(self, source_name: str, base_dir: Optional[Path] = None) -> list[Path]:
        """file_pattern glob으로 파일 탐색.

        Args:
            source_name: 데이터 소스 이름
            base_dir: 상대 경로 기준 디렉토리 (None이면 프로젝트 루트 자동 탐색)
        """
        pattern = self._get_source(source_name).get("file_pattern", "")
        if not pattern:
            return []

        if base_dir is None:
            # 프로젝트 루트 = config 파일 2단계 상위
            base_dir = self._config_path.parents[1]

        return sorted((base_dir / pattern).parent.glob((base_dir / pattern).name))

    def list_sources(self) -> list[str]:
        """등록된 데이터 소스 이름 목록 반환."""
        return list(self._sources.keys())

    def get_description(self, source_name: str) -> str:
        """데이터 소스 설명 반환."""
        return self._get_source(source_name).get("description", "")
