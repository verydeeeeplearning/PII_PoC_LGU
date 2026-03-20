"""Phase 1 tests: preprocessing_config.yaml 존재 및 구조 검증."""
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "preprocessing_config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_yaml_loads_without_error():
    assert CONFIG_PATH.exists(), f"preprocessing_config.yaml 없음: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_sumologic_column_map_keys_count(cfg):
    col_map = cfg["data_sources"]["sumologic_server_i"]["column_map"]
    assert len(col_map) == 15


def test_pk_file_fields_exactly_four(cfg):
    fields = cfg["data_sources"]["sumologic_server_i"]["pk_file_fields"]
    assert len(fields) == 4


def test_join_label_cols_contains_pk_file(cfg):
    label_cols = cfg["join"]["label_cols"]
    assert "pk_file" in label_cols


def test_s1_window_size_is_60(cfg):
    window = cfg["parsing"]["s1_parser"]["window_size"]
    assert window == 60


def test_datetime_formats_non_empty(cfg):
    formats = cfg["parsing"]["label_loader"]["datetime_formats"]
    assert len(formats) > 0


def test_csv_encoding_candidates_utf8_first(cfg):
    candidates = cfg["parsing"]["loader"]["csv_encoding_candidates"]
    assert candidates[0] == "utf-8"


def test_on_unrecognized_column_default_warn(cfg):
    val = cfg["data_sources"]["sumologic_server_i"]["on_unrecognized_column"]
    assert val == "warn"
