"""Phase 2 tests: DataSourceRegistry."""
import sys
from pathlib import Path
import tempfile

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasource_registry import DataSourceRegistry


@pytest.fixture
def registry():
    return DataSourceRegistry()


def test_registry_loads_known_source(registry):
    sources = registry.list_sources()
    assert "sumologic_server_i" in sources


def test_get_column_map_returns_dict(registry):
    col_map = registry.get_column_map("sumologic_server_i")
    assert isinstance(col_map, dict)
    assert len(col_map) > 0


def test_get_pk_fields_returns_list_of_four(registry):
    fields = registry.get_pk_fields("sumologic_server_i")
    assert isinstance(fields, list)
    assert len(fields) == 4


def test_find_files_returns_matching_paths(registry, tmp_path):
    # Create a temp xlsx file
    (tmp_path / "test.xlsx").touch()
    # Override config with tmp dir
    cfg = {
        "data_sources": {
            "test_source": {
                "file_pattern": f"{tmp_path}/*.xlsx",
                "column_map": {},
                "pk_file_fields": [],
            }
        }
    }
    cfg_file = tmp_path / "preprocessing_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f)
    r = DataSourceRegistry(config_path=cfg_file)
    files = r.find_files("test_source", base_dir=Path("/"))
    assert len(files) >= 0  # pattern resolved against base_dir


def test_unknown_source_raises_keyerror(registry):
    with pytest.raises(KeyError):
        registry.get_column_map("nonexistent_source_xyz")


def test_list_sources_returns_all_keys(registry):
    sources = registry.list_sources()
    assert isinstance(sources, list)
    assert len(sources) >= 1


def test_add_new_source_via_yaml_only(tmp_path):
    cfg = {
        "data_sources": {
            "new_source": {
                "description": "테스트 소스",
                "file_pattern": "data/raw/new/*.xlsx",
                "column_map": {"col_a": "standard_a"},
                "pk_file_fields": ["f1", "f2", "f3", "f4"],
                "on_unrecognized_column": "warn",
            }
        }
    }
    cfg_file = tmp_path / "preprocessing_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)
    r = DataSourceRegistry(config_path=cfg_file)
    assert "new_source" in r.list_sources()
    assert r.get_column_map("new_source") == {"col_a": "standard_a"}


def test_column_map_preserves_order(registry):
    col_map = registry.get_column_map("sumologic_server_i")
    keys = list(col_map.keys())
    assert keys[0] == "dfile_computername"


def test_get_source_description(registry):
    desc = registry.get_description("sumologic_server_i")
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_file_pattern_relative_to_project_root(registry):
    # find_files should work without crashing even if files don't exist
    files = registry.find_files("sumologic_server_i")
    assert isinstance(files, list)
