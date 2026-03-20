"""Unit tests for src/data/loader.py"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    DatasetLoader,
    create_dataset_directories,
    load_config,
    load_label_data,
    load_raw_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_csv(tmp_path: Path, name: str = "test.csv", rows: int = 3) -> Path:
    """Write a small CSV and return its path."""
    p = tmp_path / name
    df = pd.DataFrame({"col_a": range(rows), "col_b": [f"v{i}" for i in range(rows)]})
    df.to_csv(p, index=False, encoding="utf-8")
    return p


def _make_parquet(tmp_path: Path, name: str = "test.parquet") -> Path:
    pytest.importorskip("pyarrow", reason="pyarrow not installed — skipping parquet tests")
    p = tmp_path / name
    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_parquet(p, index=False)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# load_raw_data
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadRawData:
    def test_csv_round_trip(self, tmp_path):
        p = _make_csv(tmp_path)
        df = load_raw_data(str(p))
        assert len(df) == 3
        assert list(df.columns) == ["col_a", "col_b"]

    def test_parquet_round_trip(self, tmp_path):
        p = _make_parquet(tmp_path)
        df = load_raw_data(str(p))
        assert len(df) == 2
        assert "x" in df.columns

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
            load_raw_data(str(p))

    def test_cp949_fallback(self, tmp_path):
        """UnicodeDecodeError on utf-8 triggers cp949 retry."""
        p = tmp_path / "korean.csv"
        # Write file with cp949 encoding
        p.write_bytes("이름,값\n홍길동,42\n".encode("cp949"))
        df = load_raw_data(str(p), encoding="utf-8")
        assert len(df) == 1
        assert "이름" in df.columns

    def test_utf8_csv_returns_dataframe(self, tmp_path):
        p = _make_csv(tmp_path, rows=5)
        df = load_raw_data(str(p), encoding="utf-8")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5


# ─────────────────────────────────────────────────────────────────────────────
# load_label_data
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadLabelData:
    def test_loads_and_returns_dataframe(self, tmp_path):
        p = tmp_path / "labels.csv"
        df_orig = pd.DataFrame({"text": ["a", "b"], "label": ["TP", "FP"]})
        df_orig.to_csv(p, index=False)
        df = load_label_data(str(p), label_column="label")
        assert len(df) == 2
        assert "label" in df.columns

    def test_missing_label_column_still_returns(self, tmp_path):
        p = _make_csv(tmp_path)
        df = load_label_data(str(p), label_column="nonexistent")
        assert isinstance(df, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# load_config
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_dict_when_file_exists(self):
        config = load_config()
        assert isinstance(config, dict)

    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        """Monkeypatch PROJECT_ROOT so load_config looks in a temp dir."""
        import src.data.loader as loader_mod
        monkeypatch.setattr(loader_mod, "PROJECT_ROOT", tmp_path)
        result = load_config()
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# DatasetLoader — load_dataset_a / b / c  (explicit file_path)
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetLoaderExplicitPath:
    @pytest.fixture
    def loader(self):
        return DatasetLoader(config={})

    def test_load_dataset_a_explicit_path(self, loader, tmp_path):
        p = _make_csv(tmp_path)
        df = loader.load_dataset_a(file_path=str(p))
        assert len(df) == 3

    def test_load_dataset_b_explicit_path(self, loader, tmp_path):
        p = _make_csv(tmp_path)
        df = loader.load_dataset_b(file_path=str(p))
        assert len(df) == 3

    def test_load_dataset_c_explicit_path(self, loader, tmp_path):
        p = _make_csv(tmp_path)
        df = loader.load_dataset_c(file_path=str(p))
        assert len(df) == 3


# ─────────────────────────────────────────────────────────────────────────────
# DatasetLoader — no-file → empty DataFrame
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetLoaderNoFile:
    @pytest.fixture
    def loader(self, tmp_path):
        """Loader whose data_dir points to an empty tmp dir."""
        empty = tmp_path / "raw"
        empty.mkdir()
        inst = DatasetLoader(config={
            "dataset_a": {"path": str(empty / "dataset_a"), "file_pattern": "*.csv"},
            "dataset_b": {"path": str(empty / "dataset_b"), "file_pattern": "*.xlsx"},
            "dataset_c": {"path": str(empty / "dataset_c"), "file_pattern": "*.xlsx"},
        })
        return inst

    def test_load_dataset_a_no_file_returns_empty(self, loader):
        df = loader.load_dataset_a()
        assert df.empty

    def test_load_dataset_b_no_file_returns_empty(self, loader):
        df = loader.load_dataset_b()
        assert df.empty

    def test_load_dataset_c_no_file_returns_empty(self, loader):
        df = loader.load_dataset_c()
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# DatasetLoader.load_multiple_files
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadMultipleFiles:
    @pytest.fixture
    def loader(self):
        return DatasetLoader(config={})

    def test_single_file(self, loader, tmp_path):
        _make_csv(tmp_path, "a.csv", rows=4)
        df = loader.load_multiple_files(str(tmp_path), pattern="*.csv")
        assert len(df) == 4
        assert "_source_file" in df.columns

    def test_multiple_files_concatenated(self, loader, tmp_path):
        _make_csv(tmp_path, "a.csv", rows=2)
        _make_csv(tmp_path, "b.csv", rows=3)
        df = loader.load_multiple_files(str(tmp_path), pattern="*.csv")
        assert len(df) == 5

    def test_empty_directory_returns_empty(self, loader, tmp_path):
        df = loader.load_multiple_files(str(tmp_path), pattern="*.csv")
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# DatasetLoader._normalize_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeColumns:
    @pytest.fixture
    def loader(self):
        return DatasetLoader(config={})

    def test_no_mapping_returns_unchanged(self, loader):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = loader._normalize_columns(df, mapping={})
        assert list(result.columns) == ["a", "b"]

    def test_mapping_renames_columns(self, loader):
        # mapping = {표준명: 원본명} → reverse: {원본명: 표준명}
        df = pd.DataFrame({"original_col": [1, 2]})
        mapping = {"standard_col": "original_col"}
        result = loader._normalize_columns(df, mapping=mapping)
        assert "standard_col" in result.columns
        assert "original_col" not in result.columns

    def test_partial_mapping_only_renames_present(self, loader):
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        mapping = {"new_a": "col_a", "new_c": "col_c"}  # col_c not in df
        result = loader._normalize_columns(df, mapping=mapping)
        assert "new_a" in result.columns
        assert "col_b" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# DatasetLoader.get_dataset_info
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDatasetInfo:
    def test_returns_all_three_keys(self):
        loader = DatasetLoader(config={})
        info = loader.get_dataset_info()
        assert set(info.keys()) == {"dataset_a", "dataset_b", "dataset_c"}

    def test_each_entry_has_required_fields(self):
        loader = DatasetLoader(config={})
        info = loader.get_dataset_info()
        for key in ("dataset_a", "dataset_b", "dataset_c"):
            entry = info[key]
            assert "name" in entry
            assert "path" in entry
            assert "file_count" in entry
            assert isinstance(entry["files"], list)


# ─────────────────────────────────────────────────────────────────────────────
# create_dataset_directories
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateDatasetDirectories:
    def test_creates_expected_paths(self, tmp_path, monkeypatch):
        import src.data.loader as loader_mod
        monkeypatch.setattr(loader_mod, "PROJECT_ROOT", tmp_path)
        create_dataset_directories()
        assert (tmp_path / "data" / "raw" / "dataset_a").exists()
        assert (tmp_path / "data" / "raw" / "dataset_b").exists()
        assert (tmp_path / "data" / "raw" / "dataset_c").exists()
        assert (tmp_path / "data" / "processed").exists()
        assert (tmp_path / "data" / "features").exists()

    def test_idempotent(self, tmp_path, monkeypatch):
        """Calling twice should not raise."""
        import src.data.loader as loader_mod
        monkeypatch.setattr(loader_mod, "PROJECT_ROOT", tmp_path)
        create_dataset_directories()
        create_dataset_directories()  # second call should be fine
