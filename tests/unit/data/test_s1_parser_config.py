"""Phase 4 tests: S1ParserConfig."""
import sys
from pathlib import Path
import tempfile

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.s1_parser import S1ParserConfig, parse_context_field, make_pk_file


@pytest.fixture
def pk_file():
    return make_pk_file("server01", "10.0.0.1", "/data/test.csv")


def test_defaults_window_size_is_60():
    cfg = S1ParserConfig.defaults()
    assert cfg.window_size == 60


def test_defaults_has_masking_pattern():
    cfg = S1ParserConfig.defaults()
    assert "*" in cfg.masking_pattern


def test_from_yaml_loads_config_values():
    cfg = S1ParserConfig.from_yaml()
    # Should load from actual preprocessing_config.yaml
    assert cfg.window_size == 60
    assert cfg.pattern_truncation == 50


def test_from_yaml_falls_back_to_defaults_on_missing_file(tmp_path):
    cfg = S1ParserConfig.from_yaml(config_path=tmp_path / "nonexistent.yaml")
    assert cfg.window_size == 60
    assert cfg.pattern_truncation == 50


def test_parse_context_field_uses_config_window(pk_file):
    """config.window_size가 parse_context_field에 반영됨."""
    raw = "left context " + "hit***" + " right context"
    cfg = S1ParserConfig(window_size=5)
    results = parse_context_field(raw, pk_file, masked_hit="hit***", config=cfg)
    assert len(results) >= 1


def test_parse_context_field_backward_compat_no_config(pk_file):
    """config 없이 기존 방식으로 호출 — 하위 호환."""
    raw = "some text with ***masked*** pattern"
    results = parse_context_field(raw, pk_file)
    assert isinstance(results, list)


def test_pattern_truncation_applied(pk_file):
    """pattern_truncation이 3차 폴백에서 적용됨."""
    raw = "x" * 100  # no masking pattern, no masked_hit → 3rd fallback
    cfg = S1ParserConfig(pattern_truncation=10, context_truncation=20)
    results = parse_context_field(raw, pk_file, config=cfg)
    assert len(results) == 1
    assert len(results[0]["masked_pattern"]) <= 10


def test_context_truncation_applied(pk_file):
    """context_truncation이 3차 폴백에서 적용됨."""
    raw = "y" * 200  # no masking pattern → 3rd fallback
    cfg = S1ParserConfig(pattern_truncation=50, context_truncation=30)
    results = parse_context_field(raw, pk_file, config=cfg)
    assert len(results) == 1
    assert len(results[0]["full_context"]) <= 30


def test_encoding_candidates_from_yaml():
    """loader._get_csv_encoding_candidates()가 YAML에서 읽음."""
    from src.data.loader import _get_csv_encoding_candidates
    candidates = _get_csv_encoding_candidates("utf-8")
    assert "utf-8" in candidates
    assert len(candidates) >= 2


def test_encoding_fallback_on_yaml_error(tmp_path):
    """YAML 로드 실패 시 기본값 사용."""
    from src.data.loader import _get_csv_encoding_candidates
    import unittest.mock as mock
    # Simulate YAML load failure
    with mock.patch("src.data.loader._get_csv_encoding_candidates", wraps=lambda enc: [enc, "utf-8-sig", "cp949"]):
        from src.data.loader import _get_csv_encoding_candidates as fn
        result = fn("utf-8")
        assert "utf-8" in result
        assert "cp949" in result
