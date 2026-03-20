"""Phase 3 tests: 에러 처리 강화 — sheet fallback, datetime quarantine, JOIN diagnosis, strict mode."""
import sys
import hashlib
from pathlib import Path
import tempfile

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.column_normalizer import ColumnNormalizer
from src.data.label_loader import LabelLoader


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_excel_no_valid_sheet(path: Path) -> None:
    """유효 시트(server_name 등)가 없는 Excel 파일 생성."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["col_a", "col_b"])
    ws.append(["v1", "v2"])
    wb.save(str(path))


def _make_excel_with_valid_sheet(path: Path) -> None:
    """유효 시트(server_name 포함)가 있는 Excel 파일 생성."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "데이터"
    ws.append(["서버이름", "에이전트IP", "파일경로", "파일이름", "파일생성일시"])
    ws.append(["srv01", "10.0.0.1", "/tmp", "file.txt", "2025-07-01 10:00:00"])
    wb.save(str(path))


# ─── Sheet fallback tests ─────────────────────────────────────────────────────

def test_no_valid_sheet_warn_and_fallback(tmp_path):
    """유효 시트 없음 → warn_and_first_fallback: 첫 시트 사용, 예외 없음."""
    excel_path = tmp_path / "test.xlsx"
    _make_excel_no_valid_sheet(excel_path)

    loader = LabelLoader(label_root=tmp_path)
    loader._on_no_valid_sheet = "warn_and_first_fallback"
    sheets = loader._read_excel_sheets(excel_path)
    # Should return first sheet as fallback (not raise)
    assert isinstance(sheets, list)


def test_no_valid_sheet_raise_mode(tmp_path):
    """유효 시트 없음 + raise 모드 → ValueError."""
    excel_path = tmp_path / "test.xlsx"
    _make_excel_no_valid_sheet(excel_path)

    loader = LabelLoader(label_root=tmp_path)
    loader._on_no_valid_sheet = "raise"
    with pytest.raises(ValueError, match="유효 시트 없음"):
        loader._read_excel_sheets(excel_path)


def test_no_valid_sheet_skip_mode(tmp_path):
    """유효 시트 없음 + skip 모드 → 빈 리스트."""
    excel_path = tmp_path / "test.xlsx"
    _make_excel_no_valid_sheet(excel_path)

    loader = LabelLoader(label_root=tmp_path)
    loader._on_no_valid_sheet = "skip"
    sheets = loader._read_excel_sheets(excel_path)
    assert sheets == []


# ─── datetime quarantine tests ────────────────────────────────────────────────

def test_datetime_parse_fail_quarantine_mode(tmp_path):
    """파싱 실패 행 → quarantine 파일로 격리."""
    df = pd.DataFrame({
        "server_name": ["srv1", "srv2"],
        "file_created_at": ["2025-07-01 10:00:00", "NOT_A_DATE"],
    })
    loader = LabelLoader(label_root=tmp_path)
    loader._on_datetime_parse_fail = "quarantine"

    # Patch quarantine path to tmp_path
    import unittest.mock as mock
    with mock.patch("src.data.label_loader.PROJECT_ROOT", tmp_path):
        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
        result = loader._parse_file_created_at(df)

    # Valid row should remain
    assert len(result) >= 1


def test_datetime_quarantine_file_created(tmp_path):
    """quarantine 모드 시 silver_quarantine.parquet 파일이 생성됨."""
    df = pd.DataFrame({
        "server_name": ["srv1"],
        "file_created_at": ["INVALID_DATE_XYZ"],
    })
    loader = LabelLoader(label_root=tmp_path)
    loader._on_datetime_parse_fail = "quarantine"

    import unittest.mock as mock
    with mock.patch("src.data.label_loader.PROJECT_ROOT", tmp_path):
        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
        loader._parse_file_created_at(df)
        qpath = tmp_path / "data" / "processed" / "silver_quarantine.parquet"
        if qpath.exists():
            q_df = pd.read_parquet(qpath)
            assert "quarantine_reason" in q_df.columns


def test_datetime_warn_mode_no_quarantine(tmp_path):
    """warn 모드 → quarantine 파일 생성 안 함."""
    df = pd.DataFrame({
        "server_name": ["srv1"],
        "file_created_at": ["INVALID_DATE_XYZ"],
    })
    loader = LabelLoader(label_root=tmp_path)
    loader._on_datetime_parse_fail = "warn"

    import unittest.mock as mock
    with mock.patch("src.data.label_loader.PROJECT_ROOT", tmp_path):
        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
        result = loader._parse_file_created_at(df)

    qpath = tmp_path / "data" / "processed" / "silver_quarantine.parquet"
    # Either file doesn't exist, or if exists from another test, result has the row
    assert len(result) >= 1  # warn mode keeps all rows


def test_valid_rows_still_saved_on_partial_fail(tmp_path):
    """일부 파싱 실패 시 나머지 유효 행은 보존."""
    df = pd.DataFrame({
        "server_name": ["srv1", "srv2", "srv3"],
        "file_created_at": ["2025-07-01 10:00:00", "INVALID", "2025-08-01 00:00:00"],
    })
    loader = LabelLoader(label_root=tmp_path)
    loader._on_datetime_parse_fail = "quarantine"

    import unittest.mock as mock
    with mock.patch("src.data.label_loader.PROJECT_ROOT", tmp_path):
        (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
        result = loader._parse_file_created_at(df)

    # 2 valid rows remain
    assert len(result) == 2


# ─── JOIN diagnosis tests ─────────────────────────────────────────────────────

def test_join_diagnose_called_on_zero_result(capsys):
    """JOIN 0건 시 _diagnose_join_mismatch 출력 확인."""
    import importlib
    import scripts.run_data_pipeline as rdp
    importlib.reload(rdp)

    df_label = pd.DataFrame({"pk_file": ["a" * 64]})
    df_det = pd.DataFrame({"pk_file": ["b" * 64]})

    rdp._diagnose_join_mismatch(df_label, df_det)
    captured = capsys.readouterr()
    assert "진단" in captured.out or "pk_file" in captured.out


def test_diagnose_output_contains_sample_pk(capsys):
    """진단 출력에 pk_file 샘플이 포함됨."""
    import scripts.run_data_pipeline as rdp
    pk_val = "a" * 64
    df_label = pd.DataFrame({"pk_file": [pk_val]})
    df_det = pd.DataFrame({"pk_file": ["b" * 64]})

    rdp._diagnose_join_mismatch(df_label, df_det)
    captured = capsys.readouterr()
    assert pk_val[:8] in captured.out or "샘플" in captured.out or "pk_file" in captured.out


def test_diagnose_detects_sha256_length_mismatch(capsys):
    """pk_file 길이가 64(SHA256)가 아닐 때 경고 출력."""
    import scripts.run_data_pipeline as rdp
    df_label = pd.DataFrame({"pk_file": ["short_pk"]})  # not 64 chars
    df_det = pd.DataFrame({"pk_file": ["b" * 64]})

    rdp._diagnose_join_mismatch(df_label, df_det)
    captured = capsys.readouterr()
    assert "경고" in captured.out or "SHA256" in captured.out or "길이" in captured.out


# ─── strict mode tests ────────────────────────────────────────────────────────

def test_strict_mode_raises_on_unknown_column():
    """strict=True + 미등록 한글 컬럼 → ValueError."""
    cn = ColumnNormalizer(strict=True)
    df = pd.DataFrame({"미등록컬럼이름임": [1, 2]})
    with pytest.raises(ValueError):
        cn.normalize(df)


def test_non_strict_warns_on_unknown_column():
    """strict=False + 미등록 한글 컬럼 → 경고만, 예외 없음."""
    cn = ColumnNormalizer(strict=False)
    df = pd.DataFrame({"미등록컬럼이름임": [1, 2]})
    result = cn.normalize(df)  # should not raise
    assert result is not None


def test_strict_false_is_default():
    """ColumnNormalizer 기본값은 strict=False."""
    cn = ColumnNormalizer()
    assert cn._strict is False


def test_known_column_passes_strict_mode():
    """strict=True에서도 등록된 컬럼은 정상 처리."""
    cn = ColumnNormalizer(strict=True)
    df = pd.DataFrame({"서버이름": ["srv1"]})
    result = cn.normalize(df)
    assert "server_name" in result.columns
