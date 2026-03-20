"""Unit tests for scripts/run_data_pipeline.py"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import module under test
import scripts.run_data_pipeline as pipeline_mod
from scripts.run_data_pipeline import parse_args, save_validation_report


# ─────────────────────────────────────────────────────────────────────────────
# parse_args
# ─────────────────────────────────────────────────────────────────────────────

class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py"])
        args = parse_args()
        assert args.dataset == "all"
        assert args.validate_only is False
        assert args.skip_validation is False

    def test_dataset_b(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py", "--dataset", "b"])
        args = parse_args()
        assert args.dataset == "b"

    def test_validate_only(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py", "--validate-only"])
        args = parse_args()
        assert args.validate_only is True

    def test_skip_validation(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py", "--skip-validation"])
        args = parse_args()
        assert args.skip_validation is True

    def test_invalid_dataset_raises(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py", "--dataset", "z"])
        with pytest.raises(SystemExit):
            parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# save_validation_report
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveValidationReport:
    def test_creates_file(self, tmp_path):
        report_path = tmp_path / "report.txt"
        save_validation_report({}, report_path)
        assert report_path.exists()

    def test_contains_header(self, tmp_path):
        report_path = tmp_path / "report.txt"
        save_validation_report({}, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "데이터 검증 리포트" in content

    def test_basic_section_written(self, tmp_path):
        report_path = tmp_path / "report.txt"
        report = {
            "basic": {
                "n_rows": 100,
                "missing_columns": {},
                "n_duplicates": 2,
                "label_distribution": {"TP": 50, "FP": 50},
            }
        }
        save_validation_report(report, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "[기본 검증]" in content
        assert "100" in content

    def test_masking_section_written(self, tmp_path):
        report_path = tmp_path / "report.txt"
        report = {
            "masking": {
                "total_rows": 200,
                "masked_rows": 180,
                "masking_rate": 0.9,
                "context_length_stats": {"mean": 50.0, "max": 200},
                "exposed_phone": 0,
                "exposed_email": 1,
                "exposed_jumin": 0,
            }
        }
        save_validation_report(report, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "[마스킹 검증]" in content
        assert "90.00%" in content

    def test_pattern_type_section_written(self, tmp_path):
        report_path = tmp_path / "report.txt"
        report = {
            "pattern_type": {
                "total_rows": 100,
                "mismatch_rate": 0.05,
                "mismatch_details": {"EMAIL->PHONE": 5},
            }
        }
        save_validation_report(report, report_path)
        content = report_path.read_text(encoding="utf-8")
        assert "[패턴 타입 검증]" in content
        assert "5.00%" in content

    def test_empty_report_does_not_raise(self, tmp_path):
        report_path = tmp_path / "report.txt"
        save_validation_report({}, report_path)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# main() — no datasets → early return
# ─────────────────────────────────────────────────────────────────────────────

def _make_empty_loader():
    """Return a DatasetLoader mock whose load_dataset_* return empty DataFrames."""
    mock_loader = MagicMock()
    mock_loader.load_dataset_a.return_value = pd.DataFrame()
    mock_loader.load_dataset_b.return_value = pd.DataFrame()
    mock_loader.load_dataset_c.return_value = pd.DataFrame()
    return mock_loader


class TestMainNoData:
    def test_main_exits_gracefully_when_no_datasets(self, monkeypatch, tmp_path, capsys):
        """main() --source label 모드에서 데이터 없음 시 오류 메시지 출력 후 정상 종료."""
        # --source label 강제: LabelLoader.load_all() 패치로 빈 DataFrame 반환
        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py", "--source", "label"])

        from src.data.label_loader import LabelLoader
        monkeypatch.setattr(LabelLoader, "load_all", lambda self: pd.DataFrame())
        monkeypatch.setattr(pipeline_mod, "ensure_dirs", lambda *a: None)

        pipeline_mod.main()  # Must not raise

        captured = capsys.readouterr()
        assert "없습니다" in captured.out or "오류" in captured.out


class TestMainWithData:
    """main() end-to-end with a minimal synthetic DataFrame."""

    def _setup_mocks(self, monkeypatch, tmp_path, df_b: pd.DataFrame) -> None:
        """Patch all external IO so main() runs against in-memory data."""
        from src.utils.constants import TEXT_COLUMN, LABEL_COLUMN

        # Ensure df_b has required columns
        if TEXT_COLUMN not in df_b.columns:
            df_b[TEXT_COLUMN] = "test text"

        mock_loader = MagicMock()
        mock_loader.load_dataset_a.return_value = pd.DataFrame()
        mock_loader.load_dataset_b.return_value = df_b
        mock_loader.load_dataset_c.return_value = pd.DataFrame()

        monkeypatch.setattr(pipeline_mod, "DatasetLoader", lambda config: mock_loader)
        monkeypatch.setattr(pipeline_mod, "load_config", lambda: {})
        monkeypatch.setattr(pipeline_mod, "ensure_dirs", lambda *a: None)

        # Stub out validation functions
        basic_mock = {"n_rows": len(df_b), "missing_columns": {}, "n_duplicates": 0, "label_distribution": {}}
        monkeypatch.setattr(pipeline_mod, "validate_data", lambda df, **kw: basic_mock)
        monkeypatch.setattr(pipeline_mod, "get_available_pk", lambda df, cfg: [])
        monkeypatch.setattr(pipeline_mod, "create_composite_pk", lambda df, cols: df)

        # Redirect report and output paths to tmp_path
        (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
        (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(pipeline_mod, "REPORT_DIR", tmp_path / "reports")
        monkeypatch.setattr(pipeline_mod, "PROCESSED_DATA_DIR", tmp_path / "processed")

        # Stub preprocessor and save
        monkeypatch.setattr(pipeline_mod, "preprocess_dataframe", lambda df, **kw: df)
        monkeypatch.setattr(pipeline_mod, "save_processed", lambda df, path: None)

    def test_main_runs_with_dataset_b(self, monkeypatch, tmp_path):
        from src.utils.constants import TEXT_COLUMN, LABEL_COLUMN

        df_b = pd.DataFrame({
            TEXT_COLUMN: ["hello@example.com", "test 123"],
            LABEL_COLUMN: ["FP-더미데이터", "FP-숫자나열/코드"],
        })

        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py", "--skip-validation"])
        self._setup_mocks(monkeypatch, tmp_path, df_b)
        pipeline_mod.main()  # must not raise

    def test_validate_only_does_not_call_save(self, monkeypatch, tmp_path):
        from src.utils.constants import TEXT_COLUMN, LABEL_COLUMN

        df_b = pd.DataFrame({
            TEXT_COLUMN: ["data"],
            LABEL_COLUMN: ["FP-더미데이터"],
        })

        save_called = []

        monkeypatch.setattr(sys, "argv", ["run_data_pipeline.py", "--validate-only", "--skip-validation"])
        self._setup_mocks(monkeypatch, tmp_path, df_b)
        monkeypatch.setattr(pipeline_mod, "save_processed", lambda df, path: save_called.append(True))

        pipeline_mod.main()
        assert save_called == [], "save_processed should NOT be called in --validate-only mode"
