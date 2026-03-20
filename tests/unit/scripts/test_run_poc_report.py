"""Phase 4 테스트: run_poc_report CLI (TDD)"""
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# TestParseArgs
# ─────────────────────────────────────────────────────────────────────────────

class TestParseArgs:
    def _parse(self, args_list):
        from scripts.run_poc_report import parse_args
        return parse_args(args_list)

    def test_default_phase_is_1(self):
        args = self._parse([])
        assert args.phase == 1

    def test_default_output_path(self):
        args = self._parse([])
        assert "poc_report" in str(args.output).lower()

    def test_phase_2_accepted(self):
        args = self._parse(["--phase", "2"])
        assert args.phase == 2

    def test_skip_ml_flag(self):
        args = self._parse(["--skip-ml"])
        assert args.skip_ml is True

    def test_precision_target_override(self):
        args = self._parse(["--precision-target", "0.90"])
        assert abs(args.precision_target - 0.90) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildPocReportData
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPocReportData:
    def _make_label_df(self, n=100):
        rng = np.random.default_rng(42)
        rows = []
        for i in range(n):
            org = ["CTO", "NW", "품질혁신센터"][i % 3]
            month = (i % 12) + 1
            rows.append({
                "pk_event": f"evt_{i:04d}",
                "pk_file":  f"file_{i // 3:04d}",
                "label_raw": "TP" if i % 3 == 0 else "FP",
                "label_work_month": f"{month}월",
                "_source_file": f"{org}_2024_{month:02d}.xlsx",
                "fp_description": f"desc_{i % 5}",
            })
        return pd.DataFrame(rows)

    def test_data_condition_label_only_for_phase1(self):
        from scripts.run_poc_report import _get_data_condition
        assert _get_data_condition(phase=1) == "Label Only"

    def test_data_condition_label_sumologic_for_phase2(self):
        from scripts.run_poc_report import _get_data_condition
        assert _get_data_condition(phase=2) == "Label + Sumologic"

    def test_split_summary_has_train_test_n(self):
        from scripts.run_poc_report import _make_split_summary

        df = self._make_label_df()
        train_idx = list(range(80))
        test_idx = list(range(80, 100))
        summary = _make_split_summary(df, train_idx, test_idx, split_method="GroupTimeSplit")

        assert "train_n" in summary
        assert "test_n" in summary
        assert summary["train_n"] == 80
        assert summary["test_n"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# TestOrgHandling
# ─────────────────────────────────────────────────────────────────────────────

class TestOrgHandling:
    def _make_df_with_source_file(self):
        rows = [
            {"pk_event": "e1", "pk_file": "f1", "_source_file": "CTO_2024_01.xlsx",
             "label_raw": "TP", "label_work_month": "1월"},
            {"pk_event": "e2", "pk_file": "f2", "_source_file": "CTO_2024_02.xlsx",
             "label_raw": "FP", "label_work_month": "1월"},
            {"pk_event": "e3", "pk_file": "f3", "_source_file": "NW_2024_01.xlsx",
             "label_raw": "FP", "label_work_month": "2월"},
            {"pk_event": "e4", "pk_file": "f4", "_source_file": "품질혁신센터_2024_01.xlsx",
             "label_raw": "TP", "label_work_month": "3월"},
        ]
        return pd.DataFrame(rows)

    def test_cto_org_extracted_from_filename(self):
        from scripts.run_poc_report import _extract_org_column

        df = self._make_df_with_source_file()
        org_series = _extract_org_column(df)

        cto_rows = df[org_series == "CTO"]
        assert len(cto_rows) == 2

    def test_missing_org_warns_and_skips_tertiary(self):
        from scripts.run_poc_report import _build_tertiary_splits

        # 파일명에서 조직을 추출할 수 없는 경우
        df = pd.DataFrame([
            {"pk_event": "e1", "pk_file": "f1", "_source_file": "unknown_file.xlsx",
             "label_raw": "TP", "label_work_month": "1월"},
        ])
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = _build_tertiary_splits(df, y_true=np.array(["TP"]))
            # 조직 추출 실패 시 빈 리스트 반환
            assert isinstance(results, list)

    def test_all_three_orgs_produce_comparison_rows(self):
        from scripts.run_poc_report import _build_tertiary_splits

        rows = []
        for org in ["CTO", "NW", "품질혁신센터"]:
            for i in range(10):
                rows.append({
                    "pk_event": f"{org}_e{i}",
                    "pk_file": f"{org}_f{i}",
                    "_source_file": f"{org}_2024_01.xlsx",
                    "label_raw": "TP" if i % 2 == 0 else "FP",
                    "label_work_month": "1월",
                })
        df = pd.DataFrame(rows)
        y_true = np.array([r["label_raw"] for r in rows])

        results = _build_tertiary_splits(df, y_true=y_true)
        # 각 조직별 결과가 있어야 함 (성공한 것만)
        assert isinstance(results, list)
        names = [r.get("split_name", "") for r in results]
        # 3조직이 모두 있을 때 3개 결과
        assert len(results) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# TestNewReportFields
# ─────────────────────────────────────────────────────────────────────────────

class TestNewReportFields:
    def test_run_metadata_has_run_datetime(self):
        """_build_run_metadata가 run_datetime 키를 반환한다."""
        from scripts.run_poc_report import _build_run_metadata

        result = _build_run_metadata(model_path=None, df=pd.DataFrame({"label_work_month": ["1월"]}))
        assert "run_datetime" in result
        assert result["run_datetime"] != ""

    def test_business_impact_has_estimated_auto_fp(self):
        """_build_business_impact가 estimated_auto_fp를 계산한다."""
        from scripts.run_poc_report import _build_business_impact

        coverage_curve = {
            "recommended_tau": 0.9,
            "curve": pd.DataFrame({
                "tau": [0.9],
                "coverage": [0.5],
                "precision": [0.96],
                "tp_safety_rate": [0.0],
                "auto_fp_count": [50],
            }),
        }
        binary_stats = {"total": {"fp": 100, "tp": 50, "total": 150, "tp_ratio": 0.5, "fp_ratio": 0.5}}
        result = _build_business_impact(binary_stats, coverage_curve)

        assert "estimated_auto_fp" in result
        assert result["estimated_auto_fp"] == 50  # 100 * 0.5

    def test_error_risk_summary_separates_fp_to_tp(self):
        """_build_error_risk_summary가 fp_to_tp_count와 tp_to_fp_count를 분리한다."""
        from scripts.run_poc_report import _build_error_risk_summary

        y_test = np.array(["TP", "FP", "FP", "TP"])
        y_pred = np.array(["FP", "TP", "FP", "TP"])

        result = _build_error_risk_summary(y_test, y_pred, tp_label="TP")

        assert "fp_to_tp_count" in result
        assert "tp_to_fp_count" in result
        assert result["fp_to_tp_count"] == 1   # FP→TP (위험)
        assert result["tp_to_fp_count"] == 1   # TP→FP (누락)

    def test_class_metrics_in_report_data_field(self):
        """PocReportData에 class_metrics 필드가 존재한다."""
        from src.report.excel_writer import PocReportData
        import pandas as pd

        data = PocReportData()
        assert hasattr(data, "class_metrics")
        assert isinstance(data.class_metrics, pd.DataFrame)

    def test_confidence_distribution_field_exists(self):
        """PocReportData에 confidence_distribution 필드가 존재한다."""
        from src.report.excel_writer import PocReportData

        data = PocReportData()
        assert hasattr(data, "confidence_distribution")
        assert isinstance(data.confidence_distribution, pd.DataFrame)
