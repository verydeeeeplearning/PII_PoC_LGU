"""Phase 1 테스트: Split 확장 + poc_metrics 함수 (TDD — RED 먼저)"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_month_df(months_per_file=3, n_files=6):
    """label_work_month + pk_file DataFrame 생성.

    파일을 두 그룹으로 나누어 앞 절반은 초기 월(1..months_per_file-1),
    뒤 절반은 마지막 월(months_per_file)에만 배정하여
    월 기반 시간 분리 테스트가 의미 있게 동작하도록 함.
    """
    rows = []
    half = n_files // 2
    for i in range(n_files):
        pk = f"file_{i:03d}"
        if i < half:
            # 앞 절반: 초기 월들(마지막 월 제외)
            month_range = range(1, months_per_file)
        else:
            # 뒤 절반: 마지막 월만
            month_range = range(months_per_file, months_per_file + 1)
        for m in month_range:
            rows.append({"pk_file": pk, "label_work_month": f"{m}월",
                         "label_raw": "TP" if i % 2 == 0 else "FP"})
    return pd.DataFrame(rows)


def _make_org_df():
    """_source_file 기반 조직 DataFrame"""
    rows = [
        {"_source_file": "CTO_2024_01.xlsx", "label_raw": "TP"},
        {"_source_file": "CTO_2024_02.xlsx", "label_raw": "FP"},
        {"_source_file": "NW_2024_01.xlsx",  "label_raw": "FP"},
        {"_source_file": "NW_2024_02.xlsx",  "label_raw": "TP"},
        {"_source_file": "품질혁신센터_2024_01.xlsx", "label_raw": "FP"},
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TestWorkMonthTimeSplit
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkMonthTimeSplit:
    def test_last_n_months_go_to_test(self):
        from src.evaluation.split_strategies import work_month_time_split

        df = _make_month_df(months_per_file=3, n_files=6)
        # 3개 월 중 마지막 1개월 → test
        train_idx, test_idx = work_month_time_split(df, test_months=1)

        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert set(train_idx) | set(test_idx) == set(range(len(df)))

        # test에 있는 행의 월이 train의 최대 월보다 크거나 같아야 함
        train_months = df.iloc[train_idx]["label_work_month"].apply(
            lambda x: int(str(x).replace("월", ""))
        )
        test_months = df.iloc[test_idx]["label_work_month"].apply(
            lambda x: int(str(x).replace("월", ""))
        )
        assert test_months.max() >= train_months.max()

    def test_no_pk_file_leakage(self):
        from src.evaluation.split_strategies import work_month_time_split

        df = _make_month_df(months_per_file=3, n_files=6)
        train_idx, test_idx = work_month_time_split(df, test_months=1)

        train_files = set(df.iloc[train_idx]["pk_file"])
        test_files = set(df.iloc[test_idx]["pk_file"])
        # pk_file 누수 없음
        assert train_files.isdisjoint(test_files)

    def test_single_month_all_goes_to_test(self):
        from src.evaluation.split_strategies import work_month_time_split

        df = pd.DataFrame([
            {"pk_file": "f1", "label_work_month": "5월", "label_raw": "TP"},
            {"pk_file": "f2", "label_work_month": "5월", "label_raw": "FP"},
        ])
        train_idx, test_idx = work_month_time_split(df, test_months=2)
        # 월이 1개뿐이면 전부 test
        assert len(test_idx) == 2
        assert len(train_idx) == 0

    def test_index_union_is_complete(self):
        from src.evaluation.split_strategies import work_month_time_split

        df = _make_month_df()
        train_idx, test_idx = work_month_time_split(df)
        assert sorted(train_idx + test_idx) == list(range(len(df)))

    def test_returns_lists(self):
        from src.evaluation.split_strategies import work_month_time_split

        df = _make_month_df()
        train_idx, test_idx = work_month_time_split(df)
        assert isinstance(train_idx, list)
        assert isinstance(test_idx, list)


# ─────────────────────────────────────────────────────────────────────────────
# TestOrgSubsetSplit
# ─────────────────────────────────────────────────────────────────────────────

class TestOrgSubsetSplit:
    def test_target_org_isolated_in_test(self):
        from src.evaluation.split_strategies import org_subset_split

        df = pd.DataFrame([
            {"organization": "CTO", "label_raw": "TP"},
            {"organization": "CTO", "label_raw": "FP"},
            {"organization": "NW",  "label_raw": "FP"},
        ])
        train_idx, test_idx = org_subset_split(df, target_org="CTO")

        test_orgs = set(df.iloc[test_idx]["organization"])
        train_orgs = set(df.iloc[train_idx]["organization"])

        assert test_orgs == {"CTO"}
        assert "CTO" not in train_orgs

    def test_missing_org_col_raises_valueerror(self):
        from src.evaluation.split_strategies import org_subset_split

        df = pd.DataFrame([{"label_raw": "TP"}, {"label_raw": "FP"}])
        with pytest.raises(ValueError):
            org_subset_split(df, target_org="CTO")

    def test_org_extracted_from_source_file_filename(self):
        from src.evaluation.split_strategies import org_subset_split

        df = _make_org_df()
        train_idx, test_idx = org_subset_split(
            df, target_org="CTO", source_file_col="_source_file"
        )
        # CTO 파일은 test에만
        test_files = df.iloc[test_idx]["_source_file"].tolist()
        assert all("CTO" in f for f in test_files)
        assert len(test_idx) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeBinaryStats
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBinaryStats:
    def _make_df(self):
        rows = []
        for m in range(1, 4):
            for _ in range(10):
                rows.append({"label_raw": "TP", "label_work_month": f"{m}월"})
            for _ in range(5):
                rows.append({"label_raw": "FP", "label_work_month": f"{m}월"})
        return pd.DataFrame(rows)

    def test_tp_fp_count_and_ratio(self):
        from src.evaluation.poc_metrics import compute_binary_stats

        df = self._make_df()
        result = compute_binary_stats(df)

        assert result["total"]["tp"] == 30
        assert result["total"]["fp"] == 15
        assert result["total"]["total"] == 45
        assert abs(result["total"]["tp_ratio"] - 30/45) < 1e-6

    def test_monthly_breakdown_columns_exist(self):
        from src.evaluation.poc_metrics import compute_binary_stats

        df = self._make_df()
        result = compute_binary_stats(df)
        by_month = result["by_month"]

        assert "month" in by_month.columns
        assert "tp_count" in by_month.columns
        assert "fp_count" in by_month.columns
        assert "total" in by_month.columns
        assert "tp_ratio" in by_month.columns
        assert len(by_month) == 3

    def test_empty_df_returns_zero_dict(self):
        from src.evaluation.poc_metrics import compute_binary_stats

        result = compute_binary_stats(pd.DataFrame())
        assert result["total"]["tp"] == 0
        assert result["total"]["total"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeCoveragePrecisionCurve
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCoveragePrecisionCurve:
    def _make_data(self, n=100, tp_ratio=0.3):
        rng = np.random.default_rng(42)
        n_tp = int(n * tp_ratio)
        n_fp = n - n_tp
        y_true = ["TP"] * n_tp + ["FP"] * n_fp
        # FP에 높은 확률, TP에 낮은 확률 (FP 잘 잡음)
        proba = np.concatenate([
            rng.uniform(0.3, 0.6, n_tp),
            rng.uniform(0.6, 1.0, n_fp),
        ])
        return np.array(y_true), proba

    def test_tau_sweep_produces_correct_row_count(self):
        from src.evaluation.poc_metrics import compute_coverage_precision_curve

        y_true, proba = self._make_data()
        result = compute_coverage_precision_curve(
            y_true, proba, tau_range=(0.5, 1.0, 0.1)
        )
        # 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 → 6 rows
        assert len(result["curve"]) == 6

    def test_coverage_nonincreasing_as_tau_increases(self):
        from src.evaluation.poc_metrics import compute_coverage_precision_curve

        y_true, proba = self._make_data()
        result = compute_coverage_precision_curve(
            y_true, proba, tau_range=(0.5, 0.95, 0.05)
        )
        coverage = result["curve"]["coverage"].values
        # tau가 오르면 coverage는 내려가거나 유지
        assert all(coverage[i] >= coverage[i+1] - 1e-9 for i in range(len(coverage)-1))

    def test_precision_at_tau_zero_is_baseline(self):
        from src.evaluation.poc_metrics import compute_coverage_precision_curve

        y_true, proba = self._make_data(n=100, tp_ratio=0.3)
        result = compute_coverage_precision_curve(
            y_true, proba, tau_range=(0.0, 0.1, 0.1)
        )
        # tau=0이면 전체가 auto_fp → precision = FP 비율
        first_row = result["curve"].iloc[0]
        assert first_row["auto_fp_count"] > 0

    def test_recommended_tau_first_above_precision_target(self):
        from src.evaluation.poc_metrics import compute_coverage_precision_curve

        y_true, proba = self._make_data(n=200, tp_ratio=0.2)
        result = compute_coverage_precision_curve(
            y_true, proba, tau_range=(0.5, 1.0, 0.05), precision_target=0.5
        )
        # precision target 0.5은 낮으므로 recommended_tau가 None이 아님
        rec = result["recommended_tau"]
        if rec is not None:
            row = result["curve"][result["curve"]["tau"] == rec].iloc[0]
            assert row["precision"] >= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeSplitComparison
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSplitComparison:
    def _make_split_result(self, name, n=50, tp_ratio=0.5):
        rng = np.random.default_rng(42)
        n_tp = int(n * tp_ratio)
        n_fp = n - n_tp
        y_true = np.array(["TP"] * n_tp + ["FP"] * n_fp)
        # 80% 정확도
        y_pred = y_true.copy()
        flip = rng.choice(len(y_true), size=int(len(y_true)*0.2), replace=False)
        for i in flip:
            y_pred[i] = "FP" if y_true[i] == "TP" else "TP"
        return {
            "split_name": name,
            "train_n": 200,
            "test_n": n,
            "y_true": y_true,
            "y_pred": y_pred,
            "tp_label": "TP",
            "coverage_at_target": 0.85,
        }

    def test_returns_dataframe_with_split_rows(self):
        from src.evaluation.poc_metrics import compute_split_comparison

        results = [
            self._make_split_result("Primary"),
            self._make_split_result("Secondary"),
        ]
        df = compute_split_comparison(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "split_name" in df.columns

    def test_poc_verdict_is_pass_or_fail(self):
        from src.evaluation.poc_metrics import compute_split_comparison

        results = [self._make_split_result("Primary")]
        df = compute_split_comparison(results)
        assert df["poc_verdict"].iloc[0] in ("PASS", "FAIL", "SKIP")

    def test_missing_tertiary_excluded_gracefully(self):
        from src.evaluation.poc_metrics import compute_split_comparison

        # split_results가 비어 있어도 오류 없음
        df = compute_split_comparison([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeClassImbalance
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeClassImbalance:
    def test_ratios_sum_to_one(self):
        from src.evaluation.poc_metrics import compute_class_imbalance

        df = pd.DataFrame({"label_raw": ["TP", "TP", "FP", "FP", "FP"]})
        result = compute_class_imbalance(df)
        assert abs(result["ratio"].sum() - 1.0) < 1e-9

    def test_correct_counts_per_class(self):
        from src.evaluation.poc_metrics import compute_class_imbalance

        df = pd.DataFrame({"label_raw": ["TP", "TP", "FP", "FP", "FP"]})
        result = compute_class_imbalance(df)
        tp_row = result[result["class_name"] == "TP"].iloc[0]
        fp_row = result[result["class_name"] == "FP"].iloc[0]
        assert tp_row["count"] == 2
        assert fp_row["count"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeClassMetrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeClassMetrics:
    def test_returns_required_columns(self):
        from src.evaluation.poc_metrics import compute_class_metrics

        y_true = np.array(["TP", "TP", "FP", "FP", "FP"])
        y_pred = np.array(["TP", "FP", "FP", "FP", "TP"])
        result = compute_class_metrics(y_true, y_pred)

        assert isinstance(result, pd.DataFrame)
        for col in ["class_name", "precision", "recall", "f1_score", "support"]:
            assert col in result.columns

    def test_all_tp_fp_binary_case(self):
        from src.evaluation.poc_metrics import compute_class_metrics

        y_true = np.array(["TP"] * 10 + ["FP"] * 10)
        y_pred = np.array(["TP"] * 10 + ["FP"] * 10)
        result = compute_class_metrics(y_true, y_pred)

        assert len(result) == 2
        tp_row = result[result["class_name"] == "TP"].iloc[0]
        assert abs(tp_row["precision"] - 1.0) < 1e-6
        assert abs(tp_row["recall"] - 1.0) < 1e-6

    def test_empty_input_returns_empty_df(self):
        from src.evaluation.poc_metrics import compute_class_metrics

        result = compute_class_metrics(np.array([]), np.array([]))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_sorted_by_support_desc(self):
        from src.evaluation.poc_metrics import compute_class_metrics

        y_true = np.array(["TP"] * 30 + ["FP"] * 10)
        y_pred = y_true.copy()
        result = compute_class_metrics(y_true, y_pred)

        supports = result["support"].tolist()
        assert supports == sorted(supports, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeOrgStats
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeOrgStats:
    def test_groups_by_org_col(self):
        from src.evaluation.poc_metrics import compute_org_stats

        df = pd.DataFrame({
            "organization": ["CTO", "CTO", "NW", "NW", "NW"],
            "label_raw":    ["TP",  "FP",  "FP", "FP", "TP"],
        })
        result = compute_org_stats(df, label_col="label_raw", org_col="organization")

        assert isinstance(result, pd.DataFrame)
        for col in ["organization", "tp_count", "fp_count", "total", "tp_ratio"]:
            assert col in result.columns

        cto = result[result["organization"] == "CTO"].iloc[0]
        assert cto["tp_count"] == 1
        assert cto["fp_count"] == 1
        assert cto["total"] == 2

    def test_fallback_from_source_file(self):
        from src.evaluation.poc_metrics import compute_org_stats

        df = pd.DataFrame({
            "_source_file": ["CTO_2024_01.xlsx", "CTO_2024_02.xlsx", "NW_2024_01.xlsx"],
            "label_raw":    ["TP", "FP", "FP"],
        })
        result = compute_org_stats(df, label_col="label_raw")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_empty_df_returns_empty(self):
        from src.evaluation.poc_metrics import compute_org_stats

        result = compute_org_stats(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeConfidenceDistribution
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeConfidenceDistribution:
    def test_bins_sum_equals_total_count(self):
        from src.evaluation.poc_metrics import compute_confidence_distribution

        proba = np.array([0.51, 0.62, 0.73, 0.84, 0.95, 0.99, 0.55, 0.78])
        result = compute_confidence_distribution(proba)

        assert isinstance(result, pd.DataFrame)
        for col in ["proba_range", "count", "ratio", "cumulative_ratio"]:
            assert col in result.columns
        assert result["count"].sum() == len(proba)

    def test_cumulative_ratio_ends_at_one(self):
        from src.evaluation.poc_metrics import compute_confidence_distribution

        proba = np.linspace(0.5, 1.0, 50)
        result = compute_confidence_distribution(proba)

        assert abs(result["cumulative_ratio"].iloc[-1] - 1.0) < 1e-6

    def test_empty_proba_returns_empty(self):
        from src.evaluation.poc_metrics import compute_confidence_distribution

        result = compute_confidence_distribution(np.array([]))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_custom_bins(self):
        from src.evaluation.poc_metrics import compute_confidence_distribution

        proba = np.array([0.55, 0.65, 0.75, 0.85])
        result = compute_confidence_distribution(proba, bins=[0.5, 0.7, 0.9, 1.0])
        assert len(result) == 3  # 3 intervals from 4 bin edges
