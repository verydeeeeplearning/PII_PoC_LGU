"""Unit tests for src/features/meta_features.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.meta_features import (
    build_meta_features,
    compute_file_aggregates_label,
    extract_datetime_features,
    extract_detection_features,
    extract_fname_features,
    merge_file_aggregates_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# extract_fname_features
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractFnameFeatures:
    def test_has_date_yyyymmdd(self):
        r = extract_fname_features("report_20250315.log")
        assert r["fname_has_date"] == 1

    def test_has_date_yyyymm(self):
        r = extract_fname_features("access_202503.log")
        assert r["fname_has_date"] == 1

    def test_has_date_yyyy_mm_dd(self):
        r = extract_fname_features("log-2025-03-15.gz")
        assert r["fname_has_date"] == 1

    def test_no_date(self):
        r = extract_fname_features("application.jar")
        assert r["fname_has_date"] == 0

    def test_has_hash_hex16(self):
        r = extract_fname_features("abc1234567890abcd.tmp")
        assert r["fname_has_hash"] == 1

    def test_no_hash_short(self):
        r = extract_fname_features("abc123.txt")
        assert r["fname_has_hash"] == 0

    def test_has_rotation_num_dot(self):
        r = extract_fname_features("access.log.1")
        assert r["fname_has_rotation_num"] == 1

    def test_has_rotation_num_dash_num(self):
        r = extract_fname_features("syslog-2025")
        assert r["fname_has_rotation_num"] == 1

    def test_has_rotation_num_zero_pad(self):
        r = extract_fname_features("app.log-001")
        assert r["fname_has_rotation_num"] == 1

    def test_no_rotation(self):
        r = extract_fname_features("customer_data.xlsx")
        assert r["fname_has_rotation_num"] == 0

    def test_empty_string(self):
        r = extract_fname_features("")
        assert r["fname_has_date"] == 0
        assert r["fname_has_hash"] == 0
        assert r["fname_has_rotation_num"] == 0

    def test_none_input(self):
        r = extract_fname_features(None)
        assert r["fname_has_date"] == 0

    def test_keys_present(self):
        r = extract_fname_features("test.log")
        assert {"fname_has_date", "fname_has_hash", "fname_has_rotation_num"} <= r.keys()


# ─────────────────────────────────────────────────────────────────────────────
# extract_detection_features
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractDetectionFeatures:
    def _row(self, pattern_count=0, ssn_count=0, phone_count=0, email_count=0):
        return pd.Series({
            "pattern_count": pattern_count,
            "ssn_count": ssn_count,
            "phone_count": phone_count,
            "email_count": email_count,
        })

    def test_log1p_zero(self):
        r = extract_detection_features(self._row(0))
        assert r["pattern_count_log1p"] == pytest.approx(0.0)

    def test_log1p_nonzero(self):
        r = extract_detection_features(self._row(10))
        assert r["pattern_count_log1p"] == pytest.approx(math.log1p(10))

    def test_bin_0_to_5(self):
        r = extract_detection_features(self._row(3))
        assert r["pattern_count_bin"] == 0

    def test_bin_5_to_20(self):
        r = extract_detection_features(self._row(10))
        assert r["pattern_count_bin"] == 1

    def test_bin_20_to_100(self):
        r = extract_detection_features(self._row(50))
        assert r["pattern_count_bin"] == 2

    def test_bin_100_to_1k(self):
        r = extract_detection_features(self._row(500))
        assert r["pattern_count_bin"] == 3

    def test_bin_1k_plus(self):
        r = extract_detection_features(self._row(5000))
        assert r["pattern_count_bin"] == 4

    def test_is_mass_detection_false(self):
        r = extract_detection_features(self._row(9999))
        assert r["is_mass_detection"] == 0

    def test_is_mass_detection_true(self):
        r = extract_detection_features(self._row(10001))
        assert r["is_mass_detection"] == 1

    def test_is_extreme_detection_false(self):
        r = extract_detection_features(self._row(99999))
        assert r["is_extreme_detection"] == 0

    def test_is_extreme_detection_true(self):
        r = extract_detection_features(self._row(100001))
        assert r["is_extreme_detection"] == 1

    def test_pii_type_ratio_ssn_dominant(self):
        r = extract_detection_features(self._row(100, ssn_count=90, phone_count=5, email_count=5))
        # ssn / (ssn+phone+email+1)
        expected = 90 / (90 + 5 + 5 + 1)
        assert r["pii_type_ratio"] == pytest.approx(expected)

    def test_pii_type_ratio_all_zero(self):
        r = extract_detection_features(self._row(0, 0, 0, 0))
        # 0 / (0+0+0+1) = 0
        assert r["pii_type_ratio"] == pytest.approx(0.0)

    def test_missing_pattern_count(self):
        row = pd.Series({"ssn_count": 5})
        r = extract_detection_features(row)
        assert r["pattern_count_log1p"] == pytest.approx(0.0)

    def test_keys_present(self):
        r = extract_detection_features(self._row())
        assert {
            "pattern_count_log1p", "pattern_count_bin",
            "is_mass_detection", "is_extreme_detection", "pii_type_ratio"
        } <= r.keys()


# ─────────────────────────────────────────────────────────────────────────────
# extract_datetime_features
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractDatetimeFeatures:
    def test_hour_extracted(self):
        ts = pd.Timestamp("2025-07-15 14:30:00")
        r = extract_datetime_features(ts)
        assert r["created_hour"] == 14

    def test_weekday_tuesday(self):
        ts = pd.Timestamp("2025-07-15")  # 화요일 = 1
        r = extract_datetime_features(ts)
        assert r["created_weekday"] == 1

    def test_is_weekend_weekday(self):
        ts = pd.Timestamp("2025-07-15")  # 화요일
        r = extract_datetime_features(ts)
        assert r["is_weekend"] == 0

    def test_is_weekend_saturday(self):
        ts = pd.Timestamp("2025-07-19")  # 토요일
        r = extract_datetime_features(ts)
        assert r["is_weekend"] == 1

    def test_is_weekend_sunday(self):
        ts = pd.Timestamp("2025-07-20")  # 일요일
        r = extract_datetime_features(ts)
        assert r["is_weekend"] == 1

    def test_month_extracted(self):
        ts = pd.Timestamp("2025-07-15")
        r = extract_datetime_features(ts)
        assert r["created_month"] == 7

    def test_nat_returns_minus_one(self):
        r = extract_datetime_features(pd.NaT)
        assert r["created_hour"] == -1
        assert r["created_weekday"] == -1
        assert r["is_weekend"] == -1
        assert r["created_month"] == -1

    def test_none_returns_minus_one(self):
        r = extract_datetime_features(None)
        assert r["created_hour"] == -1

    def test_keys_present(self):
        r = extract_datetime_features(pd.Timestamp("2025-01-01"))
        assert {"created_hour", "created_weekday", "is_weekend", "created_month"} <= r.keys()


# ─────────────────────────────────────────────────────────────────────────────
# build_meta_features
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMetaFeatures:
    def _make_df(self):
        return pd.DataFrame({
            "file_name": ["access.log.1", "data.xlsx"],
            "pattern_count": [50, 3],
            "ssn_count": [10, 1],
            "phone_count": [5, 0],
            "email_count": [5, 0],
            "file_created_at": [
                pd.Timestamp("2025-07-15 14:00:00"),
                pd.Timestamp("2025-07-19 09:00:00"),
            ],
        })

    def test_returns_dataframe(self):
        df = self._make_df()
        result = build_meta_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self):
        df = self._make_df()
        result = build_meta_features(df)
        assert len(result) == len(df)

    def test_fname_features_included(self):
        df = self._make_df()
        result = build_meta_features(df)
        assert "fname_has_date" in result.columns

    def test_detection_features_included(self):
        df = self._make_df()
        result = build_meta_features(df)
        assert "pattern_count_log1p" in result.columns
        assert "is_mass_detection" in result.columns

    def test_datetime_features_included(self):
        df = self._make_df()
        result = build_meta_features(df)
        assert "created_hour" in result.columns
        assert "is_weekend" in result.columns

    def test_missing_columns_graceful(self):
        df = pd.DataFrame({"file_name": ["test.log"]})
        result = build_meta_features(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_original_df_not_modified(self):
        df = self._make_df()
        original_cols = list(df.columns)
        build_meta_features(df)
        assert list(df.columns) == original_cols

    def test_rotation_num_value(self):
        # is_log_file은 path_features.py 단일 소스 관리 (meta_features에서 제거됨)
        # 대신 fname_has_rotation_num으로 로그 로테이션 파일 판별 가능
        df = self._make_df()
        result = build_meta_features(df)
        assert result["fname_has_rotation_num"].iloc[0] == 1   # access.log.1 → rotation
        assert result["fname_has_rotation_num"].iloc[1] == 0   # data.xlsx → no rotation


# ─────────────────────────────────────────────────────────────────────────────
# compute_file_aggregates_label
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFileAggregatesLabel:
    def _make_df(self):
        return pd.DataFrame({
            "pk_file": ["pk1", "pk1", "pk2", "pk2", "pk2"],
            "ssn_count": [1, 0, 0, 0, 0],
            "phone_count": [0, 1, 0, 0, 0],
            "email_count": [0, 0, 1, 1, 0],
        })

    def test_returns_dataframe(self):
        df = self._make_df()
        agg = compute_file_aggregates_label(df)
        assert isinstance(agg, pd.DataFrame)

    def test_pk_file_index_unique(self):
        df = self._make_df()
        agg = compute_file_aggregates_label(df)
        assert agg["pk_file"].nunique() == len(agg)

    def test_file_event_count_pk1(self):
        df = self._make_df()
        agg = compute_file_aggregates_label(df)
        row = agg[agg["pk_file"] == "pk1"].iloc[0]
        assert row["file_event_count"] == 2

    def test_file_event_count_pk2(self):
        df = self._make_df()
        agg = compute_file_aggregates_label(df)
        row = agg[agg["pk_file"] == "pk2"].iloc[0]
        assert row["file_event_count"] == 3

    def test_file_pii_diversity_pk1(self):
        # pk1 has ssn (row0) and phone (row1) → diversity = 2
        df = self._make_df()
        agg = compute_file_aggregates_label(df)
        row = agg[agg["pk_file"] == "pk1"].iloc[0]
        assert row["file_pii_diversity"] == 2

    def test_file_pii_diversity_pk2(self):
        # pk2 has email in row2 and row3 → 1 type
        df = self._make_df()
        agg = compute_file_aggregates_label(df)
        row = agg[agg["pk_file"] == "pk2"].iloc[0]
        assert row["file_pii_diversity"] == 1

    def test_missing_pk_file_raises(self):
        df = pd.DataFrame({"ssn_count": [1]})
        with pytest.raises((KeyError, ValueError)):
            compute_file_aggregates_label(df)

    def test_missing_pii_columns_graceful(self):
        df = pd.DataFrame({"pk_file": ["pk1", "pk1"]})
        agg = compute_file_aggregates_label(df)
        assert "file_event_count" in agg.columns
        assert agg.loc[agg["pk_file"] == "pk1", "file_event_count"].iloc[0] == 2


# ─────────────────────────────────────────────────────────────────────────────
# merge_file_aggregates_label
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeFileAggregatesLabel:
    def test_merges_correctly(self):
        df = pd.DataFrame({
            "pk_file": ["pk1", "pk1", "pk2"],
            "value": [1, 2, 3],
        })
        agg = pd.DataFrame({
            "pk_file": ["pk1", "pk2"],
            "file_event_count": [2, 1],
            "file_pii_diversity": [1, 0],
        })
        result = merge_file_aggregates_label(df, agg)
        assert "file_event_count" in result.columns
        assert result[result["pk_file"] == "pk1"]["file_event_count"].iloc[0] == 2

    def test_left_join_preserves_rows(self):
        df = pd.DataFrame({"pk_file": ["pk1", "pk2", "pk3"]})
        agg = pd.DataFrame({
            "pk_file": ["pk1"],
            "file_event_count": [5],
            "file_pii_diversity": [1],
        })
        result = merge_file_aggregates_label(df, agg)
        assert len(result) == 3

    def test_unmatched_rows_get_nan(self):
        df = pd.DataFrame({"pk_file": ["pk1", "pk_unknown"]})
        agg = pd.DataFrame({
            "pk_file": ["pk1"],
            "file_event_count": [3],
            "file_pii_diversity": [2],
        })
        result = merge_file_aggregates_label(df, agg)
        assert pd.isna(result.loc[result["pk_file"] == "pk_unknown", "file_event_count"].iloc[0])
