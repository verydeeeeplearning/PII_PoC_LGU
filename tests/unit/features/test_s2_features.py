"""
Phase 2 — RED Phase Tests: S2 Feature Engineering (Architecture.md §6)

테스트 대상 (구현 전 — 모두 실패해야 함):
    src/features/text_prep.py
    src/features/keyword_flags.py
    src/features/path_features.py
    src/features/synthetic.py
    src/features/file_agg.py
    src/features/schema_validator.py
    src/features/tabular_features.py (기존 모듈 확장)

테스트 범위:
    2.1 make_raw_text — 고엔트로피 placeholder 치환
    2.2 make_shape_text — 숫자/문자/구분자 형태화
    2.3 make_path_text — 경로 토큰화
    2.4 keyword_flags — 25+개 has_* 플래그
    2.5 extract_path_features — 경로 구조 피처 8개
    2.6 extract_tabular_features — log1p / is_mass / is_extreme
    2.7 build_synthetic_features — Tier0=합성변수 없음, Tier1=있음
    2.8 compute_file_aggregates — 누수 없이 train agg
    2.9 validate_feature_schema — 차원 불일치 시 False
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 구현 전이므로 import 실패는 expected ──────────────────────────────────────
from src.features.text_prep import (
    make_raw_text,
    make_shape_text,
    make_path_text,
)
from src.features.keyword_flags import compute_keyword_flags
from src.features.path_features import extract_path_features
from src.features.synthetic import build_synthetic_features
from src.features.file_agg import (
    compute_file_aggregates,
    merge_file_aggregates,
)
from src.features.schema_validator import (
    save_feature_schema,
    validate_feature_schema,
)

# 기존 모듈 확장 함수 (tabular_features.py)
from src.features.tabular_features import extract_tabular_features


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_silver_df():
    """S1 파서 출력 형태의 silver_detections DataFrame"""
    return pd.DataFrame({
        "pk_event": [f"evt_{i:04d}" for i in range(6)],
        "pk_file": ["file_aaa"] * 3 + ["file_bbb"] * 3,
        "file_path": [
            "/var/lib/docker/overlay2/abc123/file.txt",
            "/var/log/hadoop/datanode.log",
            "/data/customer/crm_export.xlsx",
            "/usr/share/doc/redhat/license.txt",
            "/tmp/test_sample.csv",
            "/home/user/report_20240101.csv",
        ],
        "full_context_raw": [
            "email: user****@example.com stored in db",
            "hadoop-cmf-hdfs DATANODE server01.log entry",
            "customer john.doe@acme.com contact info",
            "redhat copyright (c) 2024 redhat.com",
            "test dummy@test.com sample data",
            "xpiryDate=1704067200 duration:86400",
        ],
        "masked_hit": [
            "user****@example.com",
            "DATANODE****",
            "john.doe@acme.com",
            "****@redhat.com",
            "dummy@test.com",
            "xpiryDate=17040672",
        ],
        "masked_pattern": [
            "user****@example.com",
            "DATANODE****",
            "john.doe@acme.com",
            "****@redhat.com",
            "dummy@test.com",
            "xpiryDate=1704067200",
        ],
        "inspect_count": [500, 233498, 10, 150000, 200, 5000],
        "pii_type_inferred": ["email", "unknown", "email", "email", "email", "unknown"],
        "email_domain": ["example.com", None, "acme.com", "redhat.com", "test.com", None],
        "parse_status": ["OK"] * 6,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.1: make_raw_text — 고엔트로피 placeholder 치환
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeRawText:
    def test_num10_placeholder(self):
        """10자리 숫자 → <NUM10>"""
        result = make_raw_text("timestamp 1704067200 found")
        assert "<NUM10>" in result
        assert "1704067200" not in result

    def test_num13_placeholder(self):
        """13자리 숫자 → <NUM13>"""
        result = make_raw_text("ts=1704067200000 ms")
        assert "<NUM13>" in result

    def test_date8_placeholder(self):
        """8자리 날짜형 숫자 → <DATE8>"""
        result = make_raw_text("date=20240101 log")
        assert "<DATE8>" in result

    def test_hash_placeholder(self):
        """32자+ 헥사 문자열 → <HASH>"""
        md5 = "a94a8fe5ccde17c5d936f47bf5f3e348"
        result = make_raw_text(f"hash={md5} value")
        assert "<HASH>" in result
        assert md5 not in result

    def test_hex_placeholder(self):
        """0x로 시작하는 헥사 → <HEX>"""
        result = make_raw_text("addr=0x7ffde1234abc offset")
        assert "<HEX>" in result

    def test_mask_placeholder(self):
        """***{3+} → <MASK>"""
        result = make_raw_text("email: user****@example.com")
        assert "<MASK>" in result

    def test_lowercase_applied(self):
        """소문자 변환 확인"""
        result = make_raw_text("User EMAIL Context")
        assert result == result.lower()

    def test_keyword_preserved(self):
        """키워드(도메인, 경로 토큰)는 그대로 유지"""
        result = make_raw_text("contact at lguplus.co.kr domain")
        assert "lguplus" in result

    def test_empty_input(self):
        assert make_raw_text("") == ""

    def test_none_input(self):
        assert make_raw_text(None) == ""

    def test_longer_num_before_shorter(self):
        """13자리는 <NUM13>, 10자리는 <NUM10> — 겹침 없음"""
        result = make_raw_text("a=1704067200000 b=1704067200")
        assert "<NUM13>" in result
        assert "<NUM10>" in result


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.2: make_shape_text — 형태 추상화
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeShapeText:
    def test_digits_to_zero(self):
        """숫자 → '0'"""
        result = make_shape_text("170603")
        assert result == "000000"

    def test_ascii_alpha_to_a(self):
        """영문 → 'a'"""
        result = make_shape_text("abc")
        assert result == "aaa"

    def test_hangul_to_ga(self):
        """한글 → '가'"""
        result = make_shape_text("홍길동")
        assert result == "가가가"

    def test_special_chars_preserved(self):
        """특수문자 유지: @, -, ., *"""
        result = make_shape_text("****@lgu.co.kr")
        assert "****" in result
        assert "@" in result
        assert "." in result

    def test_architecture_example(self):
        """Architecture.md §6.2 예시: xpiryDate=170603*****"""
        result = make_shape_text("xpiryDate=170603*****")
        assert result == "aaaaaaaaa=000000*****"

    def test_architecture_example_email(self):
        """Architecture.md §6.2 예시: ****@bdp.lguplus.co.kr"""
        result = make_shape_text("****@bdp.lguplus.co.kr")
        assert result == "****@aaa.aaaaaaa.aa.aa"

    def test_empty_input(self):
        assert make_shape_text("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.3: make_path_text — 경로 토큰화
# ─────────────────────────────────────────────────────────────────────────────

class TestMakePathText:
    def test_basic_path_tokenization(self):
        """/var/log/hadoop/abc → var log hadoop abc"""
        result = make_path_text("/var/log/hadoop/abc")
        assert "var" in result
        assert "log" in result
        assert "hadoop" in result

    def test_empty_tokens_removed(self):
        """중복 슬래시/연속 구분자 처리"""
        result = make_path_text("//var//log//")
        assert "//" not in result

    def test_dot_separator(self):
        """점 구분자로 분리: file.log.1 → file log 1"""
        result = make_path_text("file.log.1")
        tokens = result.split()
        assert "file" in tokens
        assert "log" in tokens

    def test_underscore_separator(self):
        """언더스코어 분리: hadoop_cmf_hdfs → hadoop cmf hdfs"""
        result = make_path_text("hadoop_cmf_hdfs")
        assert "hadoop" in result
        assert "cmf" in result

    def test_lowercase_output(self):
        """/VAR/LOG → var log"""
        result = make_path_text("/VAR/LOG/File.TXT")
        assert result == result.lower()

    def test_empty_input(self):
        assert make_path_text("") == ""

    def test_none_input(self):
        assert make_path_text(None) == ""


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.4: compute_keyword_flags — 25+개 has_* 플래그
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordFlags:
    def test_returns_dataframe(self):
        texts = pd.Series(["timestamp 170603", "bytes 45"])
        result = compute_keyword_flags(texts)
        assert isinstance(result, pd.DataFrame)

    def test_at_least_25_flags(self):
        """25개 이상 플래그 생성"""
        texts = pd.Series(["some text"])
        result = compute_keyword_flags(texts)
        assert result.shape[1] >= 25

    def test_has_timestamp_kw_detected(self):
        texts = pd.Series(["xpiryDate=170603 duration:86400"])
        result = compute_keyword_flags(texts)
        assert "has_timestamp_kw" in result.columns
        assert result["has_timestamp_kw"].iloc[0] == 1

    def test_has_byte_kw_detected(self):
        texts = pd.Series(["size: 45 bytes 141022"])
        result = compute_keyword_flags(texts)
        assert result["has_byte_kw"].iloc[0] == 1

    def test_has_os_copyright_kw_detected(self):
        texts = pd.Series(["copyright redhat.com apache.org"])
        result = compute_keyword_flags(texts)
        assert result["has_os_copyright_kw"].iloc[0] == 1

    def test_has_dev_kw_detected(self):
        texts = pd.Series(["test dummy sample example data"])
        result = compute_keyword_flags(texts)
        assert result["has_dev_kw"].iloc[0] == 1

    def test_has_domain_kw_detected(self):
        texts = pd.Series(["email from lguplus internal network"])
        result = compute_keyword_flags(texts)
        assert result["has_domain_kw"].iloc[0] == 1

    def test_negative_case_all_zero(self):
        """관련 키워드 없는 텍스트 → 플래그 0"""
        texts = pd.Series(["plain normal text without any special keywords"])
        result = compute_keyword_flags(texts)
        # 모든 플래그가 0인 행이 있어야 함
        row = result.iloc[0]
        # has_dev_kw 등은 0이어야 함
        assert row.get("has_byte_kw", 0) == 0
        assert row.get("has_os_copyright_kw", 0) == 0

    def test_all_flags_are_binary(self):
        """모든 플래그는 0 또는 1"""
        texts = pd.Series(["timestamp bytes redhat lguplus test", "plain text"])
        result = compute_keyword_flags(texts)
        assert result.isin([0, 1]).all().all()

    def test_same_length_as_input(self):
        texts = pd.Series(["a", "b", "c"])
        result = compute_keyword_flags(texts)
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.5: extract_path_features — 경로 구조 피처
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractPathFeatures:
    def test_returns_dict(self):
        result = extract_path_features("/var/log/app.log")
        assert isinstance(result, dict)

    def test_at_least_8_features(self):
        """Architecture §6.3: 8개 이상 경로 피처"""
        result = extract_path_features("/var/log/app.log")
        assert len(result) >= 8

    def test_is_docker_overlay_detected(self):
        result = extract_path_features("/var/lib/docker/overlay2/abc/file")
        assert result["is_docker_overlay"] == 1

    def test_is_docker_overlay_not_detected(self):
        result = extract_path_features("/home/user/data.csv")
        assert result["is_docker_overlay"] == 0

    def test_is_log_file_detected(self):
        result = extract_path_features("/var/log/app.log")
        assert result["is_log_file"] == 1

    def test_is_log_file_not_detected(self):
        result = extract_path_features("/data/report.xlsx")
        assert result["is_log_file"] == 0

    def test_has_license_path_detected(self):
        result = extract_path_features("/usr/share/doc/redhat/license.txt")
        assert result["has_license_path"] == 1

    def test_is_temp_or_dev_detected(self):
        result = extract_path_features("/tmp/test_file.csv")
        assert result["is_temp_or_dev"] == 1

    def test_has_business_token_detected(self):
        result = extract_path_features("/data/crm/customer_export.xlsx")
        assert result["has_business_token"] == 1

    def test_has_system_token_detected(self):
        result = extract_path_features("/var/log/kerberos/auth.log")
        assert result["has_system_token"] == 1

    def test_path_depth_counted(self):
        result = extract_path_features("/a/b/c/d/file.txt")
        assert result["path_depth"] == 5

    def test_empty_path(self):
        result = extract_path_features("")
        assert isinstance(result, dict)
        assert result["is_docker_overlay"] == 0

    def test_has_date_in_path(self):
        result = extract_path_features("/data/20240101/report.csv")
        assert result["has_date_in_path"] == 1

    def test_all_flag_values_binary(self):
        result = extract_path_features("/var/lib/docker/overlay2/test")
        binary_keys = [k for k, v in result.items() if k != "path_depth" and k != "extension"]
        for k in binary_keys:
            assert result[k] in (0, 1), f"{k}={result[k]} is not binary"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.6: extract_tabular_features — log1p / is_mass / is_extreme
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTabularFeatures:
    def test_returns_dict(self):
        row = {"inspect_count": 100, "pii_type_inferred": "email"}
        result = extract_tabular_features(row)
        assert isinstance(result, dict)

    def test_inspect_count_log1p(self):
        import math
        row = {"inspect_count": 100}
        result = extract_tabular_features(row)
        assert "inspect_count_log1p" in result
        assert abs(result["inspect_count_log1p"] - math.log1p(100)) < 1e-9

    def test_is_mass_detection_positive(self):
        """count > 10,000 → is_mass_detection=1"""
        row = {"inspect_count": 50000}
        result = extract_tabular_features(row)
        assert result["is_mass_detection"] == 1

    def test_is_mass_detection_negative(self):
        row = {"inspect_count": 100}
        result = extract_tabular_features(row)
        assert result["is_mass_detection"] == 0

    def test_is_extreme_detection_positive(self):
        """count > 100,000 → is_extreme_detection=1"""
        row = {"inspect_count": 233498}
        result = extract_tabular_features(row)
        assert result["is_extreme_detection"] == 1

    def test_is_extreme_detection_negative(self):
        row = {"inspect_count": 5000}
        result = extract_tabular_features(row)
        assert result["is_extreme_detection"] == 0

    def test_inspect_count_raw_preserved(self):
        row = {"inspect_count": 42}
        result = extract_tabular_features(row)
        assert result["inspect_count_raw"] == 42

    def test_zero_inspect_count(self):
        row = {"inspect_count": 0}
        result = extract_tabular_features(row)
        assert result["is_mass_detection"] == 0
        assert result["is_extreme_detection"] == 0
        assert result["inspect_count_log1p"] == 0.0

    def test_missing_inspect_count_defaults_zero(self):
        row = {}
        result = extract_tabular_features(row)
        assert result["inspect_count_raw"] == 0
        assert result["is_mass_detection"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.7: build_synthetic_features — Tier 0/1
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSyntheticFeatures:
    @pytest.fixture
    def base_df(self):
        return pd.DataFrame({
            "is_log_file": [1, 0, 1],
            "has_byte_kw": [1, 0, 0],
            "is_docker_overlay": [0, 1, 0],
            "is_mass_detection": [0, 1, 0],
            "is_extreme_detection": [0, 1, 0],
            "has_timestamp_kw": [1, 0, 0],
            "digit_ratio": [0.7, 0.2, 0.1],
            "has_domain_kw": [1, 0, 0],
            "has_license_path": [0, 0, 1],
            "has_os_copyright_kw": [0, 0, 1],
            "is_temp_or_dev": [0, 0, 1],
            "has_dev_kw": [0, 0, 1],
            "has_system_token": [0, 1, 0],
            "has_business_token": [1, 0, 0],
            "pii_type_inferred": ["email", "unknown", "email"],
        })

    def test_tier0_returns_empty_or_no_synth_cols(self, base_df):
        """Tier 0: 합성변수 없음 — 반환 DataFrame은 비어있거나 합성변수 컬럼 없음"""
        result = build_synthetic_features(base_df, tier="off")
        # Tier 0는 합성변수 컬럼이 0개
        assert result.shape[1] == 0 or len(result.columns) == 0

    def test_tier1_returns_synth_cols(self, base_df):
        """Tier 1 SAFE: 최소 1개 이상 합성변수 생성"""
        result = build_synthetic_features(base_df, tier="safe")
        assert result.shape[1] >= 1

    def test_tier1_log_file_and_byte_kw(self, base_df):
        """log_file_AND_byte_kw: is_log_file × has_byte_kw"""
        result = build_synthetic_features(base_df, tier="safe")
        if "log_file_AND_byte_kw" in result.columns:
            expected = base_df["is_log_file"] * base_df["has_byte_kw"]
            pd.testing.assert_series_equal(
                result["log_file_AND_byte_kw"].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
            )

    def test_tier1_docker_and_mass(self, base_df):
        """docker_AND_mass: is_docker_overlay × is_mass_detection"""
        result = build_synthetic_features(base_df, tier="safe")
        if "docker_AND_mass" in result.columns:
            expected = base_df["is_docker_overlay"] * base_df["is_mass_detection"]
            pd.testing.assert_series_equal(
                result["docker_AND_mass"].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
            )

    def test_tier2_has_more_cols_than_tier1(self, base_df):
        """Tier 2 AGGRESSIVE: Tier 1보다 많거나 같은 합성변수"""
        result_safe = build_synthetic_features(base_df, tier="safe")
        result_aggressive = build_synthetic_features(base_df, tier="aggressive")
        assert result_aggressive.shape[1] >= result_safe.shape[1]

    def test_invalid_tier_raises(self, base_df):
        with pytest.raises((ValueError, KeyError)):
            build_synthetic_features(base_df, tier="invalid_tier")

    def test_same_length_as_input(self, base_df):
        result = build_synthetic_features(base_df, tier="safe")
        assert len(result) == len(base_df)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.8: compute_file_aggregates — 누수 없이 train agg
# ─────────────────────────────────────────────────────────────────────────────

class TestFileAgg:
    @pytest.fixture
    def train_df(self):
        return pd.DataFrame({
            "pk_event": [f"evt_{i}" for i in range(6)],
            "pk_file": ["fileA", "fileA", "fileA", "fileB", "fileB", "fileC"],
            "email_domain": ["lguplus.co.kr", "lguplus.co.kr", "example.com",
                             "redhat.com", None, "test.com"],
            "has_timestamp_kw": [1, 0, 0, 0, 1, 0],
            "has_byte_kw": [0, 1, 0, 0, 0, 1],
            "pii_type_inferred": ["email", "email", "email", "email", "unknown", "email"],
        })

    @pytest.fixture
    def test_df(self):
        return pd.DataFrame({
            "pk_event": ["evt_test1", "evt_test2"],
            "pk_file": ["fileA", "fileD"],  # fileD는 학습에 없음
            "email_domain": ["lguplus.co.kr", "new.com"],
            "has_timestamp_kw": [0, 0],
            "has_byte_kw": [0, 0],
            "pii_type_inferred": ["email", "email"],
        })

    def test_returns_dataframe(self, train_df):
        result = compute_file_aggregates(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_pk_file_is_key(self, train_df):
        result = compute_file_aggregates(train_df)
        assert "pk_file" in result.columns

    def test_file_event_count_correct(self, train_df):
        result = compute_file_aggregates(train_df)
        row_a = result[result["pk_file"] == "fileA"].iloc[0]
        assert row_a["file_event_count"] == 3

    def test_file_unique_domains_counted(self, train_df):
        result = compute_file_aggregates(train_df)
        row_a = result[result["pk_file"] == "fileA"].iloc[0]
        assert row_a["file_unique_domains"] == 2  # lguplus.co.kr, example.com

    def test_merge_file_aggregates_left_join(self, train_df, test_df):
        """test fold에 train agg를 join — 학습에 없는 pk_file은 NaN"""
        agg = compute_file_aggregates(train_df)
        merged = merge_file_aggregates(test_df, agg)
        # fileD는 학습 agg에 없으므로 NaN or 0 (fillna 방식에 따라)
        row_d = merged[merged["pk_file"] == "fileD"]
        assert len(row_d) == 1

    def test_no_leakage_agg_from_train_only(self, train_df, test_df):
        """test_df만으로 agg를 계산하면 안 됨 — train_df만 전달 확인"""
        agg = compute_file_aggregates(train_df)
        # agg에 test-only pk_file(fileD)은 없어야 함
        assert "fileD" not in agg["pk_file"].values

    def test_merge_preserves_test_rows(self, train_df, test_df):
        agg = compute_file_aggregates(train_df)
        merged = merge_file_aggregates(test_df, agg)
        assert len(merged) == len(test_df)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2.9: schema_validator — feature_schema.json 생성/검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidator:
    @pytest.fixture
    def sample_X(self, tmp_path):
        """임의의 피처 행렬 (numpy array)"""
        return np.random.rand(10, 50)

    @pytest.fixture
    def sample_feature_names(self):
        return [f"feat_{i}" for i in range(50)]

    def test_save_feature_schema_returns_dict(self, sample_X, sample_feature_names, tmp_path):
        schema = save_feature_schema(sample_X, sample_feature_names,
                                     output_path=str(tmp_path / "schema.json"))
        assert isinstance(schema, dict)

    def test_save_feature_schema_stores_n_features(self, sample_X, sample_feature_names, tmp_path):
        schema = save_feature_schema(sample_X, sample_feature_names,
                                     output_path=str(tmp_path / "schema.json"))
        assert schema["n_features"] == 50

    def test_save_feature_schema_stores_feature_names(
        self, sample_X, sample_feature_names, tmp_path
    ):
        schema = save_feature_schema(sample_X, sample_feature_names,
                                     output_path=str(tmp_path / "schema.json"))
        assert schema["feature_names"] == sample_feature_names

    def test_save_creates_json_file(self, sample_X, sample_feature_names, tmp_path):
        output = tmp_path / "schema.json"
        save_feature_schema(sample_X, sample_feature_names, output_path=str(output))
        assert output.exists()

    def test_validate_returns_true_on_match(self, sample_X, sample_feature_names, tmp_path):
        schema = save_feature_schema(sample_X, sample_feature_names,
                                     output_path=str(tmp_path / "schema.json"))
        assert validate_feature_schema(sample_X, schema) is True

    def test_validate_returns_false_on_dimension_mismatch(
        self, sample_X, sample_feature_names, tmp_path
    ):
        schema = save_feature_schema(sample_X, sample_feature_names,
                                     output_path=str(tmp_path / "schema.json"))
        # 차원 불일치: 49개 컬럼
        X_wrong = np.random.rand(10, 49)
        assert validate_feature_schema(X_wrong, schema) is False

    def test_validate_returns_false_on_nan_schema(self, sample_X):
        """schema가 None이면 False"""
        assert validate_feature_schema(sample_X, None) is False
