"""
Phase 1 — RED Phase Tests: S1 Parser (Architecture.md §5)

테스트 대상:
    src/data/s1_parser.py  (구현 전 — 모두 실패해야 함)

테스트 범위:
    1. parse_context_field — 1차 마스킹 패턴(*{3,}) 기반 파싱
    2. parse_context_field — 2차 masked_hit 앵커 폴백
    3. parse_context_field — 3차 단일 이벤트 폴백 (parse_status=FALLBACK_SINGLE_EVENT)
    4. infer_pii_type — email/phone/rrn/unknown 재추론
    5. pk_event 유일성 — 동일 pk_file 내 이벤트 pk_event 중복 없음
    6. 필수 컬럼 누락 행 → silver_quarantine (quarantine_reason 포함)
    7. parse_success_rate >= 0.98 (정상 입력 기준)
    8. schema_registry rename_map — 컬럼명 자동 매핑
"""

import hashlib
import pytest
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 구현 전이므로 import 실패는 expected ──────────────────────────────────────
from src.data.s1_parser import (
    parse_context_field,
    infer_pii_type,
    make_pk_file,
    make_pk_event,
    compute_parse_success_rate,
    apply_schema_registry,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def pk_file_sample():
    return make_pk_file("server01", "10.0.0.1", "/data/report.csv")


@pytest.fixture
def sample_raw_with_masking():
    """마스킹 패턴(*{3,})이 명확하게 포함된 컨텍스트 원문"""
    return "email: user****@example.com and another abc@xyz*****.net"


@pytest.fixture
def sample_raw_no_masking_with_anchor():
    """마스킹 패턴 없음 — masked_hit 앵커로 탐색해야 함"""
    return "contact info for john.doe@acme.com stored in db"


@pytest.fixture
def sample_raw_no_masking_no_anchor():
    """마스킹도 없고 앵커도 없음 — 3차 폴백"""
    return "plain text without any detectable masking pattern here"


@pytest.fixture
def sample_schema_registry():
    return {
        "rename_map": {
            "dfile_computername": "server_name",
            "dfile_agentip": "agent_ip",
            "dfile_filedirectedpath": "file_path",
            "dfile_inspectedcontent": "masked_hit",
            "dfile_inspectcontentwithcontext": "full_context_raw",
        },
        "required_columns": ["server_name", "agent_ip", "file_path", "full_context_raw"],
        "on_missing_required": "quarantine",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: 1차 마스킹 패턴(*{3,}) 기반 파싱
# ─────────────────────────────────────────────────────────────────────────────

class TestParseByMaskingPattern:
    def test_returns_list_of_dicts(self, pk_file_sample, sample_raw_with_masking):
        results = parse_context_field(sample_raw_with_masking, pk_file_sample)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_each_event_has_required_keys(self, pk_file_sample, sample_raw_with_masking):
        results = parse_context_field(sample_raw_with_masking, pk_file_sample)
        required_keys = {"pk_event", "pk_file", "event_index",
                         "left_ctx", "masked_pattern", "right_ctx",
                         "full_context", "parse_status"}
        for event in results:
            assert required_keys.issubset(event.keys()), (
                f"Missing keys: {required_keys - event.keys()}"
            )

    def test_parse_status_is_ok(self, pk_file_sample, sample_raw_with_masking):
        results = parse_context_field(sample_raw_with_masking, pk_file_sample)
        for event in results:
            assert event["parse_status"] == "OK"

    def test_masked_pattern_contains_asterisks(self, pk_file_sample, sample_raw_with_masking):
        results = parse_context_field(sample_raw_with_masking, pk_file_sample)
        for event in results:
            assert "*" in event["masked_pattern"], (
                f"masked_pattern should contain '*': {event['masked_pattern']}"
            )

    def test_pk_file_matches_input(self, pk_file_sample, sample_raw_with_masking):
        results = parse_context_field(sample_raw_with_masking, pk_file_sample)
        for event in results:
            assert event["pk_file"] == pk_file_sample

    def test_event_index_is_sequential(self, pk_file_sample, sample_raw_with_masking):
        results = parse_context_field(sample_raw_with_masking, pk_file_sample)
        for i, event in enumerate(results):
            assert event["event_index"] == i

    def test_empty_input_returns_empty_list(self, pk_file_sample):
        assert parse_context_field("", pk_file_sample) == []

    def test_none_input_returns_empty_list(self, pk_file_sample):
        assert parse_context_field(None, pk_file_sample) == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: 2차 masked_hit 앵커 폴백
# ─────────────────────────────────────────────────────────────────────────────

class TestParseByAnchorFallback:
    def test_anchor_fallback_when_no_masking_pattern(
        self, pk_file_sample, sample_raw_no_masking_with_anchor
    ):
        masked_hit = "john.doe@acme.com"
        results = parse_context_field(
            sample_raw_no_masking_with_anchor, pk_file_sample, masked_hit=masked_hit
        )
        assert len(results) == 1
        assert results[0]["parse_status"] == "FALLBACK_ANCHOR"

    def test_anchor_fallback_masked_pattern_equals_masked_hit(
        self, pk_file_sample, sample_raw_no_masking_with_anchor
    ):
        masked_hit = "john.doe@acme.com"
        results = parse_context_field(
            sample_raw_no_masking_with_anchor, pk_file_sample, masked_hit=masked_hit
        )
        assert results[0]["masked_pattern"] == masked_hit

    def test_anchor_fallback_context_contains_masked_hit(
        self, pk_file_sample, sample_raw_no_masking_with_anchor
    ):
        masked_hit = "john.doe@acme.com"
        results = parse_context_field(
            sample_raw_no_masking_with_anchor, pk_file_sample, masked_hit=masked_hit
        )
        assert masked_hit in results[0]["full_context"]

    def test_anchor_fallback_window_60_chars(
        self, pk_file_sample
    ):
        """masked_hit 앞뒤로 최대 60자 컨텍스트 윈도우"""
        prefix = "A" * 100
        suffix = "B" * 100
        # asterisk 없는 masked_hit — 1차 파서 미작동, 2차 앵커 폴백 발동
        masked_hit = "john.doe@example.com"
        raw = prefix + masked_hit + suffix
        results = parse_context_field(raw, pk_file_sample, masked_hit=masked_hit)
        assert len(results) == 1
        assert results[0]["parse_status"] == "FALLBACK_ANCHOR"
        # 윈도우는 최대 60자 양쪽
        assert len(results[0]["left_ctx"]) <= 60
        assert len(results[0]["right_ctx"]) <= 60


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: 3차 단일 이벤트 폴백
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleEventFallback:
    def test_fallback_single_event_when_no_pattern_no_anchor(
        self, pk_file_sample, sample_raw_no_masking_no_anchor
    ):
        # masked_hit 없거나 텍스트 안에서 찾을 수 없는 경우
        results = parse_context_field(
            sample_raw_no_masking_no_anchor, pk_file_sample, masked_hit="NOT_IN_TEXT"
        )
        assert len(results) == 1
        assert results[0]["parse_status"] == "FALLBACK_SINGLE_EVENT"

    def test_fallback_single_event_event_index_zero(
        self, pk_file_sample, sample_raw_no_masking_no_anchor
    ):
        results = parse_context_field(
            sample_raw_no_masking_no_anchor, pk_file_sample, masked_hit="NOT_IN_TEXT"
        )
        assert results[0]["event_index"] == 0

    def test_fallback_single_event_no_masked_hit(
        self, pk_file_sample, sample_raw_no_masking_no_anchor
    ):
        # masked_hit 없을 때도 3차 폴백 동작
        results = parse_context_field(sample_raw_no_masking_no_anchor, pk_file_sample)
        assert len(results) == 1
        assert results[0]["parse_status"] == "FALLBACK_SINGLE_EVENT"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: PII 유형 재추론 (infer_pii_type)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferPiiType:
    def test_email_detected_by_at_sign(self):
        assert infer_pii_type("****@example.com", "user@example.com context") == "email"

    def test_phone_detected_by_pattern(self):
        result = infer_pii_type("010-****-1234", "010-****-1234 connected")
        assert result == "phone"

    def test_rrn_detected_by_pattern(self):
        result = infer_pii_type("890101-*******", "birth: 890101-*******")
        assert result == "rrn"

    def test_unknown_when_no_pattern(self):
        result = infer_pii_type("plain", "no special pattern here")
        assert result == "unknown"

    def test_email_takes_priority_over_phone(self):
        # @가 있으면 이메일 우선 (Architecture.md §5.3 순서)
        result = infer_pii_type("010****@test.com", "010****@test.com info")
        assert result == "email"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: pk_event 유일성
# ─────────────────────────────────────────────────────────────────────────────

class TestPkUniqueness:
    def test_pk_file_is_sha256_hex_64chars(self):
        pk = make_pk_file("srv01", "10.0.0.1", "/data/file.csv")
        assert isinstance(pk, str)
        assert len(pk) == 64
        # 유효한 hex 문자열
        int(pk, 16)

    def test_pk_file_deterministic(self):
        pk1 = make_pk_file("srv01", "10.0.0.1", "/data/file.csv")
        pk2 = make_pk_file("srv01", "10.0.0.1", "/data/file.csv")
        assert pk1 == pk2

    def test_pk_file_different_for_different_inputs(self):
        pk1 = make_pk_file("srv01", "10.0.0.1", "/data/file.csv")
        pk2 = make_pk_file("srv02", "10.0.0.2", "/data/other.csv")
        assert pk1 != pk2

    def test_pk_event_is_sha256_hex_64chars(self):
        pk_file = make_pk_file("srv01", "10.0.0.1", "/data/file.csv")
        pk_event = make_pk_event(pk_file, event_index=0)
        assert len(pk_event) == 64
        int(pk_event, 16)

    def test_pk_event_unique_per_event_index(self):
        pk_file = make_pk_file("srv01", "10.0.0.1", "/data/file.csv")
        pk_events = [make_pk_event(pk_file, i) for i in range(10)]
        assert len(pk_events) == len(set(pk_events)), "pk_event must be unique per event_index"

    def test_pk_event_unique_across_multiple_parses(self):
        """parse_context_field 결과의 pk_event는 동일 pk_file 내에서 중복 없어야 함"""
        pk_file = make_pk_file("srv01", "10.0.0.1", "/data/file.csv")
        raw = (
            "email: user****@example.com and phone: 010-****-1234\n"
            "another: abc****@test.com"
        )
        results = parse_context_field(raw, pk_file)
        pk_events = [r["pk_event"] for r in results]
        assert len(pk_events) == len(set(pk_events)), (
            "pk_event duplicates detected within same pk_file"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: 필수 컬럼 누락 → Quarantine
# ─────────────────────────────────────────────────────────────────────────────

class TestQuarantineOnMissingRequired:
    def test_missing_required_column_goes_to_quarantine(self, sample_schema_registry):
        # full_context_raw 누락된 행 포함
        df = pd.DataFrame({
            "dfile_computername": ["srv01", "srv02"],
            "dfile_agentip": ["10.0.0.1", "10.0.0.2"],
            "dfile_filedirectedpath": ["/a/b.csv", "/c/d.csv"],
            # full_context_raw (dfile_inspectcontentwithcontext) 누락
        })

        result_df, quarantine_df = apply_schema_registry(df, sample_schema_registry)

        # 모든 행이 quarantine으로
        assert len(quarantine_df) == 2
        assert "quarantine_reason" in quarantine_df.columns

    def test_quarantine_reason_is_schema_mismatch(self, sample_schema_registry):
        df = pd.DataFrame({
            "dfile_computername": ["srv01"],
            "dfile_agentip": ["10.0.0.1"],
            # file_path 및 full_context_raw 누락
        })

        _, quarantine_df = apply_schema_registry(df, sample_schema_registry)

        assert len(quarantine_df) > 0
        assert quarantine_df.iloc[0]["quarantine_reason"] in (
            "MISSING_REQUIRED_FIELD", "SCHEMA_MISMATCH"
        )

    def test_valid_rows_pass_through(self, sample_schema_registry):
        df = pd.DataFrame({
            "dfile_computername": ["srv01", "srv02"],
            "dfile_agentip": ["10.0.0.1", "10.0.0.2"],
            "dfile_filedirectedpath": ["/a/b.csv", "/c/d.csv"],
            "dfile_inspectcontentwithcontext": [
                "context with user****@example.com",
                "another text with abc****@test.net",
            ],
        })

        result_df, quarantine_df = apply_schema_registry(df, sample_schema_registry)

        assert len(result_df) == 2
        assert len(quarantine_df) == 0

    def test_rename_map_applied_to_result(self, sample_schema_registry):
        df = pd.DataFrame({
            "dfile_computername": ["srv01"],
            "dfile_agentip": ["10.0.0.1"],
            "dfile_filedirectedpath": ["/a/b.csv"],
            "dfile_inspectcontentwithcontext": ["some context text"],
        })

        result_df, _ = apply_schema_registry(df, sample_schema_registry)

        assert "server_name" in result_df.columns
        assert "agent_ip" in result_df.columns
        assert "file_path" in result_df.columns
        assert "full_context_raw" in result_df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: parse_success_rate >= 0.98
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSuccessRate:
    def test_success_rate_high_on_normal_input(self):
        """정상 입력에서 parse_success_rate >= 0.98"""
        original_row_count = 100
        # 정상 이벤트: 100행 → 각 행에서 1건 파싱 성공 → 100 events
        events_data = [
            {"pk_event": f"evt_{i:04d}", "parse_status": "OK"}
            for i in range(100)
        ]
        events_df = pd.DataFrame(events_data)
        rate = compute_parse_success_rate(events_df, original_row_count)
        assert rate >= 0.98, f"Expected >= 0.98, got {rate}"

    def test_success_rate_below_threshold_on_bad_input(self):
        """파싱 실패가 많으면 rate < 0.98"""
        original_row_count = 100
        # 50건만 파싱됨
        events_data = [
            {"pk_event": f"evt_{i:04d}", "parse_status": "OK"}
            for i in range(50)
        ]
        events_df = pd.DataFrame(events_data)
        rate = compute_parse_success_rate(events_df, original_row_count)
        assert rate < 0.98

    def test_success_rate_returns_float(self):
        events_df = pd.DataFrame([{"pk_event": "abc", "parse_status": "OK"}])
        rate = compute_parse_success_rate(events_df, original_row_count=10)
        assert isinstance(rate, float)

    def test_success_rate_zero_when_no_events(self):
        events_df = pd.DataFrame(columns=["pk_event", "parse_status"])
        rate = compute_parse_success_rate(events_df, original_row_count=10)
        assert rate == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Schema Registry rename_map 컬럼명 자동 매핑
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaRegistryRenameMap:
    def test_original_columns_renamed(self, sample_schema_registry):
        df = pd.DataFrame({
            "dfile_computername": ["srv01"],
            "dfile_agentip": ["10.0.0.1"],
            "dfile_filedirectedpath": ["/path/to/file.csv"],
            "dfile_inspectcontentwithcontext": ["context text user****@example.com"],
        })

        result_df, _ = apply_schema_registry(df, sample_schema_registry)

        # 원본 컬럼명은 사라지고 target 이름으로 변경
        assert "dfile_computername" not in result_df.columns
        assert "dfile_agentip" not in result_df.columns
        assert "server_name" in result_df.columns
        assert "agent_ip" in result_df.columns

    def test_extra_columns_preserved(self, sample_schema_registry):
        """rename_map에 없는 컬럼은 그대로 보존"""
        df = pd.DataFrame({
            "dfile_computername": ["srv01"],
            "dfile_agentip": ["10.0.0.1"],
            "dfile_filedirectedpath": ["/path/to/file.csv"],
            "dfile_inspectcontentwithcontext": ["context text"],
            "extra_column": ["extra_value"],  # rename_map에 없는 컬럼
        })

        result_df, _ = apply_schema_registry(df, sample_schema_registry)

        assert "extra_column" in result_df.columns
        assert result_df.iloc[0]["extra_column"] == "extra_value"

    def test_rename_preserves_values(self, sample_schema_registry):
        df = pd.DataFrame({
            "dfile_computername": ["my-server"],
            "dfile_agentip": ["192.168.1.1"],
            "dfile_filedirectedpath": ["/data/report.csv"],
            "dfile_inspectcontentwithcontext": ["some context"],
        })

        result_df, _ = apply_schema_registry(df, sample_schema_registry)

        assert result_df.iloc[0]["server_name"] == "my-server"
        assert result_df.iloc[0]["agent_ip"] == "192.168.1.1"
        assert result_df.iloc[0]["file_path"] == "/data/report.csv"
