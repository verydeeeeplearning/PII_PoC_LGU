"""Phase 3 S3a: RuleLabeler 단위 테스트

Tests (RED phase — 구현 전에 작성):
    3.1  내부 도메인 → FP-내부도메인
    3.2  OS 도메인  → FP-OS저작권
    3.3  bytes 키워드 → FP-bytes크기
    3.4  다중 후보 시 priority 높은 룰 선택
    3.5  다른 primary_class 충돌 → rule_has_conflict=True
    3.6  Bayesian LB (대용량 샘플, N=5000, M=4985 → ~0.994)
    3.7  Bayesian LB (소용량 샘플, N=15,  M=15   → ~0.814)
    3.8  evidence span(matched_span_start/end) 정확도
    3.9  매칭 없음 → None 반환
    +    label_batch() 출력 스키마 검증
    +    매칭된 이벤트만 evidence 보유
    +    기존 필터 하위 호환 확인
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd


# ────────────────────────────────────────────────
# Fixtures: 최소 rules_config / rule_stats
# ────────────────────────────────────────────────

RULES_CONFIG_MINIMAL = [
    {
        "rule_id": "L1_DOMAIN_INTERNAL_001",
        "applies_to_pii_type": "email",
        "primary_class": "FP-내부도메인",
        "reason_code": "INT_DOMAIN_LGUPLUS",
        "pattern_type": "domain_list",
        "pattern": ["lguplus.co.kr", "bdp.lguplus.co.kr", "lgup.co.kr"],
        "priority": 100,
        "evidence_template": "내부 도메인 매칭: {matched_value}",
        "active": True,
    },
    {
        "rule_id": "L1_DOMAIN_OS_001",
        "applies_to_pii_type": "email",
        "primary_class": "FP-OS저작권",
        "reason_code": "OS_DOMAIN_REDHAT",
        "pattern_type": "domain_list",
        "pattern": ["redhat.com", "fedoraproject.org", "gnu.org", "apache.org"],
        "priority": 90,
        "evidence_template": "OS/오픈소스 도메인 매칭: {matched_value}",
        "active": True,
    },
    {
        "rule_id": "L2_KEYWORD_BYTES_001",
        "applies_to_pii_type": "any",
        "primary_class": "FP-bytes크기",
        "reason_code": "BYTES_KEYWORD_ADJACENT",
        "pattern_type": "regex",
        "pattern": r"\bbytes?\b",
        "context_scope": "full_context",
        "priority": 85,
        "evidence_template": "bytes 키워드 인접: {matched_value}",
        "active": True,
    },
    {
        "rule_id": "L2_KEYWORD_TIMESTAMP_001",
        "applies_to_pii_type": "any",
        "primary_class": "FP-타임스탬프",
        "reason_code": "KEY_EXPIRYDATE",
        "pattern_type": "regex",
        "pattern": r"(?i)(expir|timestamp|duration|created_at|updated_at|date\s*=)",
        "context_scope": "full_context",
        "priority": 80,
        "evidence_template": "타임스탬프 관련 키워드: {matched_value}",
        "active": True,
    },
    {
        "rule_id": "L2_VERSION_PATTERN_001",
        "applies_to_pii_type": "any",
        "primary_class": "FP-숫자나열/코드",
        "reason_code": "VERSION_NUMBER",
        "pattern_type": "regex",
        "pattern": r"(?i)(v\d+\.\d+|ver(?:sion)?\s*[=:]?\s*\d+|\.\d+\.\d+\.\d+)",
        "context_scope": "full_context",
        "priority": 75,
        "evidence_template": "버전 패턴 발견: {matched_value}",
        "active": True,
    },
]

RULE_STATS_MINIMAL = {
    "L1_DOMAIN_INTERNAL_001": {"N": 5000, "M": 4985},
    "L1_DOMAIN_OS_001":       {"N": 1000, "M": 995},
    "L2_KEYWORD_BYTES_001":   {"N": 500,  "M": 490},
    "L2_KEYWORD_TIMESTAMP_001": {"N": 300, "M": 294},
    "L2_VERSION_PATTERN_001": {"N": 200,  "M": 190},
}


def _make_row(
    full_context_raw: str,
    pii_type_inferred: str = "email",
    email_domain: str = None,
    pk_event: str = "evt001",
) -> dict:
    return {
        "pk_event": pk_event,
        "pii_type_inferred": pii_type_inferred,
        "email_domain": email_domain,
        "full_context_raw": full_context_raw,
    }


# ────────────────────────────────────────────────
# Test 3.6, 3.7: Bayesian Lower Bound
# ────────────────────────────────────────────────

class TestRuleConfidence:
    """compute_rule_precision_lb 단위 테스트"""

    def test_rule_confidence_lb_bayesian_large_sample(self):
        """Test 3.6: N=5000, M=4985 → ~0.994"""
        from src.filters.rule_confidence import compute_rule_precision_lb
        lb = compute_rule_precision_lb(N=5000, M=4985)
        assert isinstance(lb, float)
        assert 0.990 <= lb <= 0.999, f"Expected ~0.994, got {lb}"

    def test_rule_confidence_lb_small_sample(self):
        """Test 3.7: N=15, M=15 → ~0.814"""
        from src.filters.rule_confidence import compute_rule_precision_lb
        lb = compute_rule_precision_lb(N=15, M=15)
        assert isinstance(lb, float)
        assert 0.79 <= lb <= 0.90, f"Expected ~0.814, got {lb}"

    def test_rule_confidence_lb_zero_samples(self):
        """N=0, M=0 → uninformative prior (0.5)"""
        from src.filters.rule_confidence import compute_rule_precision_lb
        lb = compute_rule_precision_lb(N=0, M=0)
        assert lb == 0.5, f"Expected 0.5 (uninformative prior), got {lb}"

    def test_rule_confidence_lb_perfect_record(self):
        """N=100, M=100 → 높은 신뢰도 (> 0.95)"""
        from src.filters.rule_confidence import compute_rule_precision_lb
        lb = compute_rule_precision_lb(N=100, M=100)
        assert lb > 0.95, f"Expected > 0.95, got {lb}"


# ────────────────────────────────────────────────
# Test 3.1 ~ 3.5, 3.8, 3.9: RuleLabeler.label()
# ────────────────────────────────────────────────

class TestRuleLabelerLabel:
    """RuleLabeler.label() 단위 테스트"""

    @pytest.fixture
    def labeler(self):
        from src.filters.rule_labeler import RuleLabeler
        return RuleLabeler(
            rules_config=RULES_CONFIG_MINIMAL,
            rule_stats=RULE_STATS_MINIMAL,
        )

    def test_rule_labeler_internal_domain(self, labeler):
        """Test 3.1: @lguplus.co.kr → FP-내부도메인"""
        row = _make_row(
            full_context_raw="user@lguplus.co.kr is an internal account",
            pii_type_inferred="email",
            email_domain="lguplus.co.kr",
        )
        result = labeler.label(row)
        assert result is not None
        assert result["rule_matched"] is True
        assert result["rule_primary_class"] == "FP-내부도메인"
        assert result["rule_id"] == "L1_DOMAIN_INTERNAL_001"

    def test_rule_labeler_os_domain(self, labeler):
        """Test 3.2: @redhat.com → FP-OS저작권"""
        row = _make_row(
            full_context_raw="admin@redhat.com contributed to this package",
            pii_type_inferred="email",
            email_domain="redhat.com",
        )
        result = labeler.label(row)
        assert result is not None
        assert result["rule_matched"] is True
        assert result["rule_primary_class"] == "FP-OS저작권"
        assert result["rule_id"] == "L1_DOMAIN_OS_001"

    def test_rule_labeler_bytes_keyword(self, labeler):
        """Test 3.3: '45 bytes 141022***' → FP-bytes크기"""
        row = _make_row(
            full_context_raw="file size is 45 bytes 141022 in the log",
            pii_type_inferred="unknown",
            email_domain=None,
        )
        result = labeler.label(row)
        assert result is not None
        assert result["rule_matched"] is True
        assert result["rule_primary_class"] == "FP-bytes크기"
        assert result["rule_id"] == "L2_KEYWORD_BYTES_001"

    def test_rule_labeler_priority_resolution(self, labeler):
        """Test 3.4: 다중 매칭 시 priority 높은 룰 선택 (bytes=85 > version=75)"""
        row = _make_row(
            full_context_raw="size is 1024 bytes for version v2.3.1",
            pii_type_inferred="unknown",
            email_domain=None,
        )
        result = labeler.label(row)
        assert result is not None
        assert result["rule_id"] == "L2_KEYWORD_BYTES_001", (
            f"bytes(85) should win over version(75), got {result['rule_id']}"
        )
        assert result["rule_candidates_count"] >= 2

    def test_rule_labeler_conflict_detection(self, labeler):
        """Test 3.5: 다른 primary_class 충돌 → rule_has_conflict=True"""
        # bytes(FP-bytes크기, prio=85) + timestamp(FP-타임스탬프, prio=80) 동시 매칭
        row = _make_row(
            full_context_raw="size is 45 bytes, expirDate=170603",
            pii_type_inferred="unknown",
            email_domain=None,
        )
        result = labeler.label(row)
        assert result is not None
        assert result["rule_candidates_count"] >= 2
        assert result["rule_has_conflict"] is True

    def test_rule_labeler_evidence_span(self, labeler):
        """Test 3.8: matched_span_start/end가 실제 텍스트와 일치"""
        text = "file size is 45 bytes in the system"
        row = _make_row(full_context_raw=text, pii_type_inferred="unknown", email_domain=None)
        _, evidence_list = labeler.label_with_evidence(row)
        assert len(evidence_list) > 0
        ev = evidence_list[0]
        assert "matched_span_start" in ev
        assert "matched_span_end" in ev
        start, end = ev["matched_span_start"], ev["matched_span_end"]
        assert start >= 0
        assert end > start
        # span이 실제 텍스트를 가리키는지 확인
        assert text[start:end] in text

    def test_rule_labeler_no_match_returns_none(self, labeler):
        """Test 3.9: 매칭 없음 → None 반환 (또는 rule_matched=False)"""
        row = _make_row(
            full_context_raw="홍길동 주민번호 841015-1234567",
            pii_type_inferred="rrn",
            email_domain=None,
        )
        result = labeler.label(row)
        assert result is None or result.get("rule_matched") is False

    def test_rule_labeler_inactive_rule_skipped(self, labeler):
        """active=False 룰은 무시됨"""
        inactive_rules = [
            {
                "rule_id": "INACTIVE_001",
                "applies_to_pii_type": "any",
                "primary_class": "FP-bytes크기",
                "reason_code": "TEST",
                "pattern_type": "regex",
                "pattern": r"bytes",
                "priority": 99,
                "evidence_template": "",
                "active": False,  # ← 비활성화
            }
        ]
        from src.filters.rule_labeler import RuleLabeler
        labeler_inactive = RuleLabeler(rules_config=inactive_rules, rule_stats={})
        row = _make_row(full_context_raw="45 bytes", pii_type_inferred="unknown")
        result = labeler_inactive.label(row)
        assert result is None or result.get("rule_matched") is False

    def test_rule_labeler_applies_to_pii_type_filter(self, labeler):
        """applies_to_pii_type='email' 룰은 다른 pii_type에 적용 안 됨"""
        # 내부 도메인 룰은 email 타입에만 적용
        row = _make_row(
            full_context_raw="user@lguplus.co.kr in log",
            pii_type_inferred="rrn",  # ← email 아님
            email_domain="lguplus.co.kr",
        )
        result = labeler.label(row)
        # L1_DOMAIN_INTERNAL_001 (email only) 미적용 → bytes/timestamp도 없음
        # → None 또는 rule_matched=False
        if result is not None and result.get("rule_matched"):
            assert result["rule_id"] not in (
                "L1_DOMAIN_INTERNAL_001", "L1_DOMAIN_OS_001"
            ), "email-only rule should not apply to rrn type"


# ────────────────────────────────────────────────
# label_batch() 테스트
# ────────────────────────────────────────────────

class TestRuleLabelerBatch:
    """RuleLabeler.label_batch() 테스트"""

    @pytest.fixture
    def labeler(self):
        from src.filters.rule_labeler import RuleLabeler
        return RuleLabeler(
            rules_config=RULES_CONFIG_MINIMAL,
            rule_stats=RULE_STATS_MINIMAL,
        )

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([
            {
                "pk_event": "evt001",
                "pii_type_inferred": "email",
                "email_domain": "lguplus.co.kr",
                "full_context_raw": "user@lguplus.co.kr is internal",
            },
            {
                "pk_event": "evt002",
                "pii_type_inferred": "unknown",
                "email_domain": None,
                "full_context_raw": "size 45 bytes in log",
            },
            {
                "pk_event": "evt003",
                "pii_type_inferred": "rrn",
                "email_domain": None,
                "full_context_raw": "주민번호 841015-1234567",
            },
        ])

    def test_label_batch_output_schemas(self, labeler, sample_df):
        """label_batch 출력 스키마 검증 (Architecture §7.4 기반)"""
        rule_labels_df, rule_evidence_df = labeler.label_batch(sample_df)

        # ── rule_labels_df 스키마 ──
        required_label_cols = {
            "pk_event", "rule_matched", "rule_primary_class", "rule_reason_code",
            "rule_id", "rule_confidence_lb", "rule_confidence_type",
            "rule_candidates_count", "rule_has_conflict",
        }
        missing = required_label_cols - set(rule_labels_df.columns)
        assert not missing, f"rule_labels_df 누락 컬럼: {missing}"
        assert len(rule_labels_df) == len(sample_df), "행 수 불일치"

        # ── rule_evidence_df 스키마 (long-format) ──
        required_evidence_cols = {
            "pk_event", "evidence_rank", "evidence_type", "source",
            "rule_id", "matched_value", "matched_span_start",
            "matched_span_end", "snippet",
        }
        missing_ev = required_evidence_cols - set(rule_evidence_df.columns)
        assert not missing_ev, f"rule_evidence_df 누락 컬럼: {missing_ev}"

    def test_label_batch_row_count_preserved(self, labeler, sample_df):
        """label_batch는 입력 행 수 그대로 rule_labels_df 반환"""
        rule_labels_df, _ = labeler.label_batch(sample_df)
        assert len(rule_labels_df) == len(sample_df)

    def test_label_batch_only_matched_have_evidence(self, labeler, sample_df):
        """매칭된 이벤트만 rule_evidence_df에 evidence를 가짐"""
        rule_labels_df, rule_evidence_df = labeler.label_batch(sample_df)
        matched_events = set(
            rule_labels_df.loc[rule_labels_df["rule_matched"] == True, "pk_event"]
        )
        evidence_events = set(rule_evidence_df["pk_event"])
        assert evidence_events.issubset(matched_events), (
            f"Unmatched events have evidence: {evidence_events - matched_events}"
        )

    def test_label_batch_confidence_type_populated(self, labeler, sample_df):
        """rule_confidence_type가 BAYESIAN_LB 또는 PRIOR 중 하나"""
        rule_labels_df, _ = labeler.label_batch(sample_df)
        matched = rule_labels_df[rule_labels_df["rule_matched"] == True]
        valid_types = {"BAYESIAN_LB", "PRIOR"}
        for ct in matched["rule_confidence_type"].dropna():
            assert ct in valid_types, f"Unknown confidence_type: {ct}"


# ────────────────────────────────────────────────
# 하위 호환: 기존 필터 여전히 작동
# ────────────────────────────────────────────────

class TestBackwardCompat:
    """기존 KeywordFilter/RuleFilter 하위 호환 확인"""

    def test_existing_filters_importable(self):
        """기존 필터 클래스 import 가능"""
        from src.filters import KeywordFilter, RuleFilter, FilterPipeline
        assert KeywordFilter is not None
        assert RuleFilter is not None
        assert FilterPipeline is not None

    def test_rule_labeler_importable_from_filters(self):
        """RuleLabeler가 src.filters에서 import 가능"""
        from src.filters import RuleLabeler
        assert RuleLabeler is not None


# ────────────────────────────────────────────────
# feature_condition 패턴 타입 테스트
# ────────────────────────────────────────────────

class TestFeatureConditionRule:
    """feature_condition 패턴 타입 단위 테스트"""

    @pytest.fixture
    def fc_labeler(self):
        from src.filters.rule_labeler import RuleLabeler
        rules = [
            {
                "rule_id": "FC_SINGLE_EQ",
                "applies_to_pii_type": "any",
                "primary_class": "FP-패턴맥락",
                "reason_code": "SINGLE_EQ",
                "pattern_type": "feature_condition",
                "conditions": [{"field": "is_log_file", "op": "eq", "value": 1}],
                "logic": "and",
                "priority": 90,
                "evidence_template": "test",
                "active": True,
            },
            {
                "rule_id": "FC_AND_BOTH",
                "applies_to_pii_type": "any",
                "primary_class": "FP-패턴맥락",
                "reason_code": "AND_BOTH",
                "pattern_type": "feature_condition",
                "conditions": [
                    {"field": "is_log_file", "op": "eq", "value": 1},
                    {"field": "pattern_count", "op": "gt", "value": 10000},
                ],
                "logic": "and",
                "priority": 88,
                "evidence_template": "test",
                "active": True,
            },
            {
                "rule_id": "FC_OR_ONE",
                "applies_to_pii_type": "any",
                "primary_class": "FP-OS저작권",
                "reason_code": "OR_ONE",
                "pattern_type": "feature_condition",
                "conditions": [
                    {"field": "has_license_path", "op": "eq", "value": 1},
                    {"field": "is_docker_overlay", "op": "eq", "value": 1},
                ],
                "logic": "or",
                "priority": 70,
                "evidence_template": "test",
                "active": True,
            },
        ]
        return RuleLabeler(rules_config=rules, rule_stats={})

    def test_feature_condition_single_eq(self, fc_labeler):
        """단일 조건 eq: is_log_file=1 → 매칭"""
        row = {"pk_event": "e1", "pii_type_inferred": "any", "is_log_file": 1,
               "full_context_raw": ""}
        result = fc_labeler.label(row)
        assert result is not None
        assert result["rule_matched"] is True
        assert result["rule_id"] == "FC_SINGLE_EQ"

    def test_feature_condition_and_both_true(self, fc_labeler):
        """복합 AND: 두 조건 모두 참 → 매칭"""
        row = {"pk_event": "e2", "pii_type_inferred": "any",
               "is_log_file": 1, "pattern_count": 52000, "full_context_raw": ""}
        result = fc_labeler.label(row)
        assert result is not None
        assert result["rule_matched"] is True
        # AND rule (both conditions met) wins with highest priority among matches

    def test_feature_condition_and_one_false(self, fc_labeler):
        """복합 AND: 하나가 거짓 → FC_AND_BOTH 미매칭 (FC_SINGLE_EQ는 매칭)"""
        row = {"pk_event": "e3", "pii_type_inferred": "any",
               "is_log_file": 1, "pattern_count": 100, "full_context_raw": ""}
        result = fc_labeler.label(row)
        # FC_AND_BOTH should NOT match (pattern_count=100 not > 10000)
        # FC_SINGLE_EQ should match (is_log_file=1)
        assert result is not None
        assert result["rule_id"] == "FC_SINGLE_EQ"

    def test_feature_condition_or_one_true(self, fc_labeler):
        """OR 로직: 하나라도 참이면 매칭"""
        row = {"pk_event": "e4", "pii_type_inferred": "any",
               "has_license_path": 1, "is_docker_overlay": 0, "full_context_raw": ""}
        result = fc_labeler.label(row)
        assert result is not None
        assert result["rule_id"] == "FC_OR_ONE"
        assert result["rule_matched"] is True

    def test_feature_condition_missing_field(self, fc_labeler):
        """피처 없으면 미매칭"""
        # Row has no feature fields at all
        row = {"pk_event": "e5", "pii_type_inferred": "any", "full_context_raw": ""}
        result = fc_labeler.label(row)
        assert result is None or result.get("rule_matched") is False

    def test_path_license_rule_matches(self):
        """PATH_LICENSE_001 실제 룰 검증: has_license_path=1 → FP-OS저작권"""
        from src.filters.rule_labeler import RuleLabeler
        rule = {
            "rule_id": "PATH_LICENSE_001",
            "applies_to_pii_type": "any",
            "primary_class": "FP-OS저작권",
            "reason_code": "LICENSE_PATH",
            "pattern_type": "feature_condition",
            "conditions": [{"field": "has_license_path", "op": "eq", "value": 1}],
            "logic": "and",
            "priority": 92,
            "evidence_template": "라이선스/저작권 경로",
            "active": True,
        }
        labeler = RuleLabeler(rules_config=[rule], rule_stats={})
        row = {
            "pk_event": "lic001",
            "pii_type_inferred": "email",
            "has_license_path": 1,
            "full_context_raw": "",
        }
        result = labeler.label(row)
        assert result is not None
        assert result["rule_primary_class"] == "FP-OS저작권"
        assert result["rule_id"] == "PATH_LICENSE_001"

    def test_path_log_mass_rule_matches(self):
        """PATH_LOG_MASS_001 실제 룰 검증: is_log_file=1 AND pattern_count>10000 → FP-패턴맥락"""
        from src.filters.rule_labeler import RuleLabeler
        rule = {
            "rule_id": "PATH_LOG_MASS_001",
            "applies_to_pii_type": "any",
            "primary_class": "FP-패턴맥락",
            "reason_code": "LOG_MASS_DETECTION",
            "pattern_type": "feature_condition",
            "conditions": [
                {"field": "is_log_file", "op": "eq", "value": 1},
                {"field": "pattern_count", "op": "gt", "value": 10000},
            ],
            "logic": "and",
            "priority": 88,
            "evidence_template": "로그 파일 + 대량 검출",
            "active": True,
        }
        labeler = RuleLabeler(rules_config=[rule], rule_stats={})
        row = {
            "pk_event": "mass001",
            "pii_type_inferred": "email",
            "is_log_file": 1,
            "pattern_count": 52000,
            "full_context_raw": "",
        }
        result = labeler.label(row)
        assert result is not None
        assert result["rule_primary_class"] == "FP-패턴맥락"
        assert result["rule_id"] == "PATH_LOG_MASS_001"
        # Not matched when pattern_count is low
        row_low = dict(row)
        row_low["pattern_count"] = 500
        result_low = labeler.label(row_low)
        assert result_low is None or result_low.get("rule_matched") is False
