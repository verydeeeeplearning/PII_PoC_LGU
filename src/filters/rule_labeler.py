"""S3a RULE Labeler - Architecture v1.2 §7.4

패러다임 전환:
    기존: FilterPipeline -> filtered_df / passed_df (삭제)
    신규: RuleLabeler    -> rule_labels + rule_evidence (라벨 + 근거 출력)

출력 스키마:
    rule_labels_df  : pk_event, rule_matched, rule_primary_class, rule_reason_code,
                      rule_id, rule_confidence_lb, rule_confidence_type,
                      rule_candidates_count, rule_has_conflict
    rule_evidence_df: pk_event, evidence_rank, evidence_type, source,
                      rule_id, matched_value, matched_span_start,
                      matched_span_end, snippet  (long-format)
"""

from __future__ import annotations

import operator as op
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.filters.rule_confidence import compute_rule_precision_lb


# ── 스니펫 추출 윈도우 ──
_SNIPPET_WINDOW = 40

_OPS = {
    "eq": op.eq,
    "ne": op.ne,
    "gt": op.gt,
    "gte": op.ge,
    "lt": op.lt,
    "lte": op.le,
}


@dataclass
class RuleMatch:
    """단일 룰 매칭 결과"""
    rule_id: str
    primary_class: str
    reason_code: str
    matched_value: str
    matched_span_start: int
    matched_span_end: int
    snippet: str
    priority: int


def _extract_snippet(text: str, start: int, end: int, window: int = _SNIPPET_WINDOW) -> str:
    """매칭 위치 주변 컨텍스트 추출"""
    s = max(0, start - window)
    e = min(len(text), end + window)
    return text[s:e]


class RuleLabeler:
    """rules.yaml 기반 RULE Labeler (Architecture §7.4).

    기존 KeywordFilter / RuleFilter 와 병존하며, src/filters/* 네임스페이스 내에서
    하위 호환을 유지한다.

    Args:
        rules_config: 룰 정의 목록 (rules.yaml에서 파싱된 list[dict])
        rule_stats  : 룰별 실적 통계 {rule_id: {"N": int, "M": int}}
    """

    def __init__(self, rules_config: list[dict], rule_stats: dict):
        # active=True인 룰만 사용
        self._rules = [r for r in rules_config if r.get("active", True)]
        self._rule_stats = rule_stats
        # regex 룰은 미리 컴파일
        for rule in self._rules:
            if rule.get("pattern_type") == "regex":
                rule["_compiled"] = re.compile(
                    rule["pattern"], re.IGNORECASE | re.DOTALL
                )

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def label(self, row: dict) -> Optional[dict]:
        """단일 행 라벨링.

        Returns:
            rule_label dict if any rule matches, None otherwise.
        """
        label_result, _ = self.label_with_evidence(row)
        return label_result

    def label_with_evidence(self, row: dict) -> tuple[Optional[dict], list[dict]]:
        """단일 행 라벨링 + evidence 목록.

        Returns:
            (label_dict | None, evidence_list)
        """
        matches = self._collect_matches(row)
        if not matches:
            return None, []

        # priority 내림차순 정렬
        matches.sort(key=lambda m: m.priority, reverse=True)
        best = matches[0]

        classes = {m.primary_class for m in matches}
        has_conflict = len(classes) > 1

        confidence_lb, confidence_type = self._get_confidence(best.rule_id)

        label_result = {
            "rule_matched": True,
            "rule_primary_class": best.primary_class,
            "rule_reason_code": best.reason_code,
            "rule_id": best.rule_id,
            "rule_confidence_lb": confidence_lb,
            "rule_confidence_type": confidence_type,
            "rule_candidates_count": len(matches),
            "rule_has_conflict": has_conflict,
        }

        evidence_list = [
            {
                "evidence_rank": rank,
                "evidence_type": "RULE_MATCH",
                "source": "RULE",
                "rule_id": m.rule_id,
                "matched_value": m.matched_value,
                "matched_span_start": m.matched_span_start,
                "matched_span_end": m.matched_span_end,
                "snippet": m.snippet,
            }
            for rank, m in enumerate(matches)
        ]

        return label_result, evidence_list

    def label_batch(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """배치 라벨링.

        Args:
            df: silver_detections 또는 silver_features_base DataFrame
                필수 컬럼: pk_event, pii_type_inferred, full_context_raw
                선택 컬럼: email_domain

        Returns:
            (rule_labels_df, rule_evidence_df)
        """
        label_rows: list[dict] = []
        evidence_rows: list[dict] = []

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            pk_event = row_dict.get("pk_event", "")

            label_result, ev_list = self.label_with_evidence(row_dict)

            if label_result is None:
                label_rows.append({
                    "pk_event": pk_event,
                    "rule_matched": False,
                    "rule_primary_class": None,
                    "rule_reason_code": None,
                    "rule_id": None,
                    "rule_confidence_lb": None,
                    "rule_confidence_type": None,
                    "rule_candidates_count": 0,
                    "rule_has_conflict": False,
                })
            else:
                label_rows.append({"pk_event": pk_event, **label_result})
                for ev in ev_list:
                    evidence_rows.append({"pk_event": pk_event, **ev})

        rule_labels_df = pd.DataFrame(label_rows)

        if evidence_rows:
            rule_evidence_df = pd.DataFrame(evidence_rows)
        else:
            rule_evidence_df = pd.DataFrame(columns=[
                "pk_event", "evidence_rank", "evidence_type", "source",
                "rule_id", "matched_value", "matched_span_start",
                "matched_span_end", "snippet",
            ])

        return rule_labels_df, rule_evidence_df

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────

    def _collect_matches(self, row: dict) -> list[RuleMatch]:
        """모든 룰을 순회하며 매칭 결과 수집"""
        matches = []
        for rule in self._rules:
            match = self._match_pattern(rule, row)
            if match is not None:
                matches.append(match)
        return matches

    def _match_pattern(self, rule: dict, row: dict) -> Optional[RuleMatch]:
        """pii_type 필터 후 패턴 타입에 따라 분기"""
        applies_to = rule.get("applies_to_pii_type", "any")
        if applies_to != "any":
            if row.get("pii_type_inferred", "") != applies_to:
                return None

        pattern_type = rule.get("pattern_type", "regex")
        if pattern_type == "domain_list":
            return self._match_domain_list(rule, row)
        if pattern_type == "regex":
            return self._match_regex(rule, row)
        if pattern_type == "feature_condition":
            return self._match_feature_condition(rule, row)
        return None

    def _match_domain_list(self, rule: dict, row: dict) -> Optional[RuleMatch]:
        """domain_list 패턴: email_domain 직접 비교 후 full_context_raw에서 탐색"""
        domains: list[str] = rule.get("pattern", [])
        text: str = row.get("full_context_raw", "") or ""

        # 1차: email_domain 컬럼 직접 비교 (빠른 경로)
        email_domain = row.get("email_domain", "") or ""
        for domain in domains:
            if email_domain.lower() == domain.lower():
                start, end = self._find_domain_span(text, domain)
                return RuleMatch(
                    rule_id=rule["rule_id"],
                    primary_class=rule["primary_class"],
                    reason_code=rule["reason_code"],
                    matched_value=f"@{domain}",
                    matched_span_start=start,
                    matched_span_end=end,
                    snippet=_extract_snippet(text, start, end),
                    priority=rule["priority"],
                )

        # 2차: full_context_raw에서 "@domain" 탐색
        for domain in domains:
            m = re.search(r"@" + re.escape(domain), text, re.IGNORECASE)
            if m:
                start, end = m.span()
                return RuleMatch(
                    rule_id=rule["rule_id"],
                    primary_class=rule["primary_class"],
                    reason_code=rule["reason_code"],
                    matched_value=m.group(),
                    matched_span_start=start,
                    matched_span_end=end,
                    snippet=_extract_snippet(text, start, end),
                    priority=rule["priority"],
                )
        return None

    def _match_regex(self, rule: dict, row: dict) -> Optional[RuleMatch]:
        """regex 패턴: full_context_raw에서 탐색.

        requires_context: true인 룰은 full_context_raw가 없을 때 즉시 None 반환.
        Phase 1(label-only) 환경에서 불필요한 순회를 방지한다.
        """
        text: str = row.get("full_context_raw", "") or ""
        # requires_context 룰: Phase 1에서 full_context_raw 없으면 조기 종료
        if rule.get("requires_context", False) and not text:
            return None
        compiled = rule.get("_compiled")
        if not compiled or not text:
            return None

        m = compiled.search(text)
        if not m:
            return None

        start, end = m.span()
        return RuleMatch(
            rule_id=rule["rule_id"],
            primary_class=rule["primary_class"],
            reason_code=rule["reason_code"],
            matched_value=m.group(),
            matched_span_start=start,
            matched_span_end=end,
            snippet=_extract_snippet(text, start, end),
            priority=rule["priority"],
        )

    def _match_feature_condition(self, rule: dict, row: dict) -> Optional[RuleMatch]:
        """feature_condition 패턴: 피처 컬럼 값 조건 기반 매칭"""
        conditions = rule.get("conditions", [])
        logic = rule.get("logic", "and")

        results = [self._eval_condition(cond, row) for cond in conditions]
        matched = all(results) if logic == "and" else any(results)

        if not matched:
            return None

        matched_value = ", ".join(
            f"{c['field']}={row.get(c['field'], 'N/A')}"
            for c in conditions
        )
        return RuleMatch(
            rule_id=rule["rule_id"],
            primary_class=rule["primary_class"],
            reason_code=rule["reason_code"],
            matched_value=matched_value,
            matched_span_start=0,
            matched_span_end=0,
            snippet=matched_value,
            priority=rule["priority"],
        )

    def _eval_condition(self, cond: dict, row: dict) -> bool:
        """단일 조건 평가"""
        field_val = row.get(cond["field"])
        if field_val is None:
            return False
        fn = _OPS.get(cond["op"])
        if fn is None:
            return False
        try:
            return bool(fn(field_val, cond["value"]))
        except (TypeError, ValueError):
            return False

    def _get_confidence(self, rule_id: str) -> tuple[float, str]:
        """(confidence_lb, confidence_type) 반환"""
        stats = self._rule_stats.get(rule_id, {})
        N = stats.get("N", 0)
        M = stats.get("M", 0)
        if N == 0:
            return 0.5, "PRIOR"
        return compute_rule_precision_lb(N, M), "BAYESIAN_LB"

    @staticmethod
    def _find_domain_span(text: str, domain: str) -> tuple[int, int]:
        """full_context_raw에서 도메인 위치 탐색 (없으면 0, len(domain) 반환)"""
        m = re.search(r"@" + re.escape(domain), text, re.IGNORECASE)
        if m:
            return m.span()
        m = re.search(re.escape(domain), text, re.IGNORECASE)
        if m:
            return m.span()
        return 0, len(domain)

    # ──────────────────────────────────────────────
    # Factory: rules.yaml + rule_stats.json 로드
    # ──────────────────────────────────────────────

    @classmethod
    def from_config_files(
        cls,
        rules_yaml_path: str,
        rule_stats_json_path: str,
    ) -> "RuleLabeler":
        """YAML/JSON 파일에서 RuleLabeler 인스턴스 생성.

        Args:
            rules_yaml_path    : config/rules.yaml 경로
            rule_stats_json_path: config/rule_stats.json 경로
        """
        import json
        import yaml  # type: ignore[import]

        with open(rules_yaml_path, encoding="utf-8") as f:
            rules_data = yaml.safe_load(f)
        rules_config = rules_data.get("rules", [])

        with open(rule_stats_json_path, encoding="utf-8") as f:
            rule_stats = json.load(f)
        # _comment 키 제거
        rule_stats.pop("_comment", None)

        return cls(rules_config=rules_config, rule_stats=rule_stats)
