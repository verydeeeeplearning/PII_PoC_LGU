"""S4 Auto Adjudicator - Architecture v1.2 §8.13

NEEDS_REVIEW 판정을 파일 수준 컨텍스트로 자동 판별 (Tier 1).

전략:
    1. 동일 파일 내 이벤트 다수결 -> 과반수 클래스 채택
    2. 과반수 없음 또는 컨텍스트 없음 -> TP (보수적 결정)
"""

from __future__ import annotations

from collections import Counter

from src.utils.constants import LABEL_TP


class AutoAdjudicator:
    """파일 컨텍스트 기반 자동 판별기."""

    def adjudicate(
        self,
        review: dict,
        context: dict | None = None,
    ) -> dict:
        """NEEDS_REVIEW 이벤트를 자동 판별.

        Args:
            review : 판별 대상 이벤트 dict (pk_event, pk_file, primary_class 등)
            context: {"file_decisions": [{pk_event, primary_class}, ...]}
                     동일 파일 내 다른 이벤트의 판정 목록

        Returns:
            {"adjudicated_class": str}
        """
        ctx = context or {}
        file_decisions: list[dict] = ctx.get("file_decisions", [])

        if not file_decisions:
            # 컨텍스트 없음 -> TP (보수적 결정)
            return {"adjudicated_class": LABEL_TP}

        counts = Counter(d["primary_class"] for d in file_decisions)
        majority_class, majority_count = counts.most_common(1)[0]
        total = len(file_decisions)

        # 과반수(> 50%) 조건
        if majority_count > total / 2:
            return {"adjudicated_class": majority_class}

        # 과반수 없음 -> TP (보수적)
        return {"adjudicated_class": LABEL_TP}
