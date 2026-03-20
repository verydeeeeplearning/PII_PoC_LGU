## 7. Stage S3a: RULE Labeler

### 7.1 기능

RULE 라벨러는 이 시스템의 **"가장 값싼 확실성"**을 제공하는 핵심 모듈이다. 룰이 매칭되면 해당 건에 대해 primary_class + reason_code + evidence + confidence를 완전하게 생성한다.

**수행 작업:**

1. 모든 룰을 순회하여 매칭 후보를 수집 (다중 후보 가능)
2. priority + specificity + rule_confidence_lb로 최종 1개 선택
3. 선택된 룰에 대해 primary_class, reason_code, evidence 생성
4. rule_confidence_lb (경험적 정밀도 하한) 부여

### 7.2 왜 "라벨러"인가 (필터와의 차이)

| 설계 | 동작 | 설명가능성 | 사후 검증 | 리스크 |
|------|------|-----------|-----------|--------|
| **필터** | 매칭 건 삭제 | ❌ 삭제된 건은 설명 불가 | ❌ "필터가 틀렸다"를 발견할 방법 없음 | 높음 |
| **라벨러** | 매칭 건에 라벨+증거 부여 | ✅ 증거가 데이터로 남음 | ✅ 사후 정밀도 추적 가능 | 낮음 |

### 7.3 룰 정의 포맷

**`rules.yaml` (운영 가능한 설정 파일)**

```yaml
rules:
  - rule_id: "L1_DOMAIN_INTERNAL_001"
    applies_to_pii_type: "email"
    primary_class: "FP-이메일패턴"
    reason_code: "INT_DOMAIN_LGUPLUS"
    pattern_type: "domain_list"
    pattern:
      - "lguplus.co.kr"
      - "bdp.lguplus.co.kr"
      - "lgup.co.kr"
    priority: 100
    evidence_template: "내부 도메인 매칭: {matched_value}"
    
  - rule_id: "L1_DOMAIN_OS_001"
    applies_to_pii_type: "email"
    primary_class: "FP-라이브러리"
    reason_code: "OS_DOMAIN_REDHAT"
    pattern_type: "domain_list"
    pattern:
      - "redhat.com"
      - "fedoraproject.org"
      - "gnu.org"
      - "apache.org"
    priority: 90
    evidence_template: "OS/오픈소스 도메인 매칭: {matched_value}"
    
  - rule_id: "L2_KEYWORD_BYTES_001"
    applies_to_pii_type: "any"
    primary_class: "FP-숫자패턴"
    reason_code: "BYTES_KEYWORD_ADJACENT"
    pattern_type: "regex"
    pattern: "\\bbytes?\\b"
    context_scope: "full_context"
    priority: 85
    evidence_template: "bytes 키워드 인접: {matched_value}"
    
  - rule_id: "L2_KEYWORD_TIMESTAMP_001"
    applies_to_pii_type: "any"
    primary_class: "FP-숫자패턴"
    reason_code: "KEY_EXPIRYDATE"
    pattern_type: "regex"
    pattern: "(?i)(expir|timestamp|duration|created_at|updated_at|date\\s*=)"
    context_scope: "full_context"
    priority: 80
    evidence_template: "타임스탬프 관련 키워드: {matched_value}"
```

**왜 YAML인가:**
- 코드 수정 없이 룰 추가/삭제가 가능하다. 새로운 내부 도메인이 생기면 yaml에 한 줄 추가하면 된다.
- 룰을 버전 관리(Git)할 수 있어 "언제 어떤 룰이 추가됐는지" 추적이 가능하다.
- 운영팀(비개발자)도 구조를 이해하고 수정할 수 있다.

### 7.4 룰 엔진 구현

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RuleMatch:
    rule_id: str
    primary_class: str
    reason_code: str
    matched_value: str
    matched_span_start: int
    matched_span_end: int
    snippet: str
    priority: int
    confidence_lb: float  # 경험적 정밀도 하한

class RuleLabeler:
    """
    모든 룰을 돌려서 매칭 후보를 모으고,
    priority/신뢰도로 최종 1개를 결정.
    다중 후보가 있으면 top2를 보존하여 충돌 감지에 활용.
    """
    def __init__(self, rules_config: list, rule_stats: dict):
        self.rules = rules_config
        self.rule_stats = rule_stats  # {rule_id: {N, M, precision_lb}}
    
    def label(self, row: dict) -> Optional[dict]:
        candidates = []
        
        for rule in self.rules:
            # PII 유형 필터링
            if rule['applies_to_pii_type'] != 'any':
                if row['pii_type_inferred'] != rule['applies_to_pii_type']:
                    continue
            
            # 패턴 매칭
            match_result = self._match_pattern(rule, row)
            if match_result:
                stats = self.rule_stats.get(rule['rule_id'], {})
                candidates.append(RuleMatch(
                    rule_id=rule['rule_id'],
                    primary_class=rule['primary_class'],
                    reason_code=rule['reason_code'],
                    matched_value=match_result['value'],
                    matched_span_start=match_result['start'],
                    matched_span_end=match_result['end'],
                    snippet=match_result['snippet'],
                    priority=rule['priority'],
                    confidence_lb=stats.get('precision_lb', 0.5),
                ))
        
        if not candidates:
            return None
        
        # priority가 가장 높은 후보 선택
        candidates.sort(key=lambda c: c.priority, reverse=True)
        best = candidates[0]
        
        return {
            'rule_label': best.primary_class,
            'rule_reason_code': best.reason_code,
            'rule_confidence_lb': best.confidence_lb,
            'rule_id': best.rule_id,
            'rule_matched_value': best.matched_value,
            'rule_snippet': best.snippet,
            'rule_candidates_count': len(candidates),
            'rule_has_conflict': len(candidates) > 1 and 
                candidates[0].primary_class != candidates[1].primary_class,
        }
```

### 7.5 룰 신뢰도 산출 방법

룰은 deterministic이지만, "항상 맞는다"는 보장은 없다. 새로운 로그 포맷, 새로운 도메인 등 예외가 존재할 수 있다. 따라서 룰의 신뢰도를 "경험적 정밀도 하한"으로 정의한다.

**산출 방식: Bayesian Lower Bound**

```python
from scipy import stats as sp_stats

def compute_rule_precision_lb(N: int, M: int, alpha: float = 0.05) -> float:
    """
    룰 r이 validation set에서:
    - 매칭된 건수 N
    - 그중 라벨이 맞은 건수 M
    
    Beta(1+M, 1+N-M) 사후분포의 alpha 하한을 반환.
    
    왜 단순 M/N이 아닌가:
    - N이 작으면 p_hat = M/N이 불안정하다.
    - 보수적 하한(lower bound)을 사용하면,
      표본 크기까지 반영하여 안전한 신뢰도를 제공할 수 있다.
    - 보안 운영에서는 과신보다 보수적 추정이 안전하다.
    """
    a = 1 + M           # Beta 사전분포 + 관측된 성공
    b = 1 + (N - M)     # Beta 사전분포 + 관측된 실패
    lb = sp_stats.beta.ppf(alpha, a, b)
    return round(lb, 4)
```

**예시:**
| 룰 | N (매칭) | M (정답) | 단순 정밀도 | 95% 하한 (confidence_lb) |
|----|---------|---------|-----------|-------------------------|
| L1_DOMAIN_INTERNAL_001 | 5,000 | 4,985 | 0.997 | 0.994 |
| L2_KEYWORD_BYTES_001 | 120 | 118 | 0.983 | 0.949 |
| L2_NEW_RULE_001 | 15 | 15 | 1.000 | 0.814 |

마지막 행을 보면, 15건 중 15건 맞았어도 신뢰도는 0.814에 불과하다. 이것이 "N이 작으면 과신하지 않는" Bayesian 하한의 장점이다.

**Rationale:**

- 룰이 deterministic이라고 해서 신뢰도를 1.0으로 주면 위험하다. 현실에서는 룰의 예외가 항상 존재하며, 특히 초기 배포 시점에는 룰이 검증되지 않은 상태다.
- 검증셋에서의 precision을 측정하고, 표본 수를 반영한 보수적 하한을 신뢰도로 주면, 운영팀이 "이 룰은 거의 확실" vs "이 룰은 아직 불안"을 명확히 구분할 수 있다.
- 이 수치는 Phase 2(피드백 루프)에서 현업 피드백이 쌓이면 자동으로 업데이트되어, 룰의 신뢰도가 시간이 지남에 따라 정밀해진다.

### 7.6 Auto-Rule-Promoter (v1.2 추가)

원칙 G(Zero-Human-in-the-Loop)에 따라, ML이 반복적으로 고확신으로 분류하는 패턴을 자동으로 룰로 승격한다. 사람이 새 도메인/패턴을 발견하여 `rules.yaml`에 수동 추가할 필요가 없다.

```python
class AutoRulePromoter:
    """
    ML 고확신 반복 패턴을 자동으로 RULE로 승격하거나,
    정밀도가 떨어진 기존 룰을 자동 비활성화한다.
    """
    MIN_OBSERVATIONS = 500     # 최소 관측 건수
    MIN_ML_CONFIDENCE = 0.90   # ML 최소 확신도
    MIN_HOLDOUT_PRECISION = 0.95  # holdout 최소 정밀도
    LOOKBACK_MONTHS = 3        # 패턴 누적 기간
    DEACTIVATION_THRESHOLD = 0.70  # 룰 비활성화 임계값

    def find_promotion_candidates(self, ml_history: pd.DataFrame) -> list:
        """
        최근 3개월간 ML이 동일 클래스로 분류한 케이스 중
        confidence >= 0.90 AND 동일 feature pattern 반복 N건 이상인 후보 추출.
        """
        recent = ml_history[
            ml_history['prediction_month'] >= self._lookback_start()
        ]
        candidates = (
            recent[recent['ml_top1_proba'] >= self.MIN_ML_CONFIDENCE]
            .groupby(['feature_pattern_hash', 'ml_top1_class_name'])
            .agg(count=('pk_event', 'count'),
                 avg_confidence=('ml_top1_proba', 'mean'))
            .reset_index()
        )
        return candidates[candidates['count'] >= self.MIN_OBSERVATIONS]

    def validate_and_promote(self, candidate, holdout_set,
                             rules_config: list) -> dict:
        """
        후보 룰의 holdout precision 검증 후 자동 승격.
        """
        # holdout에서 해당 패턴 매칭 → precision 검증
        matched = self._match_pattern_on_holdout(candidate, holdout_set)
        if len(matched) == 0:
            return {'action': 'skip', 'reason': 'no_holdout_match'}

        precision = matched['correct'].mean()
        if precision < self.MIN_HOLDOUT_PRECISION:
            return {'action': 'skip', 'reason': f'precision={precision:.3f}'}

        # 자동 룰 생성
        new_rule = {
            'rule_id': f"AUTO_{datetime.now():%Y%m%d%H%M}_{candidate['ml_top1_class_name'][:6]}",
            'primary_class': candidate['ml_top1_class_name'],
            'reason_code': f"AUTO_PROMOTED_{candidate['feature_pattern_hash'][:8]}",
            'pattern_type': candidate.get('pattern_type', 'feature_hash'),
            'pattern': candidate['feature_pattern_hash'],
            'priority': 50,  # 자동 생성 룰은 수동 룰보다 낮은 우선순위
            'rule_confidence_lb': self._compute_precision_lb(
                len(matched), int(matched['correct'].sum())
            ),
            'auto_generated': True,
            'generated_at': datetime.now().isoformat(),
        }

        rules_config.append(new_rule)
        return {'action': 'promoted', 'rule': new_rule, 'precision': precision}

    def deactivate_degraded_rules(self, rule_stats: dict,
                                  rules_config: list) -> list:
        """
        precision_lb < 0.70인 기존 룰을 자동 비활성화 (삭제가 아닌 비활성화).
        """
        deactivated = []
        for rule in rules_config:
            stats = rule_stats.get(rule['rule_id'], {})
            if stats.get('precision_lb', 1.0) < self.DEACTIVATION_THRESHOLD:
                rule['active'] = False
                rule['deactivated_at'] = datetime.now().isoformat()
                rule['deactivation_reason'] = (
                    f"precision_lb={stats['precision_lb']:.3f} < {self.DEACTIVATION_THRESHOLD}"
                )
                deactivated.append(rule['rule_id'])
        return deactivated
```

**Auto-Rule-Promoter 흐름:**

```
1. ML이 최근 3개월간 동일 클래스로 분류한 케이스 중
   confidence ≥ 0.90 AND 동일 feature pattern 반복 500건+ → 후보 추출

2. 후보 룰의 holdout precision 검증
   → precision ≥ 0.95 → 채택

3. 자동으로 rules.yaml에 추가
   - rule_id = "AUTO_{timestamp}_{class}"
   - rule_confidence_lb = holdout precision의 95% 하한
   - Git auto-commit (감사 추적)

4. 반대 방향: precision_lb < 0.70인 기존 룰 → 자동 비활성화 (삭제 아님)
```

### 7.7 출력 스키마

**`rule_labels.parquet`** (1행=1검출 이벤트, 룰 매칭 결과)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pk_event` | string | 검출 이벤트 ID |
| `rule_matched` | bool | 룰 매칭 여부 |
| `rule_primary_class` | string (nullable) | 룰이 부여한 primary_class |
| `rule_reason_code` | string (nullable) | 룰이 부여한 reason_code |
| `rule_id` | string (nullable) | 매칭된 룰 ID |
| `rule_confidence_lb` | float (nullable) | 룰 경험적 정밀도 하한 |
| `rule_confidence_type` | string | `"rule_precision_lb_95"` (고정) |
| `rule_candidates_count` | int | 매칭된 룰 후보 수 |
| `rule_has_conflict` | bool | 다중 후보 간 클래스 충돌 여부 |

**`rule_evidence.parquet`** (N행=1검출 이벤트, long-format)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pk_event` | string | 검출 이벤트 ID |
| `evidence_rank` | int | 근거 우선순위 (1부터) |
| `evidence_type` | string | `"RULE_MATCH"` |
| `source` | string | `"RULE"` |
| `rule_id` | string | 매칭된 룰 ID |
| `matched_value` | string | 매칭된 문자열 (예: `bdp.lguplus.co.kr`) |
| `matched_span_start` | int | full_context 내 시작 위치 |
| `matched_span_end` | int | full_context 내 끝 위치 |
| `snippet` | string | 근거 포함 문자열 |

---
