## 9. Stage S4: Decision Combiner

### 9.1 기능

RULE 라벨러(S3a)와 ML 라벨러(S3b)의 결과를 결합하여, **최종 라벨 + 신뢰도 + risk_flag**를 확정한다. 이 단계에서 "TP를 FP로 보내는 리스크 최소화" 정책을 코드 레벨에서 강제한다.

### 9.2 결합 규칙

```python
# 임계값 (하이퍼파라미터, config로 관리)
# Wave 4 C5: threshold_policy.json에서 자동 로드 (models/final/threshold_policy.json)
# 아래는 fallback 기본값. run_training.py Step 6c에서 Coverage-Precision Curve 기반으로
# 최적 tau를 계산하여 threshold_policy.json에 저장.
# Wave 5 C2: slice_thresholds (server_env별 tau) 추가 저장
# Wave 5 C1: easy_fp_suppressor 조건도 threshold_policy.json에 포함
# run_report.py는 threshold_policy.json을 우선 로드하여 재계산 없이 사용
TAU_RULE_CONF = 0.85       # 룰 신뢰도 임계값
TAU_ML_CONF = 0.70         # ML 확신 임계값
TAU_ML_MARGIN = 0.20       # ML 마진 임계값
TAU_ML_ENTROPY = 1.5       # ML 엔트로피 임계값
TAU_TP_OVERRIDE = 0.40     # TP 확률 override 임계값

def combine_decisions(rule_result: dict, ml_result: dict, tau_config: dict = None) -> dict:
    """
    Decision Combiner 로직.
    tau_config: 임계값 오버라이드 dict (simulate_combiner 등에서 전달). None이면 모듈 상단 기본값 사용.

    핵심 원칙: "룰도 모델도 틀릴 수 있다"를 인정하고,
    운영 원칙(애매하면 TP)을 시스템이 자동으로 지키게 만든다.
    """

    # === Case 0 (v1.1 추가): OOD 감지 또는 극도의 불확실 → UNKNOWN ===
    is_ood = (
        ml_result and (
            ml_result.get('ood_flag', False) or
            (ml_result['ml_top1_proba'] < 0.35 and
             rule_result is None and
             ml_result['ml_entropy'] > 2.0)
        )
    )

    if is_ood:
        return {
            'primary_class': 'UNKNOWN',
            'reason_code': 'OOD_OR_EXTREME_UNCERTAINTY',
            'decision_source': 'REJECT',
            'confidence': 0.0,
            'confidence_type': 'none',
            'risk_flag': 'OOD_SUSPECTED',
        }

    # === Case 1: RULE 라벨이 존재하고 신뢰도가 높은 경우 ===
    if rule_result and rule_result['rule_confidence_lb'] >= TAU_RULE_CONF:

        # v1.1 개선: RULE confidence에 따라 TP override 조건을 차등 적용
        # 기존 TAU_TP_OVERRIDE = 0.40은 RULE 고확신 FP까지 빈번하게 override할 위험
        if rule_result['rule_confidence_lb'] >= 0.90:
            tp_override_threshold = 0.60  # RULE 고확신 → ML override를 어렵게
        elif rule_result['rule_confidence_lb'] >= 0.80:
            tp_override_threshold = 0.50  # RULE 중확신 → 중간 임계값
        else:
            tp_override_threshold = TAU_TP_OVERRIDE  # RULE 저확신 → 기존 임계값 유지

        # ML이 TP 위험을 강하게 제기하면 리뷰로 올림
        # Wave 3 수정: ml_tp_proba는 TP 클래스의 보정 확률 (LabelEncoder 동적 인덱스)
        if ml_result and ml_result['ml_tp_proba'] >= tp_override_threshold:
            return {
                'primary_class': 'TP-실제개인정보',  # 보수적으로 TP 처리
                'reason_code': 'RULE_ML_CONFLICT',
                'decision_source': 'HYBRID_OVERRIDE',
                'confidence': ml_result['ml_tp_proba'],
                'confidence_type': 'ml_calibrated_proba',
                'risk_flag': 'RULE_CONFLICT_WITH_ML',
            }

        # RULE 라벨 확정
        return {
            'primary_class': rule_result['rule_primary_class'],
            'reason_code': rule_result['rule_reason_code'],
            'decision_source': f"RULE_{rule_result['rule_id'][:2]}",
            'confidence': rule_result['rule_confidence_lb'],
            'confidence_type': 'rule_precision_lb_95',
            'risk_flag': None,
        }

    # === Case 2: RULE이 없거나 신뢰도가 낮은 경우 → ML 사용 ===
    if ml_result:
        ml_class = ml_result['ml_top1_class_name']
        ml_conf = ml_result['ml_top1_proba']
        ml_margin = ml_result['ml_margin']
        ml_entropy = ml_result['ml_entropy']
        
        # 애매함 감지 → TP로 보수적 override
        is_ambiguous = (
            ml_conf < TAU_ML_CONF or
            ml_margin < TAU_ML_MARGIN or
            ml_entropy > TAU_ML_ENTROPY
        )
        
        if ml_class != 'TP-실제개인정보' and is_ambiguous:
            return {
                'primary_class': 'TP-실제개인정보',
                'reason_code': 'AMBIGUOUS_ML_PREDICTION',
                'decision_source': 'ML_L3',
                'confidence': ml_conf,
                'confidence_type': 'ml_calibrated_proba',
                'risk_flag': 'TP_SAFE_OVERRIDE',
            }
        
        # ML 라벨 확정
        return {
            'primary_class': ml_class,
            'reason_code': ml_result.get('ml_reason_code', ml_class),
            'decision_source': 'ML_L3',
            'confidence': ml_conf,
            'confidence_type': 'ml_calibrated_proba',
            'risk_flag': 'NEEDS_REVIEW' if ml_conf < 0.85 else None,
        }
    
    # === Case 3: RULE도 ML도 없음 (fallback) ===
    return {
        'primary_class': 'TP-실제개인정보',
        'reason_code': 'NO_CLASSIFICATION_AVAILABLE',
        'decision_source': 'FALLBACK',
        'confidence': 0.0,
        'confidence_type': 'none',
        'risk_flag': 'TP_SAFE_OVERRIDE',
    }
```

### 9.3 임계값 시뮬레이션 프레임워크 (v1.1 추가)

5개 임계값의 상호작용이 복잡하므로, 개별 튜닝이 아닌 전체 Decision Combiner 출력 분포를 시뮬레이션하는 프레임워크를 추가한다.

```python
def simulate_combiner(rule_labels, ml_predictions, tau_config):
    """
    임계값 조합의 전체 효과를 시뮬레이션.
    출력: 클래스별 비율, risk_flag 비율, 추정 정확도
    """
    results = []
    for _, (rule, ml) in enumerate(zip(rule_labels, ml_predictions)):
        decision = combine_decisions(rule, ml, tau_config)
        results.append(decision)

    return {
        'class_distribution': Counter(r['primary_class'] for r in results),
        'risk_flag_distribution': Counter(r['risk_flag'] for r in results),
        'auto_process_rate': sum(1 for r in results if r['risk_flag'] is None) / len(results),
        'review_rate': sum(1 for r in results if r['risk_flag'] in ['NEEDS_REVIEW', 'TP_SAFE_OVERRIDE', 'OOD_SUSPECTED']) / len(results),
    }
```

### 9.4 Rationale

**"TP를 FP로 보내면 안 된다"를 말로만 두면 사고 난다. 코드로 강제해야 한다.**

- 프로젝트의 보수적 원칙("애매한 경우 정탐 처리")은 모델의 목적함수(loss function)와 별개의 **운영 정책**이다. 모델은 최적의 확률을 내고, 정책은 후처리에서 명시적으로 구현해야 한다.
- confidence, margin, entropy 세 가지 지표를 조합하면, "진짜 확신" vs "가짜 확신"을 효과적으로 구분할 수 있다. 단일 지표(max_proba)만 쓰면 놓치는 케이스가 있다.

**RULE과 ML이 동시에 존재하는 이유:**

- RULE은 설명이 완전하고 비용이 낮지만, 커버리지에 한계가 있다 (새로운 패턴, 교집합 영역).
- ML은 피처 간 상호작용을 자동으로 학습해 "룰의 틈새"를 잡는 역할을 한다.
- 둘을 결합하면 "룰이 틀릴 수 있는 영역"(RULE_CONFLICT_WITH_ML)을 조기에 잡아내어 리스크를 줄인다. 이것은 룰의 품질 개선에도 직접적으로 기여한다.

**risk_flag를 데이터로 남기는 이유:**

- 운영팀이 "왜 이건 리뷰로 갔는지"를 설명해야 하고, 다음 개선(룰 강화/모델 재학습)에서 "어떤 종류의 애매함이 늘었는지"를 추적해야 한다.
- risk_flag의 분포를 월별로 모니터링하면, 시스템의 건강도를 파악할 수 있다 (NEEDS_REVIEW 비율 증가 = 모델 드리프트 신호).

### 9.5 Auto-Adjudicator — NEEDS_REVIEW 자동 판정 (v1.2 추가)

원칙 G(Zero-Human-in-the-Loop)에 따라, `risk_flag=NEEDS_REVIEW` 케이스를 사람이 검토하는 대신, 4단 자동 판정 로직으로 처리한다.

```python
class AutoAdjudicator:
    """
    NEEDS_REVIEW 케이스를 4단 자동 판정으로 처리.
    최종 fallback은 보수적 TP 처리 (안전).
    """

    def adjudicate(self, event: dict, file_events: pd.DataFrame,
                   rule_result: dict, ml_result: dict,
                   historical_db: 'HistoricalPatternDB') -> dict:

        # Step 1: File-level Consensus (파일 전파)
        if self._check_file_consensus(event, file_events):
            return {
                'primary_class': event['file_consensus_class'],
                'reason_code': 'AUTO_FILE_CONSENSUS',
                'risk_flag': 'AUTO_ADJUDICATED',
                'adjudication_step': 1,
            }

        # Step 2: RULE↔ML Agreement (교차 합의)
        if self._check_cross_agreement(rule_result, ml_result):
            agreed_class = rule_result['rule_primary_class']
            return {
                'primary_class': agreed_class,
                'reason_code': 'AUTO_CROSS_AGREEMENT',
                'risk_flag': 'AUTO_ADJUDICATED',
                'adjudication_step': 2,
            }

        # Step 3: Historical Pattern Match (과거 패턴 유사도)
        hist_match = historical_db.find_nearest(event['dense_features'])
        if hist_match and hist_match['distance'] < hist_match['threshold']:
            return {
                'primary_class': hist_match['class'],
                'reason_code': 'AUTO_HISTORICAL_MATCH',
                'risk_flag': 'AUTO_ADJUDICATED',
                'adjudication_step': 3,
            }

        # Step 4: Conservative Default (보수적 기본값)
        # 위 3단계를 모두 통과하지 못하면 → TP로 확정 (안전)
        return {
            'primary_class': 'TP-실제개인정보',
            'reason_code': 'AUTO_CONSERVATIVE_TP',
            'risk_flag': 'AUTO_TP_CONSERVATIVE',
            'adjudication_step': 4,
        }

    def _check_file_consensus(self, event, file_events) -> bool:
        """같은 pk_file 내 다른 이벤트들의 고확신 FP 합의 확인."""
        same_file = file_events[
            (file_events['pk_file'] == event['pk_file']) &
            (file_events['confidence'] >= 0.85) &
            (file_events['risk_flag'].isna())  # 이미 확정된 건들
        ]
        if len(same_file) < 3:
            return False
        consensus_class = same_file['primary_class'].mode()
        if len(consensus_class) == 1 and consensus_class[0].startswith('FP-'):
            event['file_consensus_class'] = consensus_class[0]
            return True
        return False

    def _check_cross_agreement(self, rule_result, ml_result) -> bool:
        """RULE과 ML이 동일 FP 클래스를 지목하고 합산 confidence 충분한지 확인."""
        if not rule_result or not ml_result:
            return False
        if rule_result['rule_primary_class'] != ml_result['ml_top1_class_name']:
            return False
        combined_conf = (rule_result['rule_confidence_lb'] +
                         ml_result['ml_top1_proba']) / 2
        return combined_conf >= 0.75
```

**핵심:** "사람이 검토"를 "시스템이 보수적으로 TP 처리"로 대체. 실제 개인정보를 놓치지 않으므로 안전하다.

### 9.6 UNKNOWN (OOD) 자동 처리 (v1.2 추가)

원칙 G에 따라, UNKNOWN 케이스도 사람 검토 큐가 아닌 자동 처리로 전환한다.

```python
class UnknownAutoProcessor:
    """
    UNKNOWN (OOD) 케이스를 자동 처리.
    보수적 원칙: FP에 가깝지 않으면 TP로 확정.
    """
    DISTANCE_THRESHOLD = 2.0  # Mahalanobis 거리 임계값

    def process(self, event: dict,
                class_centroids: dict,
                auto_rule_promoter: 'AutoRulePromoter') -> dict:

        # Step 1: Nearest-Class Assignment
        nearest = self._find_nearest_class(event['dense_features'],
                                           class_centroids)
        if (nearest['class'].startswith('FP-') and
                nearest['distance'] < self.DISTANCE_THRESHOLD):
            return {
                'primary_class': nearest['class'],
                'reason_code': 'OOD_AUTO_ASSIGN',
                'confidence': 'low',
                'risk_flag': 'OOD_AUTO_ASSIGNED',
            }

        # Step 2: Conservative TP Default
        return {
            'primary_class': 'TP-실제개인정보',
            'reason_code': 'OOD_UNKNOWN_TO_TP',
            'confidence': 0.0,
            'risk_flag': 'OOD_CONSERVATIVE_TP',
        }

    def _find_nearest_class(self, features, centroids) -> dict:
        """각 클래스 centroid와의 거리 비교."""
        min_dist, nearest_class = float('inf'), None
        for cls, centroid in centroids.items():
            dist = np.linalg.norm(features - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_class = cls
        return {'class': nearest_class, 'distance': min_dist}
```

**UNKNOWN 자동 처리 흐름:**

```
Step 1: 학습 데이터 클래스별 centroid와 거리 비교
  → 가장 가까운 FP 클래스 AND 거리 임계값 이내
  → 해당 FP 클래스로 자동 배정 (confidence=low)

Step 2: 어떤 FP에도 가깝지 않으면 → TP로 자동 확정 ("모르면 TP")

Step 3: UNKNOWN→auto-assign 누적 패턴 → Auto-Rule-Promoter(§7.6)에 전달
  → 동일 패턴 500건+ 축적 시 자동 룰 생성
```

### 9.7 Auto-Tuner — 임계값 자동 최적화 (v1.2 추가)

원칙 G에 따라, TAU 임계값을 사람이 실험적으로 조정하는 대신, Rolling validation 기반으로 자동 최적화한다.

```python
from itertools import product

class AutoTuner:
    """
    5개 임계값(TAU)을 자동으로 최적화.
    목적: auto_process_rate 최대화 (안전 제약 하에서).
    """
    SEARCH_SPACE = {
        'TAU_RULE_CONF': [0.80, 0.85, 0.90, 0.95],
        'TAU_ML_CONF': [0.60, 0.65, 0.70, 0.75, 0.80],
        'TAU_ML_MARGIN': [0.15, 0.20, 0.25, 0.30],
        'TAU_ML_ENTROPY': [1.0, 1.5, 2.0],
        'TAU_TP_OVERRIDE': [0.35, 0.40, 0.45, 0.50],
    }

    CONSTRAINTS = {
        'tp_recall': 0.95,        # 최소 TP Recall
        'fp_precision_est': 0.90, # 최소 추정 FP Precision
        'review_rate': 0.30,      # 최대 리뷰율
    }

    def optimize(self, rule_labels, ml_predictions,
                 validation_data, current_tau: dict) -> dict:
        """
        Grid Search로 최적 TAU 조합을 찾는다.
        안전장치: 새 TAU가 TP_recall을 하락시키면 이전 TAU 유지.
        """
        best_tau = current_tau
        best_auto_rate = -1
        current_scores = self._evaluate(rule_labels, ml_predictions,
                                        validation_data, current_tau)

        for combo in product(*self.SEARCH_SPACE.values()):
            tau_candidate = dict(zip(self.SEARCH_SPACE.keys(), combo))
            scores = self._evaluate(rule_labels, ml_predictions,
                                    validation_data, tau_candidate)

            # 안전 제약 확인
            if not self._meets_constraints(scores):
                continue

            # 안전장치: TP_recall 하락 금지
            if scores['tp_recall'] < current_scores['tp_recall']:
                continue

            if scores['auto_process_rate'] > best_auto_rate:
                best_auto_rate = scores['auto_process_rate']
                best_tau = tau_candidate

        # 변경 이력 자동 기록
        self._log_tau_change(current_tau, best_tau, best_auto_rate)

        return best_tau

    def _meets_constraints(self, scores) -> bool:
        return (scores['tp_recall'] >= self.CONSTRAINTS['tp_recall'] and
                scores['fp_precision_est'] >= self.CONSTRAINTS['fp_precision_est'] and
                scores['review_rate'] <= self.CONSTRAINTS['review_rate'])
```

**Auto-Tuner 흐름:**

```
1. 매월 실행 후, 최근 3개월 검증 데이터에서 Grid Search
2. 목적함수: maximize(auto_process_rate)
   subject to:
     TP_recall ≥ 0.95
     estimated_fp_precision ≥ 0.90
     review_rate ≤ 0.30
3. 안전장치:
   - 새 TAU로 전환 시 이전 대비 TP_recall 하락 금지 → 하락 시 이전 TAU 유지 (롤백)
   - 변경 이력 자동 기록
```

---

## 10. Stage S5: Output Writer

### 10.1 기능

Decision Combiner의 결과를 최종 운영 출력 형태로 구성한다. **테이블 2개**로 구성하여 "결론"과 "근거 상세"를 분리한다.

### 10.2 predictions_main (1행=1검출, 결론)

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `pk_event` | string | 검출 이벤트 ID | `a3f2c8d1e9b0` |
| `pk_file` | string | 파일 단위 ID | `7b4e1a9c3d` |
| `server_name` | string | 서버명 (원본 매핑용) | `SVR-001` |
| `agent_ip` | string | 에이전트 IP | `172.21.56.48` |
| `file_path` | string | 파일 경로 | `/var/log/hadoop/...` |
| `file_name` | string | 파일명 | `hdfs-datanode.log` |
| `pii_type_inferred` | string | 재추론된 PII 유형 | `email` |
| `primary_class` | string | 최종 라벨 (7클래스(6 FP+1 TP) + UNKNOWN) | `FP-이메일패턴` |
| `reason_code` | string | 세부 사유코드 (7+알파) | `INT_DOMAIN_LGUPLUS` |
| `decision_source` | string | 결정 출처 | `RULE_L1` |
| `confidence` | float | 신뢰도 (0~1) | `0.97` |
| `confidence_type` | string | 신뢰도 유형 | `rule_precision_lb_95` |
| `ml_top1_class` | string (nullable) | ML 예측 1순위 | `FP-이메일패턴` |
| `ml_top1_proba` | float (nullable) | ML 1순위 보정 확률 | `0.92` |
| `ml_top2_class` | string (nullable) | ML 예측 2순위 | `TP-실제개인정보` |
| `ml_top2_proba` | float (nullable) | ML 2순위 보정 확률 | `0.04` |
| `ml_margin` | float (nullable) | margin (top1-top2) | `0.88` |
| `ml_entropy` | float (nullable) | 확률 엔트로피 | `0.35` |
| `risk_flag` | string (nullable) | 리스크 플래그 | `None` |
| `ood_flag` | bool | OOD 판정 (엔트로피 임계값 초과 시) | `False` |
| `model_version` | string | 모델 버전 | `v1.0.0` |
| `ruleset_version` | string | 룰셋 버전 | `v2.1` |
| `run_id` | string | 실행 ID | `run_202602_01` |
| `run_date` | datetime | 실행 일시 | `2026-02-15T10:30:00` |

### 10.3 prediction_evidence (N행=1검출, 근거 상세)

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `pk_event` | string | 검출 이벤트 ID | `a3f2c8d1e9b0` |
| `evidence_rank` | int | 근거 우선순위 (1부터) | `1` |
| `evidence_type` | string | 근거 유형 | `RULE_MATCH` |
| `source` | string | 출처 (RULE/ML) | `RULE` |
| `rule_id` | string (nullable) | 룰 ID | `L1_DOMAIN_INTERNAL_001` |
| `feature_name` | string (nullable) | 피처명 | `email_domain` |
| `matched_value` | string | 매칭 값 또는 설명 | `bdp.lguplus.co.kr` |
| `snippet` | string (nullable) | 원문 내 증거 문자열 | `****@bdp.lguplus.co.kr` |
| `matched_span_start` | int (nullable) | full_context 내 시작 위치 | `4` |
| `matched_span_end` | int (nullable) | full_context 내 끝 위치 | `24` |
| `weight_or_contribution` | float (nullable) | 기여도 (SHAP) | `0.15` |

### 10.4 Rationale

**왜 테이블 2개인가 (Main + Evidence):**

- 근거는 1개로 고정되지 않는다. 하나의 검출 건에 대해 "도메인 매칭(RULE)" + "로그 파일 경로(ML)" + "대량 검출 플래그(ML)" 등 여러 근거가 동시에 존재할 수 있다.
- evidence를 main 테이블에 컬럼으로 넣으면 `evidence_1`, `evidence_2`, ... 같은 컬럼 폭발이 발생하고, 근거 개수가 달라질 때 처리가 어렵다.
- long-format evidence 테이블로 분리하면 근거를 원하는 만큼 늘릴 수 있고, 추후 분석("어떤 근거가 가장 자주 쓰였나", "어떤 근거가 오분류와 상관이 높은가")도 가능하다.
- 엑셀/리포트/대시보드에서 main 테이블로 "결론"을, evidence 테이블로 "근거 상세"를 붙여서 보여줄 수 있다.

**왜 pk_event로 연결하는가:**

- PK가 있어야 원본 데이터와 조인이 가능하다 (문제 정의서의 핵심 설계 원칙).
- predictions_main ↔ prediction_evidence ↔ silver_detections(원본) 간의 조인이 모두 pk_event로 이루어지므로, end-to-end 추적이 가능하다.

---
