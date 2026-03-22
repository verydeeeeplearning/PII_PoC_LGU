# Decision Combiner 종합 검증 리포트

**작성일:** 2026-03-22
**목적:** `decision_combiner_analysis.md`(이하 "원본 분석")의 주장을 코드·데이터 근거로 독립 검증하고, 누락된 관점을 보완한 종합 평가

---

## 1. 원본 분석 주장별 검증 결과

### 1.1 "ML 단독 F1=0.78, DC 적용 후 ~0.78" — 근거 불충분

| 항목 | 원본 주장 | 코드 검증 결과 | 판정 |
|------|----------|---------------|------|
| ML F1-macro | 0.78 | `outputs/classification_report.txt` → **F1-macro=0.9676** (test 31건: FP 15, TP 16) | **검증 불가** |
| DC F1 | ~0.78 | run_report.py의 벡터화 평가 결과는 별도 저장 안 됨 | **검증 불가** |

**문제점:**
- 현재 `threshold_policy.json`의 `f1_macro: 1.0`과 `classification_report.txt`의 0.9676은 **mock/dummy 데이터 기반** (총 31건). 실 서버 10M행에서의 0.78은 이 저장소 산출물로는 재현할 수 없다.
- 원본 분석이 참조한 "temporal split, ~400만 행"은 실 서버 환경에서의 결과로 추정되며, 로컬 저장소의 mock 데이터와는 다른 맥락이다.
- **결론:** 0.78이라는 수치 자체는 검증 불가하나, DC가 개선을 못 하는 **구조적 원인** 분석은 코드로 확인 가능하다.

### 1.2 "RULE 커버리지 < 0.5%" — 구조적으로 정확

`config/rule_stats.json` 검증 결과:

| 룰 유형 | 룰 수 | N>0 (실적 있음) | N=0 (실적 없음) | Phase 1 동작 |
|---------|-------|----------------|----------------|-------------|
| **L1 (도메인)** | 3 | 2개 (INTERNAL N=50, OS N=50) | 1개 (DUMMY N=0) | 부분 동작 |
| **L2 (정규식)** | 6 | 0개 | **6개 전부** | **전혀 동작 안 함** |
| **L3 (피처조건)** | 7 | 5개 (LICENSE N=50, DOCKER N=50, DEVICE N=30, MASS N=30, HADOOP N=20, ROTATION N=30) | 2개 (CRON N=0, TEMP_DEV N=0) | 부분 동작 |

**L2 비활성 코드 확인** (`rule_labeler.py:267-268`):
```python
if rule.get("requires_context", False) and not text:
    return None  # full_context_raw 없으면 즉시 종료
```
`rules.yaml`에서 L2 6개 룰 모두 `requires_context: true` → Phase 1에서 100% skip. **원본 분석 정확.**

**Easy FP Suppressor 수치** (`threshold_policy.json`):
- test set에서 3건 / 31건 = 9.7% (mock 데이터)
- 원본 분석의 "Train easy FP: ~9,500건 / 6,086,734건 (0.16%)"은 실 서버 결과로, 로컬에서 재현 불가

**판정:** RULE 커버리지가 극히 낮다는 구조적 주장은 **정확**. 구체적 수치(0.2%, 0.5%)는 실 데이터 의존.

### 1.3 "RULE-ML 신호 중복" — 완전히 정확

`run_training.py:440-466` `_KEEP_COLS` 검증:

| L3 룰 피처 | ML 입력 포함 여부 | 코드 위치 |
|-----------|-----------------|----------|
| `is_system_device` | **포함** | line 451 |
| `is_docker_overlay` | **포함** | line 450 |
| `has_license_path` | **포함** | line 451 |
| `is_package_path` | **포함** | line 452 |
| `is_log_file` | **포함** | line 450 |
| `has_cron_path` | **포함** | line 452 |

L3 룰이 사용하는 **모든 피처가 ML 입력에 동일하게 포함**되어 있다. ML은 이 피처들의 비선형 조합까지 학습하므로, L3 룰이 잡는 케이스를 ML이 이미 잡고 있을 가능성이 매우 높다. **원본 분석 정확.**

### 1.4 "보수적 임계값으로 N=0 룰 Case 1 미진입" — 정확

`decision_combiner.py:71`:
```python
if rule_matched and rule_conf >= thr["rule_conf"]:  # 0.85
```

`rule_stats.json`에서 N=0인 룰 → `precision_lb=0.500` (Beta prior) → 0.85 미달 → Case 1 진입 불가.

해당 룰: `L1_DOMAIN_DUMMY_001`, `PATH_CRON_LOG_001`, `PATH_TEMP_DEV_001` → **3개 룰이 매칭되어도 최종 판정에 영향 없음**. **원본 분석 정확.**

### 1.5 "Case별 비율: 99.5%가 Case 2" — 구조적으로 타당

코드 구조상:
1. Case 0 (OOD): `ood_flag` + 고엔트로피 조건 → 극소수
2. Case 1 (RULE 고확신): 커버리지 < 0.5% × confidence ≥ 0.85 필터 → 극소수
3. Case 2 (ML): 나머지 전부 → 압도적 다수
4. Case 3 (Fallback): ML도 없는 경우 → 사실상 0%

**구조적으로 99%+ 가 Case 2라는 주장은 타당.** 정확한 비율은 실 데이터 필요.

---

## 2. 원본 분석이 누락한 문제점 (독립 발견)

### 2.1 [심각] run_report.py 평가 로직 ≠ decision_combiner.py 추론 로직

**이것이 원본 분석에서 완전히 빠진 가장 중요한 문제다.**

| 항목 | `decision_combiner.py` (추론) | `run_report.py:451` (평가) |
|------|------------------------------|---------------------------|
| RULE 임계값 | `rule_conf >= 0.85` | `_rule_conf >= 0.5` |
| Case 분기 | 4-Case (OOD→RULE→ML→Fallback) | 2-Case (RULE vs ML) |
| TP override | `ml_tp_proba >= 0.60` | `_ml_proba > 0.6` (동일) |
| ambiguous ML | `ml_tp_proba >= 0.40 → TP` | **없음** |
| OOD 처리 | `ood_flag → UNKNOWN` | **없음** |
| entropy 처리 | `entropy >= 2.5 → UNKNOWN` | **없음** |
| risk_flag | 5종 (bool/str) | **없음** |

**결과:** 학습 후 평가(run_report.py)와 실제 추론(run_inference.py → decision_combiner.py)이 **다른 로직으로 판정**한다. 리포트의 DC 성능 수치는 추론 시 실제 성능과 다를 수 있다.

특히 `run_report.py`에서 RULE 임계값을 0.5로 낮춰놓아서, **평가 시에는 RULE Case 1 진입률이 추론 시보다 높게 나온다.** 이는 DC 성능을 과대평가하는 방향의 오류다.

### 2.2 [심각] threshold_policy.json — 생성만 되고 추론에 미반영

| 아티팩트 | 생성 | 추론 시 사용 | 상태 |
|---------|------|------------|------|
| `recommended_fp_tau` | `run_training.py` Step 6c | `combine_decisions()`에서 참조 안 함 | **사문서** |
| `slice_thresholds` | `run_training.py` Step 6c | `run_inference.py`에서 로드 안 함 | **사문서** |
| `easy_fp_suppressor` | `run_training.py` Step 6c | `combine_decisions()`에 로직 없음 | **사문서** |

**결과:** Step 6c에서 최적 tau를 계산하고, 서버 환경별 임계값을 분리 저장해도, 추론 파이프라인은 항상 하드코딩된 `_DEFAULT_THRESHOLDS`만 사용한다. threshold_policy.json은 리포트 시각화용으로만 소비된다.

### 2.3 [심각] 아키텍처 문서와 구현의 광범위한 괴리

| 아키텍처 컴포넌트 | 구현 상태 | 구현률 |
|-----------------|----------|--------|
| `combine_decisions()` 핵심 4-Case | 구현됨 (단, entropy 임계값 불일치: 아키텍처 1.5 vs 코드 2.5) | 85% |
| `AutoAdjudicator` 4단 판정 | Step 1(majority vote)만 구현 | **25%** |
| `UnknownAutoProcessor` | **미구현** (코드 0줄) | **0%** |
| `AutoTuner` 자동 임계값 최적화 | **미구현** (코드 0줄) | **0%** |
| `simulate_combiner()` 시뮬레이션 | **미구현** | **0%** |
| `Auto-Rule-Promoter` 자동 룰 승격 | **미구현** | **0%** |
| Dynamic TP override (rule_conf 기반 차등) | **미구현** (flat 0.60 사용) | **0%** |

**원본 분석에서는 이 괴리를 "4.3 장기 — Auto-Rule-Promoter"에서만 간략히 언급했으나, 실제로는 아키텍처 §9.5~9.7의 핵심 컴포넌트 대부분이 미구현 상태다.**

### 2.4 [중간] 현재 테스트 데이터의 대표성 부재

`threshold_policy.json`:
- `auto_fp_count: 12`, `f1_macro: 1.0` — mock 데이터 31건 기반
- `slice_thresholds: {}` — server_env 다양성 없음 (빈 dict)
- Coverage-Precision Curve: 모든 tau에서 coverage=1.0, precision=1.0

**이 수치로는 어떤 DC 개선안의 효과도 측정할 수 없다.** 실 데이터(10M행)에서의 평가가 선행되어야 한다.

### 2.5 [중간] rule_stats.json의 N값이 수동 기입 추정

L1, L3 룰의 N값이 20, 30, 50 같은 라운드 넘버로, 실제 데이터에서 자동 집계된 것이 아니라 **수동 추정치**일 가능성이 높다. 이는 Bayesian confidence_lb의 신뢰성을 약화시킨다.

---

## 3. 원본 분석의 개선 방안 평가

### 3.1 "§4.1 단기 — RULE 커버리지 확대" 평가

| 제안 | 타당성 | 실효성 | 리스크 |
|------|--------|--------|--------|
| **A) L2 → file_name/path 대안 룰** | 높음 | **중간** — 이미 ML 입력에 동일 피처 포함이므로 신호 중복 문제는 해결 안 됨 | 낮음 |
| **B) L3 피처 조건 룰 추가** | 높음 | **낮음~중간** — 같은 신호 중복 문제. 커버리지 5→10%로 올려도 ML이 이미 학습한 패턴 | 낮음 |
| **C) rule_stats.json 수동 업데이트** | 높음 | **높음** — N=0인 3개 룰의 Case 1 진입을 즉시 가능하게 함 | 중간 (잘못된 N/M → 오분류) |

**핵심 지적:** 원본 분석은 "RULE 커버리지 확대"를 최우선으로 제안하지만, **RULE-ML 신호 중복 문제가 해결되지 않는 한 RULE 커버리지를 올려도 DC의 추가 가치는 제한적**이다.

RULE이 ML에 실질적 가치를 더하려면:
1. **ML이 보지 못하는 신호**를 사용하는 룰을 만들거나 (예: full_context_raw 기반 L2 → Phase 2)
2. **ML의 오분류 패턴을 분석**하여 ML이 틀리는 지점에 특화된 룰을 만들어야 한다

단순히 같은 피처로 룰 개수만 늘리면 "ML이 이미 맞추는 건"을 룰도 맞추는 것에 불과하다.

### 3.2 "§4.2 중기 — Sumologic JOIN 데이터" 평가

**가장 실효성 높은 제안.** full_context_raw가 존재하면:
- L2 6개 룰 활성화 → ML이 보지 못하는 **텍스트 패턴** 신호 추가
- 이는 ML 입력 피처와 겹치지 않는 **독립적 신호**이므로 DC의 실질적 가치 발생

다만 `--source detection` 파이프라인이 필요하며, 실 서버에서의 JOIN 성공률(pk_file 일치율)이 관건이다.

### 3.3 "§4.3 장기 — Auto-Rule-Promoter" 평가

아키텍처 §7.8에 설계만 있고 미구현. 컨셉은 타당하나:
- **전제조건:** ML 예측의 고확신 패턴이 반복적으로 나타나야 함
- **리스크:** 잘못된 패턴이 룰로 승격되면 FP→TP 오분류 자동 확대
- **우선순위:** 다른 기반 작업(threshold 연동, 평가 로직 통일) 이후가 적절

---

## 4. 종합 평가 및 권장 사항

### 4.1 원본 분석의 강점

1. **핵심 원인 진단이 정확** — RULE 커버리지 < 0.5%, L2 비활성, 신호 중복은 모두 코드로 확인됨
2. **Case별 비율 추정이 구조적으로 타당** — 99.5% Case 2라는 결론은 코드 구조에서 자연스럽게 도출
3. **단기/중기/장기 구분이 실용적** — Sumologic JOIN을 중기로 배치한 것은 적절

### 4.2 원본 분석의 약점

1. **F1 수치 검증 불가** — 0.78이라는 핵심 수치의 출처/재현 경로가 없음
2. **평가↔추론 로직 불일치를 간과** — DC 성능이 정확히 측정되고 있는지 자체가 의문
3. **아키텍처 미구현 범위를 과소 보고** — Auto-Rule-Promoter만 언급, 실제로는 5개 컴포넌트 미구현
4. **신호 중복 문제의 해결책이 부족** — "RULE 커버리지 확대"를 권장하면서 동일 피처 사용 문제를 근본적으로 해결하지 않음
5. **mock 데이터 한계 미언급** — threshold_policy.json의 수치가 mock 31건 기반이라는 점이 빠져 있음

### 4.3 실행 권장 순서

**즉 코드베이스에서 확인된 실제 문제를 기준으로 우선순위를 재정립한다.**

#### Phase A: 측정 기반 정비 (선행 필수)

DC를 개선하기 전에, **DC의 성능을 정확히 측정할 수 있는 환경**부터 만들어야 한다.

| # | 작업 | 이유 | 난이도 |
|---|------|------|--------|
| A1 | **run_report.py 벡터화 로직을 `combine_decisions()` 호출로 대체** | 평가↔추론 로직 일치시켜야 DC 성능 수치를 신뢰할 수 있음 | 중간 |
| A2 | **threshold_policy.json을 추론 시 로드하여 `combine_decisions(thresholds=...)` 전달** | 학습된 tau가 실제 판정에 반영되어야 Step 6c의 의미가 있음 | 낮음 |
| A3 | **실 데이터 기반 rule_stats.json 자동 갱신 파이프라인** | N=0 룰 해소 + 수동 추정치 대체 → Bayesian confidence 신뢰성 확보 | 중간 |

#### Phase B: DC 로직 고도화 (A 완료 후)

| # | 작업 | 기대 효과 | 난이도 |
|---|------|----------|--------|
| B1 | **ML 후처리 3-Zone Decision** (원본 분석 미제안) — 중확신 구간(0.60~0.80)에서 entropy/margin 보조 신호로 FP 구제 | FP Recall 개선 (ML 자체 성능 향상) | 중간 |
| B2 | **File-Level Consensus 전파** — 동일 pk_file 내 고확신 판정을 저확신 이벤트에 전파 | NEEDS_REVIEW 비율 감소 | 중간 |
| B3 | **Dynamic TP override 구현** — 아키텍처 §9.2의 rule_conf 구간별 차등 임계값 | RULE↔ML 충돌 해소 정밀도 향상 | 낮음 |

#### Phase C: RULE 독립 가치 확보 (B와 병행 가능)

| # | 작업 | 기대 효과 | 난이도 |
|---|------|----------|--------|
| C1 | **ML 오분류 패턴 분석 → 특화 룰 생성** — ML이 틀리는 케이스를 분석하여, ML이 학습 못 한 패턴을 룰로 보완 | DC의 실질적 추가 가치 | 높음 (데이터 분석 필요) |
| C2 | **Sumologic JOIN 활성화** (원본 §4.2) — L2 룰 6개 활성화로 텍스트 기반 독립 신호 확보 | RULE 커버리지 대폭 확대 | 중간 (인프라 의존) |
| C3 | **미사용 피처 기반 L3 룰 추가** — `has_backup_path`, `has_database_path`, `has_cicd_path`, `digit_ratio`, `max_digit_run` 등 path_features.py에 추출되지만 _KEEP_COLS에 미포함된 피처 활용 | ML과 중복되지 않는 독립 신호 | 낮음 |

#### Phase D: 아키텍처 미구현 컴포넌트 (C 이후)

| # | 작업 | 기대 효과 | 난이도 |
|---|------|----------|--------|
| D1 | **AutoAdjudicator Step 2~3 구현** (교차 합의 + 피처 유사도) | NEEDS_REVIEW 자동 해소율 향상 | 중간 |
| D2 | **AutoTuner 경량 버전** — ml_conf × ml_margin 2차원 탐색 | 데이터 분포에 맞는 최적 임계값 자동 발견 | 중간 |
| D3 | **UnknownAutoProcessor** — OOD 케이스 자동 처리 | UNKNOWN 큐 제거 (Zero-Human 원칙 충족) | 중간 |
| D4 | **Auto-Rule-Promoter** (원본 §4.3) | 장기 자동화 | 높음 |

---

## 5. 핵심 결론

### 5.1 원본 분석은 "DC가 왜 안 되는가"를 잘 설명했지만, "DC를 어떻게 고쳐야 하는가"에서 근본 원인을 놓쳤다

원본 분석의 처방은 "RULE 커버리지 확대"인데, 이는 **증상 치료**다. 같은 피처를 쓰는 룰을 추가해봐야 ML과 동일한 결론을 내므로 DC의 추가 가치는 미미하다.

**근본 치료:**
1. RULE이 ML과 **다른 신호**를 봐야 한다 (full_context_raw, 미사용 피처)
2. DC가 **양쪽의 불확실성을 정량적으로 결합**해야 한다 (hard switch → soft vote)
3. ML이 **틀리는 지점**에 RULE을 배치해야 한다 (오분류 패턴 특화)

### 5.2 그러나 그 어떤 개선보다 "측정 정비"가 먼저다

현재 상태에서는:
- 평가(run_report.py)와 추론(decision_combiner.py)이 **다른 로직**을 사용
- threshold_policy.json이 **추론에 반영되지 않음**
- 테스트 데이터가 **31건 mock** 기반

**정확한 측정 없이는 어떤 개선안도 효과를 검증할 수 없다.** Phase A가 모든 후속 작업의 전제조건이다.

### 5.3 현실적 PoC 관점

원본 분석의 마지막 문장은 올바르다:

> "PoC 관점 권장: 현재 ML 단독 F1=0.78이 목표(0.70) 초과. Decision Combiner는 RULE 커버리지 확대 후 고도화 단계에서 추가 개선 수단으로 활용."

ML 단독 성능이 목표를 이미 초과한 상황이라면, DC 고도화의 ROI는 다른 작업(데이터 품질 개선, 피처 추가, 모델 튜닝) 대비 낮을 수 있다. DC 고도화는 **ML 성능이 천장에 부딪힌 후** 추가 개선 수단으로 투입하는 것이 합리적이다.

단, **Phase A(측정 정비)만큼은 DC 고도화와 무관하게 즉시 수행해야 한다.** 평가↔추론 로직 불일치는 시스템 전체의 신뢰성 문제이기 때문이다.

---

*END OF DOCUMENT*
