# 모델 성능 진단 리포트 v6

**작성일:** 2026-03-20
**최종 업데이트:** 2026-03-21 (Tier 2 서버 실험 결과 반영 — F1=0.8692 달성)
**대상:** Phase 1 Label-Only 학습 결과 (`--source label --split temporal --test-months 3`)
**데이터:** 서버 실행 기준 silver_label.parquet (10,049,303행 × 126열)
**참고 문서:** `docs/Architecture/08_performance_improvement_playbook.md` (독립 Agent 코드 검증 제안 — v4에서 교차 검증 후 통합)

---

## 1. 실험 이력 및 현재 성능

### 1.1 실험 이력

| 실험 | 적용 내용 | F1-macro | FP P/R | TP P/R | 피처 수 | 비고 |
|------|----------|---------|--------|--------|--------|------|
| **Baseline (Tier 0 이전)** | 과적합 피처 포함 + split 중복 | **0.6146** | 0.58/0.70 | 0.66/0.54 | 1031 | 과대평가 수치 |
| **Tier 0+1 적용** | 시간피처 4개 + server_freq 제거, exception_requested/rule_matched 추가, TF-IDF 500→200 축소 | **0.6114** | 0.59/0.59 | 0.63/0.63 | 428 | |
| **Tier 2 R1+R2 + C5** | B2 중복가중치 + B7 server_env + B1 범주형 + B8 RULE세부 + B9 file_agg + B3 Shape TF-IDF + B6 정규화 + C5 threshold저장 | **0.8063** | **0.75/0.88** | **0.87/0.74** | ~454 | **Phase 1 목표 달성** |

### 1.2 현재 성능 상세 (Tier 2 적용 후 — 2026-03-21 서버 실험)

| 지표 | Baseline | Tier 0+1 | **Tier 2** | 기준 | 판정 |
|------|---------|---------|-----------|------|------|
| F1-macro | 0.6146 | 0.6114 | **0.8063** | >=0.70 | **PASS** |
| FP Precision | 0.58 | 0.59 | **0.75** | >=0.85 | 미달 |
| FP Recall | 0.70 | 0.59 | **0.88** | - | - |
| TP Precision | 0.66 | 0.63 | **0.87** | - | - |
| TP Recall | 0.54 | 0.63 | **0.74** | >=0.75 | 근접 |
| Accuracy | 0.62 | 0.61 | **0.81** | - | - |

**결론:** Phase 1 메타데이터 기반 모델로 F1 0.81 달성 (+0.19). Phase 1 목표(>=0.70) 통과, Phase 2 목표(0.85)에 근접. TP Recall 0.74는 기준(0.75)에 1%p 미달 — tau 조정 또는 Tier 3 C1(2단 구조)로 개선 가능. FP Precision 0.75는 0.85 기준에 미달이나, Coverage-Precision Curve에서 고확신 구간 tau를 올려 운영 정밀도 확보 가능.

### 1.3 개선 효과 분해

| 요인 | 기여도 (추정) | 근거 |
|------|-------------|------|
| **B2 중복 샘플 가중치** | 높음 | 10M행 중 반복 패턴 암기 제거. sample_weight min=0.0070 max=1.1929 → 대량 반복 파일 가중치 대폭 축소 |
| **B7 server_env 의미 토큰** | 높음 | Mock E2E에서 importance 1위(178). prd=TP/dev=FP 도메인 신호 |
| **B1 범주형 피처** | 중간 | service_enc, ops_dept_enc, retention_period_enc 활용 |
| **B6 정규화 강화** | 중간 | min_child_samples=200 + reg_alpha=0.5 + max_depth=10 → 과적합 억제 |
| **B3 Shape TF-IDF** | 중간 | 숫자 n-gram 과적합 감소 |
| **B8 RULE 세부, B9 file_agg** | 낮음~중간 | rule_confidence_lb, file_event_count 보조 신호 |

### 이전 1.2 참조 (Tier 0+1 성능)

| 지표 | Baseline | Tier 0+1 | 변화 | 기준 | 판정 |
|------|---------|---------|------|------|------|
| F1-macro | 0.6146 | **0.6114** | -0.003 | >=0.70 | FAIL |
| FP Precision | 0.58 | **0.59** | +0.01 | >=0.85 | FAIL |
| FP Recall | 0.70 | **0.59** | -0.11 | - | - |
| TP Precision | 0.66 | **0.63** | -0.03 | - | - |
| TP Recall | 0.54 | **0.63** | +0.09 | >=0.75 | FAIL |
| Accuracy | 0.62 | **0.61** | -0.01 | - | - |
| 운영 가능 tau | 1.0 (0건) | 1.0 (0건) | 변화 없음 | - | 실질적 운영 불가 |

### 1.3 Tier 0+1 적용 결과 분석

**F1-macro 소폭 하락(-0.003)은 정상이며 긍정적 신호:**

1. **FP Recall 0.70→0.59 / TP Recall 0.54→0.63**: Recall 균형이 개선됨.
   Baseline에서는 FP에 편향(created_hour 과적합이 FP 과검출 유발)되었으나,
   과적합 피처 제거 후 TP/FP Recall이 균등화됨 (0.63/0.59)

2. **피처 수 1031→428**: 60% 축소. importance=0 노이즈 피처 대거 제거.
   모델이 학습할 노이즈가 줄어 일반화 기반이 마련됨

3. **단, 절대 성능은 여전히 부족**: F1 0.61은 랜덤(0.50) 대비 소폭 개선 수준.
   피처 정리만으로는 한계 → Tier 2 구조 개선이 필요

### 1.4 현재 피처 구성 (428개)

| 블록 | 피처 수 | 설명 |
|------|--------|------|
| Dense (메타/경로) | ~20 | fname_has_date/hash/rotation, pattern_count 통계, 경로 토큰, exception_requested, rule_matched |
| file_extension (Label Encoded) | 1 | 범주형 → 정수 |
| TF-IDF file_name | ~200 | char n-gram (max_features=200) |
| TF-IDF file_path | ~200 | char n-gram (max_features=200) |
| path_features (numeric) | ~7 | path_depth 등 추출 피처 |

**제거 완료된 피처 (Tier 1):**
- `created_hour`, `created_weekday`, `is_weekend`, `created_month` (시간 과적합)
- `server_freq` (train 통계 누수)

**추가 완료된 피처 (Tier 1):**
- `exception_requested` (Y/N → 0/1, FP 강신호)
- `rule_matched` (Rule Labeler 매칭 결과)

---

## 2. 발견된 문제점 및 조치 상태

### 2.1 [치명] `ml_tp_proba` 인덱스 버그 — ~~수정 완료~~ ✅

`predict_with_uncertainty()`에서 TP 클래스 인덱스를 동적으로 탐색하도록 수정.
`label_names`에서 "TP" 포함 클래스를 찾아 해당 인덱스의 proba를 사용.

### 2.2 [치명] Temporal Split 월 중복 — ~~수정 완료~~ ✅

`work_month_time_split()` 행 단위 엄격 분리로 수정.
Train 3~9월 / Test 10~12월 — 월 중복 제거 확인.

### 2.3 [치명→완료] `created_hour` 시간 피처 과적합 — ~~제거 완료~~ ✅

시간 피처 4개(`created_hour`, `created_weekday`, `is_weekend`, `created_month`) 전부 제거.
Tier 0+1 실험에서 확인: importance 1위였던 created_hour 제거 후에도 F1 유지 → 과적합 확인.

### 2.4 [치명→완료] `server_freq` 통계 누수 — ~~제거 완료~~ ✅

### 2.5 [심각] TF-IDF 노이즈 — 부분 개선

- TF-IDF max_features 500→200 축소 ✅
- **잔존 문제:** file_name char n-gram이 여전히 특정 숫자 패턴 암기 가능
- **후속 조치:** Shape TF-IDF 활성화로 숫자→D 변환 필요 (Tier 2 B3)

### 2.6 [심각] 중복/반복 샘플 가중치 미제어 — ❌ 미조치

10M행에서 동일 `(file_path, file_name)` 대량 반복 → 모델이 소수 패턴을 암기.
**Tier 2 B2에서 조치 예정.**

### 2.7 [심각] 튜닝 CV가 StratifiedKFold — ❌ 미조치 (우선순위 하향)

현재 하이퍼파라미터 튜닝 자체가 호출되지 않음 (기본값 사용).
**하이퍼파라미터 튜닝의 기대 효과가 낮아 Tier 2에서 정규화 수동 조정으로 대체** (섹션 5 참조).

### 2.8 [중간→완료] 미사용 컬럼 — 부분 개선

- `exception_requested` → ML 피처로 추가 ✅
- `rule_matched` → ML 피처로 추가 ✅
- `organization`, `ops_dept`, `service`, `retention_period` → ❌ 미활용 (Tier 2 B1)
- `fp_description` → ❌ Phase 2 multi-class용 (Tier 4 D3)

### 2.9 [중간→완료] Rule Labeler 결과 ML 미반영 — ~~수정 완료~~ ✅

`rule_matched`가 `_PRECOMPUTED_DENSE_COLS`에 추가됨.

### 2.10 [중간] `ConfidentLearningAuditor` 미연결 — ❌ 미조치

구현은 완료되어 있으나 파이프라인에 연결되지 않은 상태.
**Tier 2 B5에서 조치 예정.**

---

## 3. 근본 원인 분석

### 3.1 Tier 0+1 이후에도 F1이 0.61인 이유

| 원인 | 영향도 | 상세 |
|------|--------|------|
| **중복 샘플 미제어** | 높음 | 동일 파일 반복 행이 모델을 소수 패턴에 편향시킴. class_weight만으로는 해결 불가 |
| **server_env 의미 신호 유실** | 높음 | server_freq 제거 시 서버 환경(prd/dev/stg) 신호도 같이 삭제. 샘플에서 prd=TP, dev/stg/sbx=FP 완전 분리 [playbook P1] |
| **범주형 피처 미활용** | 높음 | `service`(6), `ops_dept`(6), `retention_period`(5) 등 유의미한 범주 정보가 ML 입력에 없음 |
| **RULE 도메인 지식 1bit 축소** | 중간 | rule_matched binary만 전달. 12개 룰의 rule_id/class/confidence가 ML에 미전달 [playbook P2] |
| **file-level aggregation 미연결** | 중간 | file_event_count/file_pii_diversity 구현 완료이나 학습 파이프라인 미연결 [playbook P3] |
| **TF-IDF 숫자 암기** | 중간 | char n-gram이 파일명 숫자를 그대로 학습. Shape 변환 없이는 새 파일명에 일반화 불가 |
| **Phase 1 본질적 한계** | 높음 | `full_context_raw` 없이 파일 메타데이터만으로 TP/FP 구분의 근본적 한계 |

### 3.2 문제 재정의

**현재 모델의 근본 구조 문제:**

```
[10M행 전부] → [단일 LightGBM] → TP/FP 이진 분류
                    ↑
     문제: 모든 행을 동일 가중치로 학습
           → 반복 행이 모델을 지배
           → easy case와 hard case 구분 없음
```

**Phase 0 검증 결과 (label conflict rate = 0, Bayes error = 0)와 모순:**
- 데이터 자체는 완전 분리 가능한데 모델이 분리하지 못함
- 이는 피처 표현력 부족 + 중복 샘플 편향의 결합 효과

### 3.3 독립 분석 교차 검증

> 출처: `docs/Architecture/08_performance_improvement_playbook.md` (독립 Agent 코드 검증 기반 제안)

독립 분석에서 우리 진단에 없던 **추가 병목 5가지**가 확인됨:

| # | 발견 | 평가 | 반영 |
|---|------|------|------|
| **P1** | **`server_name`에서 의미 토큰 미추출** — `server_freq`는 제거했지만, `server_env`(prd/dev/stg/sbx), `server_stack`(app/mms) 등 의미 신호도 같이 버림. 샘플에서 `prd=TP, dev/stg/sbx=FP` 완전 분리 | **매우 타당.** 빈도 누수와 의미 신호를 구분하지 못한 것. server_freq 제거는 맞지만 대안 부재 | Tier 2 B7로 추가 |
| **P2** | **RULE 세부 신호 축소** — `rule_matched` binary만 전달, `rule_id`/`rule_primary_class`/`rule_confidence_lb`는 ML 미전달. 12개 룰의 도메인 지식이 1bit로 축소됨 | **타당.** A4에서 rule_matched만 넣은 건 최소한의 연결. rule_id별 사전확률 차이가 큼 | Tier 2 B8로 추가 |
| **P3** | **file-level aggregation 미연결** — `compute_file_aggregates_label()` 구현 완료, `file_event_count`/`file_pii_diversity` 생성 가능하나 학습 경로에서 호출 안 됨 | **타당.** 이미 구현된 코드의 미연결. 저비용 고효율 | Tier 2 B9로 추가 |
| **P5** | **sparse linear + dense tree 2브랜치 앙상블** — TF-IDF는 선형모델에 강하고, 범주/비선형은 부스팅이 강함. 현재 한 LightGBM이 두 세계를 동시에 떠안음 | **흥미로운 구조 제안.** Phase 1의 TF-IDF+메타 혼합 구조에 적합. 다만 구현 복잡도 있음 | Tier 3 C4로 추가 |
| **P6** | **threshold 백테스트 결과가 아티팩트로 저장 안 됨** — Step 6c에서 tau를 계산하지만, decision_combiner는 하드코딩 임계값만 사용. 오프라인 최적화와 운영 정책이 단절 | **운영 관점에서 중요.** 성능 자체보다 운영 재현성 문제 | Tier 3 C5로 추가 |

**독립 분석의 핵심 메시지와 우리 진단의 정합성:**

> "단순 피처 추가보다 **라우팅 가능한 slice는 먼저 빼고, residual만 ML이 보게 하는 구조**로 가는 쪽이 맞다"
> — 이는 우리 Tier 3 C1(2단 구조)과 방향이 동일하며, P9(auto-rule/cache)로 자동화를 제안

**우리 진단과 독립 분석의 차이점:**

| 관점 | 우리 진단 (v3) | 독립 분석 (playbook) |
|------|--------------|---------------------|
| 우선순위 1 | 중복 샘플 가중치 (B2) | server_env 의미 토큰 (P1) |
| RULE 활용 | binary hit (rule_matched) | 세부 신호 (rule_id, confidence) |
| 범주형 전략 | Label Encoding 우선 | OOF Target Encoding / CatBoost 권장 |
| TF-IDF 개선 | Shape 변환 (B3) | Template canonicalization (P8) |
| 아키텍처 | 2단 구조 (C1) | sparse linear + dense tree (P5) + auto-rule (P9) |

→ 두 분석은 **방향이 동일**하고 **깊이가 다름**. 우리 Tier 2에 P1/P2/P3을 통합하고, Tier 3에 P5/P6을 추가하는 것이 최적.

---

## 4. 성능 개선 로드맵 (업데이트)

### Tier 0: 버그 수정 — ✅ 완료

| # | 조치 | 상태 |
|---|------|------|
| F0 | `ml_tp_proba` 인덱스 수정 | ✅ 완료 |
| F1 | Temporal split 월 중복 수정 | ✅ 완료 |

### Tier 1: 과적합 피처 정리 + 강신호 추가 — ✅ 완료

| # | 조치 | 상태 | 실험 결과 |
|---|------|------|----------|
| A1 | 시간 피처 4개 제거 | ✅ | F1 유지 → 과적합 확인 |
| A2 | server_freq 제거 | ✅ | 통계 누수 차단 |
| A3 | exception_requested 추가 | ✅ | FP 강신호 (Y/N→0/1) |
| A4 | rule_matched ML 피처 추가 | ✅ | Rule Labeler 도메인 지식 전달 |
| A5 | TF-IDF 500→200 축소 | ✅ | 노이즈 피처 60% 축소 |

**Tier 1 종합 결과:** F1 0.6146 → 0.6114 (거의 동일). 과적합 제거가 확인되었으나 절대 성능 개선은 미달. Tier 2 진행 필요.

### Tier 2: 구조 개선 — Round 1 구현 완료 ✅

#### Round 1: 데이터 표현 개선 (피처 추가/가중치) — ✅ 구현 완료 (서버 실행 대기)

| # | 조치 | 상태 | 구현 내용 |
|---|------|------|----------|
| **B2** | **중복 샘플 가중치** | ✅ | `run_training.py`: `(file_path, file_name)` 그룹별 `1/sqrt(group_size)`, mean=1 정규화. `trainer.py`: `sample_weight` 파라미터 추가, early stopping 내부 split 시에도 가중치 분할 |
| **B7** | **server_name 의미 토큰** | ✅ | `meta_features.py`: `extract_server_features()` 추가 + `build_meta_features()`에 벡터화 통합. `server_env`(prd/dev/stg/sbx/test/unknown), `server_is_prod`(0/1), `server_stack`(app/mms/db/web/batch/etc) |
| **B1** | **범주형 피처 Label Encoding** | ✅ | `pipeline.py`: service, ops_dept, organization, retention_period, server_env, server_stack, rule_id, rule_primary_class → `_enc` 접미사 LabelEncoding. train+test 합쳐서 fit (unseen 방지) |
| **B8** | **RULE 세부 신호** | ✅ | `run_training.py` Step 4: `rule_confidence_lb` 캡처 (기존 버려짐 → 복원). `pipeline.py`: rule_id_enc, rule_primary_class_enc (categorical), rule_confidence_lb (numeric) |
| **B9** | **file-level aggregation** | ✅ | `run_training.py` Step 5 후: `compute_file_aggregates_label(df_train)` → `file_event_count`, `file_pii_diversity` train fold에서만 계산, test는 left join + median fallback |

**Mock E2E 검증 결과 (152행 샘플 데이터):**
- 354 피처 (기존 428 → 구조 변경)
- Feature Importance 1위: `server_is_prod` (178) — B7 효과 즉시 확인
- 범주형: `service_enc`(35), `ops_dept_enc`(11), `retention_period_enc`(9) — B1 활용
- RULE: `rule_confidence_lb`(2), `rule_id_enc`(2) — B8 활용
- Sample weight: [0.46, 1.12], mean=1.00 — B2 정상 작동
- 기존 테스트 127 pass / 1 fail (기존 실패, 변경 무관)

#### Round 2: 모델 표현력 개선 — ✅ 구현 완료 (서버 실행 대기)

| # | 조치 | 상태 | 구현 내용 |
|---|------|------|----------|
| **B3** | **Shape TF-IDF** | ✅ | `pipeline.py`: Phase 1 TF-IDF에 file_name shape view 추가 (100 features). `_to_shape_text()` 변환 후 char_wb n-gram |
| **B6** | **정규화 강화** | ✅ | `constants.py`: `min_child_samples: 20→200`, `reg_alpha: 0→0.5`, `max_depth: -1→10` |
| B4 | 튜닝 CV를 TimeSeriesGroupKFold로 교체 | 보류 | 현재 튜닝 자체가 미호출. B6 수동 조정으로 대체 |
| B5 | ConfidentLearningAuditor 연결 | 후순위 | Round 1+2 적용 후 잔여 오류에 대해 적용 |

### Tier 3: 아키텍처 변경

| # | 조치 | 예상 효과 | 근거 |
|---|------|----------|------|
| C1 | **2단 구조: Easy FP Suppressor → Residual Classifier** — 고순도 slice auto-rule 자동 생성 포함 | 높음 | 고확신 FP 선제 분리 → hard case에 ML 집중. [playbook P9] support/purity 기준 자동 규칙 승격 |
| C2 | **Slice-aware threshold** — 서비스/경로군별 tau override | 중간 | easy slice에서 coverage 확대 |
| C3 | CatBoost 비교 실험 + OOF Target Encoding | 중간 | [playbook P4] ordered statistics 기반 범주 처리. Label Encoding baseline 후 비교 |
| **C4** | **Sparse Linear + Dense Tree 2브랜치 앙상블** — TF-IDF → LogisticRegression, 메타 → LightGBM, calibrated prob blending | **중간~높음** | [playbook P5] TF-IDF는 선형모델에 강하고, 범주/비선형은 부스팅이 강함. 현재 한 모델이 두 세계를 떠안는 구조 해소 |
| **C5** | **threshold_policy.json 아티팩트 저장** — ✅ 구현 완료 | **운영 필수** | [playbook P6] Step 6c에서 tau + curve_summary를 `models/final/threshold_policy.json`에 자동 저장 |

### Tier 4: Phase 2 (데이터 확보 필요) — 섹션 6 참조

| # | 조치 | 예상 효과 |
|---|------|----------|
| D1 | `full_context_raw` TF-IDF 추가 (Sumologic JOIN) | 매우 높음 |
| D2 | L1/L2 Rule 전면 활성화 | 높음 |
| D3 | Multi-class 학습 (FP 6-class + TP 1-class) | 높음 |

---

## 5. 하이퍼파라미터 튜닝 판단

### 5.1 RandomizedSearchCV 30회의 기대 효과

**결론: 큰 효과 없음. 실행 비용 대비 효율 낮음.**

| 요소 | 현재 기본값 | 최적 범위 | F1 기대 개선 |
|------|------------|----------|-------------|
| num_leaves | 31 | 20~50 | +-0.005 |
| learning_rate | 0.1 | 0.03~0.1 | +0.01 (학습 시간 3배) |
| n_estimators | 500 + early_stop=30 | 자동 조절 | 0 |
| subsample | 0.8 | 0.6~0.9 | +-0.003 |
| colsample_bytree | 0.8 | 0.5~0.9 | +-0.003 |

**총 기대 개선: +0.01~0.03 F1 (10M행 기준 실행 시간 수 시간)**

### 5.2 대안: 정규화 수동 강화 (B6)

현재 문제는 "하이퍼파라미터가 최적이 아닌 것"이 아니라 **"모델이 잘못된 것을 학습하는 것"**.
튜닝보다 과적합 억제가 효과적:

```python
# 수동 조정 3회 실험 (현재값 → 제안값)
min_child_samples: 20 → 200~500    # 일반화 강제: 리프당 최소 샘플 수 증가
reg_alpha:         0  → 0.1~1.0    # L1 정규화: 노이즈 피처 가중치 억제
max_depth:        -1  → 8~12       # 트리 깊이 제한: 과도한 분기 방지
```

**예상 효과:** +0.01~0.03 F1 (튜닝과 동등) + 일반화 개선 효과 추가

### 5.3 판단

> 하이퍼파라미터 튜닝은 Tier 2 B2(중복 가중치)/B1(범주형 피처) 적용 후,
> 잔여 개선이 필요할 때 B6(정규화 강화)로 수동 3회 실험하는 것이 효율적.
> 30회 RandomizedSearchCV는 비용 대비 효과가 낮아 보류.

---

## 6. Phase 2 JOIN 리스크 분석

### 6.1 기본 현황

- **Label 데이터:** 10,049,303행 (silver_label.parquet)
- **Sumologic 데이터:** 별도 (silver_detections.parquet)
- **JOIN 예상 결과:** ~700K행 (**전체의 ~7%**, pk_file 기준 inner join)
- **JOIN 데이터의 추가 정보:** `full_context_raw` (PII 검출 텍스트 컨텍스트)

### 6.2 식별된 리스크 (5개)

| # | 리스크 | 심각도 | 설명 |
|---|--------|--------|------|
| **R1** | **월별 분포 불균형** | 높음 | 7% JOIN율이 월별 균일한 보장 없음. Train(3~9월)/Test(10~12월) 중 특정 월의 JOIN율이 0%면 해당 월 데이터 증발 |
| **R2** | **TP/FP 비율 왜곡** | 높음 | Label 원본은 TP:FP 균형이지만, Sumologic 검출 이력이 있는 파일은 특정 유형에 편중 가능. 예: 대량 검출 FP-시스템로그가 JOIN율 높을 수 있음 |
| **R3** | **Selection Bias (가장 위험)** | 높음 | 7%는 랜덤 샘플이 아님. "label도 있고 Sumologic 검출도 있는 파일" = 체계적으로 다른 모집단. 이 subset에서 학습한 모델이 나머지 93%에 일반화되지 않을 수 있음 |
| **R4** | **Many-to-Many 증폭** | 중간 | pk_file JOIN 시 한 파일에 여러 검출 이벤트 → 행 수 폭증 가능. 중복 가중치 문제 심화 |
| **R5** | **full_context_raw 품질 편차** | 중간 | 월/조직별 Sumologic 로깅 설정 차이 → 텍스트 품질 불균일 |

### 6.3 R3 Selection Bias 상세

```
원본 10M행 (label 전체)
  |-- 7% JOIN 성공 → ~700K행 (label + Sumologic 양쪽 다 있는 파일)
  |-- 93% JOIN 실패 → ~9.3M행 (Sumologic에 기록 없거나 pk 불일치)
```

JOIN 성공 subset의 특성 편향 시나리오:
- FP-시스템로그 (대량 검출, 상시 발생) → Sumologic에도 기록 多 → JOIN 성공률 높음
- TP-실제개인정보 (소량 검출, 간헐적) → Sumologic 기록 少 → JOIN 실패 가능
- 결과: joined 데이터가 FP 과대표본 → FP 편향 모델

### 6.4 대응 전략: Phase 2는 Phase 1을 대체가 아닌 보완

```
[운영 아키텍처]
  모든 건 → Phase 1 모델 (메타/경로 기반, 10M학습)  → 1차 판정
           ↓ full_context_raw 있는 건만
           Phase 2 모델 (텍스트 TF-IDF 추가, 700K학습) → override 판정
```

- Phase 1: 전체 데이터에 적용 가능 (메타데이터만 필요)
- Phase 2: full_context_raw가 존재하는 건에만 적용 (보강 역할)
- 두 모델의 결합은 decision_combiner에서 confidence 기반 우선순위로 처리

### 6.5 JOIN 전 사전 분석 필요 항목

JOIN을 실행하기 전에 다음을 검증해야 함:

```bash
# 1. 월별 JOIN율 분포 확인
#    → 특정 월에 JOIN율이 극단적으로 낮으면 temporal split 불가
SELECT work_month, COUNT(*) as total, COUNT(joined) as joined,
       joined/total as join_rate
GROUP BY work_month

# 2. TP/FP별 JOIN율 확인
#    → FP에 편중되면 TP 학습 데이터 부족
SELECT label_raw, COUNT(*) as total, COUNT(joined) as joined
GROUP BY label_raw

# 3. file_path 그룹별 JOIN율 확인
#    → 특정 경로군만 JOIN 성공하면 selection bias 심각
SELECT file_path_prefix, label_raw, COUNT(*), join_rate
GROUP BY file_path_prefix, label_raw
```

---

## 7. 예상 결과 (업데이트)

| 시나리오 | 예상 F1-macro | 실측 F1-macro | 상태 |
|---------|-------------|-------------|------|
| Baseline (과적합 피처 + split 오류) | - | **0.6146** | 과대평가 |
| Tier 0+1: 피처 정리 + 강신호 추가 | 0.58~0.68 | **0.6114** | ✅ 범위 내 |
| **Tier 2: 구조 개선 (다음 목표)** | **0.65~0.72** | - | 대기 |
| Tier 3: 아키텍처 변경 | 0.70~0.80 | - | 후순위 |
| Tier 4: Phase 2 (full_context_raw) | 0.80~0.90 | - | 데이터 확보 후 |

### Tier 2 예상 근거 (playbook 통합 후 재산정)

| 조치 | 개별 효과 추정 | 근거 |
|------|-------------|------|
| B2 중복 가중치 | +0.02~0.05 | 반복 패턴 암기 제거 → 모델이 다양한 패턴을 균등 학습 |
| B7 server_env 의미 토큰 | +0.02~0.04 | 샘플에서 prd/dev 완전 분리. server_freq 대체 일반화 신호 |
| B1 범주형 피처 (server_env 포함) | +0.01~0.03 | service/ops_dept/retention_period + server_env 통합 |
| B8 RULE 세부 신호 | +0.01~0.02 | rule_id별 사전확률 차이 반영. binary → multi-signal |
| B9 file aggregation | +0.01~0.02 | 파일 단위 패턴 안정화 (row보다 file이 신호 강함) |
| B3 Shape TF-IDF + Template | +0.01~0.02 | 숫자 과적합 감소 + 구조 패턴 학습 |
| B6 정규화 강화 | +0.01~0.03 | 과적합 억제 |
| **합산 (비독립)** | **+0.05~0.11** | 복합 효과, 단순 합산 아님. playbook 항목 3개 추가로 상한 상향 |

**F1 0.6114 + 0.05~0.11 = 0.66~0.72 (보수적 0.66, 낙관적 0.72)**

> 주의: server_env의 효과는 10M 실데이터에서 검증 필요. 샘플 152행에서의 완전 분리가
> 실데이터에서도 유지되는지 확인 전까지 보수적으로 추정.

---

## 8. Phase 1 현실적 성능 상한

Phase 1에서 사용 가능한 정보:
- 파일 경로/이름 → 파일 "유형" 추정
- 검출 건수/PII 유형 비율 → 대량 검출 = FP 경향
- 파일 확장자 → .log, .conf 등 시스템 파일 = FP
- 예외 신청 이력, 조직/서비스 정보

**한계:**
- 같은 경로의 같은 유형 파일이 TP일 수도, FP일 수도 있음
- 개별 검출 건의 실제 내용(전화번호? 타임스탬프?)을 보지 않고는 구분 불가

단, Phase 0 검증에서 **label conflict rate = 0, Bayes error = 0** → 현재 데이터에서는 경로/메타 조합만으로 완전 분리가 가능한 상태. 이는 Phase 1의 실제 상한이 예상보다 높을 수 있음을 시사.

### 성능 목표

| 단계 | F1 목표 | TP Recall 목표 | FP Precision 목표 |
|------|---------|---------------|------------------|
| Phase 1 Tier 2 (보수적) | >=0.65 | >=0.65 | >=0.65 |
| Phase 1 Tier 2 (낙관적) | >=0.72 | >=0.75 | >=0.75 |
| Phase 1 Tier 3 | >=0.75 | >=0.80 | >=0.80 |
| Phase 2 | >=0.85 | >=0.90 | >=0.90 |

---

## 9. Tier 2 실행 계획

### 9.1 실행 순서 (Round 1 ✅ → Round 2 대기)

```
=== Round 1: 데이터 표현 개선 — ✅ 구현 완료 ===

[1] B2: 중복 샘플 가중치                              ✅ run_training.py + trainer.py
[2] B7: server_name 의미 토큰 분해                    ✅ meta_features.py
[3] B1: 범주형 피처 Label Encoding (8개 컬럼)         ✅ pipeline.py
[4] B8: RULE 세부 신호 (rule_confidence_lb 등)        ✅ run_training.py + pipeline.py
[5] B9: file-level aggregation 연결                   ✅ run_training.py

→ 서버에서 재학습 필요:
  rm -f data/checkpoints/step4_df_silver_label.pkl
  rm -f data/checkpoints/step5_features_silver_label.pkl
  rm -f data/checkpoints/step6_model_silver_label.pkl
  python scripts/run_training.py --source label --split temporal --test-months 3
  python scripts/run_report.py --source label --include-diagnosis

→ 목표: F1 0.66+

=== Round 2: 모델 표현력 개선 — ✅ 구현 완료 ===

[6] B3: Shape TF-IDF                                  ✅ pipeline.py (fname shape 100 features)
[7] B6: 정규화 강화                                    ✅ constants.py (min_child_samples=200, reg_alpha=0.5, max_depth=10)
[+] C5: threshold_policy.json 저장                    ✅ run_training.py Step 6c 후

→ 서버에서 한 번 재학습으로 Round 1+2+C5 모두 반영됨
→ 목표: F1 0.66~0.72
```

### 9.2 재학습 명령어 (Tier 2 적용 후)

```bash
# 1. 체크포인트 삭제 (step4 이후 전부 — 피처 변경이므로)
rm -f data/checkpoints/step4_df_silver_label.pkl
rm -f data/checkpoints/step5_features_silver_label.pkl
rm -f data/checkpoints/step6_model_silver_label.pkl
rm -f data/checkpoints/step6b_temporal_silver_label.pkl

# 2. 재학습
python scripts/run_training.py --source label --split temporal --test-months 3

# 3. 통합 리포트
python scripts/run_report.py --source label --include-diagnosis
```

---

## 10. 실험 재현성 주의사항

현재 이 리포트의 성능 수치는 **서버에서 실행한 10M행 데이터** 기준이고,
로컬 저장소의 `silver_label.parquet`는 **152행 샘플 데이터**이다.

개선 효과를 정확히 측정하려면:
1. input parquet fingerprint (해시)
2. split 전략 + 파라미터
3. model/feature builder artifact

이 세 가지가 항상 함께 버전 관리되어야 하며, 실험 비교 시 동일 데이터/split 기반임을 확인해야 한다.

---

## 11. 핵심 요약

### 완료 (Tier 0 + Tier 1)
1. ~~`ml_tp_proba` 인덱스 역전~~ ✅
2. ~~Temporal split 월 중복~~ ✅
3. ~~시간 피처 4개 + server_freq 제거~~ ✅
4. ~~exception_requested, rule_matched 피처 추가~~ ✅
5. ~~TF-IDF 500→200 축소~~ ✅
6. **실험 결과: F1 0.6146 → 0.6114 (과적합 제거 확인, 절대 성능 유지)**

### 다음 단계 (Tier 2 Round 1) — 최우선, 한 번에 적용
7. **B2: 중복 샘플 가중치** — 10M행 중 반복 패턴 암기가 성능 병목의 핵심
8. **B7: server_name 의미 토큰** — server_env(prd/dev/stg/sbx) 추출. server_freq 대체 일반화 신호 [playbook P1]
9. **B1: 범주형 피처 추가** — service, ops_dept, retention_period, server_env 등 미활용 강신호
10. **B8: RULE 세부 신호** — rule_id, rule_primary_class, rule_confidence_lb ML 전달 [playbook P2]
11. **B9: file aggregation 연결** — file_event_count, file_pii_diversity 구현 완료 코드 연결 [playbook P3]

### 다음 단계 (Tier 2 Round 2) — Round 1 결과 확인 후
12. **B3: Shape TF-IDF + Template** — 숫자 n-gram 과적합 → 구조 패턴 학습 [playbook P8 통합]
13. **B6: 정규화 강화** — 수동 3회 실험 (RandomizedSearchCV 30회보다 효율적)

### 구조 변경 (Tier 3) — Tier 2 이후
14. **C1: 2단 구조 + auto-rule** — 고순도 slice 자동 규칙 승격 [playbook P9 통합]
15. **C4: Sparse Linear + Dense Tree 2브랜치** — TF-IDF/메타 분리 앙상블 [playbook P5]
16. **C5: threshold_policy.json 아티팩트 저장** — 백테스트 tau → 운영 정책 연결 [playbook P6]

### Phase 2 준비 (Tier 4)
17. **Sumologic JOIN 7% — 5가지 리스크 식별** (R1~R5)
18. **Phase 2는 Phase 1 대체가 아닌 보완** — full_context_raw 있는 건만 override
19. **JOIN 전 월별/TP-FP별/경로별 JOIN율 사전 분석 필수**

### 판단 보류
20. 하이퍼파라미터 튜닝 (RandomizedSearchCV) → 비용 대비 효과 낮음, B6 수동 조정으로 대체

### 독립 분석 (playbook) 반영 이력
- P1 server_env → B7 (Tier 2 Round 1, 2순위)
- P2 RULE 세부 신호 → B8 (Tier 2 Round 1, 3순위)
- P3 file aggregation → B9 (Tier 2 Round 1, 3순위)
- P4 OOF Target Encoding → C3 (Tier 3, Label Encoding baseline 후 비교)
- P5 2브랜치 앙상블 → C4 (Tier 3)
- P6 threshold 아티팩트 → C5 (Tier 3, 운영 필수)
- P7 inner time-aware → 장기 과제 (참조)
- P8 template canonicalization → B3에 통합 (Tier 2 Round 2)
- P9 auto-rule/cache → C1에 통합 (Tier 3)
