# Decision Combiner 종합 검증 및 고도화 계획

**작성일:** 2026-03-22
**목적:** Decision Combiner의 현재 성능을 실 서버 데이터(3,963,589건)로 검증하고, 코드 구현 상의 문제점을 식별한 뒤, 일반화 가능한 고도화 방향을 수립한다.

---

## 1. 실 서버 데이터 기반 현황 (Temporal Split, ~400만 행)

### 1.1 성능 요약

| 항목 | 값 |
|------|-----|
| **ML 단독 F1-macro (Wave 6)** | **0.7790** |
| **ML 단독 F1-macro (Wave 7)** | **0.7914** (+0.012, rule_stats 정비 + PATH_HADOOP_001 비활성화) |
| **Decision Combiner F1-macro (Wave 6)** | **0.7791** (+0.0001, 사실상 무의미) |
| **DC Grid Sweep 최적 (Wave 7)** | **0.7915** (ml_conf=0.0 = ML passthrough) |
| **DC 기본 임계값 (Wave 7)** | **0.7746** (ml_conf=0.7, ml_margin=0.2 = **-0.017 악화**) |
| 전체 테스트 건수 | **~3,962,171건** (temporal split, test_months=3) |

> 출처: 실 서버 poc_report.xlsx + run_training.py Step 6d Grid Sweep 로그
>
> **Wave 7 핵심 발견:** DC 개입 시 항상 F1 하락. 35개 임계값 조합 전수 탐색 결과, ML passthrough(ml_conf=0.0)가 최적. DC의 "애매하면 TP" 로직이 ML의 정확한 FP 판정(FP recall 0.88)을 TP로 대량 전환하여 FP recall 0.73으로 급락.

### 1.2 Confusion Matrix (DC 적용 후)

| | 예측 FP | 예측 TP |
|--|---------|---------|
| **실제 FP** | ~1,734,630 | ~641,143 |
| **실제 TP** | ~72,530 | ~1,352,840 |

- **FP 중 TP로 오분류: ~641,143건** (전체 FP의 ~27%)
- **TP 중 FP로 오분류: ~72,530건** (전체 TP의 ~5%)
- FP를 잡아내는 것(FP Recall)이 핵심 과제

### 1.3 Decision Source 분포

| Source | 건수 | 비율 |
|--------|------|------|
| **ML_PREDICTION** | ~3,935,886 | **99.3%** |
| ML_OVERRIDE | ~14,437 | ~0.4% |
| RULE_HIGH_CONFIDENCE | ~12,345 | **~0.3%** |

→ **99.3%가 ML 단독 판정, RULE 기여는 0.3%에 불과.**

### 1.4 Reason Code 분포

| Reason | 건수 | 비율 |
|--------|------|------|
| ML_PREDICTION | ~3,935,886 | 99.3% |
| RULE_ML_CONFLICT | ~14,437 | ~0.4% |
| RULE_HIGH_CONFIDENCE | ~12,345 | ~0.3% |

---

## 2. RULE Labeler 실측 데이터

### 2.1 룰별 히트율 / 정밀도 (실 서버 기준)

| rule_id | hit_count | hit_rate | **precision** | dominant_class |
|---------|-----------|----------|---------------|---------------|
| PATH_HADOOP_001 | 13,102 | 0.33% | **0.117 (11.7%)** | FP-시스템로그 |
| FILE_ROTATION_001 | 11,302 | 0.29% | **0.696 (69.6%)** | FP-시스템로그 |
| PATH_TEMP_DEV_001 | 1,809 | 0.05% | 0.745 (74.5%) | FP-더미테스트 |
| PATH_DOCKER_001 | 372 | 0.01% | **0.895 (89.5%)** | FP-시스템로그 |
| PATH_PACKAGE_001 | 34 | 0.001% | **0.912 (91.2%)** | FP-라이브러리 |

**핵심 문제:** PATH_HADOOP_001이 가장 많이 매칭되는 룰(13,102건)이면서 **정밀도가 11.7%** — 10건 중 9건이 오분류. 현재 `rule_stats.json`에는 N=20, M=19 (precision_lb=0.845)로 기록되어 있으나 **실측과 완전히 괴리**된다.

### 2.2 클래스별 Rule 기여도

| class_name | rule_id_count | total_hits |
|------------|---------------|------------|
| FP-시스템로그 | 3 | 24,777 |
| FP-더미테스트 | 1 | 1,809 |
| FP-라이브러리 | 1 | 34 |

### 2.3 Rule vs ML Coverage 비교

| 항목 | 값 |
|------|-----|
| **Rule 단독 Coverage** | **0.6%** |
| ML 총 Coverage | 92.4% |
| ML 추가 Coverage (RULE 미검출 영역) | 91.8% |
| **RULE-ML 중복 검출 비율** | **90.5%** |

→ RULE이 잡는 건의 90.5%를 ML도 동시에 잡음. RULE의 독립 기여 = 0.6% × 9.5% ≈ **0.057%**.

---

## 3. Feature Importance 실측 데이터

### 3.1 그룹별 Importance 합산

| 그룹 | Importance | 비율 |
|------|-----------|------|
| **Other (범주형 메타)** | **16,697** | **55.7%** |
| Text TF-IDF | 6,432 | 21.5% |
| Path TF-IDF | 3,859 | 12.9% |
| Detection Stats | 1,505 | 5.0% |
| Path Flags | 1,119 | 3.7% |
| Server | 307 | 1.0% |
| **Filename** | **44** | **0.1%** |

### 3.2 상위 Feature 목록

| Rank | Feature | Importance | 비고 |
|------|---------|-----------|------|
| 1 | **service_enc** | 2,554 | 서비스 코드 (범주형) |
| 2 | **ops_dept_enc** | 2,416 | 운영부서 (범주형) |
| 3 | **retention_period_enc** | 2,031 | 보존기간 (범주형) |
| 4 | pattern_count_log1p | 1,088 | 검출 건수 (연속형) |
| 5 | **organization_enc** | 920 | 조직 (범주형) |
| 6 | path_depth | 724 | 경로 깊이 |
| 7 | tfidf_phase1path_logs | 388 | 경로 TF-IDF "logs" |
| 8 | tfidf_phase1path_data | 363 | 경로 TF-IDF "data" |

**핵심 관찰:**
- Top 4 중 3개(service_enc, ops_dept_enc, organization_enc)가 **조직 메타데이터** → "이 부서 = FP 많음" 단축키를 학습
- **Filename 피처 = 0.1%** — 오분류 패턴이 파일명으로 구별 가능한데도 모델이 활용하지 못함
- 범주형 메타(55.7%)가 파일 특성 피처(Path Flags 3.7% + Filename 0.1%)를 압도

---

## 4. 오분류 패턴 분석 (Error Analysis 샘플)

> **주의:** 아래는 error_analysis.csv의 **일부 샘플**이다. 전체 오분류 분포를 대표하지 않으며, 특정 패턴의 정확한 비율은 전체 데이터 분석이 필요하다.

### 4.1 샘플에서 관찰된 FP 오분류 (실제 FP → 모델이 TP로 예측)

| 파일명 유형 | 경로 패턴 | 서버 | 공통 특성 |
|-----------|----------|------|----------|
| 인프라 메시징 로그 (`*_message.log`) | `/DINAS/VMMSQM/data1/rmm/OM/log/YYYYMMDD/` | vmmsqtest03 | 날짜별 디렉토리, .log, 시스템 인프라 |
| DB 복제 로그 (`rpdb_cpsender_real*_YYYYMMDD.log`) | `/SMS/NSMS6/log/rpdb/` | vNSMS11, vNSMS13 | 날짜 접미사, .log, DB 인프라 |
| DB CDC 로그 (`dbcanal_enc*_YYYYMMDD.log`) | `/SMS/NSMS6/log/dbcanal/` | vNSMS13, vNSMS18 | 날짜 접미사, .log, DB 인프라 |
| NCP 발신 로그 (`ncpsender_UPLUS_*_YYYYMMDD.log`) | `/SMS/NSMS6/log/ncpsender/` | vNSME13 | 날짜 접미사, .log, 통신 인프라 |
| Redis 백업 (`redis_aof_backup_*.rp`) | `/DATA/redis/backup/` | vopewas03 | 타임스탬프, 백업 파일 |
| 압축 로그 (`lgpamr01_*.log.gz`) | `/SMS/NSMS6/log/teleconn/` | vNSMS11 | 날짜 접미사, 압축 로그 |

### 4.2 샘플에서 관찰된 TP 오분류 (실제 TP → 모델이 FP로 예측)

| 파일명 | 경로 | 서버 |
|--------|------|------|
| `cpmoninfo_job.log` | `/SMS_DB03/log/siatfb/` | mmsdb01 |
| `normal.cp` | `/SMS_DB03/home_backup/TRACE/` | mmsdb01 |

### 4.3 오분류 샘플의 일반화 가능한 공통 특성

개별 파일명이 아니라, **일반화 가능한 특성**으로 추출하면:

| 특성 | 해당 피처 | 현재 ML 입력 | 현재 Importance |
|------|----------|------------|----------------|
| 파일명에 날짜(YYYYMMDD) 포함 | `fname_has_date` | 포함 | 낮음 (Filename 그룹 0.1%) |
| .log 확장자 | `is_log_file` | 포함 | Path Flags 내 (3.7%) |
| /log/ 경로 하위 | path TF-IDF | 포함 | 12.9% |
| 파일명에 밑줄+숫자 반복 | `fname_has_rotation_num` | 포함 | 낮음 |
| 동일 서버/경로에서 다수 이벤트 | `file_event_count` | 포함 | Detection Stats (5.0%) |
| 시스템 인프라 경로 토큰 | `has_system_token` | 포함 | 낮음 |

**핵심:** 오분류를 구별할 피처들이 이미 ML 입력에 존재하지만, **조직 메타(55.7%)에 의해 신호가 묻히고 있음.** 피처가 부족한 게 아니라 모델이 파일 특성 피처를 제대로 활용하지 못하는 것이 문제.

### 4.4 오분류 샘플의 ML 예측 확률

오분류 행들의 `ml_tp_proba` 값이 **0.4~0.6 구간에 집중** (이미지에서 확인). 모델이 확신 없이 경계 영역에서 판정하고 있음 → Decision Combiner의 "불확실 구간" 처리 로직이 개선 대상.

---

## 5. 코드 구현 상의 문제점

### 5.1 [심각] run_report.py 평가 로직 ≠ decision_combiner.py 추론 로직

| 항목 | `decision_combiner.py` (추론) | `run_report.py:451` (평가) |
|------|------------------------------|---------------------------|
| RULE 임계값 | `rule_conf >= 0.85` | `_rule_conf >= 0.5` |
| Case 분기 | 4-Case (OOD→RULE→ML→Fallback) | 2-Case (RULE vs ML) |
| ambiguous ML → TP | `ml_tp_proba >= 0.40` | **없음** |
| OOD/entropy 처리 | 있음 | **없음** |

**결과:** 리포트의 DC F1(0.7791)은 추론 시 실제 성능과 다를 수 있다. 특히 RULE 임계값 0.5는 추론의 0.85보다 낮아서, **평가 시 RULE Case 1 진입률이 과대**.

### 5.2 [심각] threshold_policy.json — 생성만 되고 추론에 미반영

| 아티팩트 | 생성 위치 | 추론 시 사용 | 상태 |
|---------|----------|------------|------|
| `recommended_fp_tau` | run_training.py Step 6c | combine_decisions()에서 참조 안 함 | **사문서** |
| `slice_thresholds` | run_training.py Step 6c | run_inference.py에서 로드 안 함 | **사문서** |
| `easy_fp_suppressor` | run_training.py Step 6c | combine_decisions()에 로직 없음 | **사문서** |

### 5.3 [심각] rule_stats.json과 실측 정밀도의 괴리

| rule_id | rule_stats N/M | rule_stats lb | **실측 precision** | 괴리 |
|---------|---------------|---------------|-------------------|------|
| PATH_HADOOP_001 | N=20, M=19 | 0.845 | **0.117** | **7.2배 과대** |
| FILE_ROTATION_001 | N=30, M=29 | 0.875 | 0.696 | 1.3배 과대 |
| PATH_TEMP_DEV_001 | N=0, M=0 | 0.500 | 0.745 | 과소 (안전) |
| PATH_DOCKER_001 | N=50, M=49 | 0.904 | 0.895 | ≈일치 |
| PATH_PACKAGE_001 | N=20, M=19 | 0.845 | 0.912 | 과소 (안전) |

**PATH_HADOOP_001이 가장 심각:** rule_stats는 정밀도 84.5%라고 하지만 실제는 11.7%. 이 룰이 Case 1에 진입하면 **성능을 악화**시킨다.

### 5.4 [심각] 아키텍처 vs 구현 괴리

| 아키텍처 컴포넌트 (§9.5~9.7) | 구현 상태 | 구현률 |
|-----------------------------|----------|--------|
| `combine_decisions()` 핵심 4-Case | 구현됨 (entropy 임계값 불일치) | 85% |
| `AutoAdjudicator` 4단 판정 | Step 1(majority vote)만 | **25%** |
| `UnknownAutoProcessor` | **코드 0줄** | **0%** |
| `AutoTuner` 자동 임계값 최적화 | **코드 0줄** | **0%** |
| `simulate_combiner()` | **코드 0줄** | **0%** |
| `Auto-Rule-Promoter` | **코드 0줄** | **0%** |

---

## 6. 근본 원인 진단

### 6.1 왜 DC가 ML에 가치를 더하지 못하는가

```
RULE 커버리지 0.6%  ──→  99.3%가 ML 단독 판정
     ↓
RULE-ML 중복 90.5%  ──→  RULE 독립 기여 0.057%
     ↓
DC 개선 +0.0001     ──→  사실상 ML 단독과 동일
```

### 6.2 왜 ML의 FP Recall이 73%에 머무는가

```
범주형 메타 의존 55.7%  ──→  "부서X = FP" 단축키 학습
     ↓
Filename 피처 0.1%     ──→  파일명 패턴 미활용
     ↓
경계 케이스(0.4~0.6)   ──→  조직 메타로 구별 안 되는 건 → 동전 던지기
```

**핵심:** DC 고도화보다 **ML 자체의 피처 균형 문제**가 더 큰 병목이다. 다만 DC의 "불확실 구간 처리"와 "파일 수준 컨텍스트 활용"은 ML 개선과 독립적으로 가치가 있다.

---

## 7. 구현 전 검증 질문 (Phase 0)

> 독립 분석(`decision_combiner_independent_analysis.md`)에서 도출된 핵심 질문. **아래 질문에 답하기 전에 구현에 착수하면 잘못된 방향으로 최적화할 위험이 있다.**

### 7.1 DC의 이론적 개선 상한은 얼마인가

DC가 개입하는 0.7%(~27,000건)에서 **모든 판정을 완벽하게 맞춘다고 가정**해도 전역 F1은 얼마나 오르는가? 이 상한이 0.01 미만이면, DC 로직 정교화보다 DC 개입 면적 확대가 우선이다.

**확인 방법:** DC 개입 샘플(ML_OVERRIDE + RULE_HIGH_CONFIDENCE ~26,782건)의 실제 정답 분포를 집계하여, "전부 맞춘 경우"의 F1을 계산.

### 7.2 ML이 실제로 shortcut을 학습하고 있는가

범주형 메타(service_enc, ops_dept_enc 등)를 **제거한 ablation 실험**을 해야 한다. 두 가지 시나리오:

| 결과 | 해석 | 후속 조치 |
|------|------|----------|
| F1이 크게 하락 (>0.05) | 범주형 메타가 **진짜 유용한 정보**를 담고 있음 | 제거 대신 regularization으로 의존도 완화 |
| F1이 소폭 하락 또는 유지 | 범주형 메타는 **shortcut**이었고, 다른 피처가 보완 가능 | 제거 또는 대폭 약화 → 파일/경로 피처 비중 상승 기대 |

**확인 방법:** `run_training.py`의 `_KEEP_COLS`에서 `service`, `ops_dept`, `organization`, `retention_period` 4개를 제거하고 동일 split으로 재학습.

### 7.3 Rule은 ML 실패 영역에서 독립 가치가 있는가

Rule hit 샘플(~26,620건) 중 **ML이 틀린 건**만 분리하여:
- RULE이 이 영역에서 얼마나 정확한가?
- RULE 없이 ML만으로는 이 샘플들의 정확도가 어떤가?

만약 Rule hit ∩ ML 실패 영역이 극소수이면, Rule은 ML의 보완자가 아니라 **단순 중복 판정기**에 불과하다.

### 7.4 오분류는 row-level 문제인가 group-level 문제인가

- 같은 pk_file 내에서 라벨 일관성(전부 FP 또는 전부 TP)은 얼마나 강한가?
- 경계 확률(0.4~0.6) 샘플이 특정 pk_file/경로 그룹에 군집되는가?
- 파일군 단위로 판정하면 TP safety를 유지하면서 FP Recall을 얼마나 회복할 수 있는가?

**이 질문의 답이 "group-level"이면**, DC의 역할은 Rule override 엔진이 아니라 **파일 문맥 기반 후처리 계층**으로 재정의되어야 한다.

### 7.5 rule_stats 괴리의 근본 원인은 무엇인가

PATH_HADOOP_001의 rule_stats lb=0.845 vs 실측 precision=11.7%는 단순 샘플 오차로 설명되지 않는다. 가능한 가설:

1. 통계가 극소 표본에서 생성되어 대표성이 없다 (N=20은 400만 건 대비 무의미)
2. 통계 산출 시점과 평가 데이터의 도메인이 다르다 (다른 시기/서버)
3. 룰 매칭 로직과 성능 집계 로직의 기준이 다르다 (precision 분자/분모 정의 불일치)
4. 클래스 매핑이 중간에 변경되었는데 통계가 갱신되지 않았다
5. 통계 자체가 수동 추정치이다 (N=20, 30, 50 같은 라운드 넘버)

**어떤 가설이든**, 결론은 동일: **rule_stats를 단순 갱신이 아니라, 자동 집계 + 지속 검증 체계로 교체해야 한다.**

---

## 8. 고도화 계획

### Phase A: 측정 인프라 정비 (선행 필수)

DC를 고도화하기 전에, **성능을 정확히 측정할 수 있는 환경**부터 만든다.

| # | 작업 | 상세 | 난이도 |
|---|------|------|--------|
| A1 | **run_report.py DC 평가 로직을 combine_decisions() 호출로 통일** | 현재 벡터화 로직(임계값 0.5, 2-Case)을 실제 추론 로직(임계값 0.85, 4-Case)과 일치시킴. 평가↔추론 결과가 동일해야 리포트를 신뢰할 수 있음 | 중간 |
| A2 | **rule_stats.json 자동 집계 + 지속 검증 체계** | 단순 "실측 반영"이 아니라, 학습 파이프라인 내에서 매 학습 시 자동 집계 + 이전 값과의 편차 경고를 포함하는 체계. PATH_HADOOP_001의 lb=0.845 vs 실측 11.7%는 단순 샘플 오차가 아니라 **통계 체계 자체의 신뢰 부재**를 의미(근본 원인 분석: §7.5). N=20,30,50 같은 라운드 넘버도 수동 추정 의심 | 중간 |
| A3 | **threshold_policy.json → combine_decisions() 연동** | Step 6c에서 계산된 최적 tau, slice_thresholds를 추론 시 실제 적용. 현재는 생성만 되고 사문서 상태 | 낮음 |

### Phase B: RULE Labeler 정비 (해로운 룰 제거)

| # | 작업 | 상세 | 난이도 |
|---|------|------|--------|
| B1 | **PATH_HADOOP_001 비활성화 또는 조건 강화** | 실측 정밀도 11.7% → 10건 중 9건 오분류. 이 룰이 Case 1에 진입하면 성능 악화. (1) 비활성화(`active: false`), (2) 조건 추가로 정밀도 개선 (has_system_token + is_log_file + 추가 조건), 둘 중 택일 | 낮음 |
| B2 | **FILE_ROTATION_001 조건 강화 검토** | 정밀도 69.6%로 Case 1 임계값(0.85)에 미달이므로 현재 판정에 영향 없으나, rule_stats 갱신 후 lb가 변하면 영향 발생 가능. 조건 강화로 정밀도 80%+ 확보 필요 | 낮음 |

### Phase C: ML 모델 — Representation 품질 개선

**DC 고도화가 아닌 ML 자체 개선이지만, FP Recall 73% → 80%+ 달성에 가장 효과적인 경로.**

> 핵심 인식 전환: 문제는 "피처가 부족하다"가 아니라 **"좋은 피처가 묻히고 있다"**. fname_has_date, is_log_file, has_system_token 등은 이미 ML 입력에 있는데 importance가 극히 낮다. 피처를 더 넣는 것(inventory)보다, 기존 피처가 왜 활용되지 않는지(representation)를 먼저 해결해야 한다.

| # | 작업 | 근거 | 난이도 |
|---|------|------|--------|
| C1 | **범주형 메타 Ablation 실험** | service_enc + ops_dept_enc + retention_period_enc + organization_enc (합산 26.4%)를 **제거한 모델**을 학습하여 F1 변화 관찰. 하락이 크면 진짜 유용한 정보 → regularization으로 완화. 하락이 작으면 shortcut → 제거. **Phase 0 검증(§7.2) 결과에 따라 후속 조치 결정** | 중간 |
| C2 | **File Family 표현 강화** | 현재 파일명 피처가 너무 원자적(개별 binary flag). 운영 환경의 "파일 패밀리"(날짜 로테이션 로그, DB 복제 로그, 백업 시계열 등)를 구조적으로 인코딩해야 함. 방법: (1) `digit_ratio`, `max_digit_run`, `separator_count` 추가(path_features.py에 이미 추출됨), (2) 파일명 prefix 추출(첫 밑줄 앞 토큰), (3) 확장자×날짜×로그 상호작용 피처 | 중간 |
| C3 | **미사용 경로 피처 ML 입력 추가** | `has_backup_path`, `has_database_path`, `has_cicd_path` — 오분류 샘플의 공통 특성(DB, 백업, 인프라 경로)과 직접 대응. path_features.py에 추출되지만 _KEEP_COLS 미포함 | 낮음 |
| C4 | **학습 목표 재검토** | 현재 학습은 전반 평균 성능(macro F1)에 맞춰져 있을 가능성. FP 회복에 더 민감한 loss weight 또는 sample weight 조정이 FP Recall 개선에 효과적일 수 있음. 단, TP Recall 하락 감시 필수 | 중간 |

### Phase D: DC 역할 재정의 — 불확실 구간의 문맥 해소 계층

> **DC의 역할 전환:** "Rule과 ML 중 누가 맞는가를 고르는 장치" → "**ML이 불확실한 구간에서 그룹 문맥으로 판정을 보강하는 후처리 계층**". 오분류의 상당수가 0.4~0.6 경계 구간에 집중되고, 이 샘플들은 같은 pk_file/경로/서버에서 반복적으로 출현한다. 개별 행(row) 단위로는 애매하지만, 파일군(group) 단위로 보면 정체가 명확해지는 구조다.

| # | 작업 | 상세 | 난이도 |
|---|------|------|--------|
| D1 | **File-Level Consensus 전파** | 동일 pk_file 내 고확신 판정을 저확신 이벤트에 전파. 오분류 샘플에서 동일 서버+경로의 다수 이벤트가 관찰됨 → 파일 컨텍스트가 강한 신호. TP 고확신이 1건이라도 있으면 전파 차단(안전장치). **Phase 0 검증(§7.4)에서 "group-level 문제"가 확인되면 DC의 핵심 메커니즘이 됨** | 중간 |
| D2 | **3-Zone ML 후처리** | 현재 2분기(고확신→ML / 저확신→TP)를 3분기로 확장. Zone A(≥0.80): ML 신뢰, Zone B(0.60~0.80): 보조 신호(entropy, margin, top2 클래스 계열, **파일군 문맥**) 참조, Zone C(<0.60): TP 안전. Zone B에서 entropy 낮고 top1/top2 모두 FP 계열이면 FP 유지 → FP Recall 개선 | 중간 |
| D3 | **경로 패밀리 기반 일관성 검사** | 같은 경로 패턴(예: `/SMS/NSMS6/log/*/`)에서 나온 다수 이벤트의 판정 일관성 검사. 90%+가 FP 고확신이면 나머지 저확신도 FP로 전파. pk_file보다 넓은 "경로 패밀리" 단위의 문맥 활용 | 중간 |
| D4 | **AutoAdjudicator Step 2~3 구현** | 현재 Step 1(majority vote)만 구현. Step 2: RULE↔ML 교차 합의(동일 클래스 + 합산 confidence ≥ 0.75), Step 3: 피처 유사도 기반 K-NN. 아키텍처 §9.5 설계 존재 | 중간 |

### Phase E: RULE 독립 가치 확보 (장기)

**RULE이 ML과 다른 신호를 제공해야 DC에 실질적 가치가 발생한다.**

| # | 작업 | 상세 | 난이도 |
|---|------|------|--------|
| E1 | **Sumologic JOIN 활성화 (Phase 2)** | full_context_raw가 존재하면 L2 룰 6개 활성화 → ML 입력에 없는 **텍스트 패턴** 신호 추가. RULE-ML 중복(90.5%)을 구조적으로 해소하는 유일한 방법 | 중간 (인프라 의존) |
| E2 | **ML 오분류 특화 룰 설계** | 전체 error_analysis.csv를 분석하여, ML이 체계적으로 틀리는 **일반화된 조건**을 룰로 구현. 특정 파일명 하드코딩은 과적합이므로, 일반화 가능한 피처 조합(예: `fname_has_date=1 AND is_log_file=1 AND path_depth≥5`)으로 설계해야 함 | 높음 |
| E3 | **AutoTuner 경량 구현** | ml_conf × ml_margin 2차원 탐색으로 최적 임계값 자동 발견. TP Recall ≥ 0.95 제약 하에서 auto_process_rate 최대화 | 중간 |

---

## 8. 우선순위 및 기대 효과

| 순위 | 작업 | 기대 효과 | 비고 |
|------|------|----------|------|
| **0** | §7.1~7.5 검증 질문 답변 | 후속 작업 방향 결정 | §7.1 DC 상한 = **음수** (확인됨) |
| **1** | A1~A3 측정 인프라 정비 | 정확한 측정 기반 확보 | **완료** ✅ |
| **2** | B1 PATH_HADOOP_001 정비 | **+0.012 실측** (예측 +0.001~0.005 대비 2~12배) | **완료** ✅ |
| **3** | C1~C4 ML Representation 개선 | **+0.03~0.06** (FP Recall 73→80%) | **다음 우선순위** — ablation 실험 필요 |
| **4** | D1~D4 DC 문맥 해소 계층 | Grid Sweep 결과 DC 개입 = F1 악화 | **보류** — RULE 독립 신호 확보 후 재검토 |
| **5** | E1 Sumologic JOIN | RULE 커버리지 0.6→5~15% | DC 재활성화의 전제조건 |
| **6** | E2~E3 RULE 독립 가치 | 장기 | 전체 오분류 데이터 분석 필요 |

---

## 9. 평가 지표 체계

> Wave 7 변경 (2026-03-23): **F1-macro를 단독 PASS/FAIL 판정 기준**으로 통일. TP Recall, FP Precision은 참고 지표로 변경.

| 지표 | 정의 | 역할 | 제약 |
|------|------|------|------|
| **F1-macro** | TP/FP F1 평균 | **PASS/FAIL 판정 기준** | ≥ 0.70 |
| **TP Recall** | 실제 TP 중 TP로 판정된 비율 | **참고 (안전 하한선)** | ≥ 0.75 (참고) |
| **FP Precision** | FP로 판정한 것 중 실제 FP 비율 | **참고** | ≥ 0.80 (참고, 기존 0.85에서 완화) |
| **FP Recall** | 실제 FP 중 FP로 판정된 비율 | **핵심 개선 대상** — 현재 ~88%, ML 단독 기준 | 최대화 |
| **DC 기여 정확도** | DC 개입 시 F1 변화 | **DC 존재 가치 측정** — Wave 7: **음수** (DC 개입 = F1 악화) | 양수면 DC 유의미 |

**Wave 7 실측 (temporal split, ~3.96M건):**
```
ML 단독:  F1=0.7914, FP precision=0.74, FP recall=0.88, TP recall=0.71
DC 기본:  F1=0.7746, FP precision=0.76, FP recall=0.73, TP recall=0.79  (F1 -0.017)
DC 최적:  F1=0.7915 = ML passthrough (ml_conf=0.0)
```

---

## 10. 반드시 지켜야 할 원칙

1. **특정 파일명/서버명을 하드코딩하는 룰은 과적합(cheating)이다.** `ncpsender_`나 `vNSME13` 같은 구체적 패턴이 아니라, `fname_has_date AND is_log_file` 같은 일반화 가능한 조건으로 설계해야 한다.

2. **오분류 샘플은 전체의 일부다.** 이미지에서 보이는 패턴이 전체 오분류의 몇 %인지는 전체 error_analysis.csv 분석 전까지 알 수 없다. 샘플에서 보이는 패턴에 과적합하지 말 것.

3. **ML representation 개선이 DC 로직 개선보다 ROI가 높다.** 현재 F1=0.78의 병목은 DC가 아니라, ML이 범주형 메타 shortcut(55.7%)에 의존하면서 파일/경로의 구조적 패턴(File Family)을 충분히 학습하지 못하는 것.

4. **TP 안전 원칙은 비타협.** 어떤 개선도 TP Recall(현재 ~95%)을 하락시키면 안 된다. "애매하면 TP"는 운영 정책이지 최적화 대상이 아니다.

5. **구현 전에 검증.** Phase 0 질문(§7.1~7.5)에 답하지 않고 구현에 착수하면 잘못된 방향에 시간을 투자할 위험이 있다.

6. **피처 추가(inventory)보다 표현 품질(representation) 우선.** 기존 피처가 왜 묻히는지 먼저 해결해야 한다.

---

## 11. 하지 말아야 할 해석

| 위험한 결론 | 왜 위험한가 |
|-----------|-----------|
| "룰 몇 개 더 넣으면 된다" | Rule coverage 0.6%, 독립 기여 0.057%. 단순 추가는 국소 최적화 |
| "피처만 더 넣으면 된다" | 이미 존재하는 좋은 피처가 활용 안 됨. inventory가 아니라 representation 문제 |
| "F1이 안 올랐으니 DC는 의미 없다" | 개입 면적(0.7%)과 측정 체계 한계 때문에 효과가 가려졌을 수 있음 |
| "오분류 패턴을 바로 룰로 만들면 된다" | 특정 파일명 하드코딩은 과적합 |
| "범주형 메타를 무조건 제거해야 한다" | ablation 결과 보기 전에는 판단 불가 |

---

## 12. 참조 문서

| 문서 | 역할 |
|------|------|
| `docs/decision_combiner_analysis.md` | 최초 원본 분석 (다른 AI agent) |
| `docs/decision_combiner_independent_analysis.md` | 독립 구조 분석 (다른 AI agent) |
| `docs/Architecture/05_decision_output.md` | 아키텍처 설계 사양 (§9.1~9.7) |
| `src/models/decision_combiner.py` | DC 추론 구현 |
| `scripts/run_report.py:440-475` | DC 평가 구현 (추론과 불일치) |
| `config/rule_stats.json` | 룰 통계 (실측과 괴리) |

---

*END OF DOCUMENT*
