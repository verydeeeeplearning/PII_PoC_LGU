## 11. Stage S6: Monitoring & Feedback

### 11.1 기능

월 배치 실행 후 자동으로 시스템 건강도 지표를 산출하고, 현업 피드백을 수집하여 룰 신뢰도 업데이트 + 모델 재학습에 활용한다.

### 11.2 필수 KPI 3계층 12종 (v1.1 확정)

**계층 A: 파이프라인 건전성 (기술적 일반화)**

| # | KPI | 산출 방식 | 알람 임계값 | 의미 |
|---|-----|----------|-----------|------|
| 1 | `row_parse_success_rate` | #파싱 성공 원본 row / #원본 row | < 0.95 | S1이 새 포맷에 대응 못함 |
| 2 | `fallback_rate` | #폴백 이벤트 / #전체 이벤트 | > 0.10 | 마스킹/포맷 변화 감지 |
| 3 | `quarantine_count` | 격리 행 수 | > 전월 대비 3배 | 입력 스키마 변화 |
| 4 | `feature_schema_match` | 추론 피처 차원 == 학습 차원 | ≠ | 즉시 장애 알림 |

**계층 B: 분포 이동 / 신규 패턴 (통계적 일반화)**

| # | KPI | 산출 방식 | 알람 임계값 | 의미 |
|---|-----|----------|-----------|------|
| 5 | `rule_match_rate` | RULE 매칭 건수 / 전체 건수 | 전월 대비 -10%p | 새 패턴이 룰 범위 밖으로 이탈 |
| 6 | `oov_rate_raw` | raw_text vocab 밖 토큰 비율 | > 0.30 | 새 도메인/키워드 대량 유입 |
| 7 | `oov_rate_path` | path_text vocab 밖 토큰 비율 | > 0.20 | 새 경로 체계 |
| 8 | `confidence_p10` | ML confidence 10번째 백분위수 | < 0.40 | 전반적 불확실성 증가 |

**계층 C: 안전장치 동작 (리스크)**

| # | KPI | 산출 방식 | 알람 임계값 | 의미 |
|---|-----|----------|-----------|------|
| 9 | `review_rate` | (NEEDS_REVIEW + TP_SAFE_OVERRIDE + OOD_SUSPECTED) / 전체 — 사람 개입 필요 전체 비율 | > 0.35 | 자동처리율 하락 → 공수 절감 위협 |
| 10 | `ood_rate` | OOD_SUSPECTED / 전체 (review_rate 구성 요소 분리 추적) | > 0.05 | 학습 분포 밖 데이터 대량 유입 |
| 11 | `rule_conflict_rate` | RULE_CONFLICT_WITH_ML 비율 | > 0.10 | RULE-ML 불일치 증가 |
| 12 | `auto_fp_precision_est` | 자동 처리 FP의 추정 precision (샘플링 QA) | < 0.90 | 자동 처리 품질 저하 |

**산출물:** `monthly_metrics.json` (S6에서 자동 생성, Git으로 버전 관리)

**기존 모니터링 지표 (유지):**

| 지표 | 산출 방식 | 목적 |
|------|----------|------|
| 클래스 분포 | `primary_class` 별 건수/비율 (월별/서버그룹별) | 분포 변화 감지 |
| RULE 매칭률 | 전체 대비 RULE 라벨 부여 비율 + 룰별 hit rate | 룰 커버리지 추적 |
| confidence 분포 | RULE/ML 각각의 신뢰도 분포 (평균, 중앙값, P10) | 불확실성 추세 |
| risk_flag 비율 | `NEEDS_REVIEW`, `TP_SAFE_OVERRIDE`, `RULE_CONFLICT`, `OOD_SUSPECTED` 각 비율 | 애매 케이스 추세 |
| drift 신호 | 경로 토큰, 도메인, 주요 키워드 빈도 변화 | 데이터 변화 감지 |

### 11.3 Self-Validation Loop — 현업 피드백 자동 대체 (v1.2 추가)

원칙 G(Zero-Human-in-the-Loop)에 따라, 현업이 수동으로 정탐/오탐을 판단하여 피드백하는 프로세스를 RULE↔ML 교차 검증으로 대체한다.

**방법 A: Cross-Source Validation (RULE↔ML 합의율)**

```python
def compute_cross_source_validation(rule_labels, ml_predictions) -> dict:
    """
    RULE과 ML이 독립적으로 동일한 결론에 도달한 비율을 산출.
    이 합의율이 곧 '자동 검증된 precision' 프록시다.
    """
    both_classified = rule_labels.merge(
        ml_predictions, on='pk_event', how='inner'
    )
    both_classified = both_classified[
        both_classified['rule_matched'] &
        both_classified['ml_top1_proba'] >= 0.70
    ]

    agreement = (
        both_classified['rule_primary_class'] ==
        both_classified['ml_top1_class_name']
    ).mean()

    # 룰별 합의율 → rule_confidence_lb 자동 갱신
    per_rule = both_classified.groupby('rule_id').apply(
        lambda g: (g['rule_primary_class'] == g['ml_top1_class_name']).mean()
    ).to_dict()

    return {
        'overall_agreement': agreement,
        'per_rule_agreement': per_rule,
    }
```

**방법 B: Temporal Consistency (시간 일관성)**

```python
def compute_temporal_consistency(current_month, previous_month) -> dict:
    """
    동일 pk_file의 이전 월 분류 결과와 비교.
    일관 → 기존 분류 정확도에 대한 자신감 ↑
    불일치 → drift 신호 → 자동 재학습 트리거
    """
    merged = current_month.merge(
        previous_month[['pk_file', 'primary_class']],
        on='pk_file', suffixes=('_curr', '_prev'), how='inner'
    )
    consistency = (
        merged['primary_class_curr'] == merged['primary_class_prev']
    ).mean()

    return {
        'temporal_consistency': consistency,
        'drift_detected': consistency < 0.85,
        'retrain_triggered': consistency < 0.75,
    }
```

**방법 C: Confident Learning (라벨 정제)**

§22에서 상세 설명. 학습 데이터에서 모델 예측과 라벨 불일치 케이스를 자동 탐지하여 라벨을 자동 수정하거나 학습에서 제외한다.

### 11.4 Auto-Precision-Estimator — 샘플링 QA 자동 대체 (v1.2 추가)

원칙 G에 따라, 사람이 자동 처리된 FP N건을 수동 확인하는 대신, RULE↔ML 합의율을 precision 프록시로 자동 산출한다.

```python
def auto_fp_precision_estimate(rule_labels, ml_predictions) -> dict:
    """
    auto_fp_precision_est 자동 산출:
    - 분모: 자동 처리된 FP 전체 건수
    - 분자: 그 중 RULE과 ML이 모두 동일 FP 클래스를 지목한 건수
    """
    auto_fp = rule_labels[
        (rule_labels['risk_flag'].isna()) &
        (rule_labels['primary_class'].str.startswith('FP-'))
    ]

    if len(auto_fp) == 0:
        return {'precision_est': 1.0, 'sample_size': 0}

    auto_fp_with_ml = auto_fp.merge(
        ml_predictions[['pk_event', 'ml_top1_class_name']],
        on='pk_event', how='left'
    )

    agreed = (
        auto_fp_with_ml['primary_class'] ==
        auto_fp_with_ml['ml_top1_class_name']
    ).sum()

    precision_est = agreed / len(auto_fp_with_ml)

    return {
        'precision_est': round(precision_est, 4),
        'sample_size': len(auto_fp_with_ml),
        'quality_status': 'GOOD' if precision_est >= 0.90 else 'DEGRADED',
        'action': None if precision_est >= 0.90 else 'tau_conservatize',
    }
```

**동작:**
- 합의율 ≥ 0.90 → "자동 처리 품질 양호"로 판정
- 합의율 < 0.90 → 자동으로 임계값 보수화 (Auto-Tuner §9.7 연동)

### 11.5 Auto-Remediation Playbook (v1.2 추가)

원칙 G에 따라, KPI 알람 발생 시 사람이 원인 분석 후 조치하는 대신, 알람별 자동 조치를 사전에 매핑한다.

```python
AUTO_REMEDIATION_PLAYBOOK = {
    'row_parse_success_rate': {
        'condition': lambda v: v < 0.95,
        'actions': [
            'expand_fallback_levels',      # 폴백 레벨 확대
            'collect_parser_logs',          # 파서 로그 자동 수집
            'schedule_retrain',             # 재학습 예약
        ],
    },
    'fallback_rate': {
        'condition': lambda v: v > 0.10,
        'actions': [
            'auto_update_parser_config',    # parser config 패턴 확장 시도
        ],
    },
    'quarantine_count': {
        'condition': lambda v, prev: v > prev * 3,
        'actions': [
            'trigger_auto_schema_detector', # Auto-Schema-Detector 트리거
        ],
    },
    'feature_schema_match': {
        'condition': lambda v: v is False,
        'actions': [
            'halt_inference',               # 추론 즉시 중단
            'rebuild_features',             # 자동 피처 재빌드
            'update_feature_schema',        # 스키마 갱신
        ],
    },
    'oov_rate_raw': {
        'condition': lambda v: v > 0.30,
        'actions': [
            'refresh_tfidf_vocabulary',     # TF-IDF vocabulary 자동 갱신
            'trigger_retrain',              # 모델 재학습 트리거
        ],
    },
    'confidence_p10': {
        'condition': lambda v: v < 0.40,
        'actions': [
            'conservatize_tau',             # TAU 자동 보수화 (Auto-Tuner)
        ],
    },
    'review_rate': {
        'condition': lambda v: v > 0.35,
        'actions': [
            'auto_tune_tau',                # TAU 자동 조정
            'trigger_retrain_if_persistent',# 지속 시 재학습 트리거
        ],
    },
    'ood_rate': {
        'condition': lambda v: v > 0.05,
        'actions': [
            'trigger_retrain',              # 모델 재학습 트리거
            'cluster_unknowns',             # UNKNOWN 클러스터링
        ],
    },
    'rule_conflict_rate': {
        'condition': lambda v: v > 0.10,
        'actions': [
            'downgrade_conflicting_rules',  # 해당 룰 confidence 자동 하향
        ],
    },
    'auto_fp_precision_est': {
        'condition': lambda v: v < 0.90,
        'actions': [
            'conservatize_tau',             # TAU 보수화
            'trigger_retrain',              # 재학습 트리거
        ],
    },
}

class AutoRemediator:
    """KPI 알람 발생 시 자동 조치를 실행한다."""

    def check_and_remediate(self, metrics: dict,
                            prev_metrics: dict = None) -> list:
        actions_taken = []
        for kpi, playbook in AUTO_REMEDIATION_PLAYBOOK.items():
            value = metrics.get(kpi)
            if value is None:
                continue

            condition = playbook['condition']
            # 전월 대비 조건인 경우
            if 'prev' in condition.__code__.co_varnames:
                prev_value = (prev_metrics or {}).get(kpi, value)
                triggered = condition(value, prev_value)
            else:
                triggered = condition(value)

            if triggered:
                for action in playbook['actions']:
                    self._execute_action(action, kpi, value)
                    actions_taken.append({
                        'kpi': kpi, 'value': value,
                        'action': action, 'timestamp': datetime.now()
                    })

        return actions_taken
```

**알람별 자동 조치 요약:**

| KPI | 알람 조건 | 자동 조치 |
|-----|----------|----------|
| `row_parse_success_rate` < 0.95 | 파싱 실패 급증 | 폴백 확대 + 파서 로그 수집 + 재학습 예약 |
| `fallback_rate` > 0.10 | 포맷 변화 | parser config 자동 갱신 시도 |
| `quarantine_count` > 3× | 스키마 변화 | Auto-Schema-Detector 트리거 |
| `feature_schema_match` ≠ | 피처 불일치 | 추론 중단 + 피처 재빌드 + 스키마 갱신 |
| `oov_rate_raw` > 0.30 | 신규 토큰 대량 | TF-IDF vocabulary 갱신 + 재학습 |
| `confidence_p10` < 0.40 | 전반적 불확실 | TAU 자동 보수화 |
| `review_rate` > 0.35 | 자동처리율 하락 | TAU 자동 조정 또는 재학습 |
| `ood_rate` > 0.05 | OOD 대량 유입 | 재학습 + UNKNOWN 클러스터링 |
| `rule_conflict_rate` > 0.10 | RULE-ML 불일치 | 해당 룰 confidence 자동 하향 |
| `auto_fp_precision_est` < 0.90 | 품질 저하 | TAU 보수화 + 재학습 |

### 11.6 Rationale

**설명가능성의 "지속성"을 보장한다.**

- 오늘은 잘 되다가 다음 달에 갑자기 성능이 떨어지면 운영 신뢰가 깨진다. drift(도메인/경로/로그 포맷 변화)를 조기에 감지해야 한다.
- risk_flag 비율 추이는 가장 빠른 건강도 지표다. `NEEDS_REVIEW` 비율이 갑자기 증가하면 데이터 분포가 변했거나 새로운 패턴이 등장했다는 신호다.

**룰은 시간이 지날수록 바뀌는 게 정상이다.**

- 내부 도메인 추가/시스템 변경/벤더 변경 등으로 룰셋은 지속적으로 업데이트되어야 한다. 룰셋을 YAML로 버전 관리하고, 룰 신뢰도(precision_lb)를 피드백 기반으로 갱신해야 안정적이다.
- ML이 반복적으로 잡는 패턴(높은 확신으로 동일 클래스를 예측하는 반복 패턴)은 룰로 승격(promotion)시킬 수 있다. 이렇게 하면 시간이 지남에 따라 룰 커버리지가 넓어지고, ML이 처리해야 하는 애매 영역이 줄어든다.

### 11.7 "패턴 비완전"을 전제로 한 추가 안전장치

전처리는 한 번에 끝내기 어렵고, 새로운 패턴은 반드시 나온다는 전제로 아래 장치를 설계한다.

**A. 전처리 버전/추적**

모든 주요 변환에 대해 `*_raw`, `*_norm`, `*_redacted` 컬럼을 남기고, 샘플링 로그(예: 하루 1,000건)로 "변환 전/후 비교"를 저장한다. 원문이 보존되므로 전처리 로직이 바뀌어도 재파생이 가능하다.

**B. OOV/신규 토큰 모니터링**

TF-IDF vocab에 거의 안 잡히는 토큰, placeholder로도 안 잡히는 토큰을 주기적으로 수집한다. 이를 룰/키워드/정규화 패턴 확장 후보로 사용한다. 새로운 도메인, 새로운 로그 포맷이 등장하면 이 채널에서 조기에 감지된다.

**C. "Unknown-like" 신호를 피처로 남김**

- placeholder 치환이 많이 발생한 비율 (예: `<HASH>` 비율, `<NUM*>` 비율)
- `rule_has_conflict`, `rule_candidates_count`
- 이 피처들은 모델이 "이건 낯선 케이스"를 감지하는 힌트가 된다

**D. 평가 누수 방지**

이벤트 단위 랜덤 split 대신 **pk_file 기준 Group Split**을 기본으로 사용한다. 같은 파일의 반복 패턴 때문에 이벤트 단위 split은 점수를 과대평가한다.

---

## 12. 분류 체계 & 클래스 설계

### 12.1 2단 구조: Primary Class (고정) + Reason Code (확장 가능)

이 시스템의 클래스 체계는 **모델 학습 안정성**과 **운영 안정성**을 동시에 만족하기 위해 2단 구조로 설계한다.

#### 실제 데이터 확정 클래스 체계 (2026-03, `classify_fp_description.py` 기준)

실제 fp_description 611개 unique값을 정규식 규칙 기반으로 분류한 결과, 아래 **6개 FP 클래스 + TP + UNKNOWN** 체계가 확정되었다. 이는 초기 Canonical 8-class(FP-숫자나열/코드, FP-타임스탬프, FP-bytes크기 등)를 실제 데이터 분포에 맞게 재편한 것이다.

**FP 클래스 (fp_description에서 도출, 오탐 파일 전용):**

| # | 클래스명 | 설명 | 초기 설계 대비 |
|---|----------|------|--------------|
| 1 | **FP-파일없음** | 파일/경로가 이미 미존재 (삭제됨, 경로 오류) | **신규** — 운영 현실 반영 |
| 2 | **FP-이메일패턴** | `@` 문자/kerberos/내부도메인/OSS도메인이 이메일로 오인식 | FP-내부도메인 + FP-OS저작권(이메일) 통합 |
| 3 | **FP-숫자패턴** | 타임스탬프·bytes크기·일련번호·법인번호가 주민번호/전화번호 정규식에 매칭 | FP-숫자나열/코드 + FP-타임스탬프 + FP-bytes크기 **3→1 통합** |
| 4 | **FP-라이브러리** | OS 패키지(RPM/yum/apt/conda), docker 이미지, npm/pip 패키지, 오픈소스 설치파일 | FP-OS저작권 범위 대폭 확장 |
| 5 | **FP-더미테스트** | 테스트/더미/샘플/성능시험/분析용 임시파일 | FP-더미데이터 ≈ (이름 변경) |
| 6 | **FP-시스템로그** | 서비스로그·배치·DB로그·백업솔루션·인프라로그 등 운영 시스템 데이터 | FP-패턴맥락 범위 대폭 확장 |
| — | **UNKNOWN** | 위 규칙에 매핑되지 않는 fp_description | 후처리 정책 클래스 |

**TP 클래스 (정탐 파일 출처에서 직접 부여):**

| # | 클래스명 | 설명 |
|---|----------|------|
| 0 | **TP-실제개인정보** | 실제 개인정보 → 파기 대상 |

> `classify_fp_description.py`는 오탐 파일의 fp_description만 처리함. TP는 정탐 파일 출처(`25년 정탐/` 폴더)에서 자동 부여되므로 fp_description 분류 대상이 아님.

**초기 Canonical 8-class → 실제 확정 6-class 매핑:**

| 초기 설계 클래스 | 실제 확정 클래스 | 변경 이유 |
|----------------|----------------|----------|
| FP-숫자나열/코드 | FP-숫자패턴 | 타임스탬프/bytes크기와 구분 불가 → 통합 |
| FP-타임스탬프 | FP-숫자패턴 | 동일 패턴 공간 |
| FP-bytes크기 | FP-숫자패턴 | 동일 패턴 공간 |
| FP-내부도메인 | FP-이메일패턴 | 이메일 형식 오인식으로 통합 |
| FP-OS저작권 | FP-이메일패턴 / FP-라이브러리 | 이메일 부분 → FP-이메일패턴, 설치파일 부분 → FP-라이브러리 |
| FP-패턴맥락 | FP-시스템로그 | 실제 케이스 대부분이 시스템/서비스 로그 |
| FP-더미데이터 | FP-더미테스트 | 이름 명확화 |
| (없음) | FP-파일없음 | 운영 현실에서 다수 발견 → 신규 추가 |

#### Phase별 클래스 운용 전략

**Phase 1 (레이블 단독 — 경로/메타 피처로 분리 가능한 클래스만):**

EDA 분석 결과, 텍스트 없이 경로/메타 피처(I~Q)만으로는 아래 클래스 분리가 가능하다.

| Phase 1 클래스 | Phase 2 매핑 | 분리 근거 |
|---------------|-------------|----------|
| TP-실제개인정보 | TP-실제개인정보 | 정탐 파일 출처에서 직접 매핑 |
| FP-시스템로그 | FP-시스템로그 | `is_log_file` + 시스템 경로 + 대량 검출(`pattern_count > 10,000`) |
| FP-라이브러리 | FP-라이브러리 | `/usr/share/doc/`, `/usr/share/licenses/`, docker overlay 경로 |
| FP-더미테스트 | FP-더미테스트 | `/tmp/`, `/test/`, `/sample/`, `/dev/`, `/tests/` 경로 |
| FP-기타 | FP-이메일패턴, FP-숫자패턴, FP-파일없음 | **텍스트 의존** — Phase 2에서 세분화 |

**피처 공간별 클래스 분리 가능성 (Phase 1 기준):**

| 영역 | 클래스 | 분리 가능 이유 |
|------|--------|--------------|
| 분리 가능 | TP vs 전체 FP | 경로 유형(업무경로 vs 시스템경로) + 대량 검출 |
| 분리 가능 | FP-라이브러리 | `/usr/share/doc/`, `/usr/share/licenses/`, docker overlay 경로 플래그 |
| 분리 가능 | FP-시스템로그 | 시스템 로그 경로 + `pattern_count > 10,000` 조합 |
| 분리 가능 | FP-더미테스트 | `/tmp/`, `/test/`, `/sample/` 경로 |
| **분리 불가** | **FP-이메일패턴** | **이메일 도메인(@lguplus.co.kr vs @redhat.com)이 레이블 데이터에 없음** |
| **분리 불가** | **FP-숫자패턴** | **텍스트 주변 키워드(bytes, expiryDate, 일련번호)가 레이블 데이터에 없음** |
| **분리 불가** | **FP-파일없음** | **file_path 유효성은 경로 문자열만으로 판단 불가 — 런타임 확인 필요** |

> 이 분리 불가 영역은 데이터 규모 문제가 아니라 **피처 공간의 구조적 한계**다. JOIN을 통해 텍스트 피처(`dfile_inspectcontentwithcontext`, 이메일 도메인)가 추가되어야 비로소 결정 경계가 형성된다.

**Reason Code (7+알파, 확장 가능):** 룰/설명/운영을 위한 세부 분류

| Primary Class | Reason Code 예시 |
|--------------|-----------------|
| FP-파일없음 | `FILE_PATH_NOT_EXIST`, `NO_FILE_RECORD`, `PATH_INVALID` |
| FP-이메일패턴 | `INT_DOMAIN_LGUPLUS`, `INT_DOMAIN_BDP`, `KERBEROS_TOKEN`, `OS_DOMAIN_REDHAT` |
| FP-숫자패턴 | `EPOCH_13_DIGITS`, `KEY_EXPIRYDATE`, `BYTES_KEYWORD_ADJACENT`, `VERSION_NUMBER`, `SERIAL_CODE` |
| FP-라이브러리 | `OS_DOMAIN_FEDORA`, `RPM_PACKAGE`, `DOCKER_IMAGE`, `CONDA_PACKAGE`, `NPM_PACKAGE` |
| FP-더미테스트 | `JSON_TEST_EMAIL`, `CONFIG_PLACEHOLDER`, `SAMPLE_DATA`, `DEV_FIXTURE` |
| FP-시스템로그 | `LOG_STRUCTURE_FP`, `SYSTEM_ACCOUNT`, `BATCH_LOG`, `DB_LOG`, `BACKUP_LOG` |

### 12.2 UNKNOWN 운영 정책 (v1.1 추가)

7클래스(6 FP + 1 TP) 체계에 8번째 운영 클래스 `UNKNOWN`을 추가한다. 이는 모델 학습 클래스가 아니라 **후처리 정책 클래스**다.

```
모델 학습: 7클래스 (6 FP + 1 TP)
운영 출력: 8클래스 (7 + UNKNOWN)
UNKNOWN 조건: OOD 감지 OR 극도의 불확실 OR 파싱 폴백 사용
UNKNOWN 처리 (v1.2): UnknownAutoProcessor(§9.6)가 자동 처리
  → 가장 가까운 FP 클래스로 자동 배정 OR 보수적 TP 확정
  → Auto-Taxonomy-Manager(§12.3b)가 축적 패턴 자동 분석
```

UNKNOWN 케이스가 특정 패턴으로 충분히 축적되면(예: 월 500건 이상 동일 유형), Auto-Taxonomy-Manager가 자동으로 다음 중 하나를 수행한다.

- 기존 7클래스 중 하나에 Reason Code 자동 추가 (클래스 확장 불필요)
- 새로운 Primary Class 후보 자동 등록 + 재학습 트리거 (성능 하락 시 자동 롤백)

### 12.3 라벨 거버넌스 규칙 (v1.2 자동화)

Dataset B/C에서 현업이 새 라벨을 만들 가능성에 대비한다. v1.2에서 원칙 G(Zero-Human-in-the-Loop)에 따라 라벨 거버넌스를 자동화한다.

```yaml
# label_governance.yaml (v1.2)
rules:
  - name: "모든 라벨은 Canonical 7(6 FP+1 TP) + UNKNOWN으로 자동 매핑"
    action: |
      1. Auto-Mapper가 새 라벨을 기존 7 Primary Class와 퍼지 매칭
      2. 유사도 ≥ 0.7 → 해당 클래스 + 새 Reason Code 자동 생성
      3. 유사도 < 0.7 → UNKNOWN + Auto-Taxonomy-Manager로 자동 전달

  - name: "매핑 테이블은 자동 버전 관리"
    action: |
      label_mapping.yaml을 Git으로 자동 관리
      Auto-Mapper가 갱신 시 자동 commit (감사 추적)

  - name: "Taxonomy 리뷰 자동화 (월별)"
    action: |
      Auto-Taxonomy-Manager가 UNKNOWN 축적 현황을 월별 자동 분석
      클러스터링 → 자동 Reason Code 확장 또는 새 Primary Class 후보 등록
```

### 12.3a Auto-Mapper — label_mapping 자동 업데이트 (v1.2 추가)

```python
from difflib import SequenceMatcher

class AutoMapper:
    """
    새 라벨 텍스트를 기존 7 Primary Class와 퍼지 매칭하여
    자동으로 매핑한다.
    """
    SIMILARITY_THRESHOLD = 0.70

    # Primary Class 설명 (매칭 기준)
    CLASS_DESCRIPTIONS = {
        'TP-실제개인정보': ['실제', '개인정보', '진짜', 'PII', '정탐'],
        'FP-파일없음': ['파일없음', '경로없음', '미존재', 'not found', '없음'],
        'FP-이메일패턴': ['이메일', '도메인', 'kerberos', 'lguplus', '내부도메인', 'redhat', '@', '오인식'],
        'FP-숫자패턴': ['숫자', '나열', '코드', '버전', '일련번호', '타임스탬프', 'timestamp', 'bytes', '바이트', 'epoch'],
        'FP-라이브러리': ['RPM', 'docker', 'conda', 'npm', 'pip', '패키지', '설치파일', 'yum', 'apt', 'OS'],
        'FP-더미테스트': ['더미', '테스트', 'test', 'sample', '개발용', '분析'],
        'FP-시스템로그': ['시스템', '로그', '배치', 'DB', '백업', '인프라', '서비스로그', '패턴', '맥락'],
    }

    def map_label(self, new_label: str) -> dict:
        """새 라벨을 기존 Primary Class에 자동 매핑."""
        best_class, best_score = None, 0.0

        for primary_class, keywords in self.CLASS_DESCRIPTIONS.items():
            # 키워드 겹침 점수
            keyword_score = sum(
                1 for kw in keywords if kw in new_label.lower()
            ) / len(keywords)

            # edit distance 유사도
            edit_score = SequenceMatcher(
                None, new_label.lower(), primary_class.lower()
            ).ratio()

            combined = 0.6 * keyword_score + 0.4 * edit_score
            if combined > best_score:
                best_score = combined
                best_class = primary_class

        if best_score >= self.SIMILARITY_THRESHOLD:
            reason_code = self._generate_reason_code(new_label, best_class)
            return {
                'mapped_to': best_class,
                'reason_code': reason_code,
                'similarity': round(best_score, 3),
                'auto_mapped': True,
            }
        else:
            return {
                'mapped_to': 'UNKNOWN',
                'reason_code': f'UNMAPPED_LABEL_{new_label[:20]}',
                'similarity': round(best_score, 3),
                'auto_mapped': True,
                'forward_to': 'AutoTaxonomyManager',
            }

    def _generate_reason_code(self, label: str, primary_class: str) -> str:
        clean = label.upper().replace(' ', '_').replace('-', '_')[:30]
        prefix = primary_class.split('-')[1][:4].upper() if '-' in primary_class else 'TP'
        return f"AUTO_{prefix}_{clean}"
```

### 12.3b Auto-Taxonomy-Manager — 분기별 리뷰 자동화 (v1.2 추가)

```python
from sklearn.cluster import HDBSCAN
import numpy as np

class AutoTaxonomyManager:
    """
    UNKNOWN 케이스를 월별 자동 클러스터링하여
    기존 클래스에 Reason Code를 추가하거나,
    새 Primary Class 후보를 자동 등록한다.
    """
    MIN_CLUSTER_SIZE = 500
    MAX_DISTANCE_TO_EXISTING = 2.0

    def analyze_unknowns(self, unknown_events: pd.DataFrame,
                         class_centroids: dict,
                         current_taxonomy: dict) -> list:
        """월별 UNKNOWN 자동 분석."""
        if len(unknown_events) < self.MIN_CLUSTER_SIZE:
            return [{'action': 'wait', 'reason': 'insufficient_unknowns'}]

        # HDBSCAN 클러스터링
        features = np.vstack(unknown_events['dense_features'].values)
        clusterer = HDBSCAN(min_cluster_size=self.MIN_CLUSTER_SIZE)
        labels = clusterer.fit_predict(features)

        actions = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # noise
                continue

            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum()
            cluster_center = features[cluster_mask].mean(axis=0)

            if cluster_size < self.MIN_CLUSTER_SIZE:
                actions.append({
                    'action': 'wait',
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'reason': 'cluster_too_small',
                })
                continue

            # 가장 가까운 기존 Primary Class 찾기
            nearest_class, nearest_dist = self._find_nearest(
                cluster_center, class_centroids
            )

            if nearest_dist < self.MAX_DISTANCE_TO_EXISTING:
                # 기존 클래스에 새 Reason Code 추가
                reason_code = f"AUTO_CLUSTER_{cluster_id}_{nearest_class[:6]}"
                actions.append({
                    'action': 'extend_reason_code',
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'target_class': nearest_class,
                    'new_reason_code': reason_code,
                    'distance': nearest_dist,
                })
            else:
                # 새 Primary Class 후보로 자동 등록
                actions.append({
                    'action': 'propose_new_class',
                    'cluster_id': cluster_id,
                    'size': cluster_size,
                    'nearest_existing': nearest_class,
                    'distance': nearest_dist,
                    'retrain_triggered': True,
                })

        return actions

    def execute_actions(self, actions: list, taxonomy: dict,
                        model_pipeline) -> dict:
        """분석 결과에 따라 자동 실행."""
        for action in actions:
            if action['action'] == 'extend_reason_code':
                # Reason Code 자동 추가
                taxonomy['reason_codes'][action['target_class']].append(
                    action['new_reason_code']
                )
            elif action['action'] == 'propose_new_class':
                # 새 Primary Class로 재학습 트리거
                new_num_class = len(taxonomy['primary_classes']) + 1
                retrain_result = model_pipeline.retrain(
                    num_class=new_num_class
                )
                # 성능 하락 시 롤백
                if retrain_result['macro_f1'] < taxonomy['baseline_f1']:
                    model_pipeline.rollback()
                    action['action'] = 'rollback_new_class'

        return taxonomy
```

**Auto-Taxonomy-Manager 흐름:**

```
1. UNKNOWN 케이스 dense feature → HDBSCAN 클러스터링 (월별)
2. 클러스터 크기 ≥ 500건 AND 기존 Primary Class에 가까움:
   → 해당 클래스에 새 Reason Code 자동 추가
   → 해당 케이스를 해당 클래스로 재분류
   → Auto-Rule-Promoter에 패턴 전달
3. 클러스터 크기 < 500건:
   → UNKNOWN 유지 → 다음 월 재평가
4. 어떤 Primary Class에도 안 맞는 대규모 클러스터:
   → 새 Primary Class 후보로 자동 등록
   → 자동 재학습 (num_class += 1)
   → 재학습 후 성능 하락 시 → 자동 롤백
```

### 12.4 Rationale

**왜 클래스 수를 8개로 고정하는가:**

- 클래스 수를 무한히 늘리면 ML 학습이 어려워지고, 데이터 불균형이 폭발한다. 각 클래스에 충분한 학습 데이터가 있어야 Boosting 모델이 안정적으로 분류할 수 있다.
- 7클래스면 Softmax 확률 분포가 의미 있는 정보를 담을 수 있다. 20~30클래스로 늘리면 대부분의 확률이 0에 가까워져서 margin/entropy 기반 불확실성 감지가 어려워진다.

**왜 Reason Code를 별도로 두는가:**

- 새로운 오탐 유형이 발견되면 Reason Code만 추가하면 된다 (모델 재학습 없이 룰로 먼저 대응 가능).
- 운영팀은 Primary Class보다 Reason Code 수준에서 일을 한다 ("이건 RedHat 도메인이네", "이건 Kerberos 토큰이네"). 세밀한 사유코드가 있어야 구체적 조치가 가능하다.
- 결과적으로 "분류 성능(Primary Class)" + "설명 확장(Reason Code)" + "운영 유지보수(YAML 업데이트)"를 동시에 만족하는 구조다.

### 12.5 PII 유형별 주요 오탐 클래스 매핑

| PII 유형 | 주요 오탐 클래스 | 대표 Reason Code |
|----------|-----------------|-----------------|
| 주민등록번호 | FP-숫자패턴 | `VERSION_NUMBER`, `EPOCH_13_DIGITS`, `BYTES_KEYWORD_ADJACENT` |
| 이메일 | FP-이메일패턴, FP-라이브러리, FP-더미테스트 | `INT_DOMAIN_LGUPLUS`, `OS_DOMAIN_REDHAT`, `JSON_TEST_EMAIL` |
| 휴대폰번호 | FP-숫자패턴 | `SERIAL_CODE`, `FILESIZE_PATTERN` |

---
