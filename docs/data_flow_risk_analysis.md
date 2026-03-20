# Raw Data to Model: 데이터 플로우 및 일반화 성능 분석

**작성일**: 2026-03-20
**대상**: Label-only 학습 파이프라인 (Phase 1)

---

## 1. 분석 목적

모델 성능이 높게 나온다. 이것이 **실제 운영에서도 유지되는 성능인지** 확인하는 것이 목적이다.

데이터 편향 분석은 "왜 성능이 높은지"를 해석하는 보조 도구이지,
"편향이 있으니 모델이 틀렸다"는 결론을 내리기 위한 것이 아니다.

**핵심 질문**: 3~9월 데이터로 학습한 모델이 10~12월 데이터를 잘 예측하는가?

---

## 2. Raw 데이터 원천 및 구조

### 2.1 Label Data (정탐/오탐 Excel)

**원천**: 개인정보 검출 솔루션이 식별한 파일 목록을 분석가가 정탐/오탐으로 분류한 결과

**폴더 구조**:
```
data/raw/label/
  25년 정탐 (3월~12월)/3월/, 4월/, ..., 12월/   -> label_raw = "TP"
  25년 오탐 (3월~12월)/3월/, 4월/, ..., 12월/   -> label_raw = "FP"
```

**컬럼 (21개)**:

| 컬럼 | 타입 | 설명 | 모델링 사용 |
|------|------|------|------------|
| organization | str | 조직명 (CTO, NW 등) | 미사용 |
| ops_dept | str | 운영 부서 | 미사용 |
| service | str | 서비스 명칭 | 미사용 |
| label_review | str | 레이블 검수 의견 | 미사용 |
| fp_description | str | FP 사유 (611개 unique) | 미사용 |
| exception_requested | str | 예외 요청 Y/N | 미사용 |
| retention_period | str | 보관 기간 | 미사용 |
| **server_name** | str | 서버 호스트명 | **server_freq 파생** |
| **agent_ip** | str | 에이전트 IP | 미사용 (PK 구성만) |
| **pattern_count** | int | PII 검출 건수 | **pattern_count_log1p, bin 파생** |
| **file_path** | str | 파일 경로 (디렉토리) | **path 피처 12개 + TF-IDF 파생** |
| **file_name** | str | 파일명 | **fname 피처 3개 + TF-IDF 파생** |
| ssn_count | int | 주민번호 검출 수 | **pii_type_ratio 파생** |
| phone_count | int | 전화번호 검출 수 | **pii_type_ratio 파생** |
| email_count | int | 이메일 검출 수 | **pii_type_ratio 파생** |
| **file_created_at** | datetime | 파일 생성 시간 | **created_hour/weekday/month 파생** |
| **label_raw** | str | TP 또는 FP | **Target (y)** |
| label_work_month | str | 작업 월 (3월~12월) | split 기준 |
| _source_file | str | 원본 Excel 파일명 | 미사용 |
| pk_file | str | SHA256(4-field) | GroupShuffleSplit groups |
| pk_event | str | SHA256(5-field) | 미사용 |

### 2.2 Sumologic Data (검출 원본)

**원천**: Sumologic 검출 솔루션의 raw 이벤트 로그

**주요 컬럼**: `dfile_computername`, `dfile_agentip`, `dfile_filepath`, `dfile_filename`,
`dfile_patternname`, `dfile_inspectcount`, `dfile_inspectcontentwithcontext`,
`dfile_filecreatedtime`, `dfile_fileextension`, `dfile_filesize`

**Label 대비 추가 정보**: `dfile_inspectcontentwithcontext` (full_context_raw) - 검출된 텍스트 컨텍스트

| 항목 | Label Data | Sumologic Data |
|------|-----------|----------------|
| 텍스트 컨텍스트 | **없음** | `dfile_inspectcontentwithcontext` 존재 |
| PII 유형 상세 | pattern_count 합산값만 | `dfile_patternname` 개별 유형 |
| 파일 크기 | **없음** | `dfile_filesize` 존재 |
| 파일 확장자 | file_name에서 추출 | `dfile_fileextension` 별도 컬럼 |
| 검출 경과일 | **없음** | `dfile_durationdays` 존재 |
| Label | **있음** (폴더 기반) | **없음** (JOIN 필요) |

---

## 3. 데이터 플로우: Raw -> Silver -> Feature -> Model

```
[Raw Excel]
    |
    | LabelLoader.load_all()  (src/data/label_loader.py)
    |   - 폴더명으로 label_raw 부여 (정탐->TP, 오탐->FP)
    |   - ColumnNormalizer로 한글 컬럼명 영문화
    |   - file_created_at datetime 파싱
    |   - pk_file/pk_event SHA256 생성
    |
    v
[silver_label.parquet]  (21 컬럼)
    |
    | run_training.py Step 2: build_meta_features()
    |   - file_name -> fname_has_date, fname_has_hash, fname_has_rotation_num
    |   - pattern_count -> pattern_count_log1p, pattern_count_bin
    |   - pattern_count -> is_mass_detection, is_extreme_detection
    |   - ssn/phone/email_count -> pii_type_ratio
    |   - file_created_at -> created_hour, created_weekday, is_weekend, created_month
    |
    | run_training.py Step 2: extract_path_features()
    |   - file_path -> path_depth, extension, is_log_file, is_docker_overlay
    |   - file_path -> has_license_path, is_temp_or_dev, is_system_device
    |   - file_path -> is_package_path, has_cron_path, has_date_in_path
    |   - file_path -> has_business_token, has_system_token
    |
    v
[df + 메타/경로 피처]  (33+ 컬럼)
    |
    | run_training.py Step 5: build_features()
    |   - GroupShuffleSplit(groups=pk_file) train/test 분할
    |   - Phase 1 TF-IDF: file_name char(2,5) + file_path word(1,2)
    |   - Dense: create_file_path_features() (8개)
    |   - Dense: _PRECOMPUTED_DENSE_COLS (22개)
    |   - Dense: server_freq (train 기준 빈도 인코딩)
    |   - hstack(TF-IDF sparse + Dense)
    |
    v
[Feature Matrix]  (sparse, ~30+ 컬럼)
    |
    | run_training.py Step 6: train_lightgbm()
    |
    v
[best_model_v1.joblib]
```

---

## 4. 실제 모델링에 사용되는 피처 (Feature Importance 실 데이터)

| Rank | Feature | Importance | 원천 컬럼 | 파생 방법 |
|------|---------|------------|-----------|-----------|
| **1** | **server_freq** | **3884** | server_name | value_counts(normalize=True) |
| 2 | pattern_count_log1p | 945 | pattern_count | log1p() |
| 3 | path_depth | 828 | file_path | count("/") |
| 4 | created_hour | 562 | file_created_at | dt.hour |
| 5 | tfidf_phase1path_logs | 384 | file_path | TF-IDF("logs") |
| 6 | created_month | 363 | file_created_at | dt.month |
| 7 | pii_type_ratio | 350 | ssn/phone/email_count | ssn/(ssn+phone+email+1) |
| 8~28 | tfidf_phase1path_*, tfidf_fname_* 등 | 117~315 | file_path, file_name | TF-IDF tokens |

### Feature 블록 구성

| 블록 | 원천 | 피처 수 | Importance 비율 |
|------|------|---------|-----------------|
| server_freq | server_name | 1 | ~35% |
| path TF-IDF + 이진 플래그 | file_path | ~520 | ~30% |
| 검출통계 | pattern_count 등 | 6 | ~12% |
| 시간 | file_created_at | 4 | ~10% |
| fname TF-IDF | file_name | ~500 | ~14% |

---

## 5. 성능 해석: 왜 높은가

### 5.1 데이터의 구조적 특성

레이블 데이터의 핵심 특징:
- **server_name, file_path에 중복이 많음** — 같은 서버, 같은 경로에서 반복 검출
- 한번 검출 솔루션에 식별된 파일이 **방치되어 월마다 재검출**되는 경우 존재
- 결과적으로 **동일한 서버+경로 패턴이 항상 같은 label**을 가짐

이 구조에서 모델이 높은 성능을 내는 것은 자연스럽다:
- 업무 서버(CRM, 고객DB) → 실제 개인정보 → TP
- 인프라 서버(hadoop, docker, syslog) → 시스템 로그 → FP
- **서버의 용도 자체가 TP/FP를 결정하는 실제 원인이면 정당한 신호**

### 5.2 성능이 높은 것 자체가 문제는 아님

server_freq가 importance 1위(35%)라는 것은:
- 서버 용도가 TP/FP와 강하게 연관된다는 **도메인 사실을 반영**
- 이것을 "편향이니까 제거해야 한다"고 하면 오히려 모델 성능을 의미 없이 낮추는 것

### 5.3 진짜 질문: 미래 데이터에서도 유지되는가

성능이 높은 이유를 이해했으니, 실제로 걱정해야 할 시나리오:

| 시나리오 | 영향 | 발생 가능성 |
|----------|------|-------------|
| 기존 서버에서 계속 같은 패턴 검출 | 성능 유지 | 높음 |
| **새로운 서버 추가** | server_freq=0, 다른 피처로 판별 | 중간 |
| **기존 서버의 용도 변경** | 과거 학습과 불일치 | 낮음 |
| **방치 파일이 정리됨** (신규 검출만 남음) | 학습 분포와 달라질 수 있음 | 중간 |

---

## 6. 일반화 성능 검증 전략

### 6.1 핵심: Temporal Holdout (10~12월)

**가장 현실적인 검증**: 3~9월로 학습, 10~12월로 평가

이유:
- 실제 운영에서 모델은 **과거 데이터로 학습하고 미래 검출을 예측**
- 시간 순서를 지키는 split이 production 성능의 가장 정직한 추정
- 방치 파일 재검출 효과도 자연스럽게 포함 (같은 파일이 train/test 모두에 나타남 = 현실과 동일)

```
[3월] [4월] [5월] [6월] [7월] [8월] [9월] | [10월] [11월] [12월]
<-------------- Train (학습) ------------>  <--- Test (평가) --->
```

### 6.2 보조 검증: Split 전략별 성능 비교

Temporal holdout 결과를 다른 split과 비교하여 해석:

| Split 전략 | 목적 | 예상 |
|------------|------|------|
| **Temporal (10~12월)** | **운영 현실 시뮬레이션** | **가장 신뢰할 수 있는 수치** |
| Random GroupShuffleSplit | 현재 기본값 | 가장 높은 성능 (낙관적) |
| Server-level split | 새 서버 시나리오 | server_freq 의존도 측정 |

### 6.3 보조 검증: Feature Block Ablation

각 피처 블록의 기여도를 측정하여 **모델이 다양한 신호를 활용하는지** 확인:

| 실험 | 의미 |
|------|------|
| full model | 기준선 |
| full - server_freq | server 없이도 판별 가능한지 |
| server_freq only | server만으로 얼마나 되는지 |
| full - path TF-IDF | 경로 고유 토큰 의존도 |

**해석 기준**:
- server_freq 제거 후 성능이 유지되면 → 모델이 다양한 신호 활용 (건강)
- server_freq 제거 후 급락하면 → server에 과의존 (새 서버에 취약)
- 이것은 "모델이 틀렸다"가 아니라 **"취약점이 어디인지"를 파악하는 것**

---

## 7. 파생변수 생성 과정 검증

| 파생변수 | 생성 시점 | 전역 통계 사용 | 판정 |
|----------|-----------|---------------|------|
| fname_has_date/hash/rotation | Split 전 (전체 df) | 없음 (row-level regex) | 정상 |
| pattern_count_log1p/bin | Split 전 | 없음 (row-level 변환) | 정상 |
| pii_type_ratio | Split 전 | 없음 (row-level 산술) | 정상 |
| created_hour/weekday/month | Split 전 | 없음 (row-level datetime) | 정상 |
| path_depth, is_log_file 등 | Split 전 | 없음 (row-level regex) | 정상 |
| server_freq | Split 후 (train only) | train fold 빈도 | 정상 |
| Phase1 TF-IDF | Split 후 (train only) | train fold vocab | 정상 |

**결론**: 파생변수 생성 코드에 train-test 오염 없음.

---

## 8. 평가 파이프라인 알려진 버그 (별도 수정 필요)

코드 레벨에서 수정이 필요한 항목들:

| # | 문제 | 위치 | 영향 |
|---|------|------|------|
| B1 | PoC report가 학습 때와 다른 split 사용 | run_poc_report.py:489 | 리포트 수치 신뢰 불가 |
| B2 | server_freq를 전체 데이터로 재계산 | run_poc_report.py:453 | train-test 오염 |
| B5 | tp_label="TP" vs 실제 "TP-실제개인정보" 불일치 | run_poc_report.py:403 | 이진 지표 깨짐 |
| B4 | tune_model CV가 pk_file 그룹 무시 | trainer.py:279 | 하이퍼파라미터 낙관적 |

---

## 9. 실행 계획

```
1. 실 데이터로 diagnose_data_bias.py 실행
   → 데이터 구조 파악 (server/path/시간 분포)
   → Temporal split(10~12월) 성능 확인  ← 핵심
   → Feature ablation (server_freq 의존도 측정)

2. Temporal holdout 성능에 따라 판단:
   - F1 유지 (하락 < 5pp)  → 모델 신뢰 가능, 평가 코드만 수정
   - F1 하락 (5~15pp)      → server_freq 의존 완화 필요
   - F1 급락 (> 15pp)      → 피처 재설계 필요

3. 평가 파이프라인 버그 수정 (B1, B2, B5)
```
