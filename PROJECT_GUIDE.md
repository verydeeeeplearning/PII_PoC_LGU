# PII 오탐 개선 AI 모델링 프로젝트 운영 가이드

**문서 버전:** 1.6 (ML 신뢰성 보강 반영)
**최종 수정일:** 2026-03-16
**기준 문서:** `docs/Architecture/` (섹션별 분리 — 상세는 각 파일 참조)

---

## 1. 문서 목적

이 문서는 현재 리포지토리 실행 스크립트를 기준으로, 아키텍처 표준(`S0~S6`, Labeler 체인)과 실제 운영 절차를 연결하는 실행 가이드입니다.

- 코드/스크립트의 "현재 동작"
- 아키텍처 문서의 "목표 운영 형태"

두 축을 함께 기록하여 문서 간 불일치를 줄이는 것을 목표로 합니다.

---

## 2. 아키텍처 정합성 기준

### 2.1 용어 통일

- `Filter`(과거 용어) -> `Labeler`(표준 용어)
- Layer 1/2는 "제거"가 아니라 "라벨 + 근거 기록"이 목적

### 2.2 클래스 체계

학습 클래스 (2026-03 확정, 7개):

1. `TP-실제개인정보`
2. `FP-파일없음`
3. `FP-이메일패턴`
4. `FP-숫자패턴`
5. `FP-라이브러리`
6. `FP-더미테스트`
7. `FP-시스템로그`

운영 정책 클래스(추가): `UNKNOWN`

> 초기 Canonical 8-class → 실제 데이터 분석 후 재편. 상세 매핑: `docs/Architecture/06_monitoring_classification.md §12.1`

### 2.3 스테이지 맵

| Stage | 아키텍처 역할 | 현재 진입점 |
|------|----------------|------------|
| S0 | Raw Ingest + Schema 정합화 | `run_data_pipeline.py --source label\|detection\|joined` |
| S1 | Normalize & Parse | `src/data/label_loader.py`, `src/data/s1_parser.py` (SHA256 pk) |
| S2 | Feature Prep | `src/features/pipeline.py` |
| S3a | RULE Labeler | `src/filters/rule_labeler.py` (12 룰, `--use-filter` 시 활성) |
| S3b | ML Labeler | `scripts/run_training.py`, `src/models/trainer.py` |
| S4 | Decision Combiner | 설계 문서 기준 (통합 구현 후속) |
| S5 | Output Writer | 설계 문서 기준 (통합 구현 후속) |
| S6 | Monitoring/Feedback | 설계 문서 기준 (자동화 후속) |

---

## 3. 사전 조건

### 3.1 환경

- 폐쇄망(Air-gapped)
- CPU 전용 서버 (GPU 없음)
- Python 3.12 권장

### 3.2 설치 및 검증

```bash
bash scripts/setup_env.sh
python scripts/verify_env.py
```

### 3.3 데이터 배치 경로

- `data/raw/label/`: 레이블 Excel (조직별/월별 정탐·오탐 취합 보고서)
- `data/raw/dataset_a/`: Sumologic 검출 원본 xlsx (`dfile_*` 컬럼 구조)
- `data/raw/dataset_b/`: 소만사 오탐 레이블링 (레거시, 현재 미사용)
- `data/raw/dataset_c/`: 현업 피드백(참고, 미활용)

---

## 4. 설정 파일 체크리스트

### 4.1 필수 설정 파일

- `config/feature_config.yaml`
- `config/model_config.yaml`
- `config/filter_config.yaml`
- `config/rules.yaml`
- `config/schema_registry.yaml`

### 4.2 우선 확인 항목

`config/feature_config.yaml`

- `data.pk_columns.primary`
- `data.text_column`
- `data.label_column`
- `data.file_path_column`
- `datasets.dataset_a.columns_mapping`
- `datasets.dataset_b.columns_mapping`

`config/model_config.yaml`

- `xgboost.default_params`
- `lightgbm.default_params`
- `calibration`
- `ood`
- `unknown_routing`

`config/filter_config.yaml` / `config/rules.yaml`

- 내부 도메인/더미 도메인/저작권 도메인
- timestamp/bytes/version 룰 정규식
- reason_code 및 우선순위

---

## 5. 실행 런북 (S0~S6)

## S0. Raw Ingest & Validation

### 목적

- 원본 데이터 로드 및 컬럼 정규화
- PK(SHA256) 계산
- Silver Parquet 저장

### 실행 (3가지 소스)

```bash
# 레이블 Excel → silver_label.parquet (pk_file/pk_event SHA256 포함)
python scripts/run_data_pipeline.py --source label

# Sumologic xlsx → silver_detections.parquet (dfile_* 표준화 + pk_file SHA256)
python scripts/run_data_pipeline.py --source detection

# silver_label + silver_detections → silver_joined.parquet (pk_file inner JOIN)
python scripts/run_data_pipeline.py --source joined
```

### 산출물

| 소스 | 파일 | 설명 |
|------|------|------|
| label | `data/processed/silver_label.parquet` | pk_file, pk_event, label_raw, 메타 피처 |
| detection | `data/processed/silver_detections.parquet` | pk_file, full_context_raw, dfile_* 표준명 |
| joined | `data/processed/silver_joined.parquet` | 위 두 파일 pk_file 기준 inner JOIN |

### PK 알고리즘 (SHA256, 통일)

```
pk_file  = SHA256(server_name|agent_ip|file_path|file_name)         # 4-field
pk_event = SHA256(server_name|agent_ip|file_path|file_name|file_created_at)  # 5-field
```

### Sumologic 컬럼 매핑

| dfile_* 원본 컬럼 | 표준 컬럼 |
|------------------|----------|
| dfile_computername | server_name |
| dfile_agentip | agent_ip |
| dfile_filedirectedpath | file_path |
| dfile_filename | file_name |
| dfile_filecreatedtime | file_created_at |
| dfile_inspectcontentwithcontext | full_context_raw |
| dfile_patternname | pii_type_inferred |
| dfile_patterncnt | pattern_count |

---

## S1. Normalize & Parse (고급 모듈)

### 목적

- 1행=1검출 이벤트 파싱
- 3단 폴백 파싱(`OK`, `FALLBACK_ANCHOR`, `FALLBACK_SINGLE_EVENT`)
- `pk_file`, `pk_event`, `pii_type_inferred` 생성

### 구현 위치

- `src/data/s1_parser.py`
- `config/schema_registry.yaml`

### 상태

- 모듈 구현 완료
- 기본 실행 스크립트(`run_data_pipeline.py`)와의 완전 통합은 후속

---

## S2. Feature Prep

### 목적

- TF-IDF + Dense 피처 생성
- 경로/도메인/패턴 기반 파생 피처 생성
- 합성 상호작용 피처 생성

### 실행 경로

`run_training.py` 내부에서 자동 수행

### Phase 1 피처 구성 (2026-03-16 기준)

| 피처 그룹 | 구현 방식 | 피처 수 |
|-----------|----------|--------|
| file_name char TF-IDF | `char_wb`, ngram(2,5), max_features=500 | 최대 500 |
| file_path path TF-IDF | word 1-2gram, max_features=500 | 최대 500 |
| server_name 빈도 인코딩 | value_counts(normalize=True) | 1 |
| 경로/메타 Dense 피처 | path depth, extension, 길이 등 | ~27 |
| **합계 (Phase 1)** | | **~1,028** |

> TF-IDF vocab이 비어있으면(`max_df` pruning) 해당 뷰 자동 건너뜀 (로그: `[WARN] ... TF-IDF 건너뜀`)

### 현재 산출물

- `data/features/X_train.npz`
- `data/features/X_test.npz`
- `data/features/y_train.csv`
- `data/features/y_test.csv`
- `data/features/feature_names.joblib`
- `models/tfidf_vectorizer.joblib`

### 참고

`src/features/schema_validator.py`가 존재하며, 아키텍처의 `feature_schema` 고정 정책과 연결 가능합니다.

---

## S3a. RULE Labeler

### 목적

- RULE 기반 클래스 라벨 부여
- evidence(근거) long-format 출력

### 구현 위치

- `src/filters/rule_labeler.py`
- `config/rules.yaml`
- `config/rule_stats.json`

### 현재 실행 경로

- 운영 스크립트 기본 경로: `src/filters/filter_pipeline.py` (하위 호환)
- `--use-filter` 지정 시 Layer 1/2 선분류 적용

```bash
python scripts/run_training.py --use-filter
```

### 현재 산출물(필터 사용 시)

- `data/processed/keyword_filtered.csv`
- `data/processed/rule_filtered.csv`
- `outputs/filter_statistics.txt`

### 아키텍처 목표 산출물

- `rule_labels.parquet`
- `rule_evidence.parquet`

---

## S3b. ML Labeler

### 목적

- 잔여 샘플 ML 다중분류
- 모델 비교 및 최종 모델 저장

### 실행

```bash
# Phase 1: 레이블 Excel 기반 (silver_label.parquet)
python scripts/run_training.py --source label

# Phase 1.5: Sumologic JOIN 기반 (silver_joined.parquet)
python scripts/run_training.py --source detection

# RULE Labeler(S3a) 포함
python scripts/run_training.py --source label --use-filter

# 합성 확장 ON
python scripts/run_training.py --source label --use-extended-features

# Repeated GroupShuffleSplit 5회 (분산 추정 — 표본 안정성 확인)
python scripts/run_training.py --source label --n-splits 5

# 확률 보정 활성화 (isotonic, cv=3)
python scripts/run_training.py --source label --calibrate
```

### 학습 신뢰성 개선 사항 (2026-03-16)

| 항목 | 내용 |
|------|------|
| eval_set leakage 수정 | Early stopping용 eval_set을 X_train 내부 20% val split으로 교체 — 테스트셋 오염 제거 |
| Temporal Split 자동 실행 | `label_work_month` 컬럼 있으면 Step 6b에서 time split 자동 수행, `[Temporal]` vs `[Random]` F1 비교 출력 |
| Coverage-Precision Curve | Step 6c에서 τ 스윕(0.5→1.0) 자동 실행, 권장 τ (Precision ≥ 0.95) 출력 |
| 확률 보정 | `--calibrate` 시 `CalibratedClassifierCV(cv=3, isotonic)` 적용, ECE 계산 |
| SMOTE 제거 | `apply_smote()` 삭제 — Sparse TF-IDF와 비호환, dead code |

### 현재 산출물

- `models/phase1_label_lgb.joblib` — 레이블 기반 LightGBM (binary TP/FP)
- `models/phase1_label_lgb_calibrated.joblib` — 확률 보정 모델 (`--calibrate` 시)
- `models/final/best_model_v1.joblib` — 전체 파이프라인 multiclass
- `models/final/feature_schema.json` — 피처 구성 스냅샷

### 정합성 주의

`docs/Architecture/00_overview.md` §0 스냅샷과 현재 코드 기본값 모두 합성 확장 OFF(Tier0)로 정합화되어 있습니다.

---

## S4. Decision Combiner (목표 단계)

### 아키텍처 요구

- RULE/ML 결합 의사결정
- risk_flag / ood_flag / tp override 정책 반영
- `UNKNOWN` 라우팅

### 현재 상태

- 관련 설정(`config/model_config.yaml` 내 `ood`, `unknown_routing`)은 존재
- 스크립트 단일 경로에서 S4 완전 출력은 후속

---

## S5. Output Writer (목표 단계)

### 아키텍처 요구

- `predictions_main` (1행=1검출 결론)
- `prediction_evidence` (N행=1검출 근거)

### 현재 상태

평가/리포트 산출은 제공되나, S5 표준 2-테이블 출력은 통합 후속입니다.

---

## S6. Monitoring & Feedback (목표 단계)

### 아키텍처 요구

- KPI 12종 자동 산출
- `monthly_metrics.json` 생성
- Auto-remediation 루프

### 현재 상태

- `scripts/run_evaluation.py`로 모델/오분류/피처 중요도 리포트 생성 가능
- 월 KPI 자동 루프는 후속 구현

---

## S_Report. PoC 결과 Excel 리포트

### 목적

- 학습 완료 후 PoC 판정 결과를 6-sheet Excel로 자동 생성
- Primary / Secondary / Tertiary(조직별) 3종 Split 비교
- Coverage-Precision 곡선(τ 스윕) 및 Rule 기여도 분석 포함

### 실행

```bash
python scripts/run_poc_report.py
```

### 주요 옵션

```bash
python scripts/run_poc_report.py --phase 2               # Label + Sumologic
python scripts/run_poc_report.py --skip-ml               # Rule 분석만 (Feature 없을 때)
python scripts/run_poc_report.py --precision-target 0.90
python scripts/run_poc_report.py --output my_report.xlsx
```

### 산출물

- `outputs/poc_report.xlsx` (6 sheets)
  - Sheet 1: 요약 — 데이터 조건, PoC 판정(PASS/FAIL)
  - Sheet 2: 데이터 통계 — TP/FP 비율, 월별 분포, fp_description
  - Sheet 3: 모델 성능 — 3종 Split 비교표
  - Sheet 4: Coverage 곡선 — τ 스윕 테이블, 권장 τ 하이라이트
  - Sheet 5: Rule 기여도 — rule_id별 히트율·정밀도
  - Sheet 6: 오분류 분석 — 패턴 상위 15 + 샘플 200건

### 전제 조건

- `silver_label.parquet` 또는 `data/raw/label/` 디렉토리 존재
- ML 예측 포함 시: 학습 완료(`models/` 디렉토리 존재)
- Rule 분석만: `--skip-ml` 플래그 사용

---

## S_E2E. 파일 기반 Mock E2E 검증

### 목적

폐쇄망 투입 전, 실제 파일 시스템의 mock 데이터로 전체 파이프라인(생성→전처리→JOIN→학습→평가)이 정상 동작하는지 검증한다.

### 실행

```bash
# label-only 전체 E2E
python scripts/run_mock_e2e.py --mode label-only

# full E2E — Sumologic JOIN 포함
python scripts/run_mock_e2e.py --mode full

# 실 데이터 보존하면서 파이프라인만 재실행
python scripts/run_mock_e2e.py --mode full --no-generate

# 전처리(S0–S2)까지만
python scripts/run_mock_e2e.py --mode full --dry-run
```

### 스텝별 검증 (full 모드)

| Step | 실행 스크립트 | 검증 항목 |
|------|-------------|----------|
| Step 1 | generate_mock_raw_data.py | label/*.xlsx 존재, sumologic xlsx 존재 |
| Step 2A | run_data_pipeline.py --source label | silver_label.parquet, 행>0, label_raw/pk_event/pk_file 존재 |
| Step 2B | run_data_pipeline.py --source detection | silver_detections.parquet, 행>0 |
| Step 2C | run_data_pipeline.py --source joined | silver_joined.parquet, 행>0, label_raw/pk_file 존재 |
| Step 3 | run_training.py --source detection | 모델 파일 존재 |
| Step 4 | run_evaluation.py | outputs/ 파일 존재 |

### 주의

- `--no-generate` 없이 실행 시 `data/raw/label/` 디렉토리가 삭제 후 재생성
- JOIN이 동작하려면 label Excel과 Sumologic xlsx가 **같은 generate_mock_raw_data.py 실행**으로 생성되어야 함 (동일 이벤트 기반, pk_file 일치)

---

## 6. 평가 실행

```bash
python scripts/run_evaluation.py
python scripts/run_evaluation.py --include-filtered
```

현재 산출물:

- `outputs/classification_report.txt`
- `outputs/error_analysis.csv`
- `outputs/feature_importance.csv`
- `outputs/layer_performance.txt` (`--include-filtered`)
- `outputs/confusion_matrix.png`
- `outputs/feature_importance.png`

### 평가 지표 (2026-03-16 추가)

| 지표 | 설명 |
|------|------|
| PR-AUC | 이진 분류에서 F1-macro 보완 — 임계값 무관 성능 측정 |
| F1-macro 95% CI | Bootstrap n=500 신뢰구간 — 표본 변동성 정량화 |

---

## 7. 테스트 실행

```bash
# 전체
pytest tests/ -v

# E2E만
pytest tests/ -m e2e -v

# 스크립트 경유
python scripts/run_tests.py --all
```

---

## 8. 운영 체크포인트

배치 실행 전:

1. `config/*.yaml`의 컬럼/경로/룰/임계값 확인
2. `python scripts/verify_env.py` 통과 확인
3. raw 데이터 파일명/인코딩 확인

배치 실행 후:

1. `outputs/data_validation_report.txt` 확인
2. `outputs/classification_report.txt` 확인
3. `outputs/error_analysis.csv` 상위 오류 패턴 확인
4. 필터 사용 시 `outputs/filter_statistics.txt` 확인
5. PoC 리포트: `outputs/poc_report.xlsx` 열어 종합 판정(PASS/FAIL) 확인

---

## 9. 장애 대응 가이드

| 증상 | 우선 확인 |
|------|-----------|
| 데이터 로드 실패 | `config/feature_config.yaml`의 `datasets.*.path`, `file_pattern`, `encoding` |
| 텍스트 컬럼 미검출 | `data.text_column` 및 컬럼 매핑 |
| silver_detections pk_file 없음 | `--source detection` 재실행, `dfile_computername/agentip/filedirectedpath/filename` 컬럼 확인 |
| silver_joined 0행 (JOIN 실패) | label/detection이 같은 `generate_mock_raw_data.py` 실행으로 생성됐는지 확인. 다른 시점 생성 시 pk_file 불일치 |
| 학습 성능 급락 | 클래스 분포/라벨 품질/합성확장 ON-OFF 비교 |
| 필터 결과 이상 | `config/filter_config.yaml`, `config/rules.yaml` 룰 우선순위/정규식 |
| `[WARN] file_name TF-IDF 건너뜀` | file_name이 모두 동일하거나 min_df/max_df 설정으로 어휘가 비어있음 — 정상 동작, dense 피처만으로 학습 진행 |
| Calibration 실패 경고 | 훈련 샘플 수 부족 시 cv=3 fold가 너무 작음 — `--calibrate` 옵션 제거 후 재실행 |

---

## 10. 참조 문서

- 아키텍처: `docs/Architecture/`
- 프로젝트 개요: `README.md`
- 폐쇄망 시나리오: `Env_scenario.md`
- 에이전트 컨텍스트: `CLAUDE.md`
- 회의/결정 이력: `회의록.md`

---

**문서 끝**
