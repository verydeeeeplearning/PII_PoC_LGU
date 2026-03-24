# Claude Code 프로젝트 컨텍스트

이 문서는 코딩 에이전트가 본 저장소에서 작업할 때 아키텍처 기준으로 일관된 용어와 실행 절차를 유지하기 위한 운영 컨텍스트입니다.

---

## 1. 우선 참조 문서

1. `docs/Architecture/` (최상위 기준 — 섹션별 분리)
   - `00_overview.md` — 설계 원칙, 시스템 개요
   - `01_data_pipeline.md` — S0~S1 데이터 파이프라인
   - `02_feature_engineering.md` — S2 Feature Prep
   - `03_rule_labeler.md` — S3a RULE Labeler
   - `04_ml_labeler.md` — S3b ML Labeler
   - `05_decision_output.md` — S4~S5 Decision/Output
   - `06_monitoring_classification.md` — S6 모니터링, 분류체계
   - `07_quality_operations.md` — 신뢰도·인프라·평가·로드맵
2. `PROJECT_GUIDE.md` (실행 런북)
3. `README.md` (프로젝트 개요)
4. `Env_scenario.md` (폐쇄망 반입/접근 시나리오)

---

## 2. 아키텍처 기준 요약

### 2.1 용어 규칙

- 권장: `Labeler` (RULE Labeler, ML Labeler)
- 과거 용어: `Filter` (코드 네임스페이스 하위 호환으로만 유지)

### 2.2 분류 체계

**실제 데이터 확정 클래스 (2026-03, `classify_fp_description.py` 기준):**

FP 클래스(6개, fp_description 611개 분류 결과):

- `FP-파일없음` — 파일/경로 미존재 (신규)
- `FP-이메일패턴` — @문자/kerberos/내부도메인/OSS도메인 오인식
- `FP-숫자패턴` — 타임스탬프·bytes·일련번호 (기존 3-class 통합)
- `FP-라이브러리` — RPM/docker/conda/npm/pip 패키지, 오픈소스 설치파일
- `FP-더미테스트` — 테스트/더미/분析용 임시파일
- `FP-시스템로그` — 서비스로그·배치·DB·백업·인프라 운영 데이터

TP 클래스(1개, 정탐 파일 출처 자동 부여):

- `TP-실제개인정보`

운영 정책 클래스: `UNKNOWN`

> 초기 설계의 Canonical 8-class(FP-숫자나열/코드, FP-타임스탬프, FP-bytes크기, FP-내부도메인, FP-OS저작권, FP-패턴맥락)는 실제 데이터 분석 후 위 체계로 재편됨. 상세 매핑: `docs/Architecture/06_monitoring_classification.md §12.1`

### 2.3 Stage 모델

- `S0`: Raw ingest / schema canonicalization
- `S1`: Normalize & parse
- `S2`: Feature prep
- `S3a`: RULE labeler
- `S3b`: ML labeler
- `S4`: Decision combiner
- `S5`: Output writer
- `S6`: Monitoring & feedback

---

## 3. 현재 구현 스냅샷 (2026-03-24)

| 항목 | 상태 |
|------|------|
| **레이블 파이프라인** | `run_data_pipeline.py --source label` — Excel → `silver_label.parquet` (pk_file/pk_event SHA256 포함) |
| **검출 파이프라인** | `run_data_pipeline.py --source detection` — Sumologic xlsx → dfile_* 컬럼 표준화 + pk_file SHA256 → `silver_detections.parquet` |
| **JOIN 파이프라인** | `run_data_pipeline.py --source joined` — silver_label + silver_detections → pk_file 기준 inner JOIN → `silver_joined.parquet` |
| S1 파서 모듈 | `src/data/s1_parser.py` — pk_file/pk_event SHA256 64자 (MD5 → SHA256 통일 완료) |
| RULE Labeler | `src/filters/rule_labeler.py` — 11개 룰 active (PATH_HADOOP_001 비활성화, domain_list/regex/feature_condition 3유형) |
| 학습 파이프라인 | `run_training.py --source label` (silver_label 기반) / `--source detection` (silver_joined 기반) |
| **학습 Split 전략** | `--split group` (기본, GroupShuffleSplit) / `--split temporal --test-months 3` (행 단위 엄격 월 분할) / `--split server` (서버 분할) |
| 평가 파이프라인 | `scripts/run_evaluation.py` (레거시, run_report.py로 대체 권장) |
| **통합 리포트** | **`scripts/run_report.py --source label\|detection`** — eval + PoC report + 진단을 9-sheet Excel로 통합 (DC 시트 제거됨). `--include-diagnosis`로 데이터 진단 포함 |
| **데이터 진단** | **`scripts/diagnose_data_bias.py`** — Column 편향, Split Robustness, Feature Ablation 독립 실행 |
| Feature 분할 | `GroupShuffleSplit(groups=pk_file)` 기본 + temporal/server split 추가 완료 |
| **Wave 3 피처 개정** | 시간 피처 4개 + server_freq 제거, exception_requested/rule_matched 추가, Phase 1 TF-IDF 500→200 축소 |
| **Tier 2 Round 1** | B2 중복 가중치(`1/sqrt(group_size)`) + B7 server_env/stack/is_prod + B1 범주형 8개 Label Encoding + B8 rule_confidence_lb/rule_id/class + B9 file_event_count/pii_diversity |
| **Tier 2 Round 2** | B3 Shape TF-IDF (file_name shape 100 features) + B6 정규화 (constants.py: min_child_samples=200, reg_alpha=0.5, max_depth=10, 단 model_config.yaml에서 min_child_samples=20으로 override → 실제 적용값=20) |
| **Tier 3 C1** | Easy FP Suppressor — is_system_device/is_package_path+mass/is_docker_overlay/has_license_path 조건 고확신 FP 선제 분리 (purity≥95% 시 활성화) |
| **Tier 3 C2** | Slice-aware threshold — server_env별 Coverage-Precision Curve tau 개별 계산 |
| **Tier 3 C5** | threshold_policy.json 아티팩트 저장 — Step 6c tau + curve_summary → `models/final/threshold_policy.json` 자동 저장 |
| **ml_tp_proba 수정** | `predict_with_uncertainty()`에서 TP 클래스 인덱스 동적 탐색 (기존 `proba[:,0]` FP 확률 역전 버그 수정) |
| Multi-view TF-IDF | raw_text(5000) + shape_text(2000) + path_text(1000) 3-view 구현 완료 |
| 공통 시각화 | `src/utils/plot_utils.py` — `setup_plot()` 통합 (한글 폰트 경고 포함, evaluator/eda 공유) |
| Parquet 내보내기 | `scripts/run_export.py` — Silver/Feature Parquet → CSV(utf-8-sig) 또는 Excel(.xlsx) |
| S5 표준 출력 | `predictions_main`/`prediction_evidence` 통합은 후속 |
| S6 KPI 자동화 | `monthly_metrics.json` 자동 루프는 후속 |
| Phase 0 품질 검증 | `scripts/run_phase0_validation.py` — 충돌률/Bayes Error/Go-No-Go 판정 |
| 컬럼 정규화 | `src/data/column_normalizer.py` — 한글→영문 컬럼명 변환 |
| **fp_description 분류** | **`scripts/classify_fp_description.py`** — 611개 unique값 → 7-class + UNKNOWN, `fp_description_mapping.csv` 출력 |
| **Mock 원본 데이터** | **`scripts/generate_mock_raw_data.py`** — 레이블 Excel + Sumologic Excel(.xlsx) 동일 이벤트 기반 생성 (JOIN 가능) |
| **ML 파이프라인 더미 데이터** | **`scripts/generate_dummy_data.py`** — E2E 파이프라인 검증용 8-class 더미 데이터 |
| **파일 기반 E2E 검증** | **`scripts/run_mock_e2e.py`** — mock 생성 → label/detection/join → 학습 → 평가 전체 검증 (`--mode label-only\|full`) |
| PoC 리포트 (레거시) | `scripts/run_poc_report.py` — 학습 후 7-sheet Excel (run_report.py로 대체 권장) |
| PoC 지표 | `src/evaluation/poc_metrics.py` — binary_stats, coverage_precision_curve, split_comparison |
| Rule 기여도 | `src/evaluation/rule_analyzer.py` — rule_contribution, class_rule_contribution |
| Excel 작성기 | `src/report/excel_writer.py` — PocReportData dataclass + PocExcelWriter (9-sheet: 기존 7 + Feature Importance + 데이터 진단. DC 시트 제거됨) |
| Split 전략 확장 | `src/evaluation/split_strategies.py` — work_month_time_split(), org_subset_split(), server_group_split() |
| **Split 전략 CLI** | **`build_features(split_strategy="group"\|"temporal"\|"server", test_months=3)`** — `run_training.py --split temporal --test-months 3` |
| **통합 리포트** | **`scripts/run_report.py`** — eval + PoC + 진단 통합, 10-sheet Excel 생성. `--source label\|detection`, `--include-diagnosis` |
| **데이터 진단** | **`scripts/diagnose_data_bias.py`** — Column 편향 (Cramer's V, MI), Split Robustness, Feature Ablation, Column Risk Registry |
| **데이터 플로우 분석** | **`docs/data_flow_risk_analysis.md`** — Raw->Feature->Model 전체 추적 + 일반화 성능 관점 분석 |
| **표준 아티팩트 저장** | **`run_training.py` Step 7+8** — 모든 모델 아티팩트 `models/final/`(FINAL_MODEL_DIR)에 통합 저장: `phase1_label_lgb.joblib`, `best_model_v1.joblib`, `label_encoder.joblib`, `feature_builder.joblib`, `ood_detector.joblib`, `feature_schema.json`, `threshold_policy.json` |
| **FeatureBuilderSnapshot** | **`src/models/feature_builder_snapshot.py`** — fitted TF-IDF 벡터라이저 + dense 컬럼 + **categorical LabelEncoders** 통합 저장/로드. `transform()`에서 `prepare_phase1_features()` 자동 호출 → 추론 시 피처 동형성 보장 |
| **공통 피처 준비** | **`src/features/feature_preparer.py`** — `prepare_phase1_features(df)`: meta/path/RuleLabeler 피처 통합 생성. training(run_training.py Step 2-4)과 inference(FeatureBuilderSnapshot.transform) 양쪽에서 동일 함수 호출 |
| **fp_description 분류 모듈** | **`src/features/fp_classifier.py`** — MULTICLASS_RULES + `classify_fp_description()`. `--use-multiclass` 시 7-class 타깃 생성에 사용 |
| **Multi-class 학습** | **`run_training.py --use-multiclass`** — fp_description 기반 7-class (TP-실제개인정보 + FP 6-subclass) 학습. 평가 시 binary collapse |
| **TP 가중치 CLI** | **`run_training.py --tp-weight 1.5`** — TP 샘플 가중치 배수 (기본 1.0=비활성, 필요시 지정) |
| **Categorical encoding** | **pipeline.py** — train+test 합본 fit LabelEncoder. encoder가 `build_features()` return dict에 포함되어 FeatureBuilderSnapshot으로 전달 |
| **Calibration** | 제거됨 (CalibratedClassifierCV cv=3 — F1에 영향 없이 학습 시간만 증가하여 삭제) |
| **Bootstrap CI** | 제거됨 (evaluator.py n=500 — 4M행에서 ±0.0009 수준으로 실용성 없어 삭제) |
| **file_size** | `column_name_mapping.yaml`에서 `drop: false` — Label 데이터에서 전부 결측, `_KEEP_COLS`에서 제거됨. Sumologic/Joined에서만 활용 가능 |
| **Wave 7 DC 검증** | Grid Sweep(35개 임계값 조합) 결과: **DC 개입 시 항상 F1 하락**, ML passthrough(ml_conf=0.0)가 최적. RULE 커버리지 0.6%, 독립 기여 0.057%, "애매하면 TP" 로직이 ML의 정확한 FP 판정을 대량 TP 전환 |
| **PATH_HADOOP_001** | 비활성화 (`active: false`) — 실측 precision 11.7% vs rule_stats lb=0.845 (7.2배 괴리). 비활성화 후 F1 0.7790→0.7914 개선 |
| **rule_stats 자동 집계** | `run_training.py` Step 4a — 매 학습 시 rule_id별 N/M 자동 집계 + Beta lb 계산 → `rule_stats_measured.json` 별도 저장 (원본 `rule_stats.json` 보존). 이전 값 대비 큰 편차(>0.2) 경고 |
| **PoC 판정 기준 변경** | **F1-macro ≥ 0.70 단독 기준**으로 통일. TP Recall(≥0.75), FP Precision(≥0.80)은 참고 지표로 변경. FP Precision 임계값 0.85→0.80 완화 |
| **DC 시트 제거** | Wave 7 Grid Sweep 결과 DC 개입이 항상 F1 하락 확인 → 리포트에서 DC 시트(10_DecisionCombiner) 제거, DC 시뮬레이션 비활성화 |
| **threshold_policy 추론 연동** | `run_inference.py`에서 `threshold_policy.json` 로드 → `combine_decisions(thresholds=...)` 전달 |
| **file_size 제거** | `_KEEP_COLS`에서 `file_size`/`file_size_log1p` 제거 — Label 데이터에서 100% 결측, 노이즈 피처 |
| **model_config.yaml override 주의** | `model_config.yaml`이 `constants.py`를 `_deep_update()`로 override. 실제 적용값은 yaml 기준. yaml에 없는 키만 constants.py 값 사용. 현재 yaml: `min_child_samples: 20` (constants.py의 200을 override) |
| **서버 검증 성능** | temporal split (train: 3~9월, test: 10~12월), 10M행, F1-macro **0.779** → PASS |

정합성 주의:

- `docs/Architecture/00_overview.md` §0과 현재 코드 기본값은 합성 확장 OFF(Tier0)로 일치
- 합성 확장이 필요하면 `--use-extended-features`로 활성화

전략 전환 주의:

- 레이블 데이터(`data/raw/label/`)에 `full_context_raw` 텍스트 컨텍스트 없음 → Phase 1은 메타데이터 기반
- Sumologic(`data/raw/dataset_a/`) 데이터에는 `dfile_inspectcontentwithcontext` = `full_context_raw` 존재 → Phase 2에서 TF-IDF 피처 추가 예정

---

## 4. 주요 경로

**경로 상수 (`src/utils/constants.py`):**
`PROJECT_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `FEATURE_DIR`,
`MODEL_DIR`, `FINAL_MODEL_DIR`, `CHECKPOINT_DIR`,
`REPORT_DIR`, `FIGURES_DIR`, `DIAGNOSIS_DIR`, `PREDICTIONS_DIR`, `EXPORTS_DIR`

```text
config/
  feature_config.yaml
  model_config.yaml
  filter_config.yaml
  rules.yaml                 # 12개 active 룰 (domain_list/regex/feature_condition 3유형)
  rule_stats.json            # 룰별 실적 통계 (Bayesian 신뢰도 계산용)
  schema_registry.yaml
  column_name_mapping.yaml   # 한글 컬럼명 → 영문 정규화 (Phase 0-A)
  ingestion_config.yaml      # 레이블 파일 탐색 설정 (Phase 0-A)
  label_mapping.yaml         # fp_description 그룹화 (수동 작업 대기)

src/
  data/            # loader, merger, s1_parser, label_loader, column_normalizer
  features/        # text/path/tabular/synthetic
  filters/         # filter_pipeline + rule_labeler (rule_confidence 포함)
  models/          # trainer
  evaluation/      # evaluator, eda, data_quality, poc_metrics, rule_analyzer, split_strategies
  report/          # PoC 결과 Excel 작성기 (PocReportData, PocExcelWriter)
  utils/           # common, constants, plot_utils (setup_plot 통합)

scripts/
  run_pipeline.py            # ★ 전체 파이프라인 진입점 (S0→S6) --mode full|label-only
  run_data_pipeline.py       # --source detection|label|joined 지원 (단독 실행용)
  run_training.py            # --source detection|label + --split group|temporal|server
  run_report.py              # ★ 통합 리포트 (eval + PoC + 진단) --source label|detection
  run_evaluation.py          # 레거시 (run_report.py로 대체됨, run_pipeline.py에서 제거됨)
  run_poc_report.py          # 레거시 (run_report.py에서 내부 호출, 단독 실행 비권장)
  diagnose_data_bias.py      # 데이터 편향/일반화 진단 (독립 실행용)
  run_export.py              # Parquet → CSV/Excel 내보내기
  run_phase0_validation.py   # Phase 0 Go/No-Go 판정
  run_mock_e2e.py            # 파일 기반 E2E 검증 (mock 생성→파이프라인→학습→평가)
  run_tests.py
  classify_fp_description.py  # fp_description → 7-class 매핑 (result.xlsx 입력)
  generate_mock_raw_data.py   # 레이블 Excel + Sumologic Excel 동일 이벤트 기반 생성
  generate_dummy_data.py      # ML 파이프라인 E2E 검증용 8-class 더미 데이터

data/
  raw/
    label/         # 레이블 Excel (정탐/오탐 × 월 × 조직)
    dataset_a/     # Sumologic 검출 원본 xlsx
  processed/
    silver_label.parquet      # 레이블 파이프라인 산출물
    silver_detections.parquet # 검출 파이프라인 산출물 (pk_file 포함)
    silver_joined.parquet     # JOIN 산출물 (label_raw + full_context_raw 통합)

outputs/                              # REPORT_DIR — 텍스트/CSV 리포트
  poc_report.xlsx                   # ★ 통합 9-sheet Excel (run_report.py --source label)
  poc_report_detection.xlsx         # ★ Joined 모델 리포트 (run_report.py --source detection)
  classification_report.txt         # sklearn 분류 리포트
  error_analysis.csv                # 오분류 패턴 분석
  feature_importance.csv            # 피처 중요도 (CSV)
  go_no_go_report.md                # Phase 0 Go/No-Go 판정 (run_phase0_validation.py)
  fp_description_unique_list.csv    # Phase 0 fp_description 목록
  fp_description_mapping.csv        # fp_description → 7-class 매핑 (classify_fp_description.py)
  label_conflict_report.txt         # Phase 0 레이블 충돌 리포트
  figures/                          # FIGURES_DIR — 시각화 PNG
    confusion_matrix.png            #   혼동 행렬
    feature_importance.png          #   피처 중요도 바 차트
    reliability_diagram.png         #   확률 보정 신뢰도 다이어그램 (조건부)
    class_distribution.png          #   클래스 분포 (EDA)
    tp_fp_distribution.png          #   TP/FP 파이차트 (EDA)
    correlation_matrix.png          #   상관행렬 (EDA)
  diagnosis/                        # DIAGNOSIS_DIR — 데이터 진단 산출물
    column_bias_report.txt          #   Cramer's V, MI, 단일피처 F1
    column_risk_registry.csv        #   컬럼별 리스크 등급
    split_robustness_report.csv     #   temporal/server/random split F1 비교
    ablation_report.csv             #   피처 블록 제거 F1
    bias_summary.md                 #   종합 판정
  predictions/                      # PREDICTIONS_DIR — 추론 산출물
    predictions_main_{run_id}.parquet
    prediction_evidence_{run_id}.parquet
  exports/                          # EXPORTS_DIR — Parquet→CSV/Excel 내보내기
```

---

## 5. 실행 명령어

### 5.1 환경 검증

```bash
bash scripts/setup_env.sh
python scripts/verify_env.py
```

### 5.2 전체 파이프라인 (S0→S6) — 권장 진입점

`run_pipeline.py` 한 명령어로 **전처리(S0–S2) → 학습(S3–S5) → 평가(S6)** 순서를 보장한다.

```bash
# 레이블 Excel만 사용 (Phase 1 기본)
python scripts/run_pipeline.py --mode label-only

# Sumologic + 레이블 JOIN 학습 (Phase 1.5)
python scripts/run_pipeline.py --mode full

# RULE Labeler(S3a) 포함
python scripts/run_pipeline.py --mode label-only --use-filter

# 전처리(S0–S2)까지만 실행 (학습/평가 없음)
python scripts/run_pipeline.py --mode label-only --dry-run

# 평가(S6) 건너뜀
python scripts/run_pipeline.py --mode label-only --skip-eval
```

**스테이지 순서 (변경 불가):**

```
[전처리]  S0-S2
  label-only: 레이블 Excel → silver_label.parquet
  full      : 레이블 Excel → silver_label.parquet
              Sumologic xlsx → silver_detections.parquet (pk_file SHA256)
              JOIN → silver_joined.parquet (pk_file 기준 inner join)

[학습]    S3a RULE Labeler → S3b ML Labeler
  label-only: silver_label.parquet 기반
  full      : silver_joined.parquet 기반 (label_raw + full_context_raw 통합)

[판정/출력] S4 Decision combiner → S5 Output writer
[리포트]  S6 통합 리포트 (run_report.py)
  → 평가 + Feature Importance + 데이터 진단 → 9-sheet Excel (DC 시트 제거됨)
```

---

### 5.3 개별 스텝 독립 실행

전처리, 학습, 평가를 각각 따로 실행할 수 있다.

```bash
# [S0–S1] 전처리만
python scripts/run_data_pipeline.py --source label       # 레이블 Excel → silver_label.parquet
python scripts/run_data_pipeline.py --source detection   # Sumologic xlsx → silver_detections.parquet (pk_file 계산)
python scripts/run_data_pipeline.py --source joined      # silver_label + silver_detections → silver_joined.parquet

# [S3-S5] 학습만 (전처리 완료 후 실행)
python scripts/run_training.py --source label            # silver_label.parquet 기반 (GroupShuffleSplit)
python scripts/run_training.py --source detection        # silver_joined.parquet 기반
python scripts/run_training.py --source label --split temporal --test-months 3  # 마지막 3개월 holdout
python scripts/run_training.py --source label --split server                    # 서버 단위 분할
python scripts/run_training.py --source label --use-filter
python scripts/run_training.py --source label --use-extended-features

# [S6] 통합 리포트 (학습 완료 후 실행) -- 권장
python scripts/run_report.py --source label              # Label 모델 9-sheet Excel
python scripts/run_report.py --source detection          # Joined 모델 리포트
python scripts/run_report.py --source label --include-diagnosis  # 데이터 진단 포함

# [S6] 레거시 평가 (run_report.py로 대체 권장)
python scripts/run_evaluation.py
```

### 5.8 파일 기반 Mock E2E 검증

실제 파일 시스템의 mock 데이터로 전체 파이프라인 동작을 검증한다.

```bash
# label-only 전체 E2E (mock 재생성 포함)
python scripts/run_mock_e2e.py --mode label-only

# full E2E — Sumologic JOIN 포함 (mock 재생성 포함)
python scripts/run_mock_e2e.py --mode full

# 실 데이터 보존하면서 파이프라인만 재실행
python scripts/run_mock_e2e.py --mode full --no-generate

# 전처리(S0–S2)까지만 (학습/평가 없음)
python scripts/run_mock_e2e.py --mode full --dry-run
```

**주의:** `--no-generate` 없이 실행 시 `data/raw/label/` 디렉토리가 삭제 후 재생성됩니다.
실 데이터가 있는 경우 반드시 `--no-generate` 옵션을 사용하세요.

### 5.4 테스트

```bash
pytest tests/ -v
python scripts/run_tests.py --all
```

### 5.5 데이터 내보내기 (Parquet → 사람이 읽을 수 있는 형식)

```bash
python scripts/run_export.py                    # 기본: Silver/Feature → CSV (utf-8-sig)
python scripts/run_export.py --format xlsx      # Excel 출력
python scripts/run_export.py --info             # 스키마 정보만 확인
python scripts/run_export.py --max-rows 10000   # 샘플 내보내기
```

### 5.6 Phase 0 데이터 품질 검증

```bash
# 레이블 Excel 다중 파일 → silver_label.parquet
python scripts/run_data_pipeline.py --source label

# 데이터 품질 검증 + Go/No-Go 판정 리포트
python scripts/run_phase0_validation.py

# 검증만 (파일 저장 없이 콘솔 출력)
python scripts/run_phase0_validation.py --dry-run
```

### 5.7 통합 리포트 생성 (권장)

```bash
# 통합 리포트 (eval + PoC + Feature Importance를 9-sheet Excel로)
python scripts/run_report.py --source label              # Label 모델
python scripts/run_report.py --source detection          # Joined 모델
python scripts/run_report.py --source label --include-diagnosis  # 데이터 진단 포함
python scripts/run_report.py --source label --skip-ml    # Rule 분석만

# 레거시 PoC 리포트 (run_report.py로 대체 권장)
python scripts/run_poc_report.py                         # 기본: Phase 1 (Label Only)
python scripts/run_poc_report.py --phase 2               # Phase 2 (Label + Sumologic)
```

### 5.9 데이터 진단 (독립 실행)

```bash
python scripts/diagnose_data_bias.py                     # 전체 진단 (Column + Row + Split + Ablation)
python scripts/diagnose_data_bias.py --skip-ablation     # 빠른 실행 (Ablation 제외)
python scripts/diagnose_data_bias.py --skip-single-feature --skip-ablation  # 최소 실행
```

---

## 6. 코드 수정 가이드

1. 설정 우선 원칙
- 하드코딩보다 `config/*.yaml` 우선
- 컬럼/PK/룰/임계값은 설정 파일에서 조정

2. 정합성 우선 원칙
- 새 코드/문서는 S0~S6, Labeler 용어 우선
- 클래스명은 Canonical 8클래스 사용

3. 폐쇄망 제약 준수
- 외부 API/실시간 다운로드 의존 코드 금지
- 오프라인 실행 가능성을 우선 고려

4. 산출물 추적성
- 가능하면 `pk_event`, `reason_code`, evidence 형태 유지
- 운영/감사 재현 가능한 파일 기반 산출물 선호

5. 파일명/폴더명 규칙
- **소문자 + 언더스코어만 사용** — 대문자, 공백, 하이픈 금지
  - 올바름: `dataset_a`, `silver_detections.parquet`, `run_export.py`
  - 잘못됨: `Dataset A`, `silver-detections.parquet`, `runExport.py`

6. 인코딩 정책
- 서버(Linux) 원본 파일: `utf-8`
- Windows에서 공유받은 CSV: `utf-8-sig` (BOM)
- 한글 Windows 생성 파일 fallback: `cp949`
- Parquet: 바이너리 (인코딩 무관)
- 내보내기 CSV: `utf-8-sig` (Windows Excel에서 한글 깨짐 방지)

---

## 7. 문서 수정 시 체크리스트

- `docs/Architecture/` 용어와 충돌 없는지
- CLI 옵션(`--use-filter`, `--include-filtered`)이 실제 코드와 일치하는지
- 경로/파일명(`config`, `data`, `outputs`)이 실제 리포지토리와 일치하는지
- 목표(아키텍처)와 현재 구현 상태를 분리해 표기했는지

---

**작성 기준일:** 2026-03-24
