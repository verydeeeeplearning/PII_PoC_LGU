# Server-i 오탐 개선 AI 분류 시스템 — 종합 아키텍처 설계서

**대상 솔루션:** Server-i (소만사)  
**프로젝트:** PII 검출 오탐 개선 AI 모델링  
**작성일:** 2026-03-15
**Version:** 1.3
**기반 문서:** 문제 정의서 v3.4 (2026-01), 아키텍처 설계 노트 (2026-02), 수정 제안서 (2026-02-22), Zero-Human-in-the-Loop 자동화 계획 (2026-02-22)

---

## 목차

1. [설계 원칙 & Rationale](#1-설계-원칙--rationale)
2. [시스템 개요](#2-시스템-개요)
3. [데이터 파이프라인 총괄](#3-데이터-파이프라인-총괄)
4. [Stage S0: Raw Ingest & Storage](#4-stage-s0-raw-ingest--storage)
5. [Stage S1: Normalize & Parse](#5-stage-s1-normalize--parse)
6. [Stage S2: Feature Prep](#6-stage-s2-feature-prep)
7. [Stage S3a: RULE Labeler](#7-stage-s3a-rule-labeler)
8. [Stage S3b: ML Feature Builder & ML Labeler](#8-stage-s3b-ml-feature-builder--ml-labeler)
9. [Stage S4: Decision Combiner](#9-stage-s4-decision-combiner)
10. [Stage S5: Output Writer](#10-stage-s5-output-writer)
11. [Stage S6: Monitoring & Feedback](#11-stage-s6-monitoring--feedback)
12. [분류 체계 & 클래스 설계](#12-분류-체계--클래스-설계)
13. [신뢰도(Confidence) 설계](#13-신뢰도confidence-설계)
14. [설명가능성(Explainability) 설계](#14-설명가능성explainability-설계)
15. [리스크 제어 설계](#15-리스크-제어-설계)
16. [인프라 & 개발 환경](#16-인프라--개발-환경)
17. [데이터 레이어 & 중간 산출물](#17-데이터-레이어--중간-산출물)
18. [Training Pipeline vs Inference Pipeline](#18-training-pipeline-vs-inference-pipeline)
19. [평가 전략](#19-평가-전략)
20. [구현 로드맵](#20-구현-로드맵)
21. [후속 과제 & 미확정 사항](#21-후속-과제--미확정-사항)
22. [라벨 품질 감사 (v1.2 자동화)](#22-라벨-품질-감사-v12-자동화)

---

## 0. 구현 정합성 스냅샷 (2026-03)

이 문서는 **목표 아키텍처**와 **현재 리포지토리 구현**을 함께 다룬다.
용어/클래스/입출력 정합성은 아래를 기준으로 한다.

| 항목 | 기준 |
|------|------|
| Layer 1~2 용어 | 설계 용어는 "라벨러", 코드 모듈명은 하위 호환을 위해 `src/filters/*` 유지 |
| **실제 데이터 확정 클래스 (2026-03)** | `scripts/classify_fp_description.py` 실행 결과 확정. §12.1 참조 |
| 현재 구현 기준 | **`scripts/run_pipeline.py --mode label-only\|full`** (S0→S6 단일 진입점, 전처리 선행 강제) — 내부적으로 `run_data_pipeline.py` → `run_training.py` → `run_evaluation.py` 순서 실행<br>`run_data_pipeline.py --source {label\|detection\|joined}` 세 가지 독립 실행 모드 지원: `label` (레이블 Excel → `silver_label.parquet`), `detection` (Sumologic Excel → `silver_detections.parquet`, dfile_* 컬럼 매핑 + pk_file 계산, `DataSourceRegistry`가 column_map/pk_fields 관리, `--datasource` 옵션으로 소스 선택), `joined` (silver_label + silver_detections → `silver_joined.parquet`, pk_file inner join). `--mode full` 실행 시 label → detection → joined 순서로 자동 실행<br>전처리 파라미터(컬럼 매핑, 파서 설정, 인코딩, 에러 정책)는 `config/preprocessing_config.yaml`로 외부화 — 코드 수정 없이 YAML 수정만으로 데이터 포맷 변경에 대응 가능 |
| 학습 기본 실행 | `run_pipeline.py` 기본값은 필터 미적용 + 합성변수 **OFF** (Tier 0). 필터는 `--use-filter`, 합성변수는 `--use-extended-features`, 전처리만 실행(학습/평가 없음)은 `--dry-run` 옵션 사용. `--synth-tier {safe\|aggressive}`는 `run_training.py`에 전달됨 |
| 확장 설계(목표) | `prediction_evidence`, `reason_code`, `risk_flag`, `ood_flag` 중심 결합/모니터링 체계 |
| **v1.2 운영 자동화** | Zero-Human-in-the-Loop: 13개 수동 프로세스를 자동화하여 사람 역할을 Dashboard 감시 수준으로 축소 |
| 평가 분할 기본값 | pk_file Group + Time Split (이벤트 랜덤 split은 공식 지표에서 제외) |
| Open-World 방어 | S1 파서 3단 폴백 + Quarantine + OOD Score + UNKNOWN 라우팅 |
| Mock 데이터 | `scripts/generate_mock_raw_data.py` — 레이블 Excel + Sumologic 원본형 mock 생성 |
| **Mock E2E 검증** | **`scripts/run_mock_e2e.py`** — 파일 기반 E2E 검증 (mock 생성 → 전처리 → 학습 → 평가 순차 검증, `--mode label-only\|full`, `--no-generate`, `--dry-run`) |
| **전처리 설정 외부화** | **`config/preprocessing_config.yaml`** — Sumologic 컬럼 매핑(15개), pk_file fields, JOIN label_cols, S1 파서 파라미터, datetime 포맷, 인코딩 후보, 에러 처리 정책을 코드 밖으로 분리 |
| **데이터소스 레지스트리** | **`src/data/datasource_registry.py`** — `DataSourceRegistry` 클래스. `preprocessing_config.yaml`의 data_sources 섹션 관리. `--datasource <name>` CLI 옵션으로 검출 소스 선택. YAML 추가만으로 새 소스 인식 |
| **S1 파서 설정화** | **`S1ParserConfig` dataclass** (`src/data/s1_parser.py`) — `window_size`/`masking_pattern`/`pattern_truncation`/`context_truncation`을 dataclass로 관리. `from_yaml()`으로 preprocessing_config.yaml 읽음. 하위 호환 |
| **Excel 읽기 설정화** | **`ingestion_config.yaml` `excel_read` 섹션** — `header_row: 1` (Excel 2행 헤더), `skip_first_col: true` (연번 컬럼 자동 제거). CSV에는 미적용 |
| **에러 처리 정책** | `on_no_valid_sheet` (warn_and_first_fallback/raise/skip), `on_datetime_parse_fail` (quarantine → silver_quarantine.parquet / warn), JOIN 0건 시 `_diagnose_join_mismatch()` 자동 진단 |
| **컬럼명 Robust 처리** | `ColumnNormalizer.normalize()` 앞뒤 공백/탭 자동 strip → ` 서버이름`, `\t서버이름` 등 모두 정상 처리. `strict=True` 모드 추가 (미등록 한글 컬럼 → ValueError). `_is_valid_sheet()` stripped set 비교 |
| **CSV 인코딩 설정화** | `loader._get_csv_encoding_candidates()` — `preprocessing_config.yaml`에서 읽음. 실패 시 [utf-8, utf-8-sig, cp949] fallback |
| **Mock 레이블 Excel 구조** | `generate_mock_raw_data.py` — 실제 형식 재현: Row 1=타이틀, Row 2=헤더(연번 포함), Row 3+=데이터 |
| **PoC 리포트 자동 생성** | `scripts/run_poc_report.py` + `src/report/excel_writer.py` — 7-sheet Excel (1_요약/2_데이터통계/3_모델성능/4_Coverage곡선/5_Rule기여도/6_오분류분석/7_신뢰도분포). `PocReportData` dataclass + `PocExcelWriter`. `--phase 1\|2`, `--precision-target`, `--skip-ml` 옵션 지원. `_build_business_impact()`: total_fp × coverage_at_tau → Phase 1 목표 ≥40% 자동 판정. §23 상세 |
| **표준 아티팩트 저장 (Step 9)** | `run_training.py` `_run_label_mode()` 학습 완료 후 `models/final/`에 6개 표준 아티팩트 자동 저장: `best_model_v1.joblib`, `label_encoder.joblib`, `feature_builder.joblib` (FeatureBuilderSnapshot), `ood_detector.joblib`, `calibrator.joblib` (조건부), `feature_schema.json`. `run_inference.py`가 기대하는 경로/형식과 일치. `§18.3` 상세 |
| **FeatureBuilderSnapshot** | `src/models/feature_builder_snapshot.py` — fitted TF-IDF 벡터라이저 딕셔너리 + dense 컬럼 목록을 단일 객체로 캡슐화. `.transform(df)` 메서드로 새 데이터에 동일 변환 적용. `from_build_result(result)` factory로 `build_features()` 반환값에서 자동 생성. dense_columns는 `feature_names`에서 `tfidf_` 접두사 없는 항목으로 자동 추론. Phase 1 view: `phase1_fname`(file_name char n-gram), `phase1_fname_shape`(file_name shape char n-gram, Wave 4 B3), `phase1_path`(file_path word) |
| **Tier 2 Wave 4 피처 개정** | B2 중복 가중치(`1/sqrt(group_size)`) + B7 `server_env`/`server_is_prod`/`server_stack` + B1 범주형 8개 Label Encoding + B8 `rule_confidence_lb`/`rule_id_enc`/`rule_primary_class_enc` + B9 `file_event_count`/`file_pii_diversity` (train fold only) + B3 Shape TF-IDF 100 + B6 정규화 강화 (min_child_samples=200, reg_alpha=0.5, max_depth=10) |
| **threshold_policy.json** | `models/final/threshold_policy.json` — Step 6c Coverage-Precision Curve 결과 자동 저장 (추천 tau, curve summary, split 전략, F1). Tier 3 C5 구현 |

> 클래스명 표기는 §12.1의 **실제 데이터 확정 체계**를 우선 적용한다. 과거 문서의 Canonical 8-class(FP-숫자나열/코드, FP-타임스탬프, FP-bytes크기, FP-내부도메인, FP-OS저작권, FP-패턴맥락)는 실제 데이터 분석 후 재편된 클래스명으로 대체되었다.

### 0.1 2-Phase Staged Deployment (2026-02-26 전략 전환)

EDA 결과 레이블 Excel에 `full_context_raw` 텍스트 컨텍스트가 부재함을 확인. 이에 따라 아키텍처 목표(7-class + TF-IDF 텍스트 피처)를 단계적으로 접근하는 구조로 전환한다.

```
Phase 0 (즉시 — 실 데이터 투입 전)
  레이블 데이터 품질 검증 + Bayes Error 하한 계산
  → Go/No-Go 판정 → 클래스 체계 확정 입력
  산출물: go_no_go_report.md, fp_description_unique_list.csv

Phase 1 (레이블 단독 — Sumologic JOIN 대기 중 즉시 실행)
  Selective Classification: 고확신 FP 자동 처리 (precision ≥ 0.95)
  Rule Labeler (경로/파일명 기반) + 메타데이터 ML (~35 dense 피처)
  클래스: 축소 5-class (§12.1 참조)
  피처: 경로/파일명/서버/검출통계/시간 (텍스트 피처 없음)
  → v0: 파일 메타데이터 기반 분류기

Phase 2 (Sumologic JOIN 성공 시 — 병행 추진)
  텍스트 피처 추가 → coverage 확대 + 전체 7-class 세분화
  Phase 1 파이프라인 위에 피처만 확장 (S0/S3a/S4/S5/S6 재사용)
  → v1: 이벤트 컨텍스트 기반 분류기 (본 아키텍처 문서의 목표)
```

| 단계 | 핵심 제약 | 운영 가치 |
|------|----------|----------|
| Phase 1 | 텍스트 컨텍스트 없음, 파일 단위 메타 분류 | Auto-FP Coverage ≥ 40%, Precision ≥ 0.95 |
| Phase 2 | Sumologic JOIN 성공 필요 | Auto-FP Coverage ≥ 60%, 7-class 세분화 |

> **JOIN 실패/지연 시에도** Phase 1은 독립적으로 운영 가치를 제공한다. Phase 1에서 구축하는 S3a, S4, S5, S6은 Phase 2에서 코드 변경 없이 재사용된다.

---

## 1. 설계 원칙 & Rationale

이 아키텍처는 7개의 핵심 설계 원칙 위에 설계되었다. 각 원칙은 프로젝트의 고유한 제약(폐쇄망, 마스킹 데이터, 보안 운영 맥락)에서 도출되었으며, 이후 모든 기술적 선택의 근거가 된다.

### 원칙 A: 설명가능성 = 데이터 산출물

> **"설명은 UI에서 만드는 것이 아니라, 파이프라인이 증거(evidence)를 같이 출력하도록 강제한다."**

**Rationale:**

- UI/리포트는 나중에 언제든 바뀔 수 있지만, 데이터 스키마(결과 컬럼/증거 테이블)가 고정되면 운영/감사(Audit)/재현성/모델 개선/고객 커뮤니케이션이 모두 안정적으로 가능하다.
- 보안 운영 환경에서 "왜 오탐으로 판단했는가"를 레코드 단위로 추적할 수 있어야 규정 준수와 사후 검증이 가능하다.
- 설명 데이터가 구조화되어 있으면, 추후 대시보드/Splunk/엑셀 등 어떤 형태로든 시각화할 수 있다. 역은 성립하지 않는다(UI 전용 설명은 데이터로 되돌리기 어렵다).

### 원칙 B: Layer 1~2는 "필터"가 아니라 "라벨러(Labeler)"

> **"룰이 매칭되면 해당 건을 삭제하는 것이 아니라, '왜 오탐인지'를 포함한 라벨을 부여하고 결과를 남긴다."**

**Rationale:**

- 필터링(삭제/제거)은 결과가 남지 않아서 "왜 오탐인지"를 설명할 수 없다. 필터로 처리하면 감사 추적이 끊기고, 룰의 정확도를 사후에 검증할 방법이 없다.
- 라벨러는 결과를 남기므로, "이 케이스는 FP-라이브러리이며, 근거는 redhat.com 매칭이다" 같은 형태로 설명가능성을 100% 충족한다.
- 운영 리스크 측면에서도 "룰이 맞든 틀리든" 근거가 기록되므로 사후 검증(룰 정밀도 추적)이 쉽고, 룰 자체의 개선 사이클이 가능해진다.
- 라벨러 설계에서는 최종 출력이 항상 동일한 스키마(라벨+사유코드+증거+신뢰도)를 따르므로, 라벨의 출처가 RULE이든 ML이든 후속 프로세스가 동일하게 작동한다.

### 원칙 C: 불확실성을 숨기지 말고 수치화하여 리스크를 줄인다

> **"모든 예측에는 신뢰도 점수가 동반되며, 애매한 케이스는 보수적으로 처리(TP override)되도록 시스템이 강제한다."**

**Rationale:**

- 이 문제는 입력 자체가 본질적으로 제한(마스킹 + 약 10자 컨텍스트)되어 있어, 원천적으로 애매한 영역(ambiguous zone)이 존재한다. 100% 정확한 분류는 불가능하다.
- 애매함을 억지로 단정하면 리스크가 커진다. 특히 TP(실제 개인정보)를 FP로 잘못 분류하면 개인정보 유출 위험으로 이어진다.
- 시스템이 불확실성을 수치(confidence, margin, entropy)로 표현하고, 후처리에서 정책적으로 "애매하면 TP"를 강제하면 운영 안정성이 확보된다.
- RULE의 신뢰도와 ML의 신뢰도는 정의가 다르므로(경험적 정밀도 하한 vs 보정된 확률), 각각에 맞는 산출 방식을 적용해야 "같은 숫자"로 오해하는 사고를 방지할 수 있다.

### 원칙 D: 폐쇄망/대용량/월배치에서는 "중간 산출물 분리"가 곧 안정성

> **"파이프라인의 각 Stage는 독립 실행 가능하며, 중간 결과를 파일로 남긴다."**

**Rationale:**

- 파이프라인이 한 번에 길게 이어지면 중간에 실패했을 때 재시작 비용이 크다. 월 100~150만 건 규모에서 "한 번 더 돌려보자"가 부담이 된다.
- 중간 결과(파싱된 이벤트, 피처, 룰 결과, ML 결과)를 단계별로 저장하면 재실행 비용 감소, 장애 격리, 동일 입력에 대한 동일 출력 재현, 모델/룰 교체 실험이 모두 쉬워진다.
- 폐쇄망 환경에서는 디버깅이 제한적이므로, 각 단계의 산출물을 검사할 수 있는 것이 개발 생산성에 직접적으로 영향을 준다.
- Parquet 형식으로 중간 산출물을 저장하면 컬럼 기반 압축으로 스토리지 효율이 좋고, pandas/polars에서 바로 읽을 수 있어 폐쇄망 개발에 적합하다.

### 원칙 E: 피처 엔지니어링 5대 설계 원칙 — "패턴이 전부가 아님"

> **"현재 보이는 패턴만으로 설계하면 새 패턴이 추가될 때 시스템이 깨진다. 표본 밖 패턴에도 버티는 구조를 만든다."**

이 프로젝트의 검출 대상(PII 패턴)은 시간이 지남에 따라 변화한다. 새로운 로그 포맷, 새로운 도메인, 새로운 시스템이 추가되면 기존 패턴만으로는 대응할 수 없다. 따라서 피처 엔지니어링은 아래 5가지 원칙 위에 설계한다.

**E-1. 비파괴 원칙 (Non-destructive)**

원문을 최대한 보존하고, 전처리 산출물은 항상 파생 컬럼으로 만든다.

- 예: `full_context_raw`(원문) + `full_context_norm`(정규화) + `raw_text` + `shape_text`를 모두 보존
- 새 패턴이 와도 원문에서 다시 파생 가능

**E-2. 고엔트로피 값만 추상화 (Entropy Reduction)**

UUID/해시/긴 숫자열/hex/base64처럼 "거의 매번 달라지는 값"만 placeholder로 바꾸고, 그 외 구분자/키워드/구조는 유지한다.

- 단어 사전 폭발/과적합을 막으면서 구조/키워드 신호 유지
- placeholder 예: `<NUM10>`, `<NUM13>`, `<DATE8>`, `<HEX>`, `<HASH>`, `<MASK>`

**E-3. 멀티뷰 텍스트 (Semantic + Structural)**

동일 원문에서 서로 다른 관점의 피처를 추출한다.

- `raw_text`: 키워드/도메인/경로 토큰 (의미 신호)
- `shape_text`: 숫자/문자/마스킹/구분자의 "형태" (구조 신호)
- 두 뷰를 함께 사용하면 새 패턴이 들어와도 구조적으로 버틴다

**E-4. 앵커 기반 컨텍스트 윈도우링 (Anchor Windowing)**

`inspectcontentwithcontext`가 길거나 멀티라인이면 잡음이 많아질 수 있으므로, 검출값(마스킹된 hit)을 앵커로 주변만 잘라 모델에 준다.

- 새 패턴이 늘어나도 "검출 주변"이라는 문제 정의는 변하지 않음
- 컨텍스트 포맷이 달라져도 일관된 입력 보장

**E-5. 확장 가능한 룰/키워드 (설정 파일 기반)**

지금 보이는 도메인/키워드가 전부가 아니므로, 룰/키워드는 코드가 아니라 YAML/JSON 설정으로 운영 확장 가능해야 한다.

- 코드 수정 없이 패턴 추가/삭제 가능
- 버전 관리(Git)로 변경 이력 추적

### 원칙 F: 모르는 것을 모른다고 인정하는 시스템을 만든다 (Open-World Defense)

> **"파이프라인의 모든 단계는 '예상 밖 입력'에 대해 실패(crash)하거나 침묵(silent failure)하지 않고, 명시적으로 격리(quarantine)·미분류(unknown)·검토 요청(review)을 출력한다."**

**Rationale:**

현 시스템은 Closed-Set Classification(7클래스 고정, Phase 2 기준)을 전제로 한다. 그러나 운영 환경에서는 5가지 유형의 "새로운 데이터"가 필연적으로 유입된다.

| # | 새로운 데이터 유형 | 대응 방안 |
|---|------------------|----------|
| 1 | 입력 스키마/포맷 변화 | S0 Schema Registry + Quarantine |
| 2 | 마스킹/컨텍스트 포맷 변화 | S1 3단 폴백 + parse_status |
| 3 | 신규 토큰/도메인/경로 | OOV 모니터링 + 주기적 재학습 |
| 4 | 새 오탐 유형 (기존 클래스 내) | Reason Code 확장 (대응 가능) |
| 5 | 새 카테고리 (기존 클래스 밖) | UNKNOWN 라우팅 + OOD Score |

이 원칙은 이후 모든 Stage의 수정 방향에 일관성을 부여한다. 학습에 없던 데이터가 들어왔을 때 시스템이 crash하거나 silent misclassification하는 것이 아니라, 명시적으로 격리·미분류·검토 요청을 출력하도록 설계한다.

### 원칙 G: Zero-Human-in-the-Loop — 판단·수정·검토를 시스템이 수행한다 (v1.2 추가)

> **"사람의 역할을 '시스템 감시(Dashboard 확인)' 수준으로 축소하고, 모든 판단·수정·검토를 시스템이 자율적으로 수행한다."**

**Rationale:**

- v1.1까지의 설계에는 13개 프로세스에서 사람이 직접 개입(검토, 승인, 수정, 판단)해야 한다. 이는 월 배치 운영에서 병목이 되며, 인력 의존도가 높으면 확장성과 일관성이 떨어진다.
- 개인정보 보호 관점에서 **보수적 기본값(Conservative Default)** 전략을 적용하면 사람 검토 없이도 안전하다. "모르면 TP(실제 개인정보)"로 처리하면 개인정보를 놓치지 않으므로 안전측으로 기울어진다.
- 자동화의 핵심은 **교차 검증(Cross-Source Validation)**이다. RULE과 ML이 독립적으로 동일한 결론에 도달하면 사람의 확인 없이도 신뢰할 수 있다. 불일치가 발생하면 보수적으로 TP 처리한다.
- KPI 알람에 대한 자동 조치(Auto-Remediation Playbook)를 사전에 매핑해두면, 알람 발생 시 사람이 원인을 분석하고 판단할 필요 없이 시스템이 즉시 대응한다.
- 시간이 지남에 따라 ML이 반복적으로 잡는 패턴을 자동으로 룰로 승격(Auto-Rule-Promoter)하고, 새로운 스키마/라벨/분류체계를 자동으로 감지·적용(Auto-Schema-Detector, Auto-Mapper, Auto-Taxonomy-Manager)함으로써 시스템이 자율적으로 진화한다.

**이 원칙이 적용되는 13개 자동화 대상:**

| # | 수동 프로세스 | 자동화 전략 | 관련 섹션 |
|---|-------------|-----------|----------|
| 1 | NEEDS_REVIEW 검토 | Auto-Adjudicator: 4단 자동 판정 | §9 |
| 2 | UNKNOWN (OOD) 검토 | 자동 클러스터링 + 보수적 TP 기본값 | §9, §12 |
| 3 | 현업 피드백 수집 | Self-Validation Loop: RULE↔ML 교차 검증 | §11 |
| 4 | 샘플링 QA | Auto-Precision-Estimator: 합의율 프록시 | §11 |
| 5 | KPI 알람 대응 | Auto-Remediation Playbook | §11 |
| 6 | rules.yaml 업데이트 | Auto-Rule-Promoter | §7 |
| 7 | label_mapping.yaml 업데이트 | Auto-Mapper: 퍼지 매칭 자동 매핑 | §12 |
| 8 | 분기별 Taxonomy 리뷰 | Auto-Taxonomy-Manager | §12 |
| 9 | schema_registry.yaml 업데이트 | Auto-Schema-Detector | §4 |
| 10 | 임계값 (TAU) 조정 | Auto-Tuner: Rolling validation 최적화 | §9 |
| 11 | 라벨 품질 감사 | Confident Learning 자동 감사 | §22 |
| 12 | 합성변수 ablation 결정 | Automated Ablation Pipeline | §6 |
| 13 | 대안 모델 비교·선택 | Auto-Model-Selector | §8 |

---

## 2. 시스템 개요

### 2.1 프로젝트 배경 요약

Server-i는 소만사가 개발한 서버 개인정보보호 DLP 솔루션으로, 약 30,000대 서버 내 방치된 개인정보(PII)를 패턴 매칭 방식으로 검출한다. 검출 대상 PII는 3종(이메일, 휴대폰 번호, 주민등록번호)이다.

현재 검출 결과의 약 2/3가 오탐(False Positive)이며, 소만사 파견 인력이 마스킹된 컨텍스트 약 10자만으로 수작업 판단을 수행하고 있다. 이 프로젝트는 AI 모델을 통해 오탐을 자동 분류하여, 보안 운영팀의 수작업 부담을 50% 이상 절감하는 것을 목표로 한다.

### 2.2 핵심 제약 조건

| 제약 | 상세 | 아키텍처 영향 |
|------|------|---------------|
| 마스킹 데이터 Only | Server-i 자체에서 마스킹 처리 완료. 원본 확인 불가 | 모델도 마스킹 컨텍스트 약 10자로 판단해야 함 |
| GPU 미보유 | 개발 서버에 GPU 없음 (회의 확정) | Transformer/LLM 학습 불가 → CPU 기반 Boosting 모델 확정 |
| 폐쇄망 환경 | 외부 인터넷 접근 불가 | 모든 패키지 사전 반입, 오프라인 개발 |
| 월 100~150만 건 | 대용량 배치 처리 필요 | 효율적 피처 엔지니어링, 메모리 관리 필수 |
| 보수적 원칙 | 정탐(TP)을 오탐(FP)으로 잘못 분류하는 것을 최소화 | "애매하면 TP" 정책을 시스템에서 강제 |

### 2.3 핵심 재정의: 3-Layer는 "필터 체인"이 아니라 "라벨러 체인"

> 운영 스크립트 기본값은 라벨러 체인 미적용(`--use-filter` 미지정)이며, 합성변수는 기본 OFF(Tier 0)다. 합성변수 확장은 `--use-extended-features` 옵션으로 명시적으로 활성화한다. (§0 snapshot, §6.5 참조)

기존 문제 정의서의 3-Layer 접근:
```
Layer 1: Keyword 필터링 → 명확한 오탐 제거
Layer 2: Rule 필터링    → 비즈니스 로직 제거
Layer 3: ML 모델        → 나머지 분류
```

**재정의된 3-Layer 접근:**
```
Layer 1~2 = 고정밀(High-precision) 라벨러 집합
  → 룰이 맞으면 그 자체로 클래스를 결정하고, "왜 오탐인지"를 증거로 남김

Layer 3  = 잔여(룰로 결정 못한) 샘플에 대한 ML 라벨러
  → 확률 + 설명 + 불확실성 기반 보수적 처리
```

최종 출력은 항상 "라벨"이며, 라벨의 출처(decision_source)가 RULE / ML / HYBRID인지만 달라진다.

**왜 이렇게 재정의하는가:**

- 필터는 결과를 남기지 않는다. "왜 오탐인지"를 데이터로 설명하려면 필터가 아니라 라벨러여야 한다.
- 라벨러로 설계하면, RULE이든 ML이든 동일한 출력 스키마(라벨+사유+증거+신뢰도)를 따르므로 후속 프로세스(운영/감사/개선)가 통일된다.
- 필터로 처리된 건은 "룰이 잘못 걸렀다"는 사실을 사후에 발견할 방법이 없지만, 라벨러라면 prediction_evidence 테이블에서 즉시 확인할 수 있다.

### 2.4 End-to-End 흐름 요약

```
[Stage S0: Raw Ingest — Bronze]
  Dataset A: Excel(.xlsx)/CSV 모두 허용 (실 데이터는 Excel) + Label Excel (Dataset B)
  · 인코딩 자동 감지/고정 (utf-8-sig, cp949)
  · 컬럼명 통일 (Schema Canonicalization)
  · Schema Registry 기반 검증 (v1.1)
  · bronze_events.parquet, bronze_labels.parquet
        │
        ▼
[Stage S1: Normalize & Parse — Silver-S1]
  · file_path 정규화 (소문자, 구분자 통일, path_depth/extension 파생)
  · full_context 정규화 (NFKC, 줄바꿈 통일) → full_context_raw + full_context_norm 비파괴 보존
  · "1셀=다중검출" → "1행=1검출 이벤트"로 분해 (3단 폴백: 마스킹→앵커→단일이벤트)
  · left_ctx / masked_pattern / right_ctx / full_context 생성
  · 앵커 기반 local_context 생성 (masked_hit 기준 ±W chars 윈도우링)
  · PK 생성: pk_file=SHA256(server_name|agent_ip|file_path|file_name), pk_event=SHA256(+file_created_at)
  · pii_type_inferred 재추론, email_domain 추출
  · parse_status 기록 + Quarantine 분리 (v1.1)
        │
        ▼
[Stage S2: Feature Prep — Silver-S2]
  · raw_text 생성 (소문자화 + 고엔트로피 토큰 placeholder 치환: <NUM10>, <HASH>, <HEX>, <MASK>)
  · shape_text 생성 (숫자→0, 영문→a, 한글→가, 구분자/*/@ 유지)
  · path_text 생성 (경로 토큰화: /var/log/hadoop → var log hadoop)
  · 구조/통계 피처 (char_length, digit_ratio, newline_count, masking_ratio 등)
  · path parsing (확장자, 토큰, 플래그, has_date_in_path)
  · tabular 정규화 (log1p inspectcount, is_mass/extreme_detection)
  · placeholder 비율 피처 (Unknown-like 신호: <HASH> 비율, <NUM*> 비율)
  · file-level aggregation (선택): 파일 단위 통계 join
        │
        ├──────────────────────────────┐
        ▼                              ▼
[Stage S3a: RULE Labeler]      [Stage S3b: ML Feature Builder + Labeler]
  · 룰 매칭 (다중 후보)          · raw_text → word TF-IDF (개선된 token_pattern)
  · primary_class 부여           · raw_text → char TF-IDF (OOV 완화, v1.1)
  · reason_code 부여             · shape_text → char TF-IDF (3~5gram, v1.1)
  · evidence 생성                · path_text → word TF-IDF (선택)
  · rule_confidence 계산         · manual features + tabular + 피처 스키마 검증
        │                        · sparse/dense 결합 + OOD Score (v1.1)
        │                        · 7클래스 확률 출력 + 클래스별 차등 calibration
        │                              │
        └──────────┬───────────────────┘
                   ▼
         [Stage S4: Decision Combiner]
           · OOD/극도 불확실 → UNKNOWN 라우팅 (v1.1 Case 0)
           · RULE vs ML 결합 (조건부 TAU_TP_OVERRIDE, v1.1)
           · TP 안전장치 (애매하면 TP/REVIEW)
           · 최종 라벨 + confidence + risk_flag + ood_flag 확정
                   │
                   ▼
         [Stage S5: Output Writer]
           · predictions_main   (1행=1검출, 결론)
           · prediction_evidence (N행=1검출, 근거 상세)
           · PK로 원본 join 가능한 형태 유지
                   │
                   ▼
         [Stage S6: Monitoring / Feedback]
           · 3계층 12종 KPI 자동 산출 + 알람 (v1.1)
           · 월별 클래스 분포, confidence 분포, 룰 매칭률 추적
           · OOV/신규 토큰 모니터링 (패턴 확장 후보)
           · OOD 비율 추적 + UNKNOWN 축적 현황 (v1.1)
           · 전처리 변환 전/후 샘플링 로그 저장
            · Self-Validation Loop 결과 축적 → 룰 정밀도 업데이트 + 재학습
           · monthly_metrics.json 생성 (v1.1)
```

---
