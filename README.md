# Server-i PII 오탐 개선 AI 모델링 프로젝트

**대상 솔루션:** Server-i (소만사)
**프로젝트 단계:** PoC (Architecture v1.3 정합화)

---

## 프로젝트 개요

본 프로젝트는 Server-i 검출 결과 중 오탐(False Positive)을 AI로 재분류하여 운영 공수를 줄이는 것을 목표로 합니다.

- 분류 체계: **정탐 1종 + 오탐 6종 = 7클래스** (학습) + `UNKNOWN` (운영 정책 클래스)
- 입력 제약: 마스킹된 검출값 + 주변 컨텍스트(원본 PII 접근 불가)
- 운영 제약: 폐쇄망(Air-gapped), CPU 전용, 월 배치(대용량)

---

## 아키텍처 기준선

`docs/Architecture/` 기준으로 파이프라인 용어를 아래처럼 통일합니다.

- 기존 표현: `3-Layer Filter`
- 표준 표현: `RULE/ML Labeler Chain`

### S0~S6 단계

| Stage | 역할 | 현재 리포지토리 주요 구현 |
|-------|------|--------------------------|
| S0 | Raw Ingest & Schema Canonicalization | `src/data/loader.py`, `config/schema_registry.yaml` |
| S1 | Normalize & Parse (1행=1검출 이벤트) | `src/data/s1_parser.py` |
| S2 | Feature Prep | `src/features/*`, `config/feature_config.yaml` |
| S3a | RULE Labeler + Evidence | `src/filters/rule_labeler.py`, `config/rules.yaml`, `config/rule_stats.json` |
| S3b | ML Labeler (학습/추론) | `scripts/run_training.py`, `src/models/trainer.py` |
| S4 | Decision Combiner | 아키텍처 설계 완료, 스크립트 통합은 후속 |
| S5 | Output Writer (`predictions_main`, `prediction_evidence`) | 아키텍처 설계 완료, 표준 산출물 통합은 후속 |
| S6 | Monitoring & Feedback (`monthly_metrics`) | 아키텍처 설계 완료, 자동 KPI 루프는 후속 |

---

## 구현 정합성 스냅샷 (2026-03-16)

| 항목 | 기준 |
|------|------|
| 분류 클래스 | `TP-실제개인정보` + 6개 `FP-*` 클래스 (2026-03 확정) |
| 설정 파일 | `config/feature_config.yaml`, `config/model_config.yaml`, `config/filter_config.yaml`, `config/rules.yaml`, `config/schema_registry.yaml` |
| 학습 기본 경로 | `scripts/run_training.py` (필터 기본 OFF, 합성 확장 기본 OFF) |
| 정합화 상태 | `docs/Architecture/00_overview.md` §0과 현재 코드 기본값 일치 (Tier0 OFF, `--use-extended-features`로 ON) |
| **eval_set leakage 수정** | **`train_xgboost()` / `train_lightgbm()` — 내부 20% val split으로 교체 (테스트셋 오염 제거)** |
| 평가 분할 | `GroupShuffleSplit(groups=pk_file)` 구현 완료 — pk_file 없는 경우 Stratified fallback |
| Phase 1 피처 보강 | `use_phase1_tfidf=True` — file_name char TF-IDF(500) + path TF-IDF(500) + server_freq → ~1,028 피처 |
| Temporal Split 연결 | `label_work_month` 있으면 `work_month_time_split()` 자동 실행 — [Temporal] vs [Random] F1 비교 출력 |
| Coverage-Precision Curve | `compute_coverage_precision_curve()` → Step 6c에서 τ 스윕 실행, 권장 τ 자동 출력 |
| Bootstrap CI + PR-AUC | `full_evaluation()` — PR-AUC + F1-macro 95% CI (n=500) 추가 |
| Probability Calibration | `calibrate_model(cv=3, isotonic)` — `--calibrate` 옵션 시 `*_calibrated.joblib` 저장 |
| SMOTE 제거 | `apply_smote()` 삭제 (Sparse TF-IDF 비호환 dead code) |
| PoC 리포트 시스템 | `scripts/run_poc_report.py` — 6-sheet Excel 자동 생성 (poc_metrics, rule_analyzer, excel_writer) |
| Split 전략 확장 | work_month_time_split (Secondary), org_subset_split × 3조직 (Tertiary) |

---

## 빠른 시작 (폐쇄망 서버)

### 1) 환경 준비

```bash
bash scripts/setup_env.sh
python scripts/verify_env.py
```

### 2) 데이터 배치

- `data/raw/label/`: 레이블 Excel (정탐/오탐 × 월 × 조직)
- `data/raw/dataset_a/`: Sumologic 검출 원본 xlsx

### 3) 파이프라인 실행

```bash
# 레이블 Excel만 사용 (Phase 1 기본)
python scripts/run_pipeline.py --mode label-only

# Sumologic JOIN 포함 (Phase 2)
python scripts/run_pipeline.py --mode full
```

### 4) 산출물 내보내기

```bash
python scripts/run_export.py                 # Parquet → CSV (utf-8-sig)
python scripts/run_export.py --format xlsx   # Parquet → Excel
python scripts/run_export.py --info          # 스키마 정보만 확인
```

### 5) PoC 리포트 생성

```bash
python scripts/run_poc_report.py             # Label Only (Phase 1)
python scripts/run_poc_report.py --skip-ml   # Rule 분석만
```

### 선택 옵션

```bash
python scripts/run_training.py --use-filter              # RULE Labeler 포함
python scripts/run_training.py --use-extended-features   # 합성변수 확장
python scripts/run_training.py --n-splits 5              # Repeated GroupShuffleSplit (분산 추정)
python scripts/run_training.py --calibrate               # 확률 보정 (isotonic, cv=3) 활성화
python scripts/run_evaluation.py --include-filtered      # 필터 결과 포함 평가
```

---

## 클래스 체계 (2026-03 확정)

```text
TP-실제개인정보      ← 정탐 (파일 출처에서 자동 부여)
FP-파일없음          ← 파일/경로 미존재
FP-이메일패턴        ← @문자/kerberos/내부도메인/OSS도메인 오인식
FP-숫자패턴          ← 타임스탬프·bytes·일련번호 (기존 3-class 통합)
FP-라이브러리        ← RPM/docker/conda/npm/pip 패키지, 오픈소스 설치파일
FP-더미테스트        ← 테스트/더미/분析용 임시파일
FP-시스템로그        ← 서비스로그·배치·DB·백업·인프라 운영 데이터
UNKNOWN              ← 운영 정책 클래스 (OOD/미분류)
```

> 초기 설계의 Canonical 8-class는 실제 fp_description 611개 분석 후 위 체계로 재편됨.
> 상세 매핑: `docs/Architecture/06_monitoring_classification.md §12.1`

---

## 성공 기준

### 현재 PoC 게이트 (`config/feature_config.yaml`)

- Macro F1 >= 0.70
- TP Recall >= 0.75
- FP Precision >= 0.85

### 아키텍처 운영 목표

| Phase | 목표 |
|-------|------|
| Phase 1 (레이블 단독) | Auto-FP Precision ≥ 0.95 / Auto-FP Coverage ≥ 40% |
| Phase 2 (JOIN 후) | Auto-FP Coverage ≥ 60% / Macro F1 ≥ 0.85 / 공수 50%+ 절감 |

---

## 프로젝트 구조 (요약)

```text
config/           아키텍처/모델/룰/스키마 설정
src/              데이터·피처·라벨러·모델·리포트 구현
scripts/          실행 엔트리포인트
data/             raw/processed/features
models/           학습 산출물(joblib)
outputs/          리포트/시각화
tests/            단위+E2E 테스트
docs/
  Architecture/   아키텍처 설계서 (섹션별 분리)
  plans/          구현 계획 문서
```

---

## 문서 안내

| 문서 | 용도 |
|------|------|
| `docs/Architecture/` | 시스템 아키텍처 기준 (설계 원칙·Stage·클래스·로드맵) |
| `PROJECT_GUIDE.md` | 운영 실행 상세 런북 |
| `Env_scenario.md` | 폐쇄망 반입/접근 시나리오 |
| `CLAUDE.md` | 코딩 에이전트 운영 컨텍스트 |

---

**작성 기준일:** 2026-03-16
