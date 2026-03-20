# Implementation Plan: 모델 성능 신뢰성 진단 및 리포트 통합

**Status**: In Progress
**Started**: 2026-03-20
**Last Updated**: 2026-03-20

## Overview

### 문제 정의

모델 성능이 높게 나온다. 이것이 실제 운영에서도 유지되는 성능인지 확인이 필요.

**핵심 질문**: 3~9월 데이터로 학습한 모델이 10~12월 데이터를 잘 예측하는가?

데이터 편향 분석은 "왜 성능이 높은지"를 해석하는 보조 도구이며,
"편향이 있으니 모델이 틀렸다"는 결론과는 다르다.

### 성공 기준

- [ ] Temporal holdout(10~12월)으로 일반화 성능 확인
- [ ] 평가 파이프라인 알려진 버그 수정
- [ ] Training -> Report 통합 파이프라인 완성
- [ ] Label/Joined 양쪽 모두 리포트 생성 가능

---

## 현재까지 완료된 작업

### Phase A: Temporal Split 추가 (완료)

| 파일 | 변경 내용 |
|------|-----------|
| `src/features/pipeline.py` | `build_features()`에 `split_strategy` 파라미터 추가 ("group"/"temporal"/"server") |
| `src/features/pipeline.py` | `test_months` 파라미터 추가 (temporal split 시 마지막 N개월) |
| `scripts/run_training.py` | `--split` CLI 옵션 추가 (group/temporal/server) |
| `scripts/run_training.py` | `--test-months` CLI 옵션 추가 (기본 3) |

**사용법**:
```bash
# 기존 (GroupShuffleSplit) - 변경 없음
python scripts/run_training.py --source label

# Temporal holdout: 마지막 3개월 test (핵심 진단)
python scripts/run_training.py --source label --split temporal --test-months 3

# Temporal holdout: 마지막 2개월 test
python scripts/run_training.py --source label --split temporal --test-months 2

# Server split: 새 서버 시나리오
python scripts/run_training.py --source label --split server
```

### Phase B: 데이터 진단 스크립트 (완료)

| 파일 | 내용 |
|------|------|
| `scripts/diagnose_data_bias.py` | 독립 실행 가능한 진단 스크립트 |
| `docs/data_flow_risk_analysis.md` | Raw -> Feature -> Model 데이터 플로우 분석 보고서 |

**진단 항목**:
- A: Column-wise 편향 (Cramer's V, MI, 단일피처 F1)
- B: Row-wise 품질 (pk_file 충돌, 중복, 클래스 분포)
- D: Split Robustness (temporal/server/random 비교)
- E: Feature Block Ablation (server/path/time/count 블록별 제거)
- F: Column Risk Registry

**사용법**:
```bash
python scripts/diagnose_data_bias.py
python scripts/diagnose_data_bias.py --skip-ablation      # 빠른 실행
python scripts/diagnose_data_bias.py --skip-single-feature # 더 빠른 실행
```

---

## 확인된 문제점

### 코드 버그 (수정 필요)

| # | 문제 | 위치 | 영향 |
|---|------|------|------|
| B1 | PoC report가 학습 때와 다른 split 사용 | run_poc_report.py:489 | 리포트 수치 신뢰 불가 |
| B2 | server_freq를 전체 데이터로 재계산 | run_poc_report.py:453 | train-test 오염 |
| B5 | tp_label="TP" vs 실제 "TP-실제개인정보" 불일치 | run_poc_report.py:403 | 이진 지표 깨짐 |
| B4 | tune_model CV가 pk_file 그룹 무시 | trainer.py:279 | 하이퍼파라미터 낙관적 |
| B7 | run_evaluation.py가 label 모델만 로드 (하드코딩) | run_evaluation.py:242 | joined 평가 불가 |
| B8 | run_poc_report.py가 label 모델만 로드 (하드코딩) | run_poc_report.py:415 | joined 리포트 불가 |

### 구조적 이슈

| # | 문제 | 현재 상태 |
|---|------|-----------|
| S1 | eval과 report가 기능 겹침 (모델 로드+예측+F1+오분류) | 둘 다 독립 실행 |
| S2 | 진단 결과가 리포트에 미포함 | 별도 txt/csv 산출물 |
| S3 | label/joined 리포트가 분리 안 됨 | label만 지원 |

---

## Phase C: 리포트 통합 (미구현)

### C-1. 통합 리포트 스크립트 (`scripts/run_report.py`)

eval + poc_report + bias 진단을 **하나의 스크립트**로 통합.

**CLI**:
```bash
# Label 모델 리포트
python scripts/run_report.py --source label

# Joined 모델 리포트
python scripts/run_report.py --source detection

# 진단 포함
python scripts/run_report.py --source label --include-diagnosis

# 진단 제외 (빠른 실행)
python scripts/run_report.py --source label --skip-diagnosis
```

**실행 흐름**:
```
run_report.py --source label
  |
  |-- [1] 모델 로드 (best_model_v1.joblib 또는 detection_best_model_v1.joblib)
  |-- [2] 데이터 로드 (silver_label.parquet 또는 silver_joined.parquet)
  |-- [3] 피처 변환 (FeatureBuilderSnapshot.transform - 재학습 없음)
  |-- [4] 예측 수행
  |-- [5] 핵심 평가 (F1, PoC 판정, Confusion Matrix, Feature Importance)
  |-- [6] 상세 분석 (Coverage, Rule 기여도, 오분류, Split 비교)
  |-- [7] 데이터 진단 (--include-diagnosis 시)
  |       - Column 특성 분석
  |       - Split Robustness
  |       - Feature Ablation
  |-- [8] Excel 리포트 생성 (9-sheet)
  |-- [9] 개별 산출물 저장 (txt/csv/png)
```

### C-2. 통합 리포트 시트 구성 (기존 7 + 신규 2)

| # | 시트 | 출처 | 내용 |
|---|------|------|------|
| 1 | 요약 | poc_report | 데이터 조건, split, PoC 판정, 메타데이터 |
| 2 | 데이터통계 | poc_report | TP/FP 비율, 월별 분포, fp_description |
| 3 | 모델성능 | poc_report | Split 비교표, 클래스별 Precision/Recall |
| 4 | Coverage곡선 | poc_report | tau 테이블, 권장 tau |
| 5 | Rule기여도 | poc_report | rule_id별 히트율/정밀도 |
| 6 | 오분류분석 | poc_report | 패턴 상위 15 + 샘플 200건 |
| 7 | 신뢰도분포 | poc_report | 예측 확률 분포 |
| **8** | **Feature Importance** | **eval에서 이동** | **Importance 랭킹, 그룹별 합산, 시각화** |
| **9** | **데이터 진단** | **bias.py에서 이동** | **Column 특성, Split Robustness, Ablation** |

### C-3. 개별 산출물 (기존대로 유지)

```
outputs/
  poc_report.xlsx                ← 통합 리포트 (9-sheet)
  classification_report.txt     ← 텍스트 분류 리포트
  confusion_matrix.png          ← 혼동 행렬 시각화
  feature_importance.csv        ← 전체 피처 Importance
  feature_importance.png        ← Importance 바 차트
  error_analysis.csv            ← 오분류 샘플
  diagnosis/                    ← 진단 상세 (--include-diagnosis 시)
    column_bias_report.txt
    column_risk_registry.csv
    row_quality_report.txt
    split_robustness_report.csv
    ablation_report.csv
    bias_summary.md
```

### C-4. Label / Joined 분리

| 항목 | --source label | --source detection |
|------|---------------|-------------------|
| 모델 | `models/final/best_model_v1.joblib` | `models/final/detection_best_model_v1.joblib` |
| 데이터 | `silver_label.parquet` | `silver_joined.parquet` |
| 피처 스냅샷 | `feature_builder.joblib` | `detection_feature_builder.joblib` |
| 리포트 파일 | `outputs/poc_report.xlsx` | `outputs/poc_report_detection.xlsx` |

### C-5. PocReportData 확장

```python
@dataclass
class PocReportData:
    # 기존 Sheet 1~7 필드 (변경 없음)
    ...

    # Sheet 8 - Feature Importance (신규)
    feature_importance_df: pd.DataFrame    # (feature, importance) 전체
    feature_group_importance: dict         # {그룹명: 합산 importance}

    # Sheet 9 - 데이터 진단 (신규, --include-diagnosis 시)
    column_risk_registry: pd.DataFrame     # column_risk_registry.csv 내용
    split_robustness: pd.DataFrame         # split별 F1 비교
    ablation_results: pd.DataFrame         # 블록별 제거 F1
```

---

## Phase D: 평가 파이프라인 버그 수정 (미구현)

### D-1. PoC report split 수정 (B1)

run_poc_report.py가 학습 시와 동일한 split을 사용하도록 수정.
방법: 학습 시 `split_meta.json`에 train_idx/test_idx 저장 -> report가 로드.

### D-2. server_freq train-only 고정 (B2)

run_poc_report.py:453에서 전체 데이터로 server_freq 재계산하는 부분을
FeatureBuilderSnapshot에 저장된 train-only freq map을 사용하도록 변경.

### D-3. tp_label 불일치 수정 (B5)

`tp_label="TP"` -> label_raw 원본 기준("TP"/"FP") 또는 startswith("TP") 매칭.

### D-4. tune_model GroupKFold (B4)

`RandomizedSearchCV`에서 `StratifiedKFold` -> `StratifiedGroupKFold` 변경.

---

## 실행 순서

```
[완료] Phase A: Temporal split 추가 (pipeline.py, run_training.py)
[완료] Phase B: 데이터 진단 스크립트 (diagnose_data_bias.py)
         |
         v
[다음]  Phase C: 리포트 통합 (run_report.py 신규 + excel_writer.py 확장)
         |
         v
[후속]  Phase D: 평가 파이프라인 버그 수정 (B1, B2, B5, B4)
```

### 실 데이터 실행 순서

```bash
# 1. 전처리
python scripts/run_data_pipeline.py --source label

# 2. 기존 방식 학습 (기준선)
python scripts/run_training.py --source label

# 3. Temporal holdout 학습 (일반화 성능)
python scripts/run_training.py --source label --split temporal --test-months 3

# 4. 통합 리포트 생성 (Phase C 완료 후)
python scripts/run_report.py --source label --include-diagnosis

# 5. 두 F1 비교 -> 하락폭이 답
#    - 하락 < 5pp  -> 성능 유지, 모델 신뢰 가능
#    - 하락 5~15pp -> 피처 보완 검토
#    - 하락 > 15pp -> 피처 재설계 필요
```

---

## Notes & Learnings

- server_freq importance 1위(35%)는 리스크가 아닌 도메인 특성 반영일 수 있음
- 데이터 편향 분석은 해석 도구이며, "편향=모델 오류"는 아님
- 진짜 답은 temporal holdout 성능에 있음
- run_evaluation.py와 run_poc_report.py의 기능이 대부분 겹침 -> report로 통합
- label/joined 모델의 eval/report 경로가 하드코딩되어 있어 수정 필요
