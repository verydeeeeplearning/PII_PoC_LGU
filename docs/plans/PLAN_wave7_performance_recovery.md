# Implementation Plan: Wave 7 — 성능 복원 및 구조 개선

**Status**: Pending (사용자 승인 대기)
**Started**: 2026-03-21
**Last Updated**: 2026-03-21

**CRITICAL INSTRUCTIONS**: After completing each phase:
1. Check off completed task checkboxes
2. Run all quality gate validation commands
3. Verify ALL quality gate items pass
4. Update "Last Updated" date
5. Document learnings in Notes section
6. Only then proceed to next phase

DO NOT skip quality gates or proceed with failing checks

---

## Overview

### 배경

Wave 6 적용 후 F1-macro가 **0.8063 → 0.7335 (−9.0%)** 하락했다.
두 에이전트의 독립 분석을 교차 검증한 결과, 하락 원인과 추가 개선 포인트를 확정했다.

### 목표

1. **Wave 6 회귀 복원**: C2(TP 가중치 과보정) 제거 → F1 0.76+ 회복
2. **잠복 버그 수정**: B9 피처(file_event_count, file_pii_diversity)가 학습 행렬에 미포함 → 수정
3. **Categorical 인코딩 개선**: LabelEncoder → Smoothed Target Encoding + Frequency Encoding
4. **서버 토큰 정밀화**: 짧은 토큰(lb, mq, /ci/)의 오매칭 방지
5. **Calibration 제거**: CalibratedClassifierCV(cv=3) 삭제 → 학습 시간 대폭 단축
6. **Decision Combiner 평가**: RULE+ML 결합 최종 판정의 4분면(confusion matrix) 리포트 추가

### 성공 기준

- [ ] F1-macro ≥ 0.80 (Tier 2 수준 복원, temporal leak 없이)
- [ ] B9 피처가 feature_names에 실제 포함됨을 단위 테스트로 검증
- [ ] Categorical encoding이 train-only fit이면서 unseen 정보 보존
- [ ] Decision Combiner 4분면이 리포트에 포함
- [ ] 기존 테스트 전체 통과
- [ ] 추론 경로(FeatureBuilderSnapshot.transform) 동형성 유지

---

## 근거 분석

### 확정된 원인

| 변경 | 성격 | 근거 | 확신도 |
|------|------|------|--------|
| **C2 TP 1.5x** | Wave 6 회귀 | `class_weight="balanced"`(≈2.2x) + B2 + 1.5x = 3중 보정. Tier 2에 없었음. | **높음** |
| **B9 미포함** | 잠복 버그 | `build_features()` 호출 후 B9 계산 → X_train에 미반영. 코드 추적으로 확인. | **확정** |
| **C1 LabelEncoder** | 구조적 취약 | unseen→`__UNKNOWN__` 단일값 붕괴. Top4 importance 피처 영향. OOV율은 서버 확인 필요. | **중간** |
| **C3-C4 토큰** | 부차적 | 짧은 토큰(lb, mq) 오매칭 가능. 카디널리티 증가로 C1과 결합 시 악화. | **낮음** |
| **Calibration cv=3** | 시간 낭비 | 10M행에서 모델 3회 재학습. Decision Combiner는 threshold 비교라 보정 불필요. | **확정** |

### 기여도 추정

```
[전체 하락 −0.073]
  ├── C2 TP 가중치: −0.02 ~ −0.04  (확실)
  ├── C1 Categorical: −0.03 ~ −0.05  (추정, 서버 데이터 의존)
  ├── C3-C4 토큰:    −0.005 ~ −0.01  (부차적)
  └── B9 미포함:      0 (Tier 2에서도 동일 → 회귀 아닌 잠복 버그)
      → 수정 시:     +0.01 ~ +0.03 (순수 추가 개선)
```

---

## Architecture Decisions

### Key Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| C2 완전 제거 (1.0x) → CLI 옵션화 | `class_weight="balanced"` 단독으로 충분. `--tp-weight` CLI로 복원 가능 | — |
| B9를 `build_features()` 결과에 후삽입 | 누수 방지(train-only agg)를 유지하면서 X_train에 반영 | `build_features()` 내부 수정보다 결합도 낮음 |
| Smoothed Target Encoding (K-fold 없음) | unseen→global_mean fallback, smoothing으로 소수 카테고리 보호. K-fold 없이 1회 계산 | train 내 약간의 target leak (smoothing으로 완화) |
| Frequency Encoding 병행 | target-free 보조 신호, unseen→0 자연 처리 | 단독으로는 판별력 부족 |
| Calibration 삭제 | cv=3이 모델을 3회 재학습 → 10M행에서 학습시간 3배. Decision Combiner가 threshold 비교라 보정 불필요 | — |
| Decision Combiner 4분면 추가 | 최종 산출물 성능을 측정하지 않고 있었음. ML 단독 F1만으로는 실제 운영 성능 파악 불가 | 리포트 시트 1개 추가 |
| 서버 토큰 word boundary | 하이픈/언더스코어 경계 regex로 부분 매칭 방지 | 일부 네이밍 패턴에서 누락 가능 |
| Ablation 미실행 | 10M행 서버 학습에 수 시간 소요 → 변경 사항 한 번에 적용, 1회 실행으로 확인 | 개별 변경 효과 분리 불가 |

---

## Implementation Phases

모든 변경 사항을 한 번에 적용하고, 서버에서 **1회 학습**으로 검증한다.

---

### Phase 1: Calibration 삭제 + C2 제거

**Goal**: 학습 시간 단축 + 3중 가중치 과보정 해소
**Status**: Pending

#### 1-A. Calibration 삭제 (완료)

`run_training.py`에서 다음을 제거:
- `--calibrate` argparse 옵션
- Step 8 Calibration 블록 (CalibratedClassifierCV cv=3)
- Step 9에서 calibrator.joblib 저장 블록

`src/models/calibrator.py`, `trainer.py:calibrate_model()` — 라이브러리 코드이므로 유지.

**Step 번호 변경**: 기존 Step 9 → Step 8 (아티팩트 저장)

#### 1-B. C2 TP 가중치 → CLI 옵션화

**파일: `scripts/run_training.py`**

현재 코드 (line 615-624):
```python
_tp_weight_multiplier = 1.5
if _sample_weight is not None and le is not None:
    _y_train_labels = le.inverse_transform(y_train_enc)
    _tp_mask = _np.array(["TP" in str(c) for c in _y_train_labels])
    if _tp_mask.any():
        _sample_weight[_tp_mask] *= _tp_weight_multiplier
        _sample_weight = _sample_weight / _sample_weight.mean()
```

변경 후:
```python
_tp_weight_multiplier = getattr(args, "tp_weight", 1.0)
if _tp_weight_multiplier != 1.0 and _sample_weight is not None and le is not None:
    _y_train_labels = le.inverse_transform(y_train_enc)
    _tp_mask = _np.array(["TP" in str(c) for c in _y_train_labels])
    if _tp_mask.any():
        _sample_weight[_tp_mask] *= _tp_weight_multiplier
        _sample_weight = _sample_weight / _sample_weight.mean()
        print(f"\n[TP Weight] multiplier={_tp_weight_multiplier}, "
              f"TP samples={_tp_mask.sum():,}/{len(_tp_mask):,}")
else:
    print(f"\n[TP Weight] 비활성 (multiplier=1.0)")
```

argparse 추가:
```python
parser.add_argument("--tp-weight", type=float, default=1.0,
                    help="TP 샘플 가중치 배수 (기본 1.0=비활성)")
```

#### Tasks

- [x] 1.1: Calibration 코드 삭제 (run_training.py Step 8 + calibrator.joblib 저장)
- [ ] 1.2: `--tp-weight` argparse 추가 (default=1.0)
- [ ] 1.3: TP 가중치 블록을 `args.tp_weight` 기반으로 변경
- [ ] 1.4: 테스트 추가 — `test_tp_weight_default_disabled`, `test_tp_weight_custom_value`

#### Quality Gate

- [ ] `pytest tests/unit/scripts/test_run_training_label.py -v`
- [ ] `pytest tests/ -v`
- [ ] `ruff check scripts/run_training.py`

---

### Phase 2: B9 수정 — 잠복 피처 누락 해소

**Goal**: `file_event_count`, `file_pii_diversity`를 실제 학습 행렬(X_train/X_test)에 주입
**Status**: Pending

#### 문제 진단

```
run_training.py:446  _keep = [c for c in _KEEP_COLS if c in df.columns]
                     → B9 컬럼 미존재 → 제외

run_training.py:465  build_features(df_for_features) → X_train/X_test 확정
                     pipeline.py:533 `if c in df_train.columns` → B9 없음 → 조용히 제외

run_training.py:486  merge_file_aggregates_label() → df_train에 B9 추가
                     → 하지만 X_train은 이미 확정 → 반영 안 됨
```

#### 수정 방법

`build_features()` 반환 후 B9를 sparse matrix에 직접 hstack:

```python
if _b9_present:
    import scipy.sparse as _sp
    _b9_train = _df_tr[_b9_present].values.astype(_np.float64)
    _b9_test  = _df_te[_b9_present].values.astype(_np.float64)
    result["X_train"] = _sp.hstack([result["X_train"], _sp.csr_matrix(_b9_train)])
    result["X_test"]  = _sp.hstack([result["X_test"],  _sp.csr_matrix(_b9_test)])
    result["feature_names"] = result["feature_names"] + _b9_present
```

#### Tasks

- [ ] 2.1: B9 후삽입 코드 추가 (run_training.py line 478-496 수정)
- [ ] 2.2: feature_builder_snapshot.py — B9 누락 시 NaN fallback 확인
- [ ] 2.3: 테스트 추가 — `test_b9_features_in_feature_matrix`, `test_b9_no_leakage`

#### Quality Gate

- [ ] `pytest tests/unit/scripts/test_run_training_label.py -v`
- [ ] `pytest tests/test_feature_parity.py -v`
- [ ] `pytest tests/ -v`

---

### Phase 3: Categorical Encoding 개선

**Goal**: LabelEncoder → Smoothed Target Encoding + Frequency Encoding
**Status**: Pending

#### 핵심 개념

```
현재 (LabelEncoder):
  service="svcA" → 0      (의미 없는 숫자)
  service="svcB" → 1
  처음 보는 값   → 42     (__UNKNOWN__ 단일 정수 — 정보 완전 소멸)

변경 후 (Target Encoding):
  service="svcA" → 0.72   (이 서비스의 TP 비율 72%)
  service="svcB" → 0.15   (이 서비스의 TP 비율 15%)
  처음 보는 값   → 0.31   (전체 평균 TP 비율 — 정보 보존)

변경 후 (Frequency Encoding):
  service="svcA" → 0.45   (train에서 45% 출현)
  service="svcB" → 0.12   (train에서 12% 출현)
  처음 보는 값   → 0.0    (출현 안 함)
```

**K-fold 사용하지 않음**: train 통계를 1회 계산하고 smoothing(m=100)으로 소수 카테고리 과적합 방지.

#### 변경 사항

**파일: `src/features/pipeline.py`**

현재 LabelEncoder 블록 (line 502-531)을 아래로 교체:

```python
def _target_encode_column(train_series, test_series, y_binary, smoothing=100):
    """Smoothed target encoding. K-fold 없이 1회 계산."""
    global_mean = y_binary.mean()
    stats = pd.DataFrame({"cat": train_series, "y": y_binary})
    agg = stats.groupby("cat")["y"].agg(["mean", "count"])
    smoothed_map = (
        (agg["count"] * agg["mean"] + smoothing * global_mean)
        / (agg["count"] + smoothing)
    ).to_dict()

    train_enc = train_series.map(smoothed_map).fillna(global_mean)
    test_enc  = test_series.map(smoothed_map).fillna(global_mean)
    return train_enc, test_enc, smoothed_map

def _frequency_encode_column(train_series, test_series):
    """빈도 기반 인코딩. Unseen → 0."""
    freq_map = train_series.value_counts(normalize=True).to_dict()
    return train_series.map(freq_map).fillna(0.0), test_series.map(freq_map).fillna(0.0), freq_map
```

각 categorical 컬럼 × 2개 파생 피처 (`_te`, `_freq`):
```
변경 전: 8개 컬럼 × 1 (_enc) = 8개 피처
변경 후: 8개 컬럼 × 2 (_te + _freq) = 16개 피처
```

**파일: `src/models/feature_builder_snapshot.py`**

`_apply_categorical_encoding()` — 저장된 인코더 타입에 따라 분기:
- `type == "target_frequency"` → target_map/freq_map으로 인코딩
- legacy LabelEncoder → 기존 로직 유지 (하위 호환)

#### Tasks

- [ ] 3.1: `_target_encode_column()`, `_frequency_encode_column()` 함수 구현
- [ ] 3.2: pipeline.py categorical 블록 교체
- [ ] 3.3: feature_builder_snapshot.py `_apply_categorical_encoding()` 분기 추가
- [ ] 3.4: 테스트 추가 — `test_target_encoding_unseen_fallback`, `test_frequency_encoding_unseen_zero`
- [ ] 3.5: `test_inference_uses_target_frequency` — 추론 시 _te/_freq 정상 생성

#### Quality Gate

- [ ] `pytest tests/unit/features/test_pipeline_dense.py -v`
- [ ] `pytest tests/test_feature_parity.py -v`
- [ ] `pytest tests/ -v`

---

### Phase 4: 서버 토큰 정밀화

**Goal**: 짧은 토큰(lb, mq, db, /ci/, /cd/)의 오매칭 방지
**Status**: Pending

#### 변경 사항

**파일: `src/features/meta_features.py`**

짧은 토큰(≤3자)에 하이픈/언더스코어/숫자 경계 regex 적용:

```python
def _make_token_pattern(tokens):
    """짧은 토큰에 word boundary를 추가한 regex 패턴 생성."""
    parts = []
    for t in tokens:
        if len(t) <= 3:
            parts.append(rf"(?:^|[-_.\d]){re.escape(t)}(?:[-_.\d]|$)")
        else:
            parts.append(re.escape(t))
    return "|".join(parts)
```

예시:
- `"lb"` → `-lb-`, `-lb01` 매칭 O / `"album"`, `"shelby"` 매칭 X
- `"mq"` → `-mq-`, `mq-broker` 매칭 O / `"albuquerque"` 매칭 X
- `"gateway"` → 4자 이상이므로 기존 substring 매칭 유지

#### Tasks

- [ ] 4.1: `_make_token_pattern()` 구현
- [ ] 4.2: `_ENV_PATTERNS`/`_STACK_PATTERNS`에 적용
- [ ] 4.3: scalar 함수에도 동일 적용
- [ ] 4.4: 테스트 추가 — `test_server_stack_no_false_positive`, `test_server_stack_true_positive`

#### Quality Gate

- [ ] `pytest tests/unit/features/test_meta_features.py -v`
- [ ] `pytest tests/ -v`

---

### Phase 5: Decision Combiner 4분면 평가

**Goal**: RULE+ML 결합 최종 판정의 confusion matrix를 리포트에 추가
**Status**: Pending

#### 배경

현재 리포트(`run_report.py`)는 ML 단독 예측(`model.predict(X_test)`)만 평가한다.
실제 운영에서는 `S3a RULE → S3b ML → S4 Decision Combiner`를 거쳐 최종 판정이 나오는데,
이 **결합된 결과의 성능을 아무도 측정하지 않고 있다.**

#### 구현 방법

**파일: `scripts/run_report.py`**

Step 2 (ML 예측) 이후에 Decision Combiner 시뮬레이션 추가:

```python
# ── Step 2b: Decision Combiner 시뮬레이션 ──
if model is not None and X_test is not None and not df_test.empty:
    from src.models.trainer import predict_with_uncertainty
    from src.models.decision_combiner import combine_decisions

    # ML predictions (uncertainty 포함)
    _pk_events = df_test["pk_event"].tolist() if "pk_event" in df_test.columns else None
    ml_pred_df = predict_with_uncertainty(
        model, X_test, pk_events=_pk_events, label_names=list(le.classes_)
    )

    # RULE 결과 (체크포인트의 df_test에서 추출)
    _rule_cols = ["rule_matched", "rule_primary_class", "rule_id",
                  "rule_confidence_lb", "rule_has_conflict"]

    # 각 행에 combine_decisions() 적용
    dc_predictions = []
    for i in range(len(df_test)):
        rule_row = {c: df_test.iloc[i].get(c, None) for c in _rule_cols}
        rule_row.setdefault("rule_matched", False)
        rule_row.setdefault("rule_confidence_lb", 0.0)
        ml_row = ml_pred_df.iloc[i].to_dict()
        dec = combine_decisions(rule_row, ml_row)
        dc_predictions.append(dec)

    dc_df = pd.DataFrame(dc_predictions)
    # primary_class → binary (TP vs FP)
    dc_pred_binary = np.array([
        "TP" if "TP" in str(c) else "FP"
        for c in dc_df["primary_class"]
    ])
```

#### 4분면 출력

```
              예측: FP        예측: TP
실제 FP    │ TN (정확 FP)  │ FP→TP 오류   │
실제 TP    │ TP→FP 오류    │ TP (정확 TP)  │
```

추가 통계:
- **Case 분포**: Case 0(UNKNOWN) / Case 1(RULE) / Case 2(ML) / Case 3(Fallback) 각 비율
- **Decision Source 분포**: RULE / ML / ML_OVERRIDE / FALLBACK / OOD
- **ML 단독 vs Decision Combiner F1 비교**

#### 리포트 반영

**파일: `src/report/excel_writer.py`**

`PocReportData`에 Decision Combiner 필드 추가:
```python
dc_confusion_matrix: Optional[pd.DataFrame] = None
dc_case_distribution: Optional[dict] = None
dc_f1_comparison: Optional[dict] = None  # {"ml_only": 0.73, "combined": 0.78}
```

9-sheet → 10-sheet Excel: **Sheet 10 "Decision Combiner"** 추가

#### Tasks

- [ ] 5.1: `run_report.py`에 Decision Combiner 시뮬레이션 코드 추가
- [ ] 5.2: 4분면 confusion matrix + Case 분포 + F1 비교 계산
- [ ] 5.3: `PocReportData`에 DC 필드 추가
- [ ] 5.4: `PocExcelWriter`에 Sheet 10 "Decision Combiner" 작성 로직 추가
- [ ] 5.5: 테스트 추가 — `test_dc_simulation_produces_valid_predictions`

#### Quality Gate

- [ ] `pytest tests/unit/report/test_excel_writer.py -v`
- [ ] `pytest tests/ -v`
- [ ] dry-run으로 Excel에 DC 시트 생성 확인

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Target Encoding train 내 약간의 target leak | Medium | Low | smoothing=100으로 소수 카테고리 보호 |
| B9 후삽입이 sparse matrix 형상 불일치 유발 | Low | High | hstack 후 shape 검증 테스트 |
| Categorical 인코딩 변경이 FeatureBuilderSnapshot과 비호환 | Medium | High | legacy LabelEncoder 분기 유지 |
| 서버 토큰 boundary가 실제 서버명 패턴과 불일치 | Low | Low | 서버 데이터 샘플로 매칭률 확인 후 조정 |
| Calibration 제거로 predict_proba 품질 하락 | Low | Low | Decision Combiner는 threshold 비교, 보정 불필요 |
| Phase 3 인코딩 변경으로 체크포인트 무효화 | High | Low | 서버 실행 시 체크포인트 삭제 안내 |

---

## Rollback Strategy

### Phase 1 (C2 제거) 실패 시
- `--tp-weight 1.5`로 복원

### Phase 2 (B9 수정) 실패 시
- B9 후삽입 블록 주석 처리 → 기존과 동일 (B9 미포함)

### Phase 3 (Categorical) 실패 시
- feature_builder_snapshot.py에 legacy LabelEncoder 분기 유지되어 있으므로 pipeline.py만 복원

### Phase 4 (서버 토큰) 실패 시
- `_make_token_pattern()` 제거, 기존 `"|".join(_tokens)` 복원

### Phase 5 (DC 평가) 실패 시
- 리포트 코드만 관련. 학습 파이프라인에 영향 없음. 시트 제거만으로 복원.

---

## Execution Plan

**서버 1회 실행으로 검증:**

```bash
# 체크포인트 삭제 (인코딩 변경으로 무효화)
rm -f models/checkpoints/step5_features_silver_label.pkl
rm -f models/checkpoints/step6_model_silver_label.pkl

# 학습 (Phase 1-4 변경 반영)
nohup python -u scripts/run_training.py \
    --source label --split temporal --test-months 3 \
    2>&1 | tee wave7_all.log &

# 리포트 (Phase 5 DC 평가 포함)
python scripts/run_report.py --source label
```

---

## Progress Tracking

- Phase 1: 50% — Calibration 삭제 완료, C2 CLI 옵션 대기
- Phase 2: 0% — B9 피처 주입
- Phase 3: 0% — Categorical 인코딩 개선
- Phase 4: 0% — 서버 토큰 정밀화
- Phase 5: 0% — Decision Combiner 4분면 평가
- Overall: 10%

---

## Expected Outcome

| 실험 | 적용 내용 | 예상 F1 |
|------|----------|---------|
| Tier 2 (참고) | 기존 (temporal leak 포함) | 0.8063 |
| Wave 6 (현재) | C1-C7 동시 적용 | 0.7335 |
| **Wave 7** | C2 제거 + B9 수정 + Target Encoding + 토큰 정밀화 | **0.80+** |

> Wave 7에서 0.80+을 달성하면 **temporal leak 없이** Tier 2 수준을 달성/초과.

---

## Notes & Learnings

- Calibration(cv=3) 삭제 완료 (2026-03-21) — `run_training.py`에서 Step 8 + calibrator.joblib 저장 제거.
  `src/models/calibrator.py`와 `trainer.py:calibrate_model()`은 라이브러리 코드로 유지.

---

*END OF DOCUMENT*
