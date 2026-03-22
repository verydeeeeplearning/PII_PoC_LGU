## 13. 신뢰도(Confidence) 설계

### 13.1 핵심 원칙: RULE과 ML의 신뢰도는 정의가 다르다

| 구분 | 정의 | 산출 방식 | 의미 |
|------|------|----------|------|
| **RULE confidence** | "이 룰이 과거에 얼마나 맞았는가" (경험적 정밀도 하한) | Beta posterior 95% lower bound | "이 룰은 최소 97%의 확률로 맞다" |
| **ML confidence** | "이 샘플이 해당 클래스일 보정된 확률" | Isotonic/Platt calibration 후 max_proba | "이 샘플이 FP-이메일패턴일 확률이 92%다" |

### 13.2 왜 구분하는가

같은 `0.95`라도:

- RULE confidence 0.95 = "이 룰은 과거 검증에서 95%+ 정확했다" (모집단 수준)
- ML confidence 0.95 = "이 개별 샘플의 예측 확률이 95%다" (개체 수준)

이 차이를 `confidence_type` 컬럼으로 명시하여, 운영/감사 시 해석 혼동을 방지한다.

### 13.3 ML 애매함 지표

ML confidence만으로는 부족하므로, 추가 지표를 함께 출력한다.

| 지표 | 산출 | 해석 | 사용처 |
|------|------|------|--------|
| `margin` | top1_proba - top2_proba | 낮을수록 "1등과 2등이 붙어있다" (불확실) | TP override 조건 |
| `entropy` | -Σ p·log(p) | 높을수록 "여러 클래스에 분산" (불확실) | NEEDS_REVIEW 라우팅 |
| `ml_tp_proba` | TP 클래스의 보정 확률 | 높을수록 "실제 TP일 위험" | RULE 충돌 감지 |

---

## 14. 설명가능성(Explainability) 설계

### 14.1 설명가능성의 세 가지 수준

| 수준 | 출처 | 커버리지 | 비용 | 형태 |
|------|------|---------|------|------|
| **완전한 설명** | RULE | 룰 매칭 영역 전건 | 없음 | 매칭 문자열, 룰 ID, 스니펫 |
| **경량 설명** | ML 수동 피처 | ML 처리 영역 전건 | 거의 없음 | 활성화된 키워드/플래그 목록 |
| **정밀 설명** | ML SHAP | 불확실 케이스만 | 높음 | 피처별 기여도 top-K |

### 14.2 Rationale

**설명가능성을 "데이터로 내보내는" 이유:**

- 모델이 예측한 이유를 "나중에 분석할 수 있는 구조"로 남겨야, 오분류 원인 분석, 룰 승격(ML이 반복적으로 잡는 패턴을 룰로 만들기), 규정/감사 대응이 가능하다.
- prediction_evidence 테이블은 pk_event 기반으로 main 테이블과 조인되므로, "이 예측의 근거가 뭐였는가"를 언제든 추적할 수 있다.

**RULE 설명이 "가장 값싼 확실성"인 이유:**

- 룰 기반 설명은 사실상 자동이다. "이 케이스는 FP-이메일패턴이며, 근거는 `bdp.lguplus.co.kr` 도메인 매칭이다"는 사람이 봐도 납득이 100%다.
- 따라서 룰 커버리지를 최대화하는 것이 설명가능성의 ROI가 가장 높은 전략이다. ML은 룰이 못 잡는 영역만 담당하면 된다.

**TF-IDF n-gram 설명을 "증거 문자열"로 바꿔야 하는 이유:**

- SHAP이 `ngram=@lg`를 보여주면 사람은 이해하기 어렵다. 반드시 원문(마스킹 텍스트)에서 해당 substring을 하이라이트해서 evidence로 보여줘야 한다.
- 예: evidence = `"...****@bdp.lguplus.co.kr..."에서 "lguplus.co.kr" 하이라이트`

---

## 15. 리스크 제어 설계

### 15.1 TP 안전장치 체계

"정탐을 오탐으로 잘못 분류하는 것을 최소화"가 이 시스템의 최우선 운영 원칙이다. 이를 세 가지 수준에서 강제한다.

**수준 1: ML 후처리 — 애매하면 TP**

| 조건 | 동작 | risk_flag |
|------|------|-----------|
| `ml_top1_proba < τ_ml_conf` | FP → TP override | `TP_SAFE_OVERRIDE` |
| `ml_margin < τ_margin` | FP → TP override | `TP_SAFE_OVERRIDE` |
| `ml_entropy > τ_entropy` | FP → TP override | `TP_SAFE_OVERRIDE` |

**수준 2: RULE-ML 충돌 감지**

| 조건 | 동작 | risk_flag |
|------|------|-----------|
| RULE=FP인데 `ml_tp_proba > τ_tp_override` | FP → TP override | `RULE_CONFLICT_WITH_ML` |

**수준 3: OOD 감지 → UNKNOWN → 자동 처리 (v1.2 개선)**

| 조건 | 동작 | risk_flag |
|------|------|-----------|
| OOD 플래그 또는 극도의 불확실 | UnknownAutoProcessor(§9.6)가 자동 처리: 가장 가까운 FP로 배정 또는 보수적 TP 확정 | `OOD_AUTO_ASSIGNED` 또는 `OOD_CONSERVATIVE_TP` |

**수준 4: NEEDS_REVIEW → 자동 판정 (v1.2 추가)**

| 조건 | 동작 | risk_flag |
|------|------|-----------|
| ML confidence < 0.85 | Auto-Adjudicator(§9.5) 4단 자동 판정: 파일 합의 → 교차 합의 → 과거 패턴 → 보수적 TP | `AUTO_ADJUDICATED` 또는 `AUTO_TP_CONSERVATIVE` |

**수준 5: Fallback — 분류 불가 시 TP**

| 조건 | 동작 | risk_flag |
|------|------|-----------|
| RULE 미매칭 + ML 미실행/실패 | TP 처리 | `TP_SAFE_OVERRIDE` |

### 15.2 Rationale

**왜 모델 밖(후처리)에서 강제하는가:**

- ML 모델은 loss function을 최적화하여 최적의 확률을 내는 것이 역할이다. "애매하면 TP"는 비즈니스 정책이지 모델의 목적함수가 아니다.
- 정책을 모델 안에 넣으면(예: class weight 극단 조정) 모델 성능 자체가 왜곡된다. 정책은 후처리에서 명시적으로 구현해야 모델 성능과 운영 정책을 독립적으로 관리할 수 있다.
- 후처리의 임계값(τ_ml_conf, τ_margin 등)은 운영 환경에서 조정 가능하므로, 모델 재학습 없이도 리스크 수준을 제어할 수 있다.

**v1.2: Zero-Human-in-the-Loop 리스크 제어 전략:**

- v1.2에서는 사람 검토를 시스템의 보수적 자동 판정으로 대체한다. 핵심 안전장치는 **"모르면 TP"** 보수적 기본값이다.
- Auto-Adjudicator(§9.5)는 4단 자동 판정으로 NEEDS_REVIEW를 처리하며, 모든 판정이 실패하면 TP로 확정한다. 이는 사람 검토보다 일관적이고 빠르다.
- UnknownAutoProcessor(§9.6)는 UNKNOWN을 가장 가까운 FP로 배정하거나 TP로 확정한다. 개인정보 보호 관점에서 TP 기본값은 안전하다.
- Auto-Tuner(§9.7)가 임계값을 자동으로 최적화하되, TP Recall 하락은 절대 허용하지 않는 안전장치가 내장되어 있다.
- Auto-Remediation Playbook(§11.5)이 모든 KPI 알람에 대한 자동 조치를 사전에 매핑하여, 알람 발생 시 사람 개입 없이 즉시 대응한다.

---

## 16. 인프라 & 개발 환경

### 16.1 개발 서버 사양

| 구분 | 사양 |
|------|------|
| CPU | Intel Xeon E5-2620 v4 @ 2.10GHz × 2 (총 32 Threads) |
| Memory | 128 GB DDR4 |
| Disk | 3.637 TB RAID 1 |
| GPU | **미보유 (확정)** |

### 16.2 폐쇄망 환경 제약 & 대응

| 제약 | 대응 |
|------|------|
| pip/conda 실시간 다운로드 불가 | `pip download -r requirements.txt -d ./offline_packages/`로 사전 준비 |
| HuggingFace/PyTorch Hub 불가 | Transformer 사용 안 함 (CPU 기반 확정) |
| 실시간 업데이트 불가 | 패키지 버전 고정, Docker 이미지 활용 (선택) |

### 16.3 필수 패키지

```
# Core ML
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
shap>=0.43

# Data Processing  
pandas>=2.0
numpy>=1.24
scipy>=1.11
pyarrow>=14.0  # Parquet 읽기/쓰기

# Text Processing
# (scikit-learn의 TfidfVectorizer 사용, 추가 NLP 패키지 불필요)

# Utilities
pyyaml>=6.0
joblib>=1.3
tqdm>=4.65
matplotlib>=3.7
seaborn>=0.12

```

---

## 17. 데이터 레이어 & 중간 산출물

### 17.1 전체 중간 산출물 목록

| Stage | 레이어 | 산출물 | 형식 | 행 단위 | 주요 컬럼 |
|-------|--------|--------|------|---------|----------|
| S0 | Bronze | 원본 파일 (`bronze_events`, `bronze_labels`) | CSV/Excel→Parquet | 원본 그대로 | 컬럼명 통일, 인코딩/줄바꿈 정리 |
| S0 | Bronze | `schema_registry.yaml` (v1.1) | YAML | — | 입력 스키마 정의 + 누락 시 격리 정책 |
| S1 | Silver-S1 | `silver_detections.parquet` | Parquet | 1검출 이벤트 | pk_event, full_context_raw, full_context_norm, local_context_raw, masked_hit, pii_type_inferred, parse_status |
| S1 | Silver-S1 | `silver_quarantine.parquet` (v1.1) | Parquet | 격리 행 | quarantine_reason, raw_content |
| S2 | Silver-S2 | `silver_features_base.parquet` | Parquet | 1검출 이벤트 | S1 + raw_text + shape_text + path_text + 수동 피처 + 경로 피처 + tabular + placeholder 비율 |
| S2 | Silver-S2 | `silver_file_agg.parquet` (선택) | Parquet | 1파일 | pk_file + 집계 통계 |
| S2 | Silver-S2 | `silver_features_enriched.parquet` (선택) | Parquet | 1검출 이벤트 | silver_features_base + file_agg join |
| S3a | — | `rule_labels.parquet` | Parquet | 1검출 이벤트 | pk_event + 룰 라벨 + 신뢰도 |
| S3a | — | `rule_evidence.parquet` | Parquet | N:1 (long) | pk_event + 증거 상세 |
| S3b | — | `ml_predictions.parquet` | Parquet | 1검출 이벤트 | pk_event + 확률 + 애매함 지표 |
| S3b | — | `ml_evidence.parquet` | Parquet | N:1 (long) | pk_event + ML 증거 |
| S4+S5 | — | `predictions_main.parquet` | Parquet | 1검출 이벤트 | 최종 라벨 + 신뢰도 + risk_flag + ood_flag |
| S4+S5 | — | `prediction_evidence.parquet` | Parquet | N:1 (long) | 통합 증거 (RULE+ML) |
| S6 | — | `monthly_metrics.json` (v1.1) | JSON | — | 12종 KPI + 알람 결과 |

### 17.2 Rationale

각 산출물이 독립 파일로 존재하면:

- **장애 격리:** S3b(ML)가 실패해도 S1/S2/S3a의 결과는 유효하다. ML만 재실행하면 된다.
- **실험 유연성:** 동일 S2 피처에 다른 모델(XGBoost vs LightGBM)을 적용하여 비교할 수 있다.
- **디버깅:** "이 케이스가 왜 이렇게 분류됐는지" 각 단계의 산출물을 순서대로 확인하면 원인을 추적할 수 있다.
- **재현성:** 동일 입력(S0) → 동일 파서(S1) → 동일 피처(S2) → 동일 모델(S3b) → 동일 결과(S5)를 보장한다.

---

## 18. Training Pipeline vs Inference Pipeline

### 18.1 분리 이유

Training과 Inference를 분리하는 것은 ML 운영의 기본이지만, 이 프로젝트에서는 특히 중요한 이유가 있다.

- **룰 신뢰도(precision_lb)**는 Training 시점에 검증셋 기반으로 산출하여 저장한다. Inference 시점에서 즉석으로 만들 수 없다.
- **확률 보정기(calibrator)**도 Training 시점에 별도 calibration set으로 학습한다.
- **TF-IDF 벡터라이저/SVD 등 변환기**는 Training 시점에 fit하여 저장하고, Inference에서는 transform만 수행한다.

### 18.2 재현성 보장 (v1.1 추가)

```yaml
# config/reproducibility.yaml
random_seed: 42
numpy_seed: 42
sklearn_seed: 42

# 결정론적 실행 모드 (감사/재현용)
deterministic_mode: true  # true면 n_jobs=1 강제
```

```python
import random
import numpy as np

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # XGBoost/LightGBM은 파라미터로 전달

# 모델 파라미터에 seed 추가
params = {
    # ... 기존 파라미터 ...
    'random_state': config['random_seed'],
    'seed': config['random_seed'],      # LightGBM
    'n_jobs': 1 if config['deterministic_mode'] else -1,
}
```

**Rationale:** 랜덤 시드가 명시되지 않고 `n_jobs=-1`로 비결정론적 실행하면 동일 데이터에서도 다른 결과가 나올 수 있다. 감사/재현 목적에서는 결정론적 모드가 필수다.

### 18.3 Training Pipeline 산출물

모든 표준 아티팩트는 `models/final/`에 저장된다.
`_run_label_mode()` (Phase 1) 의 Step 9에서 학습 완료 후 자동 생성된다.

#### 표준 아티팩트 세트 (models/final/)

| 파일 | 형식 | 내용 | 상태 |
|------|------|------|:----:|
| `best_model_v1.joblib` | joblib dict | 학습된 모델 + label_encoder + f1_macro + 메타정보 | ✅ 구현 |
| `label_encoder.joblib` | joblib | LabelEncoder (클래스 인코더 독립 저장) | ✅ 구현 |
| `feature_builder.joblib` | joblib | `FeatureBuilderSnapshot` — 모든 TF-IDF 벡터라이저 + dense 컬럼 목록 통합. `.transform(df)` 인터페이스 제공 | ✅ 구현 |
| `ood_detector.joblib` | joblib | `OODDetector` (Mahalanobis, dense 피처 기반) | ✅ 구현 |
| `calibrator.joblib` | joblib dict | Calibration 성공 시 생성. calibrated_model + label_encoder | ✅ 구현 (조건부) |
| `feature_schema.json` | JSON | `{n_features, n_tfidf_features, n_dense_features, tfidf_views, dense_columns, saved_at}` | ✅ 구현 |

#### feature_builder.joblib — FeatureBuilderSnapshot

`FeatureBuilderSnapshot` (`src/models/feature_builder_snapshot.py`)은 `build_features()`가 반환하는 fitted TF-IDF 벡터라이저 딕셔너리와 dense 피처 컬럼 목록을 단일 객체로 캡슐화한다.

```python
# 주요 인터페이스
snapshot = FeatureBuilderSnapshot.from_build_result(build_features_result)
snapshot.save("models/final/feature_builder.joblib")

# 추론 시
builder = FeatureBuilderSnapshot.load("models/final/feature_builder.joblib")
X = builder.transform(df_silver)  # df에 file_name, file_path, dense 컬럼 필요
```

**TF-IDF view → DataFrame 컬럼 매핑:**

| view | 입력 컬럼 | 변환 | Phase |
|------|----------|------|-------|
| `phase1_fname` | `file_name` | char_wb(2~5 ngram) | Phase 1 |
| `phase1_path` | `file_path` | `_to_path_text()` 후 word(1~2 ngram) | Phase 1 |
| `raw` | `full_context_raw` | 직접 적용 | Phase 2 |
| `shape` | `full_context_raw` | `_to_shape_text()` 후 적용 | Phase 2 |
| `path` | `file_path` | `_to_path_text()` 후 word(1~2 ngram) | Phase 2 |

**dense_columns:** `feature_names`에서 `tfidf_` 접두사가 없는 항목 — `path_depth`, `file_is_log`, `server_freq` 등.

#### 참고: 이전 설계 대비 변경

| 이전 (설계 문서 v1.1) | 현재 (구현 완료) | 비고 |
|----------------------|----------------|------|
| `model.joblib` | `best_model_v1.joblib` | 메타정보 포함 dict 형식 |
| `tfidf_raw.joblib` 등 개별 파일 4개 | `feature_builder.joblib` 통합 | `FeatureBuilderSnapshot`으로 통일 |
| `label_encoders.joblib` | `label_encoder.joblib` | 단수형, 독립 파일 |
| `rules.yaml`, `rule_stats.json` | `config/`에서 직접 참조 | 학습 산출물 아님 (config 파일) |

### 18.4 Inference Pipeline 흐름

```
[월별 신규 데이터 (Dataset A)]
        │
        ▼
[Load: 모든 Training 산출물 로드 (best_model_v1, feature_builder, calibrator, ood_detector, feature_schema)]
        │
        ▼
[S1: Parse (3단 폴백)] → silver_detections.parquet + silver_quarantine.parquet
        │
        ▼
[S2: Feature Prep] → silver_features_base.parquet
        │
        ▼
[Feature Schema 검증 (v1.1)] → 차원/순서 일치 확인
        │
        ├─── [S3a: RULE Label] → rule_labels + rule_evidence
        │
        ├─── [S3b: ML Predict + OOD Score] → ml_predictions + ml_evidence
        │
        ▼
[S4: Combine (Case 0: UNKNOWN)] + [S5: Write] → predictions_main + prediction_evidence
        │
        ▼
[S6: Monitor] → monthly_metrics.json (12종 KPI)
        │
        ▼
[v1.2: Autonomous Feedback Loop]
        │
        ├─ KPI 정상 → 다음 월 대기
        │
        └─ KPI 이상 → Auto-Remediation Playbook (§11.5)
             │
             ├─ 파싱 문제 → Auto-Schema-Detector (§4.4)
             ├─ 분포 이동 → Auto-Retrainer 트리거
             ├─ 품질 저하 → Auto-Tuner (§9.7)
             ├─ 신규 패턴 → Auto-Rule-Promoter (§7.6)
             └─ UNKNOWN 축적 → Auto-Taxonomy-Manager (§12.3b)
                  │
                  ▼
             [자동 재학습 + 자동 배포]
```

### 18.5 Auto-Retrainer — 자동 재학습 루프 (v1.2 추가)

원칙 G(Zero-Human-in-the-Loop)에 따라, 모델 재학습을 사람 판단 없이 자동으로 트리거하고 실행한다.

```python
class AutoRetrainer:
    """
    KPI 알람 또는 정기 일정에 의해 모델 재학습을 자동 실행.
    안전장치: 재학습 후 성능 하락 시 자동 롤백.
    """
    RETRAIN_TRIGGERS = [
        'oov_rate_raw > 0.30',
        'confidence_p10 < 0.40',
        'review_rate > 0.35 (2개월 연속)',
        'ood_rate > 0.05',
        'auto_fp_precision_est < 0.90',
        'temporal_consistency < 0.75',
    ]

    def should_retrain(self, metrics_history: list) -> bool:
        """최근 KPI 이력을 분석하여 재학습 필요 여부 판단."""
        latest = metrics_history[-1]
        for trigger in self.RETRAIN_TRIGGERS:
            if self._evaluate_trigger(trigger, metrics_history):
                return True
        return False

    def retrain_and_validate(self, new_data, current_model_path,
                             model_pipeline) -> dict:
        """
        재학습 실행 + 검증 + 롤백 판단.
        """
        # 1. 현재 모델의 baseline 성능
        current_scores = model_pipeline.evaluate(current_model_path)

        # 2. 새 데이터를 포함하여 재학습
        new_model = model_pipeline.train(new_data)
        new_scores = model_pipeline.evaluate(new_model)

        # 3. 안전 제약 확인
        if new_scores['tp_recall'] < current_scores['tp_recall']:
            # TP Recall 하락 → 즉시 롤백
            return {
                'action': 'rollback',
                'reason': f"tp_recall: {new_scores['tp_recall']:.4f} < "
                          f"{current_scores['tp_recall']:.4f}",
                'model_path': current_model_path,
            }

        if new_scores['macro_f1'] < current_scores['macro_f1'] - 0.02:
            # macro_f1이 2%p 이상 하락 → 롤백
            return {
                'action': 'rollback',
                'reason': f"macro_f1 drop: {new_scores['macro_f1']:.4f}",
                'model_path': current_model_path,
            }

        # 4. 새 모델 채택
        new_model_path = model_pipeline.save(new_model)
        return {
            'action': 'adopt',
            'old_scores': current_scores,
            'new_scores': new_scores,
            'model_path': new_model_path,
        }
```

**자동 재학습 흐름:**

```
[KPI 알람 또는 정기 일정]
        │
        ▼
[재학습 트리거 판단]
        │
        ▼
[새 데이터 포함 재학습]
        │
        ▼
[검증: TP Recall 하락? macro_f1 하락?]
        │
        ├─ 하락 → 자동 롤백 (현재 모델 유지)
        │
        └─ 유지/상승 → 새 모델 자동 채택
             │
             ▼
        [TAU/Rule/Schema도 함께 갱신]
```

---

## 19. 평가 전략

### 19.1 평가 지표

**Phase 1 운영 지표 (Selective Classification 기준):**

| 지표 | 정의 | 목표 | 우선순위 |
|------|------|------|---------|
| **Auto-FP Precision** | 자동 FP 처리 건 중 실제 FP 비율 | ≥ 0.95 | 최우선 (비타협) |
| **TP Safety Rate** | 실제 TP 중 TP/REVIEW로 처리된 비율 | ≥ 0.99 | 최우선 |
| **Auto-FP Coverage** | 자동 FP 처리 건수 / 전체 FP 건수 | ≥ 40% | Phase 1 운영 목표 |

**Phase 2 목표 지표 (JOIN 후 전체 7-class(6 FP+1 TP) 기준):**

| 지표 | 대상 | 목표 | 이유 |
|------|------|------|------|
| **TP Recall** | 정탐(TP) 클래스 | ≥ 0.95 | 실제 개인정보 누락 방지 (최우선) |
| **Macro F1** | 전체 7클래스 | ≥ 0.85 | 클래스 간 균형 잡힌 성능 |
| **FP Precision** | 각 FP 클래스 | ≥ 0.90 | 오탐 자동 처리의 정확성 |
| **Auto-FP Coverage** | 전체 FP 대비 자동 처리 비율 | ≥ 60% | Phase 2 목표 |
| **공수 절감률** | 운영 | ≥ 50% | 비즈니스 성공 기준 |

> 현재 코드(`config/feature_config.yaml`)의 PoC 게이트는 `Macro F1 >= 0.70`, `TP Recall >= 0.75`, `FP Precision >= 0.85`이며, 위 표는 운영 목표치(상향 목표)로 관리한다.

**Coverage-Precision 트레이드오프 곡선:**

임계값(τ)을 움직이면서 Auto-FP Coverage와 Precision의 트레이드오프를 시각화한다. 이 곡선이 "Phase 1이 어디까지 가능한가"에 대한 가장 정직한 답이다.

| Precision 0.95 at Coverage | 판정 |
|---------------------------|------|
| ≥ 40% | Phase 1 단독으로 운영 가치 충분 |
| 20~40% | 부분적 가치, JOIN으로 확장 필요 |
| < 20% | 운영 임팩트 불충분, JOIN 우선 |

### 19.2 평가 분할 전략 (3종 필수, v1.1 기본값 + 2026-02-26 확장)

**v1.1 변경: Group+Time Split을 기본값으로 확정.** Event 랜덤 split은 공식 지표에서 제외한다.

**2026-02-26 추가: 3종 Split 필수 적용 (월별 데이터 10개월 확보로 Time Split 의미 있게 동작)**

```yaml
# config/evaluation_config.yaml
split_strategy:
  primary: "group_time"         # 기본값: pk_file Group + Time Split
  group_column: "pk_file"       # Group 기준 (파일 수준 누수 차단)
  time_column: "file_created_at"
  test_months: 2                # 11~12월을 테스트 (1~10월 학습)

  secondary: "label_work_month_time"  # 작업 시점 기준 Time Split
  # → 레이블 작업월 기준 분포 변화 감지

  tertiary: "leave_org_out"     # Leave-Organization-Out
  # → CTO로 학습 → NW로 테스트 등, 조직 간 일반화 능력 검증

  # event 랜덤 split은 공식 지표에서 제외
  # 디버깅/빠른 실험 목적으로만 --split=random 옵션 제공

# 2026-03-20 구현 완료:
# run_training.py --split group|temporal|server --test-months N
# build_features(split_strategy="group"|"temporal"|"server", test_months=3)

# Wave 3 개정 (2026-03-20):
# work_month_time_split()이 pk_file별 최대 월 기준 파일 단위 분할에서
# **행 단위 엄격 월 분할**로 변경됨.
# - 이전: pk_file의 max_month ∈ test_months이면 해당 파일의 모든 행이 test로 이동
#   → 5월, 6월 등 이전 월 행도 test에 포함되어 시간적 분리 깨짐
# - 변경: 각 행의 month가 test_months에 해당하면 test, 아니면 train
#   → 3~9월 train / 10~12월 test 엄격 분리
# - 동일 pk_file이 train/test 양쪽에 나타날 수 있으나 시간적 분리가 우선
#
# Step 6b (Temporal split 보조 평가) 삭제:
# temporal split 적용 시 Step 6 자체가 temporal 평가이므로
# 별도 Step 6b(test 내 최근 2개월만 재평가)는 불필요
```

| 분할 방식 | 설명 | 장점 | 단점 | 용도 |
|-----------|------|------|------|------|
| Random Split | 랜덤 8:1:1 | 간단 | ❌ 데이터 누수 위험 | ⚠️ 디버깅 전용, 공식 지표 제외 |
| **file_created_at Time + Group** | file_created_at 기준 + pk_file Group | ✅ 가장 현실적 운영 시나리오 | 데이터 충분해야 함 | ✅ **Primary (기본값)** |
| **label_work_month Time + Group** | 작업월 기준 + pk_file Group | ✅ 작업 시점 기준 분포 변화 감지 | file_created_at과 시점 불일치 가능 | Secondary |
| **Leave-Organization-Out** | 특정 조직 전체를 테스트 | ✅ 조직 간 일반화 검증 | 조직 간 분포 차이로 평가 분산 큼 | Tertiary |
| **Server Group** | server_name 기준 분리 | ✅ 가장 보수적 | 서버 간 분포 차이 | 참고용 |

**3종 Split 해석 기준:**

```
3종 결과 간 괴리 < 10%p → 일반화 능력 확보
3종 결과 간 괴리 > 10%p → 시점/조직 간 라벨 기준 불일치 신호
Primary vs Random에서 급락 → 암기(memorization) 모델 경고
Leave-Org-Out에서 성능 급락 → 조직별 모델 분리 검토
```

**월별 수량 편차 대응:**

월별 데이터 수량이 천차만별이므로, Time Split 시 단순 월 경계가 아니라 건수 기준 균형도 함께 고려한다. 학습 시 월별 샘플링 가중치 또는 stratified sampling을 검토한다.

**Rationale:**

- 경로/파일명은 매우 강력한 신호이므로, 동일 파일이 train/test에 같이 있으면 성능이 과대평가(암기)된다. pk_file 단위로 분리해야 "새로운 파일에서도 잘 되는가"를 검증할 수 있다.
- 운영에서 중요한 건 "새로운 월의 데이터에서도 잘 되는가"이므로, Time Split이 진짜 성능을 보여준다.
- `file_created_at`과 `label_work_month`가 다를 수 있으므로(6월 작업분에 4~5월 파일 포함), 두 기준 모두 측정해야 편향을 감지할 수 있다.
- Leave-Organization-Out은 "단일 모델로 전 조직 커버 가능한가"를 직접 검증한다. 차이 > 20%p이면 조직별 모델 분리 또는 조직별 룰셋 차별화를 검토한다.

### 19.3 Calibration 평가

확률 보정의 품질은 **reliability diagram**과 **Expected Calibration Error (ECE)**로 평가한다.

| 지표 | 의미 | 목표 |
|------|------|------|
| ECE (Expected Calibration Error) | 예측 확률과 실제 정답률의 괴리 | ≤ 0.05 |
| MCE (Maximum Calibration Error) | 가장 큰 괴리 | ≤ 0.10 |

---

## 20. 구현 로드맵

### 20.0 Phase 0/1/2 단계 구조 (2026-02-26 전략 전환 반영)

2-Phase Staged Deployment 전략에 따라 Step A~E 이전에 Phase 0 → Phase 1 → Phase 2 순으로 진행한다.

**Phase 0: 데이터 품질 검증 + Go/No-Go 판정**

| 작업 | 산출물 | Go/No-Go 기준 |
|------|--------|--------------|
| 전 조직 × 전 월(72+개) 합산 + 한글→영문 + PK dedup | 클린 데이터셋 + 월별/조직별 건수 분포 | — |
| 정탐/오탐 파일 교차 검증 + 라벨 충돌률 + Bayes Error 하한 | go_no_go_report.md | 충돌률 > 10% → 라벨 정제 선행 |
| fp_description unique 목록 + 빈도 + 조직 간 일관성 | fp_description_unique_list.csv | 클래스 후보 < 50건 경고 |
| 정탐 레이블 품질 확인 | 품질 판정 결과 | TP 라벨 노이즈 추정 |

Phase 0 Go/No-Go 게이트:

| 신호 | 기준 | 판정 |
|------|------|------|
| pk_event 라벨 충돌률 | > 10% | 라벨 품질 이슈 우선 해결 |
| Bayes Error 하한 | > 20% | 레이블 단독 운영 목표 미달 가능성 |
| 조직 간 FP 비율 차이 (동일 패턴) | > 20%p | 조직별 모델 분리 검토 |
| Dedup 후 유효 샘플 수 | < 10,000 | 학습 안정성 우려 |
| 월별 건수 편차 (최다/최소) | > 20배 | 시간 편향 대응 필수 |

**Phase 1: 레이블 단독 Selective Classification (즉시 실행)**

- 목표: 고확신 FP 자동 처리 (Precision ≥ 0.95), Auto-FP Coverage ≥ 40%
- 피처: ~35개 dense (경로/파일명/서버/검출통계/시간, 텍스트 없음)
- 클래스: 축소 5-class (TP, FP-시스템로그, FP-라이선스/저작권, FP-개발/테스트, FP-기타)
- 아키텍처 컴포넌트: S3a(경로/파일명 기반 룰) + S3b(dense 피처 ML) + S4 + S5 + S6

**Phase 2: JOIN 성공 시 확장 (Phase 1과 병행 추진)**

- 목표: Auto-FP Coverage ≥ 60%, 전체 7-class 세분화
- 피처: Phase 1 dense + TF-IDF 텍스트 (~1,200개 추가)
- 조건: Sumologic 재추출(JOIN 키 컬럼 포함) 성공, 5-key JOIN 매핑률 확인
- Phase 1 컴포넌트 전체 재사용 (코드 변경 없이 피처만 확장)

### 20.1 추천 구축 순서

각 단계가 독립 산출물을 남기므로, 순차적으로 진행하되 이전 단계 완료 후 다음 단계로 진행한다.

**Step A (최우선): S1 + Quarantine + Schema Registry**

| 항목 | 내용 |
|------|------|
| 기존 | 1행=1검출 이벤트 정규화, PK 안정화 |
| **v1.1 추가** | 파서 3단 폴백, Quarantine 테이블, Schema Registry, parse_status KPI |
| 전제 | Dataset A/B 샘플 수령 |
| 산출물 | `silver_detections.parquet`, `silver_quarantine.parquet`, S1 파서 코드 |
| 중요성 | **이 단계가 성공하면 이후는 다 "반복 개선"이 가능해짐** |
| 검증 | row_parse_success_rate ≥ 0.98, quarantine_count 정상, PK 유일성, pii_type_inferred 분포 |

**Step B: S2 + RULE + 합성변수 정책 확립**

| 항목 | 내용 |
|------|------|
| 기존 | 피처 엔지니어링 완성, "설명이 완전한 라벨" 확보 |
| **v1.1 추가** | 합성변수 3-Tier 정책 적용, TF-IDF 파라미터 수정, token_pattern 개선 |
| **v1.1 추가** | 인코더 Unknown 안전 설정, feature_schema.json 생성 |
| 산출물 | 피처 파일, 룰셋(rules.yaml), 룰 evidence, feature_schema.json |
| 중요성 | 룰 커버리지가 높을수록 ML 부담이 줄고, 설명가능성이 높아짐 |
| 검증 | Tier 0 baseline 성능 확보, Tier 1 ablation 결과, 룰 매칭률, 룰 정밀도 |

**Step C: S3b + S4 + 평가 체계 확립**

| 항목 | 내용 |
|------|------|
| 기존 | ML 모델 학습 + calibration + TP 안전장치 결합 |
| **v1.1 추가** | OOD Score 추가, Reject Option(UNKNOWN) 도입, TAU_TP_OVERRIDE 조건부 개선 |
| **v1.1 추가** | 대안 모델 비교 실험, Group+Time Split 기본값 확정 |
| **v1.1 추가** | file-level agg 누수 차단, 재현성 보장(시드 관리) |
| 산출물 | 학습된 모델, 보정기, ood_detector, predictions_main/evidence |
| 중요성 | 룰로 못 잡는 영역을 ML이 커버, 전체 성능 달성 |
| 검증 | 3가지 split에서의 성능 비교, F1, TP Recall, OOD 감지율, risk_flag 비율 |

**Step D: S6 + 운영 체계**

| 항목 | 내용 |
|------|------|
| 기존 | SHAP 정밀 evidence |
| **v1.1 추가** | 12종 KPI 자동 산출, 알람 임계값 설정 |
| **v1.1 추가** | 라벨 거버넌스 규칙, UNKNOWN 운영 정책 |
| **v1.1 추가** | monthly_metrics.json + 임계값 시뮬레이션 프레임워크 |
| 산출물 | SHAP 기반 evidence, monthly_metrics.json |
| 중요성 | 비용 대비 효과 최적화 + 운영 안정성 확보 |
| 검증 | 1개월 운영 시뮬레이션, KPI 알람 동작 확인 |

**Step E: Zero-Human-in-the-Loop 자동화 (v1.2 추가)**

| 항목 | 내용 |
|------|------|
| **Tier 1 (PoC 내 구현)** | Auto-Adjudicator(§9.5), UNKNOWN 자동 처리(§9.6), Auto-Model-Selector(§8.13), Automated Ablation(§6.5), Confident Learning(§22) |
| **Tier 2 (PoC 이후 우선)** | Auto-Tuner(§9.7), Auto-Precision-Estimator(§11.4), Self-Validation Loop(§11.3), Auto-Remediation Playbook(§11.5) |
| **Tier 3 (운영 안정화 후)** | Auto-Rule-Promoter(§7.6), Auto-Mapper(§12.3a), Auto-Taxonomy-Manager(§12.3b), Auto-Schema-Detector(§4.4), Auto-Retrainer(§18.5) |
| 산출물 | 13개 자동화 모듈, autonomous feedback loop |
| 중요성 | 사람 역할을 Dashboard 감시 수준으로 축소 → 운영 확장성 확보 |
| 검증 | 3개월 자율 운영 시뮬레이션, 사람 개입 없이 KPI 유지 확인 |

### 20.2 PoC 범위 요약

```
┌─────────────────────────────────────────────────┐
│   Phase 0 (즉시)                                │
│   데이터 품질 검증 + Go/No-Go 판정              │
│                                                 │
│   Phase 1 PoC (레이블 단독)                     │
│   Step A~D + Step E Tier 1 (경로/메타 피처)     │
│                                                 │
│   Phase 2 PoC (JOIN 후)                         │
│   Phase 1 확장 + 텍스트 피처 추가               │
│                                                 │
│   운영 확장 (PoC 이후)                          │
│   Step E Tier 2~3                               │
└─────────────────────────────────────────────────┘

Phase 1 완료 기준:
Auto-FP Precision ≥ 0.95 (비타협)
Auto-FP Coverage ≥ 40%
TP Safety Rate ≥ 0.99
3종 Split 간 괴리 < 10%p

Phase 2 완료 기준 (JOIN 성공 시):
Auto-FP Coverage ≥ 60% (Phase 1 대비 +20%p)
Macro F1 ≥ 0.85 (7-class)
TP Recall ≥ 0.95
50%+ 공수 절감

v1.2 추가 기준:
Zero-Human-in-the-Loop Tier 1 구현 완료
자동 판정 정확도 ≥ 사람 판정 정확도
```

---
## 21. 후속 과제 & 미확정 사항

### 21.1 Action Items

**Phase 0 (즉시 실행 가능):**

| # | 항목 | 담당 | 상태 |
|---|------|------|------|
| 1 | 레이블 Excel 72+개 파일 → `data/raw/label/` 구조 배치 | 고객 | 🔲 대기 |
| 2 | `run_pipeline.py --mode label-only --dry-run` → silver_label.parquet 생성 (전처리만) | 딜로이트 | 🔲 데이터 배치 후 즉시 |
| 3 | `run_phase0_validation.py` → Go/No-Go 판정 + fp_description 목록 산출 | 딜로이트 | 🔲 2번 완료 후 |
| 4 | fp_description_unique_list.csv 검토 → `config/label_mapping.yaml` 수동 작성 | 딜로이트+고객 | 🔲 3번 완료 후 |

**Phase 1 (실 데이터 투입 후):**

| # | 항목 | 담당 | 상태 |
|---|------|------|------|
| 5 | Rule Labeler 경로/파일명 기반 룰 설계 + `rules.yaml` 작성 | 딜로이트 | 🔲 Go/No-Go 통과 후 |
| 6 | 메타데이터 ML (~35 dense 피처) 학습 + 3종 Split 평가 | 딜로이트 | 🔲 5번 완료 후 |
| 7 | Decision Combiner + 보수적 후처리 구현 | 딜로이트 | 🔲 6번 완료 후 |
| 8 | coverage-precision 곡선 → TAU_HIGH 결정 | 딜로이트 | 🔲 7번 완료 후 |

**Phase 2 (Phase 1과 병행 준비):**

| # | 항목 | 담당 | 상태 |
|---|------|------|------|
| 9 | Sumologic 재추출 요청 (dfile_filedirectedpath, dfile_filename, dfile_filecreatedtime 포함) | 고객 | 🔲 즉시 요청 권장 |
| 10 | 5-key JOIN 시도 + 매핑률 확인 | 딜로이트 | 🔲 재추출 완료 후 |
| 11 | 텍스트 피처 추가 + Phase 1 → Phase 2 확장 | 딜로이트 | 🔲 JOIN 성공 후 |

**완료 항목:**

| # | 항목 | 상태 |
|---|------|------|
| C1 | 레이블 데이터 EDA 및 구조 분석 (`docs/eda_insight_report_v2.md`) | ✅ 완료 (2026-02-26) |
| C2 | 전략 리포트 작성 (`docs/fp_improvement_strategy_report.md`) | ✅ 완료 (2026-02-26) |
| C3 | 컬럼 정규화 (`config/column_name_mapping.yaml`, `src/data/column_normalizer.py`) | ✅ 완료 |
| C4 | 레이블 로더 (`config/ingestion_config.yaml`, `src/data/label_loader.py`) | ✅ 완료 |
| C5 | Phase 0 검증 파이프라인 (`scripts/run_phase0_validation.py`, `src/evaluation/data_quality.py`) | ✅ 완료 |
| C6 | 폐쇄망 패키지 환경 검증 | ✅ 완료 |
| C7 | GPU 미보유 확정 → CPU 기반 Boosting 모델 확정 | ✅ 완료 |
| **C8** | **fp_description 분류 완료** (`scripts/classify_fp_description.py`) — 611개 unique값 → 6-class + UNKNOWN 매핑, `fp_description_mapping.csv` 생성 | ✅ **완료 (2026-03)** |
| **C9** | **실제 클래스 체계 확정** — FP-파일없음/FP-이메일패턴/FP-숫자패턴/FP-라이브러리/FP-더미테스트/FP-시스템로그 6-class (§12.1 반영) | ✅ **완료 (2026-03)** |
| **C10** | **원본형 Mock 데이터 생성** (`scripts/generate_mock_raw_data.py`) — 레이블 Excel(조직×월×TP/FP) + Sumologic Excel(.xlsx) 원본 포맷 생성 (`--csv` 옵션으로 CSV 병행 출력), 한글 컬럼명 variant 재현 포함 | ✅ **완료 (2026-03)** |
| **C11** | **ML 파이프라인 E2E 검증용 더미 데이터** (`scripts/generate_dummy_data.py`) — 7-class 클래스당 200건 생성, `data/processed/merged_cleaned.csv` 출력 | ✅ **완료 (2026-03)** |

### 21.2 미확정 사항

| # | 항목 | 현재 상태 | 영향 |
|---|------|----------|------|
| 1 | ~~fp_description 그룹화 매핑 최종 클래스 수~~ | **완료 (2026-03)** — 6 FP 클래스 + 1 TP = 7클래스 확정 (§12.1, §21.1 C8/C9) | — |
| 2 | 조직 간 라벨링 일관성 | Phase 0 검증 후 판단 | 단일 모델 vs 조직별 모델 |
| 3 | Sumologic 재추출 가능 여부 및 일정 | 요청 예정 | Phase 2 진입 시점 |
| 4 | Sumologic 5-key JOIN 매핑률 | JOIN 시도 후 판단 | Phase 2 가능 여부 |
| 5 | 피드백 수집 인터페이스 방식 | 미정 (현행: 엑셀 자유 텍스트) | 운영 피드백 루프 설계 |
| 6 | 클래스별 분포 (데이터 불균형 정도) | Phase 0 완료 후 확인 | 학습 전략 (class weight 등) |

---

## 22. 라벨 품질 감사 (v1.2 자동화)

4건의 독립 분석 모두에서 라벨 노이즈가 모델 성능의 진짜 천장(ceiling)으로 지적되었다. 이 감사는 전 Step에 걸친 전제 조건이다.

**v1.2: Confident Learning 기반 자동 감사**

원칙 G(Zero-Human-in-the-Loop)에 따라, 수동 재라벨링과 수동 확인을 Confident Learning 기반 자동 감사로 대체한다.

```python
from sklearn.model_selection import GroupKFold
import numpy as np

class ConfidentLearningAuditor:
    """
    cleanlab 방식의 Confident Learning으로
    라벨 노이즈를 자동 추정하고 정제한다.
    반드시 GroupKFold(groups=pk_file)를 사용해 파일 단위 누수를 방지한다.
    """
    NOISE_THRESHOLD = 0.10     # 노이즈율 10% 이상이면 정제 수행
    CONFIDENCE_THRESHOLD = 0.90  # 고확신 불일치만 라벨 오류 후보로

    def audit(self, X, y, groups, model_cls, n_splits=5) -> dict:
        """
        GroupKFold(groups=pk_file) Cross-Validation으로 각 샘플의
        out-of-fold 예측 확률을 산출하여 라벨 오류 탐지.
        동일 pk_file이 train/val에 동시 노출되지 않도록 강제한다.
        """
        kf = GroupKFold(n_splits=n_splits)
        oof_proba = np.zeros((len(y), len(np.unique(y))))

        for train_idx, val_idx in kf.split(X, y, groups=groups):
            model = model_cls()
            model.fit(X[train_idx], y[train_idx])
            oof_proba[val_idx] = model.predict_proba(X[val_idx])

        # 라벨 오류 후보 탐지
        predicted_labels = oof_proba.argmax(axis=1)
        predicted_confidence = oof_proba.max(axis=1)

        label_errors = (
            (predicted_labels != y) &
            (predicted_confidence >= self.CONFIDENCE_THRESHOLD)
        )

        noise_rate = label_errors.mean()

        return {
            'noise_rate': round(noise_rate, 4),
            'error_indices': np.where(label_errors)[0].tolist(),
            'error_count': label_errors.sum(),
            'total_samples': len(y),
            'needs_cleaning': noise_rate > self.NOISE_THRESHOLD,
        }

    def clean_labels(self, X, y, audit_result: dict) -> tuple:
        """
        감사 결과에 따라 라벨을 자동 정제.
        - 방법 1: 오류 후보를 학습에서 제외
        - 방법 2: 모델 예측으로 자동 재라벨링
        """
        if not audit_result['needs_cleaning']:
            return X, y  # 정제 불필요

        error_mask = np.zeros(len(y), dtype=bool)
        error_mask[audit_result['error_indices']] = True

        # 방법 1: 오류 후보 제외 (보수적)
        X_clean = X[~error_mask]
        y_clean = y[~error_mask]

        return X_clean, y_clean

    def compute_pii_type_mismatch(self, df: pd.DataFrame) -> dict:
        """
        pattern_name_raw vs pii_type_inferred 불일치율 자동 계산.
        """
        has_both = df.dropna(subset=['pattern_name_raw', 'pii_type_inferred'])
        mismatch = (
            has_both['pattern_name_raw'] != has_both['pii_type_inferred']
        )
        return {
            'mismatch_rate': round(mismatch.mean(), 4),
            'mismatch_count': mismatch.sum(),
            'total': len(has_both),
        }
```

**자동 감사 흐름:**

```
1. K-Fold Cross-Validation으로 각 샘플의 out-of-fold 예측 확률 산출
2. 예측과 라벨이 불일치하고 confidence ≥ 0.90인 케이스 → 라벨 오류 후보
3. 라벨 오류 추정 건수 / 전체 건수 = 라벨 노이즈율 자동 추정
4. 노이즈율 > 10% → 해당 케이스를 학습 데이터에서 자동 제외
5. pattern_name_raw vs pii_type_inferred 불일치율도 자동 계산
```

라벨 노이즈율이 10% 이상이면, Macro F1 ≥ 0.85 목표 자체가 라벨 품질에 의해 제약받을 수 있으므로 Confident Learning이 자동으로 정제를 수행한다.

---

## 23. PoC 리포트 자동 생성 (run_poc_report.py)

### 23.1 개요

`scripts/run_poc_report.py`는 학습 완료 후 PoC 결과를 7-sheet Excel로 자동 생성하는 스크립트이다.
`src/report/excel_writer.py`의 `PocReportData` dataclass와 `PocExcelWriter` 클래스가 데이터 컨테이너와 Excel 작성을 담당한다.

**주요 실행 명령어:**

```bash
python scripts/run_poc_report.py                          # 기본: Phase 1, Label Only
python scripts/run_poc_report.py --phase 2                # Phase 2 (Label + Sumologic)
python scripts/run_poc_report.py --output my.xlsx         # 출력 경로 지정
python scripts/run_poc_report.py --precision-target 0.90  # τ 목표 Precision
python scripts/run_poc_report.py --skip-ml                # Feature 없을 때 Rule 분석만
```

---

### 23.2 통합 리포트 스크립트 (`scripts/run_report.py`, 2026-03-20 신규)

기존 `run_evaluation.py` + `run_poc_report.py` + `diagnose_data_bias.py`를 통합.
`run_pipeline.py`의 S6 단계에서 자동 호출된다.

```bash
python scripts/run_report.py --source label              # Label 모델 리포트
python scripts/run_report.py --source detection          # Joined 모델 리포트
python scripts/run_report.py --source label --include-diagnosis  # 데이터 진단 포함
```

**10단계 처리 흐름:**

```
Step 1:  데이터 로드 (silver_label.parquet 또는 silver_joined.parquet)
Step 2:  모델 로드 (best_model_v1.joblib 또는 detection_best_model_v1.joblib)
Step 3:  FeatureBuilderSnapshot.transform() → X_all (재학습 없음)
Step 4:  Primary Split (work_month_time_split 또는 group_time_split)
Step 5:  RuleLabeler.label_batch(df_test)  → rule_labels_df
Step 6:  predict_with_uncertainty()        → ml_predictions_df
Step 7:  핵심 지표 (F1, PoC 판정, Coverage-Precision)
Step 7b: eval 고유 산출물 (confusion_matrix.png, feature_importance.csv/png, error_analysis.csv)
Step 8:  Secondary / Tertiary Split 성능 계산
Step 9:  데이터 진단 (--include-diagnosis 시: Column Registry, Split Robustness, Ablation)
Step 10: PocExcelWriter → 10-sheet Excel + 개별 산출물 (Sheet 10: Decision Combiner RULE+ML 통합 평가)
```

**source별 모델/데이터 경로:**

| source | 모델 | 데이터 | 피처 스냅샷 | 출력 |
|--------|------|--------|------------|------|
| label | `best_model_v1.joblib` | `silver_label.parquet` | `feature_builder.joblib` | `poc_report.xlsx` |
| detection | `detection_best_model_v1.joblib` | `silver_joined.parquet` | `detection_feature_builder.joblib` | `poc_report_detection.xlsx` |

---

### 23.3 CLI 옵션 (`run_report.py`)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--source {label,detection}` | `label` | 데이터 소스 |
| `--output PATH` | source별 자동 | 출력 Excel 경로 |
| `--precision-target FLOAT` | `0.95` | Coverage-Precision 곡선의 목표 tau Precision |
| `--skip-ml` | `False` | ML 모델 없을 때 Rule 분석만 실행 |
| `--include-diagnosis` | `False` | 데이터 진단 포함 (Sheet 9) |
| `--skip-diagnosis` | `False` | 데이터 진단 제외 (빠른 실행) |
| `--model-dir PATH` | `models` | 학습된 모델 디렉토리 |
| `--label-dir PATH` | `data/raw/label` | 레이블 Excel 디렉토리 |

---

### 23.4 9-Sheet 구성 (2026-03-20 확장)

| 시트명 | 데이터 출처 | 포함 항목 |
|--------|------------|-----------|
| `1_요약` | `poc_criteria`, `split_summary`, `business_impact`, `run_metadata` | 데이터 조건, Split 건수, F1/Recall/Precision 판정, PoC PASS/FAIL, 비즈니스 임팩트 |
| `2_데이터통계` | `binary_stats`, `class_imbalance`, `fp_description_stats`, `org_stats` | TP/FP 비율, 월별 분포, fp_description 카테고리 분포, dedup 통계, 조직별 분포 |
| `3_모델성능` | `split_comparison`, `class_metrics` | 3종 Split 비교표 (Primary/Secondary/Tertiary), PASS/FAIL 색상, 클래스별 성능 |
| `4_Coverage곡선` | `coverage_curve` | tau 테이블, 권장 tau 하이라이트, 목표 Precision 기준선 |
| `5_Rule기여도` | `rule_contribution`, `class_rule_contribution`, `rule_vs_ml_coverage` | rule_id별 히트율/정밀도, 클래스x룰 교차표, Rule vs ML Coverage 비교 |
| `6_오분류분석` | `error_patterns`, `error_samples`, `error_risk_summary` | 오분류 패턴 상위 15, 샘플 200건, FP->TP/TP->FP 위험도 요약 |
| `7_신뢰도분포` | `confidence_distribution` | ML 예측 신뢰도 분포 (Histogram 데이터, 클래스별 분리) |
| **`8_FeatureImportance`** | `feature_importance_df`, `feature_group_importance` | **피처 그룹별 합산 + 상위 30 피처 랭킹** |
| **`9_데이터진단`** | `column_risk_registry`, `split_robustness`, `ablation_results` | **Column 리스크 등급, Split Robustness (temporal/server/random F1 비교), Feature Ablation** |

---

### 23.5 PocReportData 주요 필드

| 필드 | 타입 | 대응 시트 | 설명 |
|------|------|----------|------|
| `data_condition` | `str` | 1_요약 | `"Label Only"` / `"Label + Sumologic"` |
| `split_summary` | `dict` | 1_요약 | `{train_n, test_n, split_method}` |
| `poc_criteria` | `dict` | 1_요약 | `check_poc_criteria()` 반환: `{f1_macro, tp_recall, fp_precision, passes}` |
| `run_metadata` | `dict` | 1_요약 | 실행 시각, Git 커밋 해시 등 |
| `business_impact` | `dict` | 1_요약 | `_build_business_impact()` 반환 (§23.6 참조) |
| `binary_stats` | `dict` | 2_데이터통계 | `compute_binary_stats()` 반환 |
| `fp_description_stats` | `DataFrame` | 2_데이터통계 | fp_description 카테고리별 건수 |
| `split_comparison` | `DataFrame` | 3_모델성능 | 3종 Split 성능 비교 |
| `coverage_curve` | `dict` | 4_Coverage곡선 | `{curve: DataFrame, recommended_tau, precision_at_tau}` |
| `rule_contribution` | `DataFrame` | 5_Rule기여도 | `rule_contribution()` 반환 |
| `error_patterns` | `list` | 6_오분류분석 | `[(actual, predicted, count), ...]` |
| `confidence_distribution` | `DataFrame` | 7_신뢰도분포 | 클래스별 신뢰도 분포 데이터 |
| `feature_importance_df` | `DataFrame` | 8_FeatureImportance | (feature, importance) 전체 랭킹 |
| `feature_group_importance` | `dict` | 8_FeatureImportance | {그룹명: 합산 importance} |
| `column_risk_registry` | `DataFrame` | 9_데이터진단 | 컬럼별 리스크 등급 (--include-diagnosis) |
| `split_robustness` | `DataFrame` | 9_데이터진단 | temporal/server/random F1 비교 |
| `ablation_results` | `DataFrame` | 9_데이터진단 | 피처 블록 제거 F1 |

---

### 23.6 비즈니스 임팩트 자동 판정

`_build_business_impact(binary_stats, coverage_curve)` 함수가 아래 4개 값을 자동 산출한다:

| 키 | 산출 방식 | 설명 |
|----|----------|------|
| `total_fp` | `binary_stats["total"]["fp"]` | 테스트셋 전체 FP 건수 |
| `coverage_at_target` | `curve_df[tau==recommended_tau]["coverage"]` | 권장 τ에서의 Coverage 비율 |
| `estimated_auto_fp` | `total_fp × coverage_at_target` | 자동 처리 가능 추정 FP 건수 |
| `phase1_goal_40pct_met` | `coverage_at_target >= 0.40` | Phase 1 목표(Coverage ≥ 40%) 달성 여부 |

**판정 로직:**
`recommended_tau`는 `compute_coverage_precision_curve(precision_target=0.95)`가 반환하는 값이다.
`coverage_at_target ≥ 0.40`이면 `phase1_goal_40pct_met = True` → Sheet 1_요약에 PASS로 표기된다.

---

### 23.7 Rationale

**왜 7-sheet Excel인가:**
운영팀/보안팀이 Jupyter 환경 없이 결과를 즉시 열람할 수 있어야 한다. 폐쇄망 환경에서 HTML/Notebook 렌더링이 불가능하므로, openpyxl 직접 작성으로 외부 패키지 의존성을 최소화했다.

**왜 자동 생성인가:**
Phase 1 결과 리뷰는 반복 작업이다. 데이터 재수집 → 재학습 → 재평가 사이클마다 동일한 지표를 수동으로 채울 경우 오류/누락이 발생한다. `run_poc_report.py` 한 명령으로 재현 가능한 산출물을 얻는다.

**`--skip-ml` 모드:**
Feature Parquet(`models/*.parquet`)이 아직 없는 상태에서도 Rule Labeler 기여도 분석을 먼저 실행할 수 있다. Sheet 3/6/7은 빈 상태로 생성되며, Sheet 5(Rule 기여도)는 완전 작성된다.

---

**— 문서 끝 —**
