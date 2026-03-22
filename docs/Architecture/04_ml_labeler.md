## 8. Stage S3b: ML Feature Builder & ML Labeler

### 8.1 기능

RULE 라벨러가 처리하지 못한(또는 신뢰도가 낮은) 잔여 케이스에 대해 ML 모델로 7클래스 확률을 출력하고, 확률 보정(calibration) + 설명(evidence)을 생성한다.

### 8.2 ML이 처리하는 영역

RULE 라벨러가 이미 높은 신뢰도로 라벨을 부여한 케이스는 ML을 실행할 필요가 없다. 이렇게 하면 CPU 자원이 절약되고, ML이 실제로 가치를 제공하는 영역(롱테일/교집합)에 집중할 수 있다.

```python
# RULE로 결정되지 않은 케이스만 ML로 전달
ml_target_mask = (
    (~rule_labels['rule_matched']) |                    # 룰 미매칭
    (rule_labels['rule_confidence_lb'] < TAU_RULE_MIN)  # 룰 신뢰도 낮음
)
```

**Rationale:**

- 룰은 설명이 완전하고 비용이 낮다. 룰이 높은 신뢰도로 커버하는 영역에 ML을 중복 실행하는 것은 비효율적이다.
- ML은 피처 간 상호작용을 자동으로 학습해 "룰의 틈새"를 잡는 역할에 집중한다. 이 역할 분리가 전체 시스템의 효율성과 안정성을 동시에 높인다.
- 다만, Decision Combiner(S4)에서 RULE과 ML의 충돌을 감지하기 위해, 일부 RULE 매칭 케이스에 대해서도 ML을 실행할 수 있다(옵션).

### 8.3 ML 입력 피처 구성

| 피처 그룹 | 생성 방식 | 예상 차원 수 | 역할 |
|-----------|----------|-------------|------|
| 수동 피처 (S2) | 키워드/정규식/비율/길이/구조 플래그 + placeholder 비율 | 약 30개 | 도메인 지식 기반 강력한 단서 |
| word TF-IDF (raw_text) | `TfidfVectorizer(analyzer='word', token_pattern 커스텀)` | 약 500개 | 키워드/도메인/토큰 패턴 (의미 신호) |
| char TF-IDF (shape_text) | `TfidfVectorizer(analyzer='char', ngram_range=(3,6))` | 약 300개 | 구조적 패턴 (형태 신호) |
| word/char TF-IDF (path_text) | 경로 토큰 word TF-IDF 또는 단순 키워드 플래그 대체 | 약 200개 | 파일 유형 맥락 |
| Tabular | 검출 건수, 경로 깊이, 확장자 등 | 약 15개 | 메타데이터 맥락 |
| File-level aggregation (선택) | pk_file 단위 통계 | 약 10개 | 파일 수준 맥락 증폭 |
| **합계 (Phase 2 full)** | | **약 1,055개** | |

> **Phase 1 실제 피처 구성 (Tier 3 C1+C2, ~538개, F1=0.78):**
>
> | 그룹 | 피처 수 | 상세 |
> |------|--------|------|
> | TF-IDF fname char | ~200 | file_name char_wb (2,5)-gram |
> | TF-IDF fname shape | ~100 | file_name → `_to_shape_text()` → char_wb (2,5)-gram [Tier 2 B3] |
> | TF-IDF path word | ~200 | file_path → `_to_path_text()` → word (1,2)-gram |
> | Dense 메타/경로 | ~20 | fname_has_*, pattern_count_*, is_*, has_*, rule_matched |
> | Dense 서버 의미 | 3 | server_env(prd/dev/stg/sbx/test/unknown), server_is_prod, server_stack(app/mms/db/web/batch/etc) [Tier 2 B7] |
> | Dense RULE 세부 | 3 | rule_confidence_lb, rule_id_enc, rule_primary_class_enc [Tier 2 B8] |
> | Dense file 집계 | 2 | file_event_count, file_pii_diversity (df에 추가, X_train 미주입) [Tier 2 B9] |
> | Dense 범주형 _enc | 8 | service/ops_dept/organization/retention_period/server_env/stack/rule_id/class [Tier 2 B1, train+test 합본 fit] |
> | Dense 기타 | ~3 | file_extension_enc, path_depth 등 |
> | ~~exception_requested~~ | ~~1~~ | ~~제거됨 (Sumologic에 없음, 추론 불가)~~ |
> | **Phase 1 합계** | **~538** | |
>
> **Tier 3 C1 Easy FP Suppressor:** ML 학습 전에 고확신 FP를 규칙 기반으로 선제 분리. 4개 조건 (is_system_device, is_package_path+mass, is_docker_overlay, has_license_path). purity≥95% 시 활성화, suppressed 행은 ML에서 제외.
>
> **현재 구조:**
> - **학습/추론 동형성:** `prepare_phase1_features()` 공통 함수로 training(run_training.py Step 2-4)과 inference(FeatureBuilderSnapshot.transform) 통일
> - **FeatureBuilderSnapshot:** categorical LabelEncoders 저장, transform() 내부에서 전처리 자동 수행
> - **Calibration 제거:** CalibratedClassifierCV(cv=3) 삭제 (F1 무관, 학습 시간 절감)
> - **Bootstrap CI 제거:** evaluator.py n=500 삭제 (4M행에서 실용성 없음)
> - **TP 가중치:** `--tp-weight` CLI 옵션 (기본 1.0, 필요시 조정)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# raw_text용 TF-IDF (word 단위 — 의미/키워드 신호 포착)
# v1.1 수정: token_pattern을 개선하여 = : , ; 기준으로도 토큰 분리
# 기존 r'(?u)[^\s]+'는 "xpiryDate=170603*****"를 하나의 긴 토큰으로 인식하여
# vocabulary를 일회성 토큰으로 오염시킴
import re

def custom_tokenizer(text):
    """= : , ; 기준 분리 후 공백 기준 분리"""
    text = re.sub(r'[=:,;]', ' ', text)
    return text.split()

tfidf_raw = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    max_features=500,
    min_df=5,
    max_df=0.95,
    sublinear_tf=True
)

# shape_text용 TF-IDF (char 단위 — 구조/형태 신호 포착)
# v1.1 수정: n-gram 범위를 (3,5)로 줄이고 max_features를 500으로 확대
# 기존 max_features=300은 12M+ 가능 조합 중 극소수만 선택하여
# 판별력 높은 희귀 n-gram이 제외될 수 있음
tfidf_shape = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),        # v1.1: 6-gram 제거로 조합 수 감소
    max_features=500,          # v1.1: 더 많은 n-gram 유지
    min_df=5,
    sublinear_tf=True
)

# v1.1 추가: raw_text char n-gram (OOV 완화 목적)
# 고정 vocabulary 문제를 완화하기 위해 raw_text에도 char n-gram 채널을 소량 추가
# @newvendor.example.com이 word vocab에 없어도 @new, vend, .com 등이 부분 매칭
tfidf_raw_char = TfidfVectorizer(
    analyzer='char',
    ngram_range=(4, 6),
    max_features=200,
    min_df=5,
    sublinear_tf=True
)

# (선택) path_text용 TF-IDF
tfidf_path = TfidfVectorizer(
    analyzer='word',
    max_features=200,
    min_df=5,
    sublinear_tf=True
)
```

**왜 raw_text는 word TF-IDF이고 shape_text는 char TF-IDF인가:**

- **raw_text → word TF-IDF**: `raw_text`에는 `@lguplus.co.kr`, `bytes`, `Date=`, `<NUM13>` 같은 의미 있는 토큰이 포함된다. word 단위로 잡아야 이 토큰들이 그대로 피처가 된다. 커스텀 `token_pattern`으로 `@`, `.`, `/`, `=` 등 특수문자를 포함한 토큰을 살린다.
- **shape_text → char TF-IDF (3~6gram)**: `shape_text`는 `aaa@000.aaa.aa.aa` 같은 형태 문자열이므로, word boundary가 의미 없다. char n-gram이 `000.`, `@000`, `aaa.` 같은 구조적 부분 패턴을 자연스럽게 포착한다.
- 기본 word TF-IDF만 쓰면 "구조 신호(`010***`, `000000<MASK>` 등)"가 약해질 수 있다 → shape 기반 char TF-IDF가 그 구멍을 메워준다.
- 새 패턴이 늘어나도, char n-gram은 OOV(미등록 단어) 문제를 완화한다.

**왜 Transformer가 아니라 TF-IDF + Boosting인가:**

- GPU 미보유 → Transformer 학습 불가 (확정 제약)
- 이 분류 문제의 결정 경계가 "의미론적 유사성"이 아니라 "구조적/어휘적 패턴"에 있다. `bytes`, `Date`, `@lguplus` 같은 키워드의 존재 여부가 분류의 핵심이지, 문맥 내 단어 간 관계(self-attention)가 핵심이 아니다.
- 약 10자에서 Transformer의 강점인 long-range dependency가 작동할 여지가 없다. 오히려 짧은 텍스트에서는 hand-crafted feature + char n-gram이 Transformer 임베딩보다 noise-robust할 수 있다.
- Tabular 피처(파일 경로, 검출 건수 등)와의 자연스러운 결합은 Boosting 모델의 네이티브 강점이다. Boosting은 텍스트 피처와 Tabular 피처를 트리 분기에서 자연스럽게 교차(interaction)시킨다.

### 8.4 인코더 Unknown 안전 설정 (v1.1 추가)

새로운 범주값(새 확장자, 새 서버 그룹 등)이 추론 시 유입되면 인코더 설정에 따라 런타임 에러가 발생할 수 있다. 모든 범주형 인코더에 unknown 안전 설정을 강제한다.

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# OneHotEncoder 사용 시
encoder = OneHotEncoder(
    handle_unknown='infrequent_if_exist',  # unknown → infrequent 카테고리로 처리
    min_frequency=5,                        # 빈도 5 미만 → infrequent
    sparse_output=True
)

# OrdinalEncoder 사용 시
encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1  # unknown → -1로 인코딩
)
```

LightGBM/XGBoost native categorical 사용 시에도 ETL 단계에서 unknown 카테고리를 명시적으로 도입한다.

### 8.5 피처 스키마 고정 (v1.1 추가)

TF-IDF + 수동 피처 + tabular 결합 시, 학습과 추론의 컬럼 순서/차원이 1비트라도 다르면 모델 입력 shape mismatch로 장애가 발생한다. Training Pipeline 산출물에 `feature_schema.json`을 추가한다.

```json
{
  "version": "v1.0.0",
  "total_features": 1055,
  "feature_groups": {
    "manual": {"start_idx": 0, "end_idx": 29, "names": ["has_byte_kw", "..."]},
    "tfidf_raw": {"start_idx": 30, "end_idx": 529, "vocab_size": 500},
    "tfidf_raw_char": {"start_idx": 530, "end_idx": 729, "vocab_size": 200},
    "tfidf_shape": {"start_idx": 730, "end_idx": 1229, "vocab_size": 500},
    "tfidf_path": {"start_idx": 1230, "end_idx": 1429, "vocab_size": 200},
    "tabular": {"start_idx": 1430, "end_idx": 1454, "names": ["inspect_count_log1p", "..."]}
  },
  "created_at": "2026-02-15T10:30:00",
  "model_version": "v1.0.0"
}
```

Inference Pipeline에서 피처 빌드 후 스키마 검증을 강제한다.

```python
def validate_feature_schema(X, schema):
    assert X.shape[1] == schema['total_features'], \
        f"Feature dimension mismatch: {X.shape[1]} vs {schema['total_features']}"
```

### 8.6 모델 학습 설정

```python
import lightgbm as lgb

params = {
    'objective': 'multiclass',
    'num_class': 7,  # 확정 체계: 6 FP 클래스 + 1 TP 클래스 = 7 (§12.1). Phase 1(label-only)은 2-class(TP/FP)
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 7,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'verbose': -1,
    # v1.1: 재현성 보장
    'random_state': 42,
    'seed': 42,
    'n_jobs': 1,  # 결정론적 모드 (감사/재현용), 속도 우선 시 -1
}

# Wave 4 (Tier 2 B6): Phase 1 실제 운영 파라미터 (src/utils/constants.py LGB_DEFAULT_PARAMS)
# 10M행 실데이터 진단 후 과적합 억제를 위해 정규화 강화
phase1_params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'n_estimators': 500,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 200,   # 20→200: 일반화 강제 (리프당 최소 샘플 수 증가)
    'reg_alpha': 0.5,           # 0→0.5: L1 정규화 (노이즈 피처 가중치 억제)
    'max_depth': 10,            # -1→10: 트리 깊이 제한 (과적합 방지)
    'class_weight': 'balanced',
    'n_jobs': -1,
    'verbose': -1,
}
```

**Early Stopping - Inner Validation Split (eval_set 누수 방지):**

Early stopping은 `eval_set` loss를 모니터링하여 학습 중단 시점(실효 n_estimators)을 결정한다. `eval_set`에 최종 test set을 직접 전달하면, test set 정보가 모델 구조에 간접 유입되어 **모든 성능 수치가 과대추정**된다.

**구현 원칙:** train set의 20%를 내부 validation으로 분리하여 early stopping에만 사용. 최종 평가는 test set에 대해 1회만 수행.

```python
# train set 80% / inner_val 20% 분리 (early stopping 전용)
X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
    X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=42
)
model.fit(
    X_inner_train, y_inner_train,
    eval_set=[(X_inner_val, y_inner_val)],   # test set 아닌 inner val 사용
    callbacks=[early_stopping(50)],
)
# 최종 평가: X_test에 대해 1회만
y_pred = model.predict(X_test)
```

> **구현 위치:** `src/models/trainer.py` — `train_xgboost()`, `train_lightgbm()` 내 `EARLY_STOPPING_ROUNDS > 0` 분기
> **이전 문제:** eval_set에 X_test를 직접 전달 → test loss 최소 시점에서 early stop → F1 과대추정(CRITICAL)

### 8.6a Focal Loss — Hard Example 집중 학습 (Wave 3 구현)

`class_weight="balanced"`는 소수 클래스 recall을 높이지만, easy negative(명백한 FP)에도 동일 비중으로 학습하여 precision이 떨어지는 부작용이 있다. Focal Loss는 모델이 이미 잘 분류하는 샘플의 loss 기여를 줄이고, **분류 경계 근처의 hard example에 집중**한다.

```python
def focal_loss_lgb_objective(gamma: float = 2.0, alpha: float = 0.25):
    """LightGBM custom objective - Binary Focal Loss.

    gamma: focusing parameter (2.0 권장) — 높을수록 easy sample loss 기여 감소
    alpha: positive class weight (0.25 권장)
    """
    def focal_loss(y_true, y_pred):
        p = 1.0 / (1.0 + np.exp(-y_pred))
        focal_weight = np.where(y_true == 1,
                                alpha * (1 - p) ** gamma,
                                (1 - alpha) * p ** gamma)
        grad = focal_weight * (p - y_true)
        hess = focal_weight * p * (1 - p)
        return grad, hess
    return focal_loss
```

> **구현 위치:** `src/models/trainer.py` — `focal_loss_lgb_objective()` 함수
> **사용:** `LGBMClassifier(objective=focal_loss_lgb_objective(gamma=2.0, alpha=0.25))`
> **Phase 1 기본값:** class_weight="balanced" 유지(binary, 소규모 데이터). Focal Loss는 Phase 2 multiclass 또는 심한 클래스 불균형 시 활성화 권장.

### 8.6b Class Imbalance 처리 원칙 (SMOTE 비사용 근거)

| 방법 | 현재 상태 | 근거 |
|------|----------|------|
| `class_weight="balanced"` | **기본 적용** | LightGBM/XGBoost 네이티브 지원, 추가 연산 없음 |
| Focal Loss | **구현 완료 (선택 적용)** | Hard example 집중, precision 유지 |
| SMOTE | **삭제됨** | Sparse TF-IDF 행렬에서 고차원 희소 공간 보간은 의미 없음. Dead code 제거. |
| Hybrid Under-sampling | **미착수** | label_mapping.yaml(FP 서브클래스) 완성 후 진행 예정 |

**SMOTE 비사용 근거:** `apply_smote()`는 dense 피처에서만 의미 있는 보간을 수행한다. 5,000차원 TF-IDF sparse matrix에 SMOTE를 적용하면 비어있는 공간에 가상의 샘플이 생성되어 과적합만 심해진다. class_weight 또는 threshold 최적화(§8.6c 해당 없음, §6.3 Coverage-Precision 참조)로 충분히 대체 가능.

### 8.7 확률 보정 (Calibration) — 클래스별 차등 전략 (v1.1 개선)

Boosting 모델의 softmax 확률은 과신(overconfident)하는 경향이 있다. 이를 보정하지 않으면 confidence 기반 의사결정(threshold, override)이 왜곡된다.

**v1.1 변경:** Isotonic Regression이 소수 클래스(예: FP-시스템로그)에서 과적합 위험이 있으므로, 클래스별 차등 보정 전략을 적용한다.

```python
from sklearn.calibration import CalibratedClassifierCV

def build_calibrator(model, X_cal, y_cal, class_counts):
    """
    소수 클래스는 Platt Scaling (파라미터 2개),
    다수 클래스는 Isotonic Regression (비모수적) 적용.
    """
    calibrators = {}
    for class_idx, count in class_counts.items():
        if count < 200:
            # 소수 클래스: Platt Scaling (안전)
            calibrators[class_idx] = CalibratedClassifierCV(
                model, method='sigmoid', cv='prefit'
            )
        else:
            # 다수 클래스: Isotonic Regression (정밀)
            calibrators[class_idx] = CalibratedClassifierCV(
                model, method='isotonic', cv='prefit'
            )
    return calibrators

# 단순 적용 (전체 동일 보정)도 여전히 유효
calibrated_model = CalibratedClassifierCV(
    base_estimator=lgb_model,
    method='isotonic',
    cv='prefit'
)
calibrated_model.fit(X_cal, y_cal)
```

**Rationale:**

- 부스팅 확률은 "0.92니까 확실"이라고 말하기 위험하다. 보정을 하면 confidence가 "실제 정답 확률"에 더 가까워져서, threshold 기반 의사결정(TP override, NEEDS_REVIEW)이 안정적으로 작동한다.
- 보정 모델(calibrator)은 학습 파이프라인에서 생성하여 저장하고, 추론 파이프라인에서 불러와 사용한다.
- **v1.1:** 소수 클래스에 Isotonic을 적용하면 보정 데이터 부족으로 과적합될 수 있다. Platt Scaling(sigmoid)은 파라미터가 2개뿐이라 소수 클래스에서도 안전하다.

### 8.8 OOD Score — Epistemic Uncertainty 포착 (v1.1 추가)

현재 설계의 불확실성 지표(margin, entropy)는 Aleatoric Uncertainty(데이터 자체의 모호함)를 포착하지만, Epistemic Uncertainty(모델이 이 영역에 대해 무지함)를 포착하지 못한다.

**방법 1: Mahalanobis Distance (Dense 피처 대상)**

> **구현 완료:** `src/models/ood_detector.py` — `OODDetector` 클래스.
> `run_training.py` Step 9에서 학습 후 `models/final/ood_detector.joblib`로 자동 저장.
> dense 피처는 `X_train`의 뒷부분 (`n_tfidf:` 이후 컬럼)을 사용.

```python
from src.models.ood_detector import OODDetector

# Training 시점: 학습 데이터의 dense 피처 분포 적합 + 저장
n_dense = len(dense_columns)
X_dense = X_train[:, -n_dense:].toarray()
ood_detector = OODDetector(threshold_percentile=95.0)
ood_detector.fit(X_dense)
ood_detector.save("models/final/ood_detector.joblib")

# Inference 시점
ood = OODDetector.load("models/final/ood_detector.joblib")
is_ood = ood.predict(X_new_dense)   # bool array
ood_score = ood.score(X_new_dense)  # Mahalanobis distance
```

**방법 2: Leaf Node Isolation (트리 기반 OOD)**

학습 데이터에서 각 리프 노드에 도달한 샘플 수를 저장하고, 추론 시 도달 리프의 평균 샘플 수가 낮으면 OOD로 판단한다.

```python
# Training 시점
leaf_indices_train = model.predict(X_train, pred_leaf=True)
leaf_sample_counts = {}  # {tree_idx: {leaf_idx: count}}

# Inference 시점
leaf_indices_new = model.predict(X_new, pred_leaf=True)
avg_leaf_support = mean([leaf_sample_counts[t][l] for t, l in enumerate(leaf_indices_new)])
# avg_leaf_support가 낮으면 → OOD 가능성
```

### 8.9 ML 출력 & 애매함 지표

> **Wave 3 버그 수정 (2026-03-20):** `predict_with_uncertainty()`에서 `ml_tp_proba = proba[:, 0]`으로 하드코딩되어 있었으나, `LabelEncoder`는 알파벳순 정렬이므로 `classes_ = ['FP', 'TP']` → index 0은 **FP 확률**이었음. TP 확률로 사용하면 Decision Combiner의 TP 안전 override가 **역전**(FP 확률이 높을 때 TP로 override)되는 치명적 버그. **수정:** `label_names`에서 "TP" 포함 클래스의 인덱스를 동적으로 탐색하도록 변경.

```python
def ml_predict_with_uncertainty(model, calibrator, X) -> pd.DataFrame:
    """
    7클래스 보정 확률 + 애매함 지표 출력.

    단순히 max_proba만 보면 '가짜 확신'에 속을 수 있다.
    margin(1등-2등 차이)과 entropy(확률 분산)를 같이 보면
    진짜 확신 vs 가짜 확신을 구분할 수 있다.
    """
    raw_proba = model.predict_proba(X)
    cal_proba = calibrator.predict_proba(X)  # 보정된 확률

    # Top-2 클래스와 확률
    top2_idx = np.argsort(cal_proba, axis=1)[:, -2:]
    top1_class = top2_idx[:, -1]
    top2_class = top2_idx[:, -2]
    top1_proba = cal_proba[np.arange(len(X)), top1_class]
    top2_proba = cal_proba[np.arange(len(X)), top2_class]

    # TP 확률 — label_names에서 TP 인덱스를 동적으로 탐색 (Wave 3 수정)
    # LabelEncoder 알파벳순: ['FP', 'TP'] → index 1이 TP
    _tp_idx = next((i for i, n in enumerate(label_names) if "TP" in n), n_classes - 1)
    tp_proba = cal_proba[:, _tp_idx]

    # 애매함 지표
    margin = top1_proba - top2_proba
    entropy = -np.sum(cal_proba * np.log(cal_proba + 1e-10), axis=1)

    return pd.DataFrame({
        'ml_top1_class': top1_class,
        'ml_top1_proba': top1_proba,
        'ml_top2_class': top2_class,
        'ml_top2_proba': top2_proba,
        'ml_tp_proba': tp_proba,  # Wave 3: 동적 인덱스
        'ml_margin': margin,
        'ml_entropy': entropy,
    })
```

**왜 margin과 entropy가 필요한가:**

- `top1_proba = 0.85`와 `top1_proba = 0.45`는 "확신의 크기"가 다르다. 하지만 이것만으로는 부족하다.
- `top1=0.45, top2=0.40` (margin=0.05)은 "1등과 2등이 거의 붙어있다"는 뜻이고, `top1=0.45, top2=0.10` (margin=0.35)은 "1등이 확실하진 않지만 대안이 없다"는 뜻이다. 이 두 상황은 리스크가 다르다.
- entropy가 높으면 확률이 여러 클래스에 분산되어 있다는 뜻이고, 이는 모델이 "잘 모르겠다"고 말하는 것과 같다. 이런 케이스를 NEEDS_REVIEW로 라우팅해야 한다.

### 8.10 ML 설명 (evidence) — 2단계 전략

ML 설명은 비용(CPU 시간)이 있으므로, 전건 제공과 정밀 제공을 분리한다.

**A. 경량 설명 (전건 제공, 비용 무시 가능)**

"활성화된 수동 피처" + "텍스트에서 발견된 증거 substring"을 조합. 이것은 피처 값을 단순히 읽어오는 것이므로 추가 연산이 거의 없다.

```python
def generate_lightweight_evidence(row: dict, features: dict) -> list[dict]:
    """
    전건 제공 가능한 경량 설명.
    활성화된 키워드/플래그 피처를 근거로 변환.
    """
    evidences = []
    
    feature_evidence_map = {
        'has_byte_kw': ('KEYWORD_FOUND', 'bytes 키워드 발견'),
        'has_timestamp_kw': ('KEYWORD_FOUND', '타임스탬프 키워드 발견'),
        'has_domain_kw': ('KEYWORD_FOUND', '도메인 키워드 발견'),
        'has_os_copyright_kw': ('KEYWORD_FOUND', 'OS/오픈소스 키워드 발견'),
        'has_dev_kw': ('KEYWORD_FOUND', '개발/테스트 키워드 발견'),
        'is_log_file': ('PATH_FLAG', '로그 파일 경로'),
        'is_docker_overlay': ('PATH_FLAG', 'Docker overlay 경로'),
        'is_mass_detection': ('TABULAR_FLAG', '대량 검출 (10,000건 초과)'),
    }
    
    rank = 1
    for feat_name, (ev_type, description) in feature_evidence_map.items():
        if features.get(feat_name, 0) == 1:
            evidences.append({
                'evidence_rank': rank,
                'evidence_type': ev_type,
                'source': 'ML',
                'feature_name': feat_name,
                'matched_value': description,
            })
            rank += 1
    
    return evidences
```

**B. 정밀 설명 (불확실/TP 후보에만)**

SHAP(TreeSHAP) 또는 Decision Path로 상위 기여 피처 top-K를 제공. CPU 비용이 있으므로 `risk_flag=NEEDS_REVIEW` 또는 `risk_flag=TP_SAFE_OVERRIDE` 케이스에만 계산한다.

```python
import shap

def generate_shap_evidence(model, X_single, feature_names, top_k=5) -> list[dict]:
    """
    TreeSHAP으로 상위 기여 피처를 추출.
    비용이 있으므로 불확실 케이스에만 실행.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)  # shape: (1, n_features, n_classes)
    
    # 예측된 클래스에 대한 SHAP 값
    predicted_class = np.argmax(model.predict_proba(X_single), axis=1)[0]
    shap_for_class = shap_values[predicted_class][0]  # (n_features,)
    
    # 절대값 기준 상위 K개
    top_indices = np.argsort(np.abs(shap_for_class))[-top_k:][::-1]
    
    evidences = []
    for rank, idx in enumerate(top_indices, 1):
        feat_name = feature_names[idx]
        feat_value = X_single[0, idx]
        contribution = shap_for_class[idx]
        
        evidences.append({
            'evidence_rank': rank,
            'evidence_type': 'MODEL_FEATURE_IMPORTANCE',
            'source': 'ML',
            'feature_name': feat_name,
            'matched_value': f"{feat_name}={feat_value:.4f}",
            'weight_or_contribution': float(contribution),
        })
    
    return evidences
```

**Rationale (경량 vs 정밀 분리):**

- 전건에 SHAP을 계산하면 월 100~150만 건 × 수초/건으로 CPU 비용이 폭발한다.
- 경량 설명(활성화된 키워드/플래그)은 전건 제공이 가능하면서도, 운영팀 입장에서 "왜 이렇게 분류됐는지"를 80% 이상 납득할 수 있다.
- 정밀 설명(SHAP)은 모델이 "왜 틀렸는지" 또는 "왜 애매한지"를 분석할 때 가치가 높으므로, 불확실 케이스에 집중하는 것이 ROI가 좋다.

### 8.11 출력 스키마

**`ml_predictions.parquet`** (1행=1검출 이벤트)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pk_event` | string | 검출 이벤트 ID |
| `ml_top1_class` | int | 최고 확률 클래스 인덱스 |
| `ml_top1_class_name` | string | 최고 확률 클래스명 |
| `ml_top1_proba` | float | 최고 확률 (보정됨) |
| `ml_top2_class` | int | 2순위 클래스 인덱스 |
| `ml_top2_class_name` | string | 2순위 클래스명 |
| `ml_top2_proba` | float | 2순위 확률 (보정됨) |
| `ml_margin` | float | top1 - top2 |
| `ml_entropy` | float | 확률 분포 엔트로피 |
| `ml_tp_proba` | float | TP 클래스의 보정 확률 (리스크 제어용) |
| `ood_mahalanobis` | float | Dense 피처 기반 Mahalanobis distance (v1.1 추가) |
| `ood_leaf_support` | float | 평균 리프 노드 학습 샘플 수 (v1.1 추가) |
| `ood_flag` | bool | OOD 판정 (임계값 초과 시) (v1.1 추가) |

**`ml_evidence.parquet`** (N행=1검출 이벤트, long-format)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `pk_event` | string | 검출 이벤트 ID |
| `evidence_rank` | int | 근거 우선순위 |
| `evidence_type` | string | `KEYWORD_FOUND` / `PATH_FLAG` / `TABULAR_FLAG` / `MODEL_FEATURE_IMPORTANCE` |
| `source` | string | `"ML"` |
| `feature_name` | string | 피처명 |
| `matched_value` | string | 피처 값 또는 설명 문자열 |
| `weight_or_contribution` | float (nullable) | SHAP 기여도 (정밀 설명 시) |

### 8.12 대안 모델 비교 실험 프레임워크 (v1.1 추가)

LightGBM/XGBoost 단일 모델 타입에 의존하는 리스크를 줄이기 위해, PoC Step C(모델 학습) 단계에서 최소 3개 모델을 비교한다. 고차원 sparse TF-IDF 피처에서 Linear SVM이 동등하거나 우수할 수 있다.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

model_candidates = {
    'lgbm': LGBMClassifier(**lgbm_params),
    'linear_svm': SGDClassifier(loss='log_loss', class_weight='balanced'),
    'stacking': StackingClassifier(
        estimators=[
            ('lgbm', LGBMClassifier(**lgbm_params)),
            ('svm', SGDClassifier(loss='log_loss'))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    ),
}

for name, model in model_candidates.items():
    scores = evaluate(model, X_train, y_train, X_test, y_test,
                      split='group_time')
    print(f"{name}: F1={scores['macro_f1']:.4f}, "
          f"TP_Recall={scores['tp_recall']:.4f}, "
          f"Time={scores['train_time']:.1f}s")
```

Linear SVM이 TF-IDF 공간에서 우수하고 합성변수 없이도 좋은 성능을 내면, 합성변수 필요성 자체가 줄어든다.

### 8.13 Auto-Model-Selector (v1.2 추가)

원칙 G(Zero-Human-in-the-Loop)에 따라, 사람이 3개 모델 성능을 비교하여 선택하는 대신, 시스템이 자동으로 벤치마크를 수행하고 최적 모델을 선택한다.

```python
class AutoModelSelector:
    """
    안전 제약(TP Recall ≥ 0.95)을 충족하는 후보 중
    macro_f1이 가장 높은 모델을 자동 선택한다.
    """
    SAFETY_CONSTRAINTS = {
        'tp_recall': 0.95,   # 최소 TP Recall
    }

    def select_best_model(self, model_candidates: dict,
                          X_train, y_train, X_test, y_test,
                          split='group_time') -> dict:
        best_model = None
        best_score = -float('inf')
        results = []

        for name, model in model_candidates.items():
            scores = evaluate(model, X_train, y_train, X_test, y_test,
                              split=split)

            result = {'name': name, 'scores': scores, 'eligible': True}

            # 안전 제약 충족 여부
            for metric, threshold in self.SAFETY_CONSTRAINTS.items():
                if scores[metric] < threshold:
                    result['eligible'] = False
                    result['rejection_reason'] = (
                        f"{metric}={scores[metric]:.4f} < {threshold}"
                    )
                    break

            if result['eligible'] and scores['macro_f1'] > best_score:
                best_score = scores['macro_f1']
                best_model = model
                result['selected'] = True
            else:
                result['selected'] = False

            results.append(result)

        if best_model is None:
            # 모든 후보가 안전 제약 미충족 → 가장 높은 tp_recall 모델 선택
            fallback = max(results, key=lambda r: r['scores']['tp_recall'])
            best_model = model_candidates[fallback['name']]
            fallback['selected'] = True
            fallback['selection_reason'] = 'fallback_highest_tp_recall'

        # 자동 저장 + 결과 로깅
        save_model(best_model)
        log_selection_results(results)

        return {
            'selected_model': best_model,
            'results': results,
            'best_score': best_score
        }
```

**자동 선택 흐름:**

1. 모든 후보 모델에 대해 Group+Time Split 평가 수행
2. 안전 제약 (`TP Recall ≥ 0.95`) 미충족 모델 → 후보 탈락
3. 제약 충족 후보 중 `macro_f1`이 가장 높은 모델 자동 선택
4. 모든 후보가 안전 제약 미충족 시 → `tp_recall`이 가장 높은 모델로 폴백
5. 선택 사유 자동 로깅 (감사 추적)

### 8.14 평가 지표 보강 — PR-AUC 및 Bootstrap CI (Wave 3 구현)

#### PR-AUC (Precision-Recall Area Under Curve)

F1-macro는 임계값(threshold=0.5)에서의 단일 점 지표다. TP:FP 불균형이 심하면 **FP F1이 쉽게 0.95+를 달성해 전체 F1-macro를 끌어올리는 반면, TP F1이 0.60이어도 "통과"로 판정**될 수 있다.

PR-AUC는 threshold-agnostic이며 불균형 데이터에 강건하다.

```python
from sklearn.metrics import average_precision_score

# Binary case: TP 클래스를 positive로 설정
tp_proba = model.predict_proba(X_test)[:, le.transform(['TP'])[0]]
pr_auc = average_precision_score(y_test_binary, tp_proba)
print(f"PR-AUC: {pr_auc:.4f}  [권장: threshold-agnostic 지표]")
```

> **구현 위치:** `src/evaluation/evaluator.py` — `full_evaluation()` 내 PR-AUC 블록
> **보고 우선순위:** PR-AUC > F1-macro (F1-macro는 참고용으로만 표시)

#### Bootstrap Confidence Interval

단일 test set에 대한 point estimate만 보고하면, "F1=0.85"가 0.80~0.90인지 0.70~0.95인지 알 수 없다. Bootstrap으로 신뢰구간을 제공하여 의사결정 근거를 보강한다.

```python
from sklearn.utils import resample
import numpy as np

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=500, ci=0.95):
    """Bootstrap으로 metric의 신뢰구간 산출."""
    scores = []
    for _ in range(n_bootstrap):
        idx = resample(range(len(y_true)), replace=True)
        score = metric_fn(np.array(y_true)[idx], np.array(y_pred)[idx])
        scores.append(score)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lower), float(upper)

# 사용 예
mean_f1, lo, hi = bootstrap_metric(y_test_enc, y_pred, f1_macro_fn, n_bootstrap=500)
print(f"F1-macro CI: {mean_f1:.4f} ({lo:.4f}~{hi:.4f}, 95% CI, n=500)")
```

> **구현 위치:** `src/evaluation/evaluator.py` — `bootstrap_metric()` 함수, `full_evaluation()` CI 블록
> **보고 형식 예시:** `F1-macro CI: 0.8400 (0.8100~0.8700, 95% CI, n=500)`

#### 권장 평가 보고 형식

```
[TP Safety]    TP Recall      = 0.95  (0.91~0.98, 95% CI)
[Auto-FP]      Coverage @τ   = 42%   @ FP Precision >= 0.95
[Overall]      PR-AUC         = 0.89
[Reference]    F1-macro       = 0.83  (참고용)
```

### 8.x 모델 아티팩트 구조 (2026-03-19 확정)

`run_training.py`의 Step 9(label 모드) / Step 8(detection 모드)에서 `models/final/`에 자동 저장한다.

| 파일 | Label 모드 | Detection 모드 | 설명 |
|------|-----------|----------------|------|
| `best_model_v1.joblib` | O | — (detection_) | LightGBM 최종 모델 |
| `label_encoder.joblib` | O | — (detection_) | LabelEncoder |
| `feature_builder.joblib` | O | — (detection_) | FeatureBuilderSnapshot |
| `ood_detector.joblib` | O (조건부) | — (detection_, 조건부) | OODDetector (dense 피처 기반) |
| `calibrator.joblib` | O (조건부) | — | CalibratedClassifierCV |
| `feature_schema.json` | O | — (detection_) | 피처 수/뷰/dense 컬럼 목록 |
| **총 피처 수** | **~1,031** | **~8,031** | |

`detection_` 접두사 파일들은 `--source detection` 학습 시에만 생성된다.

### 8.y FeatureBuilderSnapshot 추론 인터페이스

학습 완료 후 자동 저장(`FeatureBuilderSnapshot.from_build_result(result)`). 추론/리포트 시 `builder.transform(df)` 호출로 학습 시 TF-IDF vocab 동결 보장.

```python
from src.models.feature_builder_snapshot import FeatureBuilderSnapshot

# 로드
builder = FeatureBuilderSnapshot.load("models/final/feature_builder.joblib")

# 추론 시 필요한 df 컬럼
# - Label 모드: file_name, file_path, server_name, meta/path 피처 컬럼들
# - Detection 모드: 위 + full_context_raw
X = builder.transform(df)  # TF-IDF vocab 동결, dense 컬럼 순서 보장
```

**주의:** `run_poc_report.py` Step 3에서 `FeatureBuilderSnapshot.transform()` 호출 전 반드시 `build_meta_features()` → `extract_path_features()` 순서로 df 컬럼을 준비해야 한다.

### 8.z Coverage-Precision 커브 정의 (확정)

```python
# P(FP) 기반 — top1_proba 사용 금지
# top1_proba = max(P(TP), P(FP))이므로 TP 고신뢰 예측도 포함 → precision 왜곡
_fp_cls_list = [c for c in le.classes_ if c != "TP"]
_fp_idx = list(le.classes_).index(_fp_cls_list[0])
ml_proba_fp = model.predict_proba(X)[:, _fp_idx]  # P(FP) 전용

# τ 스윕
for tau in np.arange(0.50, 1.01, 0.05):
    auto_fp_mask = ml_proba_fp >= tau          # P(FP) ≥ τ
    precision = (y_true[auto_fp_mask] != "TP").mean()
    coverage  = auto_fp_mask[y_true != "TP"].mean()
```

- `precision`: auto_fp로 분류된 건 중 실제 FP 비율 (오탐 방지 지표)
- `coverage`: 전체 실제 FP 중 자동 처리된 비율 (업무 효율 지표)
- 목표: `precision ≥ 0.95`를 만족하는 최소 τ에서 `coverage ≥ 0.40`
- **버그 수정 (2026-03-20):** `max_proba = max(P(TP), P(FP))` → `P(FP)` 전용으로 변경. max_proba 사용 시 TP 고신뢰 예측(P(TP)=0.95)도 auto_fp에 포함되어 precision이 FP 비율(~0.44) 수준으로 하락하는 오류 확인

**구현 위치:** `src/evaluation/poc_metrics.py` — `compute_coverage_precision_curve()`

---
