# 05. 모델 개발 (Model Development)

> **문서 버전**: 1.2
> **최종 수정일**: 2026-02-22
> **기반 문서**: docs/Architecture/ v1.2 §7~§9 (Stage S3a, S3b, S4)

## 1. 개요

### 1.1 문서 목적

본 문서는 Server-i 오탐 개선 PoC 프로젝트의 모델 개발 단계에서 적용할 방법론, 이론적 근거, 의사결정 기준을 정의한다. 단순히 "무엇을 할 것인가"가 아닌 "왜 그렇게 하는가"에 초점을 맞추어 PoC 수행 과정에서 일관된 의사결정이 가능하도록 한다.

### 1.2 모델링 목표 재정의 (v1.2)

Server-i DLP 시스템의 PII 검출 결과에서 오탐(False Positive)을 자동 분류하는 것이 최종 목표이다.

| 구분 | 정의 |
|------|------|
| Primary Task | Multi-class Classification (학습 8개 클래스) |
| Secondary Task | Binary Classification (정탐/오탐) via 후처리 |
| 입력 | raw_text + shape_text + path_text TF-IDF + Tabular (S2 Feature) |
| 출력 | 클래스별 확률 분포 + confidence + reason_code + risk_flag + ood_flag |

**v1.2 3-Layer 재정의 (Labeler 체인):**

```
기존: Layer 1~2 = Filter (데이터 제거) + Layer 3 = ML
v1.2: Layer 1~2 = RULE Labeler (고정밀 라벨 부여, 결과 보존)
      Layer 3   = ML Labeler (잔여 샘플 확률 기반 분류)
```

최종 출력은 항상 "라벨 + decision_source(RULE/ML/HYBRID) + 증거"다.

**Rationale**: Multi-class를 Primary로 설정하는 이유:
- 오탐의 "유형"을 파악해야 근본적인 개선 방향 도출 가능
- RULE Labeler로 처리된 건도 동일 스키마(라벨+사유+증거)를 따르므로 후속 처리 통일
- PoC 결과 기반으로 오탐 유형별 처리 규칙 정의에 활용
- UNKNOWN 라우팅으로 OOD 케이스 명시적 처리

---

## 2. Baseline 모델 전략

### 2.1 Baseline의 필요성

**이론적 배경**

ML 프로젝트에서 Baseline은 "이 문제가 얼마나 어려운가"를 측정하는 기준선이다. Baseline 없이 복잡한 모델의 성능만 보고하면, 해당 성능이 의미 있는 것인지 판단할 수 없다.

Google의 ML Engineering 가이드라인에서는 "Rule #1: Don't be afraid to launch a product without machine learning"을 강조한다. 이는 ML 모델이 Simple Heuristic 대비 유의미한 개선을 보여야만 도입 가치가 있다는 의미이다.

**Rationale**

Server-i 프로젝트에서 Baseline이 특히 중요한 이유:
1. 현재 시스템이 "검출 결과의 약 2/3 (약 66%)가 오탐"이라는 것은 이미 알려진 사실
2. 단순히 "모두 오탐"으로 예측하면 약 2/3 (약 66%) 수준 정확도 달성 가능 (무의미한 성능)
3. ML 모델이 이 Naive Baseline을 얼마나 초과하는지가 핵심 가치

### 2.2 Zero Rule Baseline

**정의**: 가장 빈번한 클래스로 모든 예측을 수행하는 전략

**적용 방법**:
- Multi-class: 가장 많은 오탐 유형으로 모든 샘플 예측
- Binary: 전체를 "오탐"으로 예측

**기대 성능 추정**:
- Binary Accuracy: 약 2/3 (약 66%) (오탐 비율과 동일)
- Multi-class Accuracy: 최다 빈도 오탐 클래스 비율과 동일

**해석 기준**: ML 모델이 Zero Rule 대비 유의미한 개선을 보이지 못하면, 해당 Feature Set 또는 모델 아키텍처 재검토 필요

### 2.3 Simple Heuristic Baseline

**정의**: 도메인 지식 기반의 규칙으로 예측을 수행하는 전략

**Server-i 맥락에서의 Heuristic 예시**:

| Heuristic | 로직 | 근거 |
|-----------|------|------|
| 경로 기반 | /test/, /sample/, /backup/ 포함 시 오탐 | 테스트/샘플 데이터가 오탐의 상당 부분 차지 |
| 패턴 반복 | 동일 패턴 10회 이상 반복 시 오탐 | 실제 개인정보는 반복 패턴으로 나타나지 않음 |
| 파일 형식 | .log, .tmp, .bak 확장자는 오탐 | 시스템 파일에서의 검출은 대부분 오탐 |
| 문서 키워드 | "테스트", "샘플", "예시" 포함 시 오탐 | 테스트 목적 문서 식별 |

**Rationale**: Simple Heuristic이 높은 성능을 보인다면:
- ML 모델 없이도 상당 부분 해결 가능 → 비용 효율적
- Heuristic을 Feature로 활용하여 ML 모델 성능 향상 가능
- 현업 담당자가 이해하기 쉬운 설명 가능한 로직 확보

**고려사항**:
- Heuristic 설계에 데이터 누수(Data Leakage) 주의 필요
- Test Set 정보를 참고하여 Heuristic을 만들면 과적합 발생
- EDA 단계에서 Train Set만으로 Heuristic 도출

### 2.4 Baseline 성능 목표 설정

| Baseline 유형 | 예상 Accuracy | 예상 F1 (정탐) | 역할 |
|--------------|---------------|----------------|------|
| Zero Rule | ~66% | 0.0 | 하한선 (이보다 낮으면 실패) |
| Simple Heuristic | TBD (EDA 후) | TBD | 중간 기준선 |
| ML 모델 | Target: 95%+ | Target: 0.7+ | 목표 성능 |

**의사결정 기준**:
- ML 모델 Accuracy < Zero Rule: Feature Engineering 전면 재검토
- ML 모델 Accuracy < Heuristic + 5%p: ML 도입 가치 재검토
- ML 모델 F1(정탐) < 0.5: 정탐 검출 능력 부족, 클래스 불균형 대응 강화

---

## 3. 모델 선정 근거

### 3.1 모델 후보군 분석

Server-i 프로젝트의 제약 조건(폐쇄망, CPU 환경, Text+Tabular 복합 입력)을 고려할 때, 아래 모델군이 후보가 된다.

#### 3.1.1 Tree 기반 앙상블 (Primary 후보)

**XGBoost (eXtreme Gradient Boosting)**

이론적 배경:
- Gradient Boosting의 정규화된 구현체
- L1(Lasso) + L2(Ridge) 정규화 내장으로 과적합 방지
- Weighted Quantile Sketch를 통한 효율적인 분할점 탐색
- Sparsity-aware 알고리즘으로 결측치 자동 처리

장점:
- 클래스 불균형에 대한 `scale_pos_weight` 파라미터 제공
- Feature Importance 추출 용이 (Gain, Cover, Weight)
- CPU 환경에서도 충분한 성능 (Histogram 기반 학습)

단점:
- LightGBM 대비 대용량 데이터에서 학습 속도 느림
- 메모리 사용량이 상대적으로 높음

**LightGBM (Light Gradient Boosting Machine)**

이론적 배경:
- Gradient-based One-Side Sampling (GOSS): 큰 Gradient를 가진 샘플에 집중
- Exclusive Feature Bundling (EFB): 희소 Feature 묶음으로 차원 축소
- Leaf-wise Growth: Level-wise 대비 손실 감소 최대화 분할

장점:
- XGBoost 대비 학습 속도 2-10배 빠름
- 메모리 효율성 우수
- 범주형 변수 직접 처리 가능 (`categorical_feature`)

단점:
- 과적합 위험이 상대적으로 높음 (Leaf-wise 특성)
- 소규모 데이터셋에서는 XGBoost와 성능 차이 미미

**CatBoost**

이론적 배경:
- Ordered Target Encoding으로 범주형 변수 처리
- Oblivious Decision Trees 사용으로 예측 속도 향상
- 학습 데이터의 시간 순서 고려한 Ordered Boosting

장점:
- 범주형 변수가 많은 경우 별도 인코딩 불필요
- 과적합에 상대적으로 강건
- GPU 지원 시 빠른 학습 가능

단점:
- 설치 파일 크기가 큼 (폐쇄망 반입 시 고려)
- 커뮤니티 및 문서가 상대적으로 적음

#### 3.1.2 선형 모델 (Baseline 보조)

**Logistic Regression**

역할:
- 해석 가능한 Baseline으로 활용
- Feature Importance의 방향성(양/음) 파악
- 규제 강도에 따른 Feature 선택 효과

적용 시나리오:
- TF-IDF 기반 Text Feature와 조합 시 효과적
- L1 규제로 불필요한 Feature 자동 제거

### 3.2 Text + Tabular 복합 입력 처리

**문제 정의**

Server-i 데이터는 Text(컨텍스트)와 Tabular(메타데이터)가 혼합된 형태이다. 이 두 유형의 Feature를 어떻게 결합하는지가 모델 성능에 직접적 영향을 미친다.

**접근 방식 비교**

| 접근법 | 설명 | 장점 | 단점 |
|--------|------|------|------|
| Feature Concatenation | Text Feature(TF-IDF 등)와 Tabular를 단순 연결 | 구현 간단, Tree 모델과 호환 | 스케일 차이로 Tree 분할 편향 가능 |
| Separate Models | Text/Tabular 각각 모델 학습 후 앙상블 | 각 유형에 최적화된 처리 | 복잡도 증가, 학습/추론 비용 증가 |
| Embedding + Tabular | Text를 임베딩으로 변환 후 Tabular와 결합 | 의미적 정보 보존 | GPU 필요, 폐쇄망 제약 |

**권장 접근법**: Feature Concatenation (PoC 단계)

Rationale:
- 폐쇄망 + CPU 환경에서 가장 실현 가능
- XGBoost/LightGBM은 Feature 스케일에 둔감 (분할 기반)
- Text에서 추출한 Rule-based Feature가 충분히 정보력 있을 가능성

고려사항:
- Text Feature 차원이 높아지면 과적합 위험 증가
- TF-IDF의 max_features 제한으로 차원 관리
- Feature Selection 단계에서 중요도 낮은 Feature 제거

### 3.3 모델 선정 의사결정 매트릭스

| 평가 기준 | 가중치 | XGBoost | LightGBM | CatBoost | Logistic |
|-----------|--------|---------|----------|----------|----------|
| 클래스 불균형 대응 | 25% | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| CPU 환경 성능 | 20% | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| 폐쇄망 설치 용이성 | 20% | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★★★ |
| Feature Importance | 15% | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★☆ |
| 해석 가능성 | 10% | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| 튜닝 복잡도 (낮을수록 좋음) | 10% | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |

**최종 권장**:
1. Primary: LightGBM (학습 속도 + 성능 균형)
2. Secondary: XGBoost (벤치마크 비교용)
3. Baseline: Logistic Regression (해석 가능한 참조점)

### 3.4 v1.2 3-Layer Labeler 아키텍처 (Stage S3a + S3b + S4)

docs/Architecture/ §2.3에 따라 3-Layer는 "필터"가 아니라 "라벨러 체인"으로 설계한다.

```
[Stage S3a: RULE Labeler — 고정밀 라벨러]
  입력: S2 Feature Prep 결과 (full_context_raw, path_text, pii_type_inferred 등)
  처리: 키워드/패턴 룰 매칭 (rules.yaml 기반)
  출력: primary_class + reason_code + evidence_spans + rule_confidence
  · 매칭된 건: 바로 S4로 전달 (decision_source="RULE")
  · 미매칭된 건: S3b로 전달

         │
         ▼

[Stage S3b: ML Labeler — 잔여 샘플 분류]
  입력: RULE이 처리 못한 샘플 + S2 Feature Matrix
  처리: LightGBM/XGBoost 8클래스 분류
         raw_text TF-IDF + shape_text TF-IDF + path_text TF-IDF + tabular
  출력: 8클래스 확률 분포 + calibrated_proba + OOD Score
  · decision_source="ML"

         │
         ▼

[Stage S4: Decision Combiner]
  Case 0: OOD Score 높거나 최대 확률 극하 → UNKNOWN 라우팅
  Case 1: RULE + ML 일치 → 해당 라벨 확정 (decision_source="HYBRID", 높은 신뢰도)
  Case 2: RULE + ML 불일치 → 보수적 처리 (RULE 우선 또는 TP override)
  Case 3: ML 단독 → TAU 임계값 기반 판정 (TAU_TP_OVERRIDE: 애매하면 TP)
  출력: final_label + confidence + risk_flag + ood_flag + decision_source
```

**핵심 설계 원칙 — "애매하면 TP":**

개인정보 유출 위험을 최소화하기 위해, 불확실한 케이스는 `TP-실제개인정보`로 보수적으로 처리한다.

```
confidence < TAU_TP_OVERRIDE → final_label = "TP-실제개인정보", risk_flag = 1
RULE ≠ ML 방향 → RULE 우선 또는 TP override
OOD Score > 임계값 → final_label = "UNKNOWN", ood_flag = 1
```

---

## 4. 클래스 불균형 대응 전략

### 4.1 문제 정의 및 영향 분석

**현황**:
- 전체 검출 결과의 약 2/3 (약 66%)가 오탐(FP)
- 정탐(TP)은 전체의 약 1/3 (약 34%)로 상대적 소수 클래스
- Multi-class 관점에서 각 오탐 유형별 분포도 불균형 예상

**클래스 불균형이 모델에 미치는 영향**:

| 영향 | 설명 | 결과 |
|------|------|------|
| Accuracy Paradox | 다수 클래스 예측만으로 높은 정확도 | 소수 클래스(정탐) 검출 실패 |
| Gradient Domination | 다수 클래스의 손실이 학습 방향 지배 | 소수 클래스 경계면 학습 부족 |
| Probability Calibration | 예측 확률이 다수 클래스로 편향 | 임계값 조정만으로는 해결 한계 |

**비즈니스 영향**:
- 정탐을 오탐으로 잘못 분류(FN) → 개인정보 유출 위험 (Critical)
- 오탐을 정탐으로 잘못 분류(FP) → 불필요한 파기 작업 (Moderate)

### 4.2 데이터 레벨 접근법

#### 4.2.1 언더샘플링 (Undersampling)

**개념**: 다수 클래스 샘플 수를 줄여 균형 맞춤

**방법론**:

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| Random Undersampling | 다수 클래스에서 무작위 제거 | 구현 간단, 학습 속도 향상 | 정보 손실, 대표성 훼손 |
| Cluster Centroids | 클러스터 중심점만 유지 | 중복 제거 효과 | 계산 비용, 정보 손실 |
| Tomek Links | 경계면 근처 다수 클래스 제거 | 결정 경계 명확화 | 적은 샘플만 제거 |
| NearMiss | 소수 클래스와 거리 기반 선택 | 어려운 샘플에 집중 | 노이즈에 민감 |

**Server-i 적용 고려사항**:
- 데이터 규모가 충분하다면 Random Undersampling 검토 가능
- 단, 정탐의 다양한 패턴을 학습하기 위해 정탐 샘플은 보존 필수
- 오탐 유형 간 불균형도 있으므로 Stratified Undersampling 적용

#### 4.2.2 오버샘플링 (Oversampling)

**개념**: 소수 클래스 샘플 수를 늘려 균형 맞춤

**방법론**:

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| Random Oversampling | 소수 클래스 단순 복제 | 구현 간단, 정보 손실 없음 | 과적합 위험, 동일 샘플 반복 |
| SMOTE | 소수 클래스 간 보간으로 합성 | 다양한 샘플 생성 | 노이즈 증폭, 비현실적 샘플 |
| SMOTE + Tomek | SMOTE 후 Tomek Links 제거 | 경계면 정리 효과 | 계산 비용 증가 |
| ADASYN | 어려운 영역에 더 많은 합성 | 경계면 집중 학습 | 노이즈 영역 과잉 생성 |

**SMOTE (Synthetic Minority Over-sampling Technique) 상세**:

이론적 배경:
1. 소수 클래스 샘플 x 선택
2. x의 k-최근접 이웃 중 하나 선택 (xnn)
3. x와 xnn 사이의 선분 상에 새로운 샘플 생성
4. x_new = x + λ × (xnn - x), where λ ∈ [0, 1]

고려사항:
- Text Feature에 SMOTE 적용 시 의미 없는 벡터 생성 가능
- TF-IDF 벡터의 보간이 실제 텍스트 의미를 반영하지 않음
- Tabular Feature에만 SMOTE 적용하거나, Feature별 분리 처리 검토

**Server-i 적용 권장**:
- 정탐(TP) 클래스에 대해 SMOTE 적용 검토
- 오탐 유형 중 극소수 클래스에도 선택적 적용
- 과적합 모니터링을 위해 Validation 성능 면밀히 추적

### 4.3 알고리즘 레벨 접근법

#### 4.3.1 Class Weight 조정

**개념**: 소수 클래스의 오분류에 더 큰 페널티 부여

**XGBoost/LightGBM 적용**:

XGBoost:
- `scale_pos_weight` 파라미터 (Binary 전용)
- 권장값: (negative 샘플 수) / (positive 샘플 수)
- Multi-class의 경우 Custom Objective Function 필요

LightGBM:
- `class_weight='balanced'` 자동 계산
- 또는 `class_weight` dict로 직접 지정
- `is_unbalance=True` 옵션으로 자동 처리

**이론적 근거**:
- 손실 함수에서 소수 클래스 샘플의 기여도 증가
- L(y, ŷ) = Σ wi × loss(yi, ŷi), where wi ∝ 1 / (클래스 빈도)
- Gradient 계산 시 소수 클래스 방향으로의 업데이트 강화

**고려사항**:
- 과도한 Weight는 다수 클래스 성능 저하 초래
- Grid Search로 최적 Weight 탐색 권장
- Binary와 Multi-class에서의 설정 방식 상이함 주의

#### 4.3.2 임계값 조정 (Threshold Tuning)

**개념**: 기본 임계값(0.5)을 조정하여 정탐/오탐 분류 기준 변경

**이론적 배경**:
- 분류 모델은 확률 P(Y=1|X)를 출력
- 기본적으로 P > 0.5면 양성(정탐)으로 분류
- 클래스 불균형 시 P 분포가 한쪽으로 치우침
- 임계값을 낮추면 Recall 증가, Precision 감소

**적용 방법**:
1. Validation Set에서 다양한 임계값으로 성능 측정
2. Precision-Recall Curve 분석
3. 비즈니스 목표에 맞는 최적 임계값 선정

**Server-i 맥락**:
- 정탐 Recall 우선 시: 임계값 낮춤 (정탐 놓치지 않도록)
- 오탐 Precision 우선 시: 임계값 높임 (불필요한 작업 최소화)
- Trade-off 분석 후 현업과 협의하여 결정

**고려사항**:
- 임계값 조정은 모델 자체를 개선하지 않음
- 근본적인 클래스 분리 능력 부족 시 효과 제한적
- Multi-class에서는 클래스별 임계값 조정 복잡성 증가

### 4.4 손실 함수 레벨 접근법

#### 4.4.1 Focal Loss

**이론적 배경**:
- Facebook(Meta)에서 Object Detection의 클래스 불균형 해결을 위해 제안
- 쉬운 샘플(잘 분류되는 다수 클래스)의 손실 기여도를 낮춤
- 어려운 샘플(경계면의 소수 클래스)에 집중

**수학적 정의**:
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

where:
- p_t: 정답 클래스의 예측 확률
- α_t: 클래스별 가중치 (Class Weight와 유사)
- γ: Focusing Parameter (γ=0이면 Cross-Entropy와 동일)
```

**γ(Gamma)의 역할**:
- γ = 0: 표준 Cross-Entropy
- γ = 2 (권장): 쉬운 샘플(p_t > 0.9) 손실 약 100배 감소
- γ > 5: 과도한 집중으로 학습 불안정

**XGBoost/LightGBM 적용**:
- Custom Objective Function으로 구현 필요
- 1차 미분(Gradient)과 2차 미분(Hessian) 직접 정의
- 구현 복잡도 증가하나 효과 검증 시 적용 가치 있음

**고려사항**:
- Focal Loss의 효과는 데이터셋에 따라 상이
- Class Weight 조정과 중복 적용 시 과보정 위험
- PoC에서는 Class Weight 우선 적용, Focal Loss는 옵션

#### 4.4.2 Cost-Sensitive Learning

**개념**: 오분류 유형별 비용을 다르게 설정

**비용 매트릭스 예시**:

|  | 예측: 정탐 | 예측: 오탐 |
|--|-----------|-----------|
| 실제: 정탐 | 0 | C_FN (높음) |
| 실제: 오탐 | C_FP (중간) | 0 |

**Server-i 맥락의 비용 설정**:
- C_FN (정탐 → 오탐 오분류): 개인정보 유출 위험 → 높은 비용
- C_FP (오탐 → 정탐 오분류): 불필요한 파기 작업 → 중간 비용

**적용 방법**:
- Sample Weight로 비용 반영
- Threshold 조정으로 간접 적용
- Multi-class에서는 클래스 쌍별 비용 정의 필요

### 4.5 클래스 불균형 대응 전략 종합

**권장 적용 순서** (누적 적용):

| 단계 | 방법 | 근거 |
|------|------|------|
| 1단계 | Stratified Sampling | 데이터 분할 시 클래스 비율 유지 |
| 2단계 | Class Weight 조정 | 구현 간단, 효과 검증 용이 |
| 3단계 | Threshold Tuning | 비즈니스 목표에 맞는 Trade-off 조정 |
| 4단계 (옵션) | SMOTE | 1-3단계로 부족 시 적용 |
| 5단계 (옵션) | Focal Loss | Custom 구현 필요, 효과 검증 후 결정 |

**의사결정 체크리스트**:
- [ ] 클래스별 샘플 수 확인 및 불균형 비율 정량화
- [ ] Stratified K-Fold 적용 여부 확인
- [ ] Class Weight 자동 계산 vs 수동 설정 결정
- [ ] Validation Set에서 임계값 탐색 범위 정의
- [ ] SMOTE 적용 시 Text/Tabular 분리 처리 여부 결정

---

## 5. 하이퍼파라미터 튜닝 전략

### 5.1 핵심 하이퍼파라미터 이해

#### 5.1.1 XGBoost 핵심 파라미터

**Tree 구조 관련**:

| 파라미터 | 설명 | 영향 | 권장 범위 |
|----------|------|------|-----------|
| max_depth | 트리 최대 깊이 | 깊을수록 복잡한 패턴 학습, 과적합 위험 | 3-10 |
| min_child_weight | 리프 노드 최소 샘플 가중치 합 | 높을수록 보수적, 과적합 방지 | 1-10 |
| gamma | 분할 최소 손실 감소량 | 높을수록 보수적, 가지치기 강화 | 0-5 |

**학습률 및 앙상블 관련**:

| 파라미터 | 설명 | 영향 | 권장 범위 |
|----------|------|------|-----------|
| learning_rate (eta) | 각 트리의 기여도 | 낮을수록 안정적, 더 많은 트리 필요 | 0.01-0.3 |
| n_estimators | 트리 개수 | 많을수록 복잡한 패턴 학습 | 100-1000 |
| subsample | 트리별 샘플링 비율 | 낮을수록 다양성 증가, 과적합 방지 | 0.6-1.0 |
| colsample_bytree | 트리별 Feature 샘플링 비율 | 낮을수록 다양성 증가 | 0.6-1.0 |

**정규화 관련**:

| 파라미터 | 설명 | 영향 | 권장 범위 |
|----------|------|------|-----------|
| reg_alpha (L1) | L1 정규화 강도 | Feature Selection 효과 | 0-10 |
| reg_lambda (L2) | L2 정규화 강도 | 가중치 크기 제한 | 0-10 |

#### 5.1.2 LightGBM 핵심 파라미터

**LightGBM 특화 파라미터**:

| 파라미터 | 설명 | XGBoost 대응 | 권장 범위 |
|----------|------|-------------|-----------|
| num_leaves | 트리당 최대 리프 수 | 2^max_depth | 20-100 |
| min_data_in_leaf | 리프 최소 샘플 수 | min_child_weight | 20-100 |
| max_bin | Feature 이산화 구간 수 | 없음 | 255 (기본값) |

**num_leaves vs max_depth 관계**:
- LightGBM은 Leaf-wise 성장으로 max_depth보다 num_leaves가 중요
- 경험적 규칙: num_leaves < 2^max_depth (과적합 방지)
- 예: max_depth=7이면 num_leaves < 128

### 5.2 탐색 전략

#### 5.2.1 Grid Search vs Random Search vs Bayesian Optimization

| 방법 | 설명 | 장점 | 단점 | 적용 시나리오 |
|------|------|------|------|--------------|
| Grid Search | 모든 조합 탐색 | 완전 탐색 보장 | 차원의 저주, 비효율적 | 파라미터 2-3개, 범위 좁음 |
| Random Search | 무작위 조합 탐색 | 효율적, 넓은 탐색 | 최적해 보장 없음 | 초기 탐색, 파라미터 많음 |
| Bayesian Optimization | 이전 결과 기반 탐색 | 효율적, 수렴 빠름 | 구현 복잡, 라이브러리 의존 | 평가 비용 높을 때 |

**이론적 배경 - Random Search 우위성 (Bergstra & Bengio, 2012)**:
- 대부분의 ML 문제에서 일부 파라미터만 성능에 크게 영향
- Grid Search는 중요하지 않은 파라미터에도 균등 할당
- Random Search는 중요 파라미터의 다양한 값 탐색에 유리

#### 5.2.2 단계적 튜닝 전략

**Phase 1: 빠른 범위 탐색**

목적: 대략적인 최적 범위 파악
방법: Random Search, 적은 반복 횟수
파라미터: 핵심 파라미터만 (learning_rate, max_depth, n_estimators)
설정: n_iter=20-50, cv=3

**Phase 2: 정밀 탐색**

목적: 최적 범위 내에서 세밀한 조정
방법: Grid Search 또는 Bayesian Optimization
파라미터: Phase 1 결과 주변 + 정규화 파라미터
설정: cv=5, 좁은 범위

**Phase 3: 최종 검증**

목적: 최적 파라미터로 최종 모델 학습
방법: 전체 학습 데이터로 학습, Hold-out 검증
추가: Early Stopping 적용

### 5.3 리소스 제약 고려

**폐쇄망 환경에서의 튜닝 전략**:

| 제약 | 대응 방안 |
|------|----------|
| Optuna 등 라이브러리 설치 어려움 | sklearn의 RandomizedSearchCV 활용 |
| 병렬 처리 제한 | n_jobs 조정, 순차 실행 |
| 시간 제약 | Early Stopping 적극 활용, 반복 횟수 제한 |

**컴퓨팅 리소스 최적화**:

| 전략 | 설명 | 효과 |
|------|------|------|
| Subsampling | 튜닝 시 전체 데이터의 일부만 사용 | 탐색 속도 향상 |
| Early Stopping | 성능 개선 없으면 조기 종료 | 불필요한 반복 방지 |
| Warm Start | 이전 모델 가중치 재사용 | 수렴 속도 향상 |

### 5.4 Early Stopping 전략

**개념**: Validation 성능이 개선되지 않으면 학습 조기 종료

**적용 방법**:

XGBoost:
- `early_stopping_rounds` 파라미터
- `eval_set` 지정 필수
- 권장값: 50-100 (learning_rate에 비례)

LightGBM:
- `callbacks=[early_stopping(stopping_rounds=N)]`
- 또는 `early_stopping_round` 파라미터

**고려사항**:
- Early Stopping은 Validation Set에 과적합될 수 있음
- 최종 모델 학습 시에는 전체 데이터 사용 권장
- 최적 n_estimators를 Early Stopping으로 결정 후, 해당 값으로 재학습

### 5.5 하이퍼파라미터 튜닝 의사결정 체크리스트

**튜닝 전**:
- [ ] Baseline 모델 성능 확보 (튜닝 효과 비교 기준)
- [ ] 탐색할 파라미터 우선순위 결정
- [ ] 리소스 예산 (시간, 메모리) 설정
- [ ] 평가 지표 및 CV 전략 확정

**튜닝 중**:
- [ ] 파라미터 간 상호작용 모니터링
- [ ] 과적합 징후 확인 (Train-Validation Gap)
- [ ] 탐색 범위 적절성 검토 (경계값에서 최적 발견 시 확장)

**튜닝 후**:
- [ ] 최적 파라미터 재현 가능성 확인
- [ ] Holdout Test Set에서 최종 검증
- [ ] 파라미터 설정 근거 문서화

---

## 6. 앙상블 전략

### 6.1 앙상블의 이론적 기반

**다양성(Diversity)과 성능의 관계**:

앙상블이 효과적이려면 개별 모델들이:
1. 단독으로도 일정 수준 이상의 성능을 가져야 함
2. 서로 다른 오류 패턴을 보여야 함 (다양성)

**수학적 근거 - Bias-Variance Decomposition**:
```
E[(y - f̂(x))²] = Bias²(f̂) + Var(f̂) + σ²

앙상블 효과:
- Bagging: Variance 감소 (평균화 효과)
- Boosting: Bias 감소 (순차적 오류 보정)
```

### 6.2 앙상블 방법론

#### 6.2.1 Voting (투표 기반)

**Hard Voting**: 다수결로 최종 클래스 결정
- 구현 간단
- 확률 정보 손실

**Soft Voting**: 확률 평균으로 최종 클래스 결정
- 확률 정보 활용
- 개별 모델의 Calibration 중요

**Weighted Voting**: 모델별 가중치 부여
- 성능 좋은 모델에 높은 가중치
- 가중치 최적화 필요

#### 6.2.2 Stacking (적층 기반)

**개념**:
1. Level 0: 여러 Base 모델 학습
2. Level 1: Base 모델 예측을 입력으로 Meta 모델 학습

**장점**:
- 서로 다른 알고리즘의 강점 결합
- 비선형적 모델 결합 가능

**단점**:
- 과적합 위험 높음 (특히 소규모 데이터)
- 학습/추론 시간 증가
- 구현 복잡도 증가

**고려사항**:
- Level 0 모델 학습 시 Hold-out 또는 K-Fold Out-of-Fold 예측 사용 필수
- Train 데이터로 예측하면 Data Leakage 발생

#### 6.2.3 Blending

**개념**: Stacking의 단순화 버전
- Level 0 모델을 Train Set으로 학습
- Validation Set의 예측으로 Meta 모델 학습

**Stacking과 차이**:
- CV 불필요로 구현 간단
- 데이터 활용 효율성 낮음

### 6.3 Server-i 프로젝트 앙상블 전략

**권장 접근법**: Weighted Soft Voting

Rationale:
- PoC 단계에서 복잡한 Stacking은 과도함
- 모델 간 다양성 확보 (LightGBM + XGBoost + Logistic)
- 확률 기반 결합으로 Threshold Tuning 유연성 유지

**앙상블 구성 예시**:

| 모델 | 역할 | 예상 가중치 |
|------|------|-------------|
| LightGBM | Primary (성능) | 0.5 |
| XGBoost | Secondary (안정성) | 0.3 |
| Logistic Regression | Baseline (해석성) | 0.2 |

**가중치 결정 방법**:
1. Validation Set 성능 비례 가중치
2. 또는 Grid Search로 최적 가중치 탐색
3. 단순 평균(동일 가중치)도 baseline으로 검토

### 6.4 앙상블 적용 시 고려사항

**언제 앙상블이 효과적인가**:
- 개별 모델 성능이 비슷할 때
- 모델 간 오류 패턴이 다를 때
- 충분한 데이터가 있을 때

**언제 앙상블을 피해야 하는가**:
- 단일 모델이 압도적으로 좋을 때
- 추론 시간/리소스 제약이 클 때
- 모델 해석성이 중요할 때

**PoC 단계 권장**:
- 개별 모델 성능 먼저 최적화
- 앙상블은 추가 개선이 필요할 때 적용
- 복잡한 앙상블(Stacking)은 현재 PoC 범위에서 제외

---

## 7. Multi-class → Binary 후처리

### 7.1 후처리 필요성

**문제 상황**:
- 모델 출력: 8개 클래스의 확률 분포
- 비즈니스 요구: 정탐/오탐 이진 판단

**후처리 없이 Binary 모델만 학습하면?**:
- 오탐 유형별 패턴 정보 손실
- 오분류 원인 분석 어려움
- 추후 오탐 유형별 자동화 어려움

### 7.2 후처리 방법론

#### 7.2.1 확률 집계 (Probability Aggregation)

**방법**: 클래스별 확률을 정탐/오탐 그룹으로 합산

```
P(정탐) = P(TP 클래스)
P(오탐) = Σ P(FP 클래스 i)  for all i
```

**장점**:
- Multi-class 모델의 확률 calibration 보존
- 임계값 조정 유연성 유지

**단점**:
- 개별 FP 클래스의 구분 정보 손실 (Binary 결과에서)

#### 7.2.2 Top-1 클래스 매핑

**방법**: 가장 높은 확률의 클래스를 선택 후 정탐/오탐으로 매핑

```
예측 클래스 = argmax(P(클래스 i))
최종 분류 = "정탐" if 예측 클래스 == TP else "오탐"
```

**장점**:
- 구현 간단
- 오탐 유형 정보 활용 가능

**단점**:
- 확률 크기 정보 손실
- 임계값 조정 어려움

### 7.3 권장 후처리 전략

**2단계 의사결정 구조**:

1단계 (Multi-class 예측):
- 모든 클래스의 확률 분포 출력
- 오탐 유형 정보 보존

2단계 (Binary 변환):
- P(정탐) = P(TP)
- P(오탐) = 1 - P(TP) = Σ P(FP_i)
- 임계값 τ 적용: P(정탐) > τ → 정탐

**임계값 τ 결정**:
- 기본값: 0.5
- Precision-Recall Trade-off에 따라 조정
- Validation Set에서 최적 τ 탐색

### 7.4 후처리 시 고려사항

**확률 Calibration**:
- Multi-class 모델의 출력 확률이 실제 확률을 반영하는지 확인
- Tree 모델은 일반적으로 Calibration이 좋지 않음
- Platt Scaling 또는 Isotonic Regression으로 보정 검토

**오탐 유형 정보 활용**:
- Binary 결과와 함께 Top-N 오탐 유형도 제공
- 담당자가 왜 오탐으로 판단했는지 이해 가능
- 예: "오탐 (사유: FP-더미데이터 85%, FP-숫자나열/코드 10%)"

---

## 8. 의사결정 포인트 및 체크리스트

### 8.1 모델 개발 단계별 의사결정 포인트

| 단계 | 의사결정 포인트 | 결정 기준 |
|------|----------------|----------|
| Baseline | Heuristic Baseline 성능이 높은가? | 높으면 ML 필요성 재검토 |
| 클래스 불균형 | SMOTE 필요한가? | Class Weight만으로 부족 시 적용 |
| 튜닝 | 탐색 범위 적절한가? | 경계에서 최적 발견 시 확장 |
| 앙상블 | 앙상블 효과 있는가? | 단일 모델 대비 개선 미미 시 생략 |

### 8.2 PoC 완료 전 필수 체크리스트

**모델 성능**:
- [ ] Zero Rule Baseline 대비 유의미한 개선 확인
- [ ] 정탐 클래스 F1 Score 목표 달성 여부
- [ ] Validation과 Test 성능 Gap 확인 (과적합 여부)

**재현성**:
- [ ] 모든 하이퍼파라미터 문서화
- [ ] Random Seed 고정
- [ ] 학습 데이터 버전 관리

**확장성**:
- [ ] 모듈화된 코드 구조
- [ ] 새로운 오탐 유형 추가 용이성 확인
- [ ] 모델 업데이트 프로세스 정의

### 8.3 Risk 및 대응 방안

| Risk | 발생 조건 | 대응 방안 |
|------|----------|----------|
| 과적합 | Train >> Validation 성능 | 정규화 강화, Early Stopping |
| 저성능 | Baseline과 유사한 성능 | Feature Engineering 재검토 |
| 클래스 불균형 미해결 | 정탐 F1 < 0.5 | SMOTE, Focal Loss 적용 |
| 학습 시간 과다 | 폐쇄망 리소스 제약 | 데이터 샘플링, 모델 단순화 |

---

## 9. v1.2 자동화 확장 (S3a/S3b/S4)

### 9.1 Auto-Rule-Promoter

docs/Architecture/ §7.6 기준으로, ML이 반복적으로 고확신 분류한 패턴을 자동으로 RULE로 승격한다.

**승격 조건:**

- 최근 3개월 누적 패턴
- `ml_top1_proba >= 0.90`
- 동일 패턴 관측 `>= 500건`
- holdout 정밀도 `>= 0.95`

**비활성화 조건:**

- 기존 룰 `precision_lb < 0.70`이면 삭제가 아닌 `active=false`로 자동 비활성화

| 단계 | 입력 | 출력 |
|------|------|------|
| 후보 추출 | ML 예측 이력 | 후보 패턴 리스트 |
| holdout 검증 | 후보 패턴 + 검증셋 | 채택/탈락 |
| 룰 반영 | 채택 후보 | `rules.yaml` 자동 갱신 + 감사 로그 |

### 9.2 Auto-Model-Selector

docs/Architecture/ §8.13 기준으로, 후보 모델 간 비교/선정을 수동이 아닌 자동 평가로 수행한다.

**선택 로직:**

1. Group+Time Split 기준 후보 모델 전체 평가
2. 안전 제약 `TP Recall >= 0.95` 미충족 모델 탈락
3. 제약 충족 후보 중 `macro_f1` 최대 모델 선택
4. 전 후보 탈락 시 `tp_recall` 최고 모델로 fallback

| 항목 | 기준값 | 목적 |
|------|--------|------|
| 안전 제약 | `TP Recall >= 0.95` | 개인정보 누락 방지 |
| 주목표 | `macro_f1` 최대화 | 다중 클래스 균형 성능 |
| 기록 | 선택/탈락 사유 자동 로그 | 감사 가능성 확보 |

### 9.3 Decision Combiner 자동 판정 연계

#### 9.3.1 Auto-Adjudicator

`risk_flag=NEEDS_REVIEW` 케이스를 4단 자동 판정으로 처리한다. (docs/Architecture/ §9.5)

1. File-level consensus
2. RULE↔ML cross agreement
3. Historical pattern match
4. 실패 시 보수적 TP 확정 (`AUTO_CONSERVATIVE_TP`)

#### 9.3.2 UNKNOWN (OOD) 자동 처리

`UNKNOWN` 케이스는 사람이 검토하지 않고 UnknownAutoProcessor로 자동 처리한다. (docs/Architecture/ §9.6)

- FP centroid와 거리 임계값 이내면 해당 FP 클래스로 자동 배정 (`OOD_AUTO_ASSIGN`)
- 그렇지 않으면 TP 보수 처리 (`OOD_UNKNOWN_TO_TP`)

#### 9.3.3 Auto-Tuner

docs/Architecture/ §9.7 기준으로 5개 TAU 임계값을 자동 탐색한다.

| 제약 조건 | 기준 |
|-----------|------|
| `tp_recall` | `>= 0.95` |
| `fp_precision_est` | `>= 0.90` |
| `review_rate` | `<= 0.30` |

안전장치: 새 임계값으로 `tp_recall`이 하락하면 즉시 기존 임계값으로 유지(롤백).

### 9.4 PoC 적용 범위

| 구분 | 적용 대상 |
|------|-----------|
| PoC 내 구현(Tier 1) | `Auto-Adjudicator`, `UNKNOWN (OOD) 자동 처리`, `Auto-Model-Selector` |
| PoC 이후 확장(Tier 2~3) | `Auto-Tuner`, `Auto-Rule-Promoter`의 운영 자동화 완전 적용 |

---

## 부록 A: 용어 정의

| 용어 | 정의 |
|------|------|
| 정탐 (TP) | 실제 개인정보를 개인정보로 올바르게 검출 |
| 오탐 (FP) | 개인정보가 아닌 것을 개인정보로 잘못 검출 |
| 미탐 (FN) | 실제 개인정보를 검출하지 못함 |
| Baseline | 성능 비교의 기준이 되는 단순 모델 또는 규칙 |
| Calibration | 예측 확률이 실제 확률을 얼마나 잘 반영하는지 |

## 부록 B: 참고 문헌

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
2. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
3. Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR.
4. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
5. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.
6. Google ML Engineering Best Practices. https://developers.google.com/machine-learning/guides/rules-of-ml
