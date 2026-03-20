# Server-i 오탐 개선 PoC 프로젝트 Master Plan

> **문서 버전**: 1.2
> **최종 수정일**: 2026-02-22
> **기반 문서**: docs/Architecture/ v1.2 (2026-02-22)
> **문서 목적**: PoC 프로젝트의 전체 구조, 단계별 의존관계, 의사결정 기준을 정의하여 체계적 수행을 보장

---

## 1. Executive Summary

### 1.1 프로젝트 개요

Server-i DLP 솔루션의 PII(Personal Identifiable Information) 검출 결과 중 약 2/3 (약 66%)가 오탐(False Positive)으로 추정되며, 이로 인해 보안 담당자의 검토 부담이 과중하고 실질적인 개인정보 유출 위험 대응이 지연되고 있다. 본 PoC는 머신러닝 기반 오탐 자동 분류 모델의 기술적 타당성을 검증하는 것을 목표로 한다.

### 1.2 PoC 범위 및 한계

**범위 내(In-Scope)**:
- 레이블링된 데이터 기반 오프라인 모델 학습 및 평가
- 핵심 Feature 식별 및 모델 성능 검증
- CPU 기반 ML 파이프라인 고도화 (GPU 미보유 환경 기준)
- 합성 변수 확장 기반 ML(기본) + 선택형 3-Layer Labeler 체인 파이프라인 검증
- Zero-Human-in-the-Loop Tier 1 자동화 구현: `Auto-Adjudicator`, `UNKNOWN (OOD) 자동 처리`, `Auto-Model-Selector`, `Confident Learning` 감사

> **v1.2 변경:** "3-Layer Filter"에서 "3-Layer Labeler"로 재정의. Layer 1~2는 데이터를 제거(필터)하지 않고, 라벨+사유코드+증거+신뢰도를 포함한 결과를 출력하는 **라벨러(Labeler)**다. (docs/Architecture/ §2.3 참조)

**범위 외(Out-of-Scope)**:
- 실시간 추론 파이프라인 구축
- Zero-Human-in-the-Loop Tier 2~3 자동화 (`Auto-Tuner`, `Self-Validation Loop`, `Auto-Remediation Playbook`, `Auto-Retrainer`, `Auto-Schema-Detector`, `Auto-Mapper`, `Auto-Taxonomy-Manager`)
- 현업 수동 피드백 루프 구현 (v1.2에서는 RULE↔ML 기반 자동 검증으로 대체)
- Dataset C (현업 피드백 데이터) 활용: 비정형 사유 구조화 공수 대비 효익 낮음 → **v1.2에서 범위 제외 확정**

### 1.3 성공 기준 요약

| 구분 | 기준 | 근거 |
|------|------|------|
| 기술적 타당성 | Multi-class F1-macro ≥ 0.70 | 클래스 불균형 환경에서 전체 클래스 균형 성능을 반영 |
| 보안 안정성 | 정탐(TP) Recall ≥ 0.75 | 실제 개인정보 누락(FN) 위험을 제어 |
| 운영 효율성 | 오탐(FP) Precision ≥ 0.85 | 잘못된 오탐 분류를 최소화하여 운영 부담 절감 |
| 재현 가능성 | 모듈화된 파이프라인 구조 확보 | 동일 PoC 절차 재실행 용이 |

---

## 2. 문제 정의 및 이론적 배경

### 2.1 문제 유형 분류

본 PoC의 ML Task는 **Multi-class Classification**이며, 최종 출력은 **Binary Decision**(정탐/오탐)으로 후처리된다.

**v1.2 클래스 체계:**

| 클래스 구분 | 수 | 용도 |
|------------|---|------|
| 학습 대상 클래스 (Canonical 8개) | 8 | 모델 학습 타겟 |
| 운영 후처리 클래스 | 1 (`UNKNOWN`) | OOD/극도 불확실 케이스 라우팅 (학습 클래스 아님) |

`UNKNOWN`은 모델이 학습하는 클래스가 아니라, Stage S4(Decision Combiner)에서 OOD Score 또는 신뢰도 극하 시 자동 라우팅되는 **후처리 정책 클래스**다.

**Multi-class 접근의 이론적 근거**:

직접적인 Binary Classification(정탐 vs 오탐) 대신 Multi-class 접근을 선택한 이유는 **오탐의 이질성(Heterogeneity)** 때문이다. 오탐은 단일한 특성을 가진 집단이 아니라, 테스트 데이터, 샘플 코드, 시스템 로그 등 서로 다른 생성 메커니즘을 가진 이질적 하위 집단의 합집합이다.

통계학적으로, 이질적 집단을 단일 클래스로 묶으면 **Simpson's Paradox** 유사 현상이 발생할 수 있다. 각 하위 집단에서는 특정 Feature가 강한 예측력을 가지더라도, 집단을 합치면 그 관계가 희석되거나 역전될 수 있다. Multi-class 접근은 각 오탐 유형의 고유한 패턴을 학습하여 이 문제를 완화한다.

**Binary 후처리의 실용적 근거**:

최종 비즈니스 의사결정은 "이 검출 결과를 담당자에게 보여줄 것인가?"라는 Binary 질문이다. Multi-class 예측 결과를 Binary로 매핑하면:
- 정탐(TP) 클래스 예측 → 담당자 검토 필요
- 모든 FP 클래스 예측 → 자동 분류/후처리

이 2단계 접근은 **Hierarchical Classification**의 단순화된 형태로, 세부 오탐 유형 정보를 보존하면서도 명확한 액션 기준을 제공한다.

### 2.2 클래스 불균형 문제

**현황 분석**:

전체 데이터의 약 2/3 (약 66%)가 오탐이라는 것은 정탐:오탐 ≈ 1:2 수준의 클래스 불균형을 의미한다. 더 복잡한 것은 Multi-class 관점에서 각 오탐 하위 클래스의 분포 역시 불균형할 가능성이 높다는 점이다.

**이론적 배경**:

클래스 불균형이 ML 모델에 미치는 영향은 크게 세 가지로 구분된다:

1. **학습 편향(Learning Bias)**: 대부분의 학습 알고리즘은 전체 오류를 최소화하도록 설계되어 있어, 다수 클래스에 편향된 결정 경계를 학습한다. 이는 Loss Function이 각 샘플에 동일한 가중치를 부여하기 때문이다.

2. **평가 메트릭 왜곡**: Accuracy는 불균형 데이터에서 무의미하다. 오탐 비율이 약 2/3인 데이터에서 모든 샘플을 오탐으로 예측해도 약 2/3 정확도를 달성한다.

3. **결정 경계의 불안정성**: 소수 클래스 근처의 결정 경계가 소수의 샘플에 의해 결정되어 일반화 성능이 저하된다.

**대응 전략 옵션**:

| 전략 | 메커니즘 | 적용 시점 | 고려사항 |
|------|----------|-----------|----------|
| **Class Weight** | Loss 계산 시 소수 클래스에 높은 가중치 부여 | 학습 | 가중치 설정이 휴리스틱에 의존 |
| **SMOTE** | 소수 클래스의 합성 샘플 생성 | 전처리 | 고차원 희소 데이터(TF-IDF)에서 비효과적일 수 있음 |
| **Stratified Sampling** | 학습/검증 분할 시 클래스 비율 유지 | 전처리 | 필수 적용 사항 |
| **Threshold Tuning** | 예측 확률 임계값 조정 | 후처리 | Precision-Recall Trade-off 제어 |
| **Focal Loss** | 쉬운 샘플의 Loss 기여도 감소 | 학습 | XGBoost/LightGBM 기본 미지원, 커스텀 구현 필요 |

**PoC 권장 접근**:

PoC 단계에서는 Stratified Sampling(필수) + Class Weight(기본) 조합으로 시작하고, 성능이 부족할 경우 SMOTE 또는 Threshold Tuning을 순차 적용한다. Focal Loss는 구현 복잡도 대비 PoC 효용이 불확실하므로 현재 범위에서는 제외한다.

### 2.3 평가 메트릭 선정

**메트릭 선정의 이론적 프레임워크**:

ML 메트릭 선정은 **비즈니스 비용 구조**에서 출발해야 한다. 본 PoC에서 발생 가능한 오류와 그 비용은:

| 오류 유형 | 설명 | 비즈니스 비용 |
|-----------|------|---------------|
| **False Negative (Type II)** | 정탐을 오탐으로 잘못 분류 | **높음** - 실제 개인정보 유출 건을 놓침 |
| **False Positive (Type I)** | 오탐을 정탐으로 잘못 분류 | **중간** - 불필요한 담당자 검토 발생 |

이 비대칭적 비용 구조는 **Recall(재현율)**의 중요성을 시사한다. 그러나 Recall만 최적화하면 모든 것을 정탐으로 예측하는 trivial solution이 발생한다. 따라서 Precision과의 균형이 필요하다.

**선정 메트릭 및 근거**:

1. **Primary: F1-macro**
   - 근거: 모든 클래스에 동일한 가중치를 부여하여 소수 클래스 성능을 명시적으로 반영
   - 계산: 각 클래스의 F1을 계산 후 단순 평균
   - 목표: ≥ 0.70

2. **Secondary: Recall@정탐 클래스**
   - 근거: 실제 정탐 중 정탐으로 예측된 비율 = 놓치는 개인정보 비율 제어
   - 목표: ≥ 0.75

3. **Secondary: Precision@오탐 클래스**
   - 근거: 오탐으로 예측한 건 중 실제 오탐의 비율 = 자동 분류 신뢰도
   - 목표: ≥ 0.85

4. **Monitoring/Diagnostic: Confusion Matrix, Classification Report, PR-AUC**
   - 근거: 클래스별 성능 분포 파악, 특정 오탐 유형의 오분류 패턴 식별

**Precision-Recall Trade-off 관리**:

정탐 Recall과 오탐 Precision은 본질적으로 Trade-off 관계에 있다. 본 PoC에서는 정탐 누락 방지(Recall)와 자동 분류 신뢰도(오탐 Precision)를 함께 만족하도록 운영한다.

```
Maximize: F1-macro
Subject to: Recall@정탐 ≥ 0.75, Precision@오탐 ≥ 0.85
```

로 정식화할 수 있으며, 실무적으로는 Threshold Tuning을 통해 이 균형점을 탐색한다.

---

## 3. 데이터 전략

### 3.1 데이터 유형 및 특성

**v1.2 데이터셋 현황:**

| 데이터셋 | 상태 | 설명 |
|---------|------|------|
| Dataset A (Server-i 검출 원본) | ✅ 사용 | PII 검출 이벤트 원본 (마스킹 완료) |
| Dataset B (소만사 레이블링) | ✅ 사용 | 7개 오탐 클래스 레이블링 완료 |
| Dataset C (현업 피드백) | ❌ **v1.2 범위 제외** | 비정형 사유 구조화 공수 대비 효익 낮음 |

> **⚠️ 패턴 종류 필드 신뢰도 이슈:** Dataset B의 `패턴 종류` 필드는 직접 신뢰 불가. 이메일(`****@redhat.com`)임에도 주민번호로 표기된 케이스 다수 발견. → 모델 피처로 직접 사용 금지, `pii_type_inferred`(검출 내역 기반 재추론) 사용.

본 PoC의 입력 데이터는 **멀티모달(Multi-modal)** 특성을 가진다:

| 데이터 유형 | 설명 | 특성 |
|-------------|------|------|
| **Text (원본)** | PII 검출 위치의 원본 텍스트 + 앞뒤 컨텍스트 | 비정형, 고차원 |
| **Text (마스킹)** | Asterisk 처리된 텍스트 + 컨텍스트 | 비정형, 패턴 정보 손실 가능 |
| **Tabular** | 파일 경로, 확장자, 서버 정보, 검출 메타데이터 | 정형, 범주형+수치형 혼합 |

**멀티모달 학습의 이론적 고려사항**:

멀티모달 데이터를 다루는 방법은 크게 세 가지이다:

1. **Early Fusion**: 모든 Feature를 단일 벡터로 결합 후 학습
   - 장점: 구현 단순, Feature 간 상호작용 학습 가능
   - 단점: 차원의 저주, 스케일 불일치 문제

2. **Late Fusion**: 각 모달리티별 모델 학습 후 예측 결합
   - 장점: 모달리티별 최적화 가능
   - 단점: Feature 간 상호작용 학습 불가

3. **Intermediate Fusion**: 각 모달리티의 중간 표현을 결합
   - 장점: 상호작용 학습 + 모달리티별 최적화 균형
   - 단점: 아키텍처 복잡도 증가

**PoC 권장 접근**:

Gradient Boosting 기반 접근에서는 **Early Fusion**이 자연스러운 선택이다. Tree 기반 모델은 Feature 스케일에 불변하고, 자동으로 Feature 간 상호작용을 학습한다. Text Feature를 수치화(TF-IDF, 키워드 존재 여부 등)한 후 Tabular Feature와 결합하여 단일 입력 행렬을 구성한다.

### 3.2 원본 vs 마스킹 데이터 선택

**의사결정 프레임워크**:

| 기준 | 원본 텍스트 | 마스킹 텍스트 |
|------|-------------|---------------|
| **정보량** | 높음 (PII 패턴 직접 학습 가능) | 낮음 (패턴 정보 손실) |
| **개인정보 노출 위험** | 높음 | 낮음 |
| **모델 일반화** | 낮을 수 있음 (특정 PII 값에 과적합 위험) | 높을 수 있음 |
| **운영 환경 일치성** | 확인 필요 | 확인 필요 |

**핵심 고려사항**:

원본/마스킹 선택은 단순히 성능 비교의 문제가 아니라, **Train-Serving Skew** 관점에서 접근해야 한다. 학습 데이터와 추론 시점의 데이터 형태가 다르면 성능 저하가 발생한다.

질문해야 할 사항:
- 실제 운영 환경에서 모델에 입력되는 데이터는 원본인가, 마스킹인가?
- 마스킹 처리는 Server-i 내부에서 이루어지는가, 모델 파이프라인에서 이루어지는가?

**권장 접근**:

운영 환경과 동일한 형태의 데이터로 학습하되, PoC에서는 두 가지 모두 실험하여 성능 차이를 측정한다. 만약 원본 데이터가 유의미하게 우수하다면, 모델 학습 환경의 보안 강화(접근 제어, 로깅)를 통해 원본 사용을 정당화할 수 있다.

### 3.3 PK 기반 원본 매핑 전략

**설계 원칙**:

Feature Engineering 과정에서 원본 데이터는 TF-IDF, 파생 변수 등으로 변환되어 원본 복원이 불가능해진다. 그러나 최종 예측 결과는 담당자에게 원본 컨텍스트와 함께 전달되어야 한다.

**해결 방안**:

모든 데이터 처리 단계에서 **Primary Key(PK)**를 유지하고, 예측 결과와 원본 데이터를 PK로 조인한다.

```
[데이터 흐름]
원본 데이터 (PK 포함)
    ├──→ Feature Engineering ──→ 모델 추론 ──→ 예측 결과 (PK 포함)
    │                                                    │
    └──→ 원본 저장소 ←─────────────────────── PK 기반 JOIN
                                                    │
                                            담당자 통보 + 후속 처리
```

**이론적 근거**:

이 설계는 **Separation of Concerns** 원칙을 따른다:
- 모델은 예측에만 집중 (Feature 공간에서 작업)
- 원본 데이터는 별도 관리 (데이터 무결성 보장)
- 예측 결과와 원본의 연결은 PK로 수행 (일관된 인터페이스)

**구현 시 고려사항**:

1. **PK 유일성 보장**: 중복 PK가 없는지 데이터 검증 단계에서 확인
2. **PK 누락 방지**: 파이프라인 각 단계에서 PK 컬럼 유지 여부 검증
3. **JOIN 성능**: 대용량 데이터의 경우 인덱싱 또는 파티셔닝 고려

---

## 4. 모델링 전략

### 4.1 모델링 접근법

2026-01 회의록 기준 GPU 미보유가 확정되어 현재 PoC는 CPU 기반 Gradient Boosting 파이프라인으로 진행한다.

| 구분 | 내용 |
|------|------|
| **Text 처리** | 멀티뷰 TF-IDF (`raw_text` + `shape_text` + `path_text`) + 키워드/패턴 매칭 |
| **모델** | XGBoost, LightGBM (CPU 기반 Gradient Boosting) |
| **학습 기본값** | 필터 미적용 + 합성변수 OFF(Tier 0); `--synth-tier safe/aggressive` 선택 적용 가능 |
| **학습 시간** | 상대적으로 짧음 |
| **추론 시간** | 상대적으로 짧음 |
| **기대 성능** | 중간 (패턴 기반) |

**v1.2 Feature Engineering 멀티뷰 접근:**

| Feature 뷰 | 설명 | 목적 |
|-----------|------|------|
| `raw_text` | 소문자화 + 고엔트로피 토큰 placeholder 치환 (`<NUM10>`, `<HASH>`, `<HEX>`, `<MASK>`) | 키워드/도메인 신호 |
| `shape_text` | 숫자→`0`, 영문→`a`, 한글→`가`, 구분자 유지 | 구조적 패턴 신호 |
| `path_text` | 파일 경로를 `/`, `.`, `_` 기준 토큰화 | 생성 맥락 신호 |

5대 Feature Engineering 원칙 (docs/Architecture/ §1 원칙 E 참조): ①비파괴, ②고엔트로피만 추상화, ③멀티뷰, ④앵커 기반 윈도우링, ⑤설정 파일 기반 확장.

**접근법의 이론적 근거**:

TF-IDF 기반 접근은 **Bag-of-Words** 가정에 기반하며, 단어의 순서와 컨텍스트 정보를 무시한다. 그러나 본 PoC의 오탐 유형이 특정 키워드(예: "test", "sample", "예시")의 존재 여부로 상당 부분 구분 가능하다면, TF-IDF 기반 접근도 충분한 성능을 달성할 수 있다. 멀티뷰 접근(raw_text + shape_text)으로 마스킹 환경에서의 정보 손실을 최소화한다.

**CPU 기반 접근의 장점**:
- 폐쇄망 환경에서 의존성 관리 단순화
- 빠른 iteration으로 다양한 Feature Engineering 실험 가능
- 해석 가능성 (Feature Importance)

### 4.2 Baseline 모델의 중요성

**이론적 배경**:

모든 ML 프로젝트는 반드시 **Baseline**과 비교해야 한다. Baseline 없이는 모델의 "좋은" 성능이 실제로 의미 있는지 판단할 수 없다.

**Baseline 유형 및 목적**:

| Baseline | 설명 | 목적 |
|----------|------|------|
| **Zero Rule** | 가장 빈번한 클래스로 모든 샘플 예측 | 클래스 불균형의 영향 측정 |
| **Random** | 클래스 분포에 따른 무작위 예측 | 우연에 의한 성능 하한선 |
| **Simple Heuristic** | 도메인 지식 기반 규칙 (예: 파일 경로에 "test" 포함 시 오탐) | 휴리스틱의 효과 측정, ML 모델의 부가 가치 정량화 |

**Simple Heuristic Baseline의 중요성**:

만약 Simple Heuristic만으로도 F1-macro 0.65를 달성한다면, 복잡한 ML 모델이 0.70을 달성하더라도 그 부가 가치는 0.05에 불과하다. 이 경우 모델 복잡도, 유지보수 비용 대비 효용을 재검토해야 한다.

반대로 Simple Heuristic이 0.40에 그친다면, ML 모델의 0.70은 75%의 상대적 성능 향상을 의미하며, 투자 정당성이 명확해진다.

### 4.3 모델 선택 근거

**Gradient Boosting 계열 선택 이유**:

XGBoost, LightGBM은 정형 데이터에서 현재까지 가장 강력한 성능을 보이는 알고리즘이다. 그 이론적 기반은:

1. **Boosting**: 약한 학습기(weak learner)를 순차적으로 학습하여 이전 학습기의 오류를 보정. Bias-Variance Trade-off에서 Bias를 점진적으로 감소.

2. **Gradient Descent in Function Space**: 잔차(residual)가 아닌 Loss의 Gradient를 직접 학습하여 다양한 Loss Function에 유연하게 대응.

3. **Regularization**: Tree의 복잡도에 대한 페널티를 부여하여 과적합 방지.

**XGBoost vs LightGBM**:

| 기준 | XGBoost | LightGBM |
|------|---------|----------|
| **트리 성장 방식** | Level-wise (균형 트리) | Leaf-wise (불균형 트리) |
| **학습 속도** | 상대적으로 느림 | 상대적으로 빠름 |
| **메모리 사용** | 상대적으로 높음 | 상대적으로 낮음 |
| **과적합 위험** | 낮음 | 높음 (작은 데이터셋에서) |
| **범주형 변수 처리** | 인코딩 필요 | Native 지원 |

**권장 접근**:

두 모델 모두 실험하여 성능을 비교한다. 데이터 규모가 크고 학습 시간이 제약이라면 LightGBM을 우선 고려하고, 데이터 규모가 작아 과적합이 우려된다면 XGBoost를 우선 고려한다.

### 4.4 하이퍼파라미터 튜닝 전략

**탐색 방법론 비교**:

| 방법 | 메커니즘 | 장점 | 단점 |
|------|----------|------|------|
| **Grid Search** | 모든 조합 탐색 | 완전 탐색 보장 | 차원의 저주, 비효율 |
| **Random Search** | 무작위 샘플링 | 고차원에서 효율적 | 최적해 보장 없음 |
| **Bayesian Optimization** | 이전 결과 기반 다음 탐색점 선택 | 효율적, 수렴 빠름 | 구현 복잡도 |

**이론적 근거**:

Random Search가 Grid Search보다 효율적인 이유는 **유효 차원(Effective Dimensionality)** 때문이다. 대부분의 하이퍼파라미터 공간에서 소수의 파라미터만 성능에 큰 영향을 미친다. Grid Search는 모든 차원을 균등하게 탐색하지만, Random Search는 중요한 차원에서 더 다양한 값을 탐색하게 된다.

**PoC 권장**:

1. **1차**: 도메인 지식과 기본값 기반 초기 설정
2. **2차**: Random Search로 넓은 범위 탐색 (50-100 iterations)
3. **3차**: 유망 영역에서 세밀한 Grid Search 또는 Bayesian Optimization

**핵심 하이퍼파라미터 (XGBoost/LightGBM)**:

| 파라미터 | 영향 | 권장 탐색 범위 |
|----------|------|----------------|
| `learning_rate` | 학습 속도, 과적합 제어 | [0.01, 0.3] |
| `max_depth` | 트리 복잡도, 과적합 제어 | [3, 10] |
| `n_estimators` | 앙상블 크기 | [100, 1000] |
| `min_child_weight` / `min_data_in_leaf` | 리프 노드 최소 샘플 수 | [1, 100] |
| `subsample` | 샘플링 비율, 과적합 제어 | [0.6, 1.0] |
| `colsample_bytree` | Feature 샘플링 비율 | [0.6, 1.0] |
| `scale_pos_weight` | 클래스 불균형 대응 | 클래스 비율 기반 설정 |

---

## 5. 평가 전략

### 5.1 데이터 분할 전략

**Hold-out vs Cross-Validation**:

| 방법 | 장점 | 단점 | 적용 상황 |
|------|------|------|-----------|
| **Hold-out** | 단순, 빠름 | 분할에 따른 분산 큼 | 대용량 데이터 |
| **K-Fold CV** | 분산 감소, 전체 데이터 활용 | 계산 비용 K배 | 중소규모 데이터 |
| **Stratified K-Fold** | K-Fold + 클래스 비율 유지 | 계산 비용 K배 | 클래스 불균형 데이터 |

**v1.2 확정 분할 기준: pk_file Group + Time Split**

> **중요:** v1.2부터 랜덤 Split은 공식 지표에서 제외. 아래 두 방식이 공식 평가 기준.

| 분할 방식 | 설명 | 용도 |
|-----------|------|------|
| **pk_file Group Split** | 동일 파일(pk_file)의 이벤트가 train/test에 분산되지 않도록 파일 단위로 분할 | Data Leakage 방지 (파일 내 이벤트 간 상관성 격리) |
| **Time Split** | 레이블링 시점 기준으로 과거 → 학습, 최근 → 테스트 | 운영 환경과 동일한 시간 순서 재현 |

두 방식 중 Time Split이 가능한 경우 우선 적용하고, 시간 정보 부재 시 pk_file Group Split을 적용한다.

**Stratified K-Fold (K=5)** 는 탐색적 실험(Feature/모델 비교)에서 활용하되, 최종 성능 보고는 위 분할 기준을 사용한다.

### 5.2 Slice-based 평가

**개념 및 필요성**:

전체 평가 메트릭이 양호하더라도 특정 하위 집단(Slice)에서 성능이 저조할 수 있다. 이는 **Aggregate Paradox**의 한 형태로, 전체 성능이 세부 성능을 대표하지 못하는 상황이다.

**권장 Slice 정의**:

| Slice 기준 | 세부 항목 | 분석 목적 |
|------------|-----------|-----------|
| **PII 유형** | 휴대폰, 이메일, 주민번호 | 특정 PII 유형에서의 성능 저하 식별 |
| **파일 확장자** | .java, .py, .log, .doc, .xls 등 | 파일 형식별 성능 차이 파악 |
| **서버 그룹** | 개발, 스테이징, 운영 | 환경별 오탐 패턴 차이 확인 |
| **오탐 클래스** | 테스트, 샘플, 개발, 문서 등 | 특정 오탐 유형의 오분류율 파악 |

**분석 방법**:

각 Slice에 대해 동일한 메트릭(Precision, Recall, F1)을 계산하고, 전체 성능과의 차이를 분석한다. 특정 Slice에서 성능이 현저히 낮다면:
1. 해당 Slice의 데이터 특성 분석
2. 추가 Feature Engineering 검토
3. Slice-specific 모델 또는 후처리 규칙 고려

### 5.3 오류 분석 프로세스

**체계적 오류 분석의 중요성**:

단순히 "F1이 0.75다"라는 결과보다 "어떤 케이스에서 실패하고 왜 실패하는가"를 이해하는 것이 모델 개선과 비즈니스 의사결정에 더 유용하다.

**오류 분석 프레임워크**:

1. **Confusion Matrix 분석**
   - 어떤 클래스가 어떤 클래스로 오분류되는가?
   - 특정 오분류 패턴이 반복되는가?

2. **오분류 샘플 심층 분석**
   - 오분류된 샘플의 원본 데이터 확인
   - 공통된 특성 또는 패턴 식별
   - Feature 값 분포 분석

3. **Feature Importance 분석**
   - 모델이 어떤 Feature에 의존하는가?
   - 예상과 다른 중요 Feature가 있는가? (Data Leakage 징후)

4. **경계 케이스 분석**
   - 예측 확률이 0.5 근처인 샘플 분석
   - 모델이 확신하지 못하는 패턴 파악

**오류 분석 결과의 활용**:

| 발견 | 가능한 조치 |
|------|-------------|
| 특정 클래스 간 혼동 빈번 | 클래스 병합 또는 추가 Feature 개발 |
| 특정 Feature에 과도한 의존 | Data Leakage 점검, Feature 제거 후 재학습 |
| 짧은 텍스트에서 성능 저하 | 텍스트 길이 기반 분기 또는 앙상블 |
| 최근 데이터에서 성능 저하 | Concept Drift 징후, 주기적 재학습 필요 |

---

## 6. 단계별 수행 계획

### 6.1 단계 정의 및 의존관계

v1.2 파이프라인은 docs/Architecture/의 **S0~S6 Stage** 체계를 따른다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PoC 프로젝트 흐름도 (v1.2)                        │
└─────────────────────────────────────────────────────────────────────────────┘

[Phase 1: 환경 구성]
    │
    ├── 1.1 폐쇄망 패키지 준비
    ├── 1.2 개발 환경 셋업
    └── 1.3 데이터 접근 권한 확보
    │
    ▼
[Phase 2: 데이터 파이프라인]  ← 대응 Stage: S0 + S1 + S2
    │
    ├── 2.1 데이터 수집 및 검증  [S0: Raw Ingest — Schema Registry, bronze_*.parquet]
    ├── 2.2 정규화 및 파싱       [S1: Normalize & Parse — 1행=1이벤트, pk_event 생성, silver_detections.parquet]
    └── 2.3 Feature Prep        [S2: Feature Prep — raw_text, shape_text, path_text, tabular 피처]
    │
    ▼
[Phase 3: 탐색적 데이터 분석]
    │
    ├── 3.1 클래스 분포 분석
    ├── 3.2 Feature 특성 파악
    └── 3.3 Data Leakage 점검 (pk_file Group Split 기준)
    │
    ▼
[Phase 4: Feature Engineering]
    │
    ├── 4.1 Text Feature 처리 (raw_text/shape_text/path_text → TF-IDF)
    ├── 4.2 Tabular Feature 처리
    └── 4.3 Feature Selection
    │
    ▼
[Phase 5: 모델 개발]  ← 대응 Stage: S3a + S3b + S4
    │
    ├── 5.1 Baseline 구축
    ├── 5.2 RULE Labeler 구현  [S3a: 키워드/룰 매칭 → 라벨+사유+증거+신뢰도]
    ├── 5.3 ML Labeler 학습    [S3b: XGBoost/LightGBM → 8클래스 확률]
    ├── 5.4 Decision Combiner  [S4: OOD/불확실 → UNKNOWN, TP override]
    └── 5.5 하이퍼파라미터 튜닝
    │
    ▼
[Phase 6: 평가 및 분석]  ← 대응 Stage: S5 + S6
    │
    ├── 6.1 오프라인 평가 (pk_file Group + Time Split 기준)
    ├── 6.2 Slice-based 평가 (PII 유형별, 파일 확장자별)
    ├── 6.3 오류 분석
    └── 6.4 Output 생성 [S5: predictions_main + prediction_evidence]
    │
    ▼
[Phase 7: 산출물 정리]
    │
    ├── 7.1 모델 카드 작성
    ├── 7.2 성능 리포트 작성
    └── 7.3 코드 및 설정 정리
    │
    ▼
[Phase 8: Zero-Human-in-the-Loop Tier 1]
    │
    ├── 8.1 NEEDS_REVIEW 자동 판정  [Auto-Adjudicator]
    ├── 8.2 UNKNOWN 자동 처리       [UnknownAutoProcessor]
    ├── 8.3 모델 자동 선택          [Auto-Model-Selector]
    └── 8.4 라벨 품질 자동 감사     [Confident Learning]
```

### 6.2 단계별 입출력 및 완료 조건

| Phase | 입력 | 출력 | 완료 조건 |
|-------|------|------|-----------|
| **1. 환경 구성** | 패키지 목록, 서버 접근 정보 | 실행 가능한 개발 환경 | 의존성 설치 완료, 테스트 스크립트 실행 성공 |
| **2. 데이터 파이프라인** | Dataset A/B 원본 | `bronze_*.parquet` → `silver_detections.parquet` + Feature 컬럼 | 데이터 무결성 검증 통과, `parse_success_rate` ≥ 95%, PK 매핑 검증 |
| **3. EDA** | silver 데이터셋 | EDA 리포트 | 클래스 분포 파악, pk_file 기준 Leakage 점검 완료 |
| **4. Feature Engineering** | silver 데이터셋, EDA 결과 | Feature Matrix (raw_text+shape_text+path_text TF-IDF, tabular) | 결측치 처리 완료, 피처 스키마 검증 통과 |
| **5. 모델 개발** | Feature Matrix, 레이블 | RULE Labeler + ML Labeler + Decision Combiner | Baseline 대비 유의미한 성능 향상, UNKNOWN 라우팅 동작 확인 |
| **6. 평가** | 학습된 모델, 테스트 데이터 | `predictions_main` + `prediction_evidence` + 평가 리포트 | pk_file Group + Time Split 기준 성공 기준 달성 여부 판정 |
| **7. 산출물** | 모든 이전 산출물 | 모델 카드, 최종 리포트 | 재현 가능성 검증 완료 |
| **8. 자동화(Tier 1)** | Phase 1~7 산출물, KPI/예측 로그 | Auto-Adjudicator 결과, UNKNOWN 자동 처리 로그, 자동 모델 선택 로그, Confident Learning 감사 리포트 | 사람 개입 없이 자동 판정 루프 동작 확인 |

### 6.3 의사결정 포인트

프로젝트 진행 중 명시적 의사결정이 필요한 지점:

| 시점 | 의사결정 사항 | 판단 기준 | 영향 |
|------|---------------|-----------|------|
| **Phase 1 완료 후** | GPU 사용 여부 | 회의록 2026-01 기준 미보유 확정 | CPU 기반 파이프라인 확정 |
| **Phase 3 완료 후** | 원본/마스킹 데이터 선택 | 운영 환경 일치성, 성능 차이 | Feature Engineering 방향 |
| **Phase 3 완료 후** | Multi-class 클래스 수 확정 | 오탐 하위 클래스 분포, 비즈니스 요구 | 모델 출력 차원 |
| **Phase 5 진행 중** | 모델 복잡도 조정 | Baseline 대비 성능, 과적합 징후 | 추가 Feature/모델 실험 여부 |
| **Phase 6 완료 후** | PoC 성공 판정 | 성공 기준 달성 여부 | 후속 프로젝트 진행 여부 |

---

## 7. 리스크 관리

### 7.1 식별된 리스크 및 대응 방안

| 리스크 | 발생 가능성 | 영향도 | 대응 방안 |
|--------|-------------|--------|-----------|
| **폐쇄망 패키지 누락** | 중간 | 높음 | 사전 의존성 분석, 패키지 목록 검증, 대체 패키지 준비 |
| **레이블 데이터 부족** | 중간 | 높음 | 클래스별 최소 샘플 수 확인, Active Learning 고려 |
| **클래스 분포 극심한 불균형** | 높음 | 중간 | 다양한 불균형 대응 기법 순차 적용 |
| **GPU 미보유(확정)** | 낮음 | 낮음 | CPU 기반 파이프라인을 표준으로 운영 |
| **Data Leakage 발견** | 낮음 | 높음 | EDA 단계에서 철저한 점검, pk_file Group Split으로 파일 내 상관성 격리 |
| **Baseline 대비 성능 향상 미미** | 낮음 | 높음 | Feature Engineering 재검토, 모델 변경, 문제 재정의 검토 |
| **시간 제약으로 튜닝 불충분** | 중간 | 낮음 | 필수 실험 우선순위 정의, Early Stopping 적극 활용 |
| **예상 밖 입력/스키마 변경 (Open-World)** | 중간 | 중간 | Schema Registry + Quarantine으로 crash 방지, OOD Score + UNKNOWN 라우팅 |
| **파싱 실패로 데이터 누락** | 중간 | 중간 | 3단 폴백 파서, `parse_success_rate` KPI 모니터링 (목표: 95% 이상) |

### 7.2 리스크 모니터링

각 Phase 완료 시점에 다음을 점검:
- 일정 준수 여부
- 품질 기준 충족 여부
- 신규 리스크 발생 여부
- 기존 리스크 상태 변화

---

## 8. 확인 필요 사항 (Open Questions)

PoC 착수 전 또는 진행 중 확정해야 할 사항:

| # | 질문 | 중요도 | 영향 범위 | 현재 상태 |
|---|------|--------|-----------|-----------|
| 1 | 오탐 세부 클래스 정의 (8-10종 추정) | 높음 | 모델 설계, 평가 | **확정 (8개 클래스)** |
| 2 | 레이블링된 데이터 규모 | 높음 | 학습 가능성, 모델 선택 | **미확정 (수령 대기)** |
| 3 | 원본 vs 마스킹 데이터 사용 결정 | 높음 | Feature Engineering | **확정 (마스킹 데이터)** |
| 4 | Tabular 필드 목록 | 중간 | Feature Engineering | **미확정 (실데이터 헤더 확인 필요)** |
| 5 | 평가 메트릭 우선순위 합의 | 높음 | 성공 기준 | **확정 (F1-macro/TP Recall/FP Precision)** |
| 6 | GPU 가용 여부 | 높음 | 모델링 접근법 | **확정 (미보유, CPU 기반)** |
| 7 | 추론 요구사항 (실시간 vs 배치) | 중간 | 모델 복잡도 제약 | **미확정** |
| 8 | 폐쇄망 반입 절차 및 소요 시간 | 중간 | 일정 계획 | **미확정 (현장 확인 필요)** |
| 9 | 대안 환경 검토 | 해당 없음 | 단일 시나리오 유지 | **폐쇄망 단일 환경 확정** |
| 10 | 데이터에 시간 정보 포함 여부 | 중간 | 데이터 분할 전략 | **확정 — pk_file Group + Time Split 적용 (v1.2 기준)** |

---

## 10. 문서 체계

본 Master Plan과 연계되는 세부 문서:

| 문서 | 목적 | 상태 |
|------|------|------|
| **docs/Architecture/** | **목표 아키텍처 설계서 (v1.2 기준 최우선 참조)** | **최신 (v1.2, 2026-02-22)** |
| 01_Environment_Setup.md | 환경 구성 상세 가이드 | 작성 완료 (v1.2 반영) |
| 02_Data_Pipeline.md | 데이터 파이프라인 설계 (S0~S2) | 작성 완료 (v1.2 + Auto-Schema-Detector/Quarantine KPI 반영) |
| 03_EDA_Guide.md | 탐색적 데이터 분석 가이드 | 작성 완료 (v1.2 반영) |
| 04_Feature_Engineering.md | Feature Engineering 상세 (멀티뷰, 5대 원칙) | 작성 완료 (v1.2 반영) |
| 05_Model_Development.md | 모델 개발 상세 (S3a/S3b/S4) | 작성 완료 (v1.2 + Auto-Rule-Promoter/Auto-Model-Selector 반영) |
| 06_Evaluation_Validation.md | 평가 및 검증 상세 (pk_file Group + Time Split) | 작성 완료 (v1.2 + 운영 자동화 검증 항목 반영) |
| 07_Deliverables_Templates.md | 산출물 템플릿 (predictions_main, prediction_evidence) | 작성 완료 (v1.2 + label_governance/Auto-Retrainer/Confident Learning 반영) |

---

## 부록 A: 용어 정의

| 용어 | 정의 |
|------|------|
| **정탐 (True Positive, TP)** | 실제 개인정보이면서 개인정보로 검출된 건 |
| **오탐 (False Positive, FP)** | 개인정보가 아니면서 개인정보로 검출된 건 |
| **PII** | Personal Identifiable Information, 개인식별정보 |
| **DLP** | Data Loss Prevention, 데이터 유출 방지 |
| **Data Leakage** | 학습 시 테스트 정보가 누출되는 현상 |
| **Concept Drift** | 시간에 따라 데이터 분포가 변화하는 현상 |
| **Train-Serving Skew** | 학습 환경과 운영 환경의 차이로 인한 성능 저하 |
| **RULE Labeler** | Layer 1~2. 키워드/룰 매칭으로 라벨+사유코드+증거+신뢰도를 출력하는 고정밀 라벨러. "필터"가 아니라 결과를 남기는 라벨러 |
| **ML Labeler** | Layer 3. RULE로 결정 못한 잔여 샘플에 대해 ML 모델(XGBoost/LightGBM)이 8클래스 확률 기반 라벨을 출력 |
| **UNKNOWN** | 학습 클래스가 아닌 운영 후처리 클래스. OOD Score 높거나 신뢰도 극하일 때 Stage S4에서 자동 라우팅됨 |
| **decision_source** | 최종 라벨의 출처: `RULE` / `ML` / `HYBRID` |
| **prediction_evidence** | 1건의 예측에 대한 N행 상세 근거 테이블 (매칭된 키워드/룰, ML 확률 분포, 신뢰도 등) |
| **confidence** | 예측의 신뢰도 점수. RULE은 경험적 정밀도 하한, ML은 보정된 확률 기반으로 계산 방식이 다름 |
| **risk_flag** | TP 오분류 위험 표시. 애매한 케이스에 자동 부여, 보수적 처리(TP override) 트리거 |
| **ood_flag** | Out-Of-Distribution 플래그. 학습 분포 밖 입력으로 판단될 때 1 |
| **parse_status** | S1 파싱 결과 상태: `OK`, `FALLBACK_ANCHOR`, `FALLBACK_SINGLE_EVENT` |
| **Bronze / Silver** | 데이터 레이어 명칭. Bronze=원본 Parquet(S0), Silver=정규화/피처 추가 Parquet(S1~S2) |
| **Schema Registry** | 입력 데이터 컬럼 구조를 등록·검증하는 YAML 기반 레지스트리. 스키마 불일치 시 crash 대신 Quarantine |
| **pk_file** | 파일 단위 고유 키: hash(server_name\|agent_ip\|file_path) |
| **pk_event** | 검출 이벤트 단위 고유 키: hash(pk_file\|masked_hit\|event_time\|row_id) |
| **OOD Score** | Out-Of-Distribution 탐지 점수. 학습 분포와의 거리를 수치화 |
| **pii_type_inferred** | Dataset B `패턴 종류` 필드의 신뢰도 이슈 대응, 검출 내역 텍스트 기반 PII 유형 재추론 결과 |

---

## 부록 B: 참고 문헌

1. Chip Huyen, "Designing Machine Learning Systems", O'Reilly, 2022
2. XGBoost Documentation: https://xgboost.readthedocs.io/
3. LightGBM Documentation: https://lightgbm.readthedocs.io/
4. Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
5. Google ML Best Practices: https://developers.google.com/machine-learning/guides/rules-of-ml

---

**문서 끝**
