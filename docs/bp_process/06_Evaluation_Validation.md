# 06. 모델 평가 및 검증 (Evaluation & Validation)

> **문서 버전**: 1.2
> **최종 수정일**: 2026-02-22
> **기반 문서**: docs/Architecture/ v1.2 §19(평가 전략)

## 1. 개요

### 1.1 문서 목적

본 문서는 Server-i 오탐 개선 PoC 프로젝트의 모델 평가 및 검증 단계에서 적용할 방법론, 평가 지표 선정 근거, 검증 전략을 정의한다. 단순히 "어떤 지표를 측정할 것인가"가 아닌 "왜 그 지표가 비즈니스 목표와 연결되는가"에 초점을 맞춘다.

> **v1.2 중요 변경:** 평가 분할 기본값이 변경됨. **이벤트 랜덤 split은 공식 지표에서 제외**. 공식 기준은 `pk_file Group Split` + `Time Split`.

### 1.2 평가의 두 가지 관점

**기술적 관점 (Technical Evaluation)**:
- 모델이 학습 데이터의 패턴을 얼마나 잘 포착했는가
- 일반화 성능이 충분한가 (과적합 여부)
- 예측 불확실성이 적절히 반영되는가

**비즈니스 관점 (Business Evaluation)**:
- 오탐 자동 분류로 운영 효율이 개선되는가
- 정탐을 놓치는 위험이 허용 가능한 수준인가
- 현업 담당자가 모델 결과를 신뢰할 수 있는가

**두 관점의 연결**:

| 기술 지표 | 비즈니스 의미 |
|----------|--------------|
| F1-macro | 8개 클래스 전반의 균형 성능 → 기술적 타당성 판단 |
| Recall (정탐) | 실제 정탐 중 정탐으로 분류된 비율 → 개인정보 유출 위험 통제 |
| Precision (오탐) | 오탐으로 분류된 것 중 실제 오탐 비율 → 자동 분류 신뢰도 |

---

## 2. 평가 지표 선정

### 2.1 기본 분류 지표 이해

#### 2.1.1 Confusion Matrix 기반 지표

**Binary Classification 관점**:

```
                    예측
                정탐        오탐
        ┌──────────┬──────────┐
  정탐  │    TP    │    FN    │  → 실제 정탐 (Actual Positive)
실      ├──────────┼──────────┤
제      │    FP    │    TN    │  → 실제 오탐 (Actual Negative)
  오탐  └──────────┴──────────┘
```

**주요 지표 정의**:

| 지표 | 수식 | 의미 |
|------|------|------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | 전체 중 올바른 예측 비율 |
| Precision (정탐) | TP/(TP+FP) | 정탐 예측 중 실제 정탐 비율 |
| Recall (정탐) | TP/(TP+FN) | 실제 정탐 중 정탐 예측 비율 |
| Specificity (오탐) | TN/(TN+FP) | 실제 오탐 중 오탐 예측 비율 |
| F1 Score | 2×(Precision×Recall)/(Precision+Recall) | Precision-Recall 조화 평균 |

#### 2.1.2 Server-i 맥락에서의 지표 해석

**정탐 관점 (Positive = 정탐 = 실제 개인정보)**:

| 오류 유형 | 의미 | 비즈니스 영향 |
|----------|------|--------------|
| FN (정탐 → 오탐) | 실제 개인정보를 오탐으로 잘못 분류 | **Critical**: 개인정보 유출 위험 |
| FP (오탐 → 정탐) | 실제 오탐을 정탐으로 잘못 분류 | **Moderate**: 불필요한 파기 작업 |

**핵심 통찰**:
- FN의 비용 > FP의 비용 → Recall 우선
- 하지만 Precision이 너무 낮으면 모든 것을 정탐으로 예측하는 것과 동일
- Trade-off 지점을 비즈니스 요구사항에 맞게 조정

### 2.2 클래스 불균형 하에서의 지표 선택

#### 2.2.1 Accuracy의 한계

**Accuracy Paradox**:

Server-i 데이터에서 오탐 비율이 약 2/3 (약 66%)라면:
- 모든 샘플을 "오탐"으로 예측 → Accuracy ≈ 66%
- 하지만 정탐 검출 능력 = 0% (무의미한 모델)

**Rationale**: 클래스 불균형 상황에서 Accuracy는 모델 성능을 과대평가할 수 있으며, 단독 지표로 사용하면 안 됨

#### 2.2.2 클래스별 지표의 중요성

**Macro vs Micro vs Weighted Average**:

| 평균 방식 | 계산 방법 | 특성 | 적용 상황 |
|----------|----------|------|----------|
| Macro | 클래스별 지표의 단순 평균 | 소수 클래스에 동등한 가중치 | 클래스별 성능 균형 중요 시 |
| Micro | 전체 샘플 기반 계산 | 다수 클래스에 유리 | 전체 정확도 중요 시 |
| Weighted | 클래스 빈도로 가중 평균 | Micro와 유사한 특성 | 클래스 분포 반영 필요 시 |

**Server-i 권장**: Macro Average

Rationale:
- 정탐(소수 클래스)의 성능이 비즈니스적으로 중요
- Macro Average는 정탐 성능을 Micro/Weighted보다 더 잘 반영
- 단, 개별 클래스 성능도 반드시 확인 필요

### 2.3 Multi-class 평가 지표

#### 2.3.1 Multi-class Confusion Matrix

8개 클래스의 전체 Confusion Matrix는 해석이 어려우므로, 계층적 분석 접근:

**Level 1: Binary (정탐 vs 오탐)**
- Multi-class 결과를 Binary로 집계
- 정탐/오탐 구분 능력 평가

**Level 2: 오탐 유형별**
- 오탐으로 분류된 샘플 중 세부 유형 분류 정확도
- 오분류 패턴 파악 (어떤 오탐 유형이 혼동되는가)

#### 2.3.2 Multi-class 지표

| 지표 | 설명 | 적용 |
|------|------|------|
| Macro F1 | 클래스별 F1의 단순 평균 | 전체 클래스 균형 성능 |
| Weighted F1 | 클래스 빈도 가중 F1 평균 | 클래스 분포 반영 성능 |
| Cohen's Kappa | 우연 일치를 보정한 일치도 | 클래스 불균형 보정된 성능 |
| Top-K Accuracy | 상위 K개 예측에 정답 포함 비율 | 후보군 추천 성능 |

### 2.4 확률 기반 평가 지표

#### 2.4.1 Log Loss (Cross-Entropy Loss)

**정의**: 예측 확률 분포와 실제 분포의 차이

**수식**:
```
Log Loss = -1/N × Σ[y_i × log(p_i) + (1-y_i) × log(1-p_i)]
```

**특성**:
- 확률 예측의 품질 평가
- 자신감 있는 오답에 큰 페널티
- 확률 Calibration 반영

**해석**:
- Log Loss가 낮을수록 좋음
- 이상적인 값: 0 (완벽한 예측)
- Random Guess (Binary): 약 0.693

#### 2.4.2 Brier Score

**정의**: 예측 확률과 실제 결과의 평균 제곱 오차

**수식**:
```
Brier Score = 1/N × Σ(p_i - y_i)²
```

**특성**:
- 확률 Calibration과 분류 능력 모두 반영
- 0-1 범위 (낮을수록 좋음)
- Log Loss보다 극단적 오류에 덜 민감

### 2.5 Ranking 기반 평가 지표

#### 2.5.1 ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

**개념**: 다양한 임계값에서 TPR과 FPR의 Trade-off 곡선 아래 면적

**ROC Curve**:
- X축: FPR (1 - Specificity) = FP / (FP + TN)
- Y축: TPR (Recall) = TP / (TP + FN)

**해석**:
- AUC = 1.0: 완벽한 분류
- AUC = 0.5: Random Guess
- AUC > 0.8: 일반적으로 좋은 성능

**클래스 불균형에서의 특성**:
- FPR이 낮아지기 어려워 AUC가 과대평가될 수 있음
- 소수 클래스 성능을 잘 반영하지 못할 수 있음

#### 2.5.2 PR-AUC (Precision-Recall - Area Under Curve)

**개념**: Precision-Recall Curve 아래 면적

**PR Curve**:
- X축: Recall = TP / (TP + FN)
- Y축: Precision = TP / (TP + FP)

**클래스 불균형에서의 장점**:
- 소수 클래스(정탐) 성능에 민감
- TN(다수인 오탐의 올바른 분류)에 영향받지 않음
- ROC-AUC보다 더 현실적인 성능 반영

**Server-i 맥락**:
- 정탐이 소수 클래스이므로 PR-AUC가 더 적합
- ROC-AUC와 함께 사용하여 다각도 평가

### 2.6 지표 선정 종합 (v1.2)

**Primary 지표** (성능 판단 기준):

| 지표 | 목적 | 목표값 |
|------|------|--------|
| Macro F1 (Multi-class) | 전체 클래스 균형 성능 | ≥ 0.70 |

**Secondary 지표** (보조 판단):

| 지표 | 목적 | 목표값 |
|------|------|--------|
| Recall (정탐) | 개인정보 유출 위험 통제 | ≥ 0.75 |
| Precision (오탐) | 오탐 자동 분류 신뢰도 | ≥ 0.85 |

**Monitoring 지표** (과적합/품질 확인):

| 지표 | 목적 |
|------|------|
| PR-AUC | 정탐-오탐 분리 추세 모니터링 |
| Log Loss | 확률 예측 품질 |
| Train-Validation Gap | 과적합 정도 |
| Calibration Error | 확률 신뢰도 |
| UNKNOWN 비율 | OOD/미분류 케이스 비율 (목표: 낮을수록 좋음) |
| RULE 매칭률 | Layer 1~2 RULE Labeler 커버리지 |
| `parse_success_rate` | S1 파싱 성공률 (목표: ≥ 95%) |

> **v1.2 평가 분할 기준**: `pk_file Group Split` + `Time Split`만 공식 지표로 사용. 이벤트 랜덤 split은 참고용으로만 사용하고 공식 결과에서 제외.

---

## 3. 교차 검증 전략

### 3.1 교차 검증의 목적

**왜 교차 검증이 필요한가**:

1. **일반화 성능 추정**: 단일 Train-Test 분할은 분할 방식에 따라 성능이 크게 달라질 수 있음
2. **분산 감소**: 여러 번의 검증으로 성능 추정의 신뢰도 향상
3. **과적합 탐지**: Train과 Validation 성능 Gap으로 과적합 여부 판단
4. **데이터 효율성**: 모든 데이터를 학습과 검증에 활용

### 3.2 v1.2 공식 평가 분할 기준

#### 3.2.0 pk_file Group Split + Time Split (공식 기준)

> **v1.2 확정:** 이벤트 랜덤 split은 공식 지표에서 제외. 아래 두 방식이 공식.

**pk_file Group Split:**

동일 파일(`pk_file`)의 이벤트가 Train/Test에 분산되지 않도록, 파일 단위로 분할한다.

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_idx, test_idx in gss.split(X, y, groups=df['pk_file']):
    X_train, X_test = X[train_idx], X[test_idx]
    # 동일 파일의 이벤트가 반드시 같은 split에 위치
```

**이유:** 동일 파일에서 나온 이벤트들은 맥락(경로, 서버, 파일 구조)을 공유한다. 이를 Train/Test에 분산하면 Data Leakage가 발생하여 성능이 과대평가된다.

**Time Split:**

레이블링 시점 기준으로 과거 → Train, 최근 → Test. Concept Drift를 반영하는 가장 현실적인 평가.

```python
# 시간 순서로 정렬 후 80:20 분할
df_sorted = df.sort_values('detection_time')
split_point = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_point]
test_df = df_sorted.iloc[split_point:]
```

**Time Split 불가 시:** pk_file Group Split 단독 적용.

#### 3.2.1 기본 K-Fold (탐색적 실험용)

**방법**:
1. 데이터를 K개의 동일한 크기의 Fold로 분할
2. 각 Iteration에서 1개 Fold를 Validation, 나머지 K-1개를 Train으로 사용
3. K번 반복 후 성능 평균 및 표준편차 계산

> **주의:** 공식 성능 지표는 3.2.0의 분할 기준을 사용. K-Fold는 Feature 탐색, 하이퍼파라미터 실험 등 탐색 목적으로 활용.

**K 값 선택 기준**:

| K 값 | 장점 | 단점 | 권장 상황 |
|------|------|------|----------|
| 5 | 계산 효율적, 적절한 분산 | Validation 크기 20% | 일반적 상황, 중간 크기 데이터 |
| 10 | 낮은 편향, 더 많은 학습 데이터 | 높은 분산, 계산 비용 | 데이터가 충분할 때 |
| N (LOOCV) | 최소 편향 | 매우 높은 분산, 계산 비용 | 매우 작은 데이터 |

#### 3.2.2 Stratified K-Fold

**개념**: 각 Fold에서 클래스 비율을 원본 데이터와 동일하게 유지

**필요성**:
- Server-i 데이터는 정탐:오탐 ≈ 1:2의 극심한 불균형
- 일반 K-Fold는 일부 Fold에 정탐이 거의 없을 수 있음
- Stratified는 모든 Fold에서 클래스 비율 유지

**적용 방법**:
- Binary 분류: 정탐/오탐 비율 유지
- Multi-class 분류: 모든 클래스 비율 유지

**고려사항**:
- 극소수 클래스(예: 특정 오탐 유형이 10개 미만)는 K-Fold 자체가 어려울 수 있음
- 이 경우 해당 클래스는 모든 Fold의 Train에 포함되도록 특별 처리

### 3.3 시간 기반 분할 고려

#### 3.3.1 Temporal Validation의 필요성

**문제 상황**:
- DLP 시스템의 오탐 패턴이 시간에 따라 변화할 수 있음
- 새로운 서버, 애플리케이션, 개발 프로젝트 추가에 따른 패턴 변화
- Random Split은 미래 데이터로 과거 예측하는 Data Leakage 발생 가능

**적용 여부 판단 기준**:

| 조건 | Temporal Validation |
|------|---------------------|
| 검출 데이터에 명확한 시간 순서 존재 | 고려 필요 |
| 시간에 따른 패턴 변화 예상 | 적용 권장 |
| PoC 데이터가 특정 시점의 Snapshot | 일반 CV로 충분 |

#### 3.3.2 Time Series Split

**방법**:
- 시간 순으로 데이터 정렬
- Train 기간 → Validation 기간 순차적 분할
- 미래 데이터가 학습에 포함되지 않도록 보장

**변형 - Expanding Window**:
- Fold 1: [T1] train, [T2] valid
- Fold 2: [T1, T2] train, [T3] valid
- Fold 3: [T1, T2, T3] train, [T4] valid

**변형 - Sliding Window**:
- Fold 1: [T1] train, [T2] valid
- Fold 2: [T2] train, [T3] valid
- Fold 3: [T3] train, [T4] valid

### 3.4 Nested Cross Validation

**개념**: 하이퍼파라미터 튜닝과 성능 평가를 분리

**구조**:
```
Outer Loop (성능 평가용):
  └── Inner Loop (하이퍼파라미터 튜닝용):
        └── K-Fold로 최적 파라미터 선택
  └── Outer Fold에서 선택된 파라미터로 평가
```

**필요성**:
- 하이퍼파라미터 튜닝에 사용된 Validation Set으로 최종 성능 측정 시 낙관적 편향 발생
- Nested CV로 튜닝과 평가를 독립적으로 수행

**고려사항**:
- 계산 비용이 K_outer × K_inner 배 증가
- PoC 단계에서는 단순 Train/Valid/Test Split으로 대체 가능
- 엄밀한 성능 평가 필요 시 적용 검토

### 3.5 교차 검증 전략 권장

**Server-i PoC 권장 구성**:

```
전체 데이터
    │
    ├── Train+Validation (80%)
    │   └── Stratified 5-Fold CV
    │       ├── Fold 1: Train(80%) / Valid(20%)
    │       ├── Fold 2: Train(80%) / Valid(20%)
    │       └── ... (5회 반복)
    │
    └── Hold-out Test (20%)
        └── 최종 성능 평가 (1회만 사용)
```

**핵심 원칙**:
1. Test Set은 모델 개발 과정에서 절대 사용하지 않음
2. 하이퍼파라미터 튜닝은 CV의 Validation 성능 기준
3. 최종 모델 선택 후 Test Set으로 단 한 번 평가

---

## 4. Slice-based 평가

### 4.1 Slice-based 평가의 필요성

**전체 성능의 함정**:
- 전체 F1 Score = 0.8이라도, 특정 조건에서는 0.3일 수 있음
- 평균 성능이 좋아도 특정 사용자/상황에서 실패하면 신뢰도 하락
- 문제가 되는 영역을 조기에 발견하여 대응 필요

**Slice의 정의**:
- 특정 조건으로 필터링된 데이터 부분집합
- 예: "휴대폰 번호 검출", "Excel 파일에서의 검출", "개발 서버에서의 검출"

### 4.2 Server-i 맥락의 Slice 정의

#### 4.2.1 PII 유형별 Slice

| Slice | 정의 | 중요도 | 근거 |
|-------|------|--------|------|
| 휴대폰 번호 | PII_TYPE = '휴대폰번호' | 높음 | 가장 빈번한 검출 유형 |
| 이메일 주소 | PII_TYPE = '이메일' | 중간 | 형식 다양성 높음 |
| 주민등록번호 | PII_TYPE = '주민번호' | 높음 | 민감도 최고, 오탐 시 리스크 |

**평가 관점**:
- 각 PII 유형별로 모델 성능이 유사한지 확인
- 특정 유형에서 성능 저하 시 해당 유형 Feature 강화 필요

#### 4.2.2 파일 형식별 Slice

| Slice | 정의 | 중요도 | 근거 |
|-------|------|--------|------|
| Excel 파일 | FILE_EXT in ['.xls', '.xlsx'] | 높음 | 정형화된 데이터, 정탐 비율 높을 가능성 |
| 텍스트 파일 | FILE_EXT in ['.txt', '.log'] | 중간 | 시스템 로그 등 오탐 많을 가능성 |
| 소스 코드 | FILE_EXT in ['.java', '.py', '.js'] | 높음 | 테스트 데이터, 변수명 등 오탐 다수 |
| 문서 파일 | FILE_EXT in ['.doc', '.pdf'] | 중간 | 컨텍스트 정보 풍부 |

**평가 관점**:
- 소스 코드 파일에서의 오탐 분류 효과가 핵심
- 문서 파일에서의 정탐 검출 정확도 확인

#### 4.2.3 서버/경로 기반 Slice

| Slice | 정의 | 중요도 | 근거 |
|-------|------|--------|------|
| 개발 서버 | SERVER_GROUP = 'DEV' | 높음 | 테스트 데이터 집중, 오탐 비율 높음 |
| 운영 서버 | SERVER_GROUP = 'PROD' | 높음 | 실제 개인정보 존재 가능성 높음 |
| 백업 경로 | PATH contains '/backup/' | 중간 | 중복 데이터, 오탐 가능성 |
| 테스트 경로 | PATH contains '/test/' | 높음 | 명확한 오탐 예상 영역 |

#### 4.2.4 오탐 유형별 Slice (Multi-class)

| Slice | 정의 | 평가 목적 |
|-------|------|----------|
| FP-더미데이터 | 예측 = 'FP-더미데이터' | 테스트/샘플 데이터 오탐 분류 정확도 |
| FP-숫자나열/코드 | 예측 = 'FP-숫자나열/코드' | 코드/식별자 패턴 오탐 분류 정확도 |
| 정탐 vs FP-더미데이터 | 두 클래스 간 혼동 | 가장 중요한 구분 능력 |

### 4.3 Slice 성능 분석 방법

#### 4.3.1 Slice별 지표 계산

**계산 방법**:
1. 전체 Test/Validation Set에서 Slice 조건에 해당하는 샘플 추출
2. 해당 Slice에서 지표 계산
3. 전체 성능과 비교하여 Gap 분석

**분석 프레임워크**:

| Slice | 샘플 수 | F1 (정탐) | Recall | Precision | 전체 대비 Gap |
|-------|---------|----------|--------|-----------|--------------|
| 전체 | 10,000 | 0.75 | 0.80 | 0.70 | - |
| 휴대폰번호 | 5,000 | 0.78 | 0.82 | 0.74 | +0.03 |
| 소스코드 | 2,000 | 0.65 | 0.70 | 0.61 | -0.10 |
| 개발서버 | 3,000 | 0.60 | 0.65 | 0.56 | -0.15 |

#### 4.3.2 Critical Slice 식별

**Critical Slice 기준**:
1. 비즈니스 중요도가 높은 영역 (예: 운영 서버)
2. 성능이 전체 평균보다 유의미하게 낮은 영역
3. 샘플 수가 충분하여 통계적으로 의미 있는 영역

**대응 전략**:
- Critical Slice에서 오분류되는 샘플의 공통 특성 분석
- 해당 특성을 반영한 Feature 추가 검토
- Slice 특화 모델 또는 규칙 적용 고려

### 4.4 Slice-based 평가 고려사항

**Slice 정의 시 주의점**:
- Slice가 너무 작으면 통계적 신뢰도 부족 (최소 100+ 샘플 권장)
- Slice 간 겹침(overlap)이 있을 수 있음 (예: 개발서버의 Excel 파일)
- 모든 가능한 Slice를 정의할 필요 없음, 비즈니스 중요도 기준 선별

**해석 시 주의점**:
- Slice 성능 차이가 모델 문제인지, 데이터 분포 문제인지 구분
- 특정 Slice에서 정탐 샘플이 극소수면 지표 신뢰도 낮음
- Slice별 클래스 분포도 함께 확인 필요

---

## 5. 오류 분석 (Error Analysis)

### 5.1 오류 분석의 목적

**목적**:
1. 모델이 실패하는 패턴 파악
2. Feature Engineering 개선 방향 도출
3. 모델 한계 인식 및 보완 전략 수립
4. 현업 담당자에게 모델 약점 설명 가능

### 5.2 오류 분석 프레임워크

#### 5.2.1 오류 유형 분류

**Binary 관점**:

| 오류 유형 | 정의 | 분석 초점 |
|----------|------|----------|
| False Negative | 정탐 → 오탐 오분류 | 왜 실제 개인정보를 놓쳤는가 |
| False Positive | 오탐 → 정탐 오분류 | 왜 오탐을 정탐으로 잘못 판단했는가 |

**Multi-class 관점**:

| 혼동 쌍 | 분석 초점 |
|---------|----------|
| 정탐 ↔ FP-더미데이터 | 샘플/테스트 데이터의 어떤 특성이 정탐과 유사한가 |
| 정탐 ↔ FP-숫자나열/코드 | 숫자/코드 패턴과 실제 개인정보 구분 기준 |
| FP-더미데이터 ↔ FP-패턴맥락 | 오탐 유형 간 구분 어려움 원인 |

#### 5.2.2 오류 샘플 분석 절차

**Step 1: 오류 샘플 추출**
- Validation/Test Set에서 오분류 샘플 식별
- 오류 유형별(FN, FP) 분리

**Step 2: 패턴 분석**
- 오류 샘플의 공통 특성 탐색
- Feature 분포 비교 (오류 vs 정분류)
- Text 컨텍스트 검토

**Step 3: 근본 원인 분류**

| 원인 유형 | 설명 | 대응 방향 |
|----------|------|----------|
| Feature 부족 | 필요한 정보가 Feature로 표현되지 않음 | Feature Engineering 추가 |
| 노이즈 | 레이블링 오류 또는 모호한 샘플 | 데이터 정제, 레이블 검토 |
| 분포 차이 | 특정 패턴이 학습 데이터에 부족 | 데이터 보강, 오버샘플링 |
| 모델 한계 | 모델이 표현하기 어려운 패턴 | 모델 변경 또는 규칙 보완 |

**Step 4: 개선 우선순위 결정**
- 빈도: 해당 오류 패턴이 얼마나 자주 발생하는가
- 영향도: 비즈니스적으로 얼마나 심각한 오류인가
- 해결 가능성: 현재 데이터/리소스로 개선 가능한가

### 5.3 Confusion Matrix 심층 분석

#### 5.3.1 Multi-class Confusion Matrix 해석

**분석 관점**:
1. 대각선 (정분류): 각 클래스별 정분류율 확인
2. 행 방향: 실제 클래스가 어떤 클래스로 잘못 예측되는가
3. 열 방향: 특정 예측이 실제로 어떤 클래스였는가

**집중 분석 영역**:
- 정탐 행: 정탐이 어떤 오탐 유형으로 잘못 분류되는가
- 정탐 열: 어떤 오탐 유형이 정탐으로 잘못 분류되는가
- 높은 혼동 쌍: 상호 오분류가 많은 클래스 쌍 식별

#### 5.3.2 정규화된 Confusion Matrix

**행 정규화 (Recall 관점)**:
- 각 행을 합이 1이 되도록 정규화
- 각 실제 클래스가 어디로 분류되는지 비율 확인
- 클래스 크기 차이에 무관한 비교 가능

**열 정규화 (Precision 관점)**:
- 각 열을 합이 1이 되도록 정규화
- 각 예측 클래스의 실제 구성 비율 확인
- 예측 신뢰도 분석에 유용

### 5.4 Feature Importance 분석

#### 5.4.1 Feature Importance 유형

**Tree 기반 모델의 Importance**:

| 유형 | 설명 | 특성 |
|------|------|------|
| Gain | Feature로 인한 손실 감소량 합계 | 예측 기여도 직접 측정 |
| Cover | Feature가 영향 미치는 샘플 수 | Feature 사용 범위 |
| Weight (Frequency) | Feature가 사용된 분할 횟수 | Feature 사용 빈도 |

**권장**: Gain 기반 Importance (예측 성능 기여도 직접 반영)

#### 5.4.2 Permutation Importance

**개념**: Feature 값을 무작위로 섞었을 때 성능 하락 정도

**장점**:
- 모델에 무관하게 적용 가능
- Feature 간 상호작용 고려
- 실제 예측 성능 기반

**단점**:
- 계산 비용 높음
- 상관된 Feature에서 과소평가 가능

**적용 방법**:
1. 기준 성능 측정 (Validation Set)
2. 각 Feature를 순서대로 Permutation
3. 성능 하락량 = Importance

#### 5.4.3 SHAP (SHapley Additive exPlanations)

**개념**: 게임 이론의 Shapley Value를 ML에 적용한 Feature 기여도

**장점**:
- 개별 샘플 수준의 설명 가능
- Feature 간 상호작용 반영
- 일관된 이론적 기반

**Server-i 적용**:
- 오분류 샘플에서 어떤 Feature가 잘못된 판단에 기여했는지 분석
- 정탐/오탐 분류에 가장 영향력 있는 Feature 식별
- 현업 담당자에게 모델 판단 근거 설명 가능

### 5.5 오류 분석 체크리스트

**분석 전**:
- [ ] 오류 샘플 추출 및 저장 (원본 데이터 + 예측 결과)
- [ ] 오류 유형별 분류 (FN, FP, 클래스별)
- [ ] 분석 대상 샘플 수 확인 (통계적 유의성)

**분석 중**:
- [ ] Confusion Matrix 시각화 및 해석
- [ ] 오류 샘플의 Feature 분포 분석
- [ ] Text 컨텍스트 직접 검토 (샘플링)
- [ ] Feature Importance 분석

**분석 후**:
- [ ] 주요 오류 패턴 문서화
- [ ] 개선 방향 우선순위 결정
- [ ] 해결 불가능한 한계 명시

---

## 6. 모델 성능 해석 및 보고

### 6.1 성능 해석 가이드라인

#### 6.1.1 통계적 유의성 고려

**신뢰 구간 계산**:
- K-Fold CV 결과의 평균 ± 표준편차 보고
- 예: F1 = 0.75 ± 0.03 (5-Fold CV)

**모델 간 비교**:
- 단순 평균 비교만으로 우열 판단 주의
- 표준편차가 겹치는 경우 통계적으로 유의미한 차이 아닐 수 있음
- Paired t-test 또는 Wilcoxon signed-rank test 고려

#### 6.1.2 과적합 판단 기준

| Train-Valid Gap | 해석 | 대응 |
|----------------|------|------|
| < 5%p | 건강한 상태 | 유지 |
| 5-10%p | 경미한 과적합 | 모니터링 |
| > 10%p | 심각한 과적합 | 정규화 강화, 모델 단순화 |

**추가 확인**:
- 학습 곡선 (Learning Curve) 분석
- 데이터 증가에 따른 성능 변화
- Early Stopping 시점 확인

### 6.2 성능 리포트 구조

#### 6.2.1 Executive Summary

**포함 내용**:
- 최종 모델 성능 요약 (Primary 지표 중심)
- 비즈니스 목표 달성 여부
- 주요 발견 사항 및 권고

**예시**:
```
Server-i 오탐 개선 모델 성능 요약
- F1-macro: 0.78 (목표: 0.70) ✓
- 정탐 Recall: 0.85 (목표: 0.75) ✓
- 오탐 Precision: 0.88 (목표: 0.85) ✓
- 오탐 자동 분류 가능 비율: 약 75%
- 권고: 소스코드 파일 대상 추가 Feature 개발 필요
```

#### 6.2.2 상세 성능 분석

**클래스별 성능**:
- Binary 및 Multi-class Confusion Matrix
- 클래스별 Precision, Recall, F1

**Slice별 성능**:
- 주요 Slice 정의 및 성능
- Critical Slice 식별 및 원인 분석

**오류 분석 요약**:
- 주요 오류 패턴 Top 5
- 개선 방향 및 우선순위

### 6.3 현업 커뮤니케이션

#### 6.3.1 비기술적 설명

**지표 설명 방법**:

| 기술 용어 | 비즈니스 표현 |
|----------|--------------|
| Recall 85% | "실제 개인정보 100건 중 85건을 정확히 찾아냄" |
| Precision 70% | "개인정보라고 판단한 것 중 70%가 실제 개인정보" |
| F1 0.75 | "전반적인 정탐 검출 능력이 100점 만점에 75점 수준" |

**Trade-off 설명**:
- "더 많은 개인정보를 찾으려면(Recall↑), 오탐을 정탐으로 잘못 판단하는 경우도 늘어남(Precision↓)"
- "현재 설정은 개인정보 유출 위험을 최소화하는 방향으로 조정됨"

#### 6.3.2 신뢰 구축을 위한 정보 제공

**제공 정보**:
1. 모델이 잘 작동하는 영역 명시
2. 모델이 취약한 영역 솔직히 공개
3. 예측 결과와 함께 신뢰도(확률) 제공
4. 이의제기 및 피드백 채널 안내

---

## 7. PoC 완료 조건 및 검증

### 7.1 기술적 완료 조건

| 조건 | 기준 | 측정 방법 |
|------|------|----------|
| Primary 지표 달성 | F1-macro ≥ 0.70 | Hold-out Test Set |
| 정탐 안정성 하한선 | Recall(정탐) ≥ 0.75 | Hold-out Test Set |
| 오탐 자동 분류 신뢰도 | Precision(오탐) ≥ 0.85 | Hold-out Test Set |
| 과적합 없음 | Train-Valid Gap < 10%p | CV 결과 비교 |
| 재현성 확보 | 동일 조건에서 성능 재현 | Seed 고정 후 재실행 |

### 7.2 비즈니스 완료 조건

| 조건 | 기준 | 검증 방법 |
|------|------|----------|
| 오탐 분류 효과 | 오탐의 70%+ 자동 분류 | Test Set 분석 |
| 정탐 유지율 | 정탐의 75%+ 정확히 분류 | Recall 확인 |
| 현업 수용성 | 샘플 검토 후 긍정적 피드백 | 현업 담당자 인터뷰 |

### 7.3 산출물 체크리스트

**필수 산출물**:
- [ ] 최종 모델 파일 (직렬화된 형태)
- [ ] 하이퍼파라미터 설정 파일
- [ ] 성능 리포트 문서
- [ ] Feature 정의 및 변환 로직 문서
- [ ] 오류 분석 결과 문서

**권장 산출물**:
- [ ] 모델 카드 (Model Card)
- [ ] Feature Importance 분석 결과
- [ ] Slice별 성능 분석 결과
- [ ] 재현 가능한 실행 스크립트

### 7.4 Model Card 템플릿

**Model Card 구성**:

```
1. 모델 개요
   - 모델명: Server-i 오탐 분류 모델 v1.0
   - 모델 유형: LightGBM Classifier
   - 학습 일자: YYYY-MM-DD
   - 버전: 1.0.0

2. 의도된 사용
   - Primary Use: Server-i DLP 검출 결과의 정탐/오탐 분류
   - 대상 사용자: 개인정보 관리 담당자
   - 사용 불가 영역: 실시간 차단 시스템 (배치 처리 전용)

3. 학습 데이터
   - 데이터 기간: YYYY-MM-DD ~ YYYY-MM-DD
   - 총 샘플 수: N개 (정탐 n개, 오탐 m개)
   - 데이터 출처: Server-i DLP 검출 로그

4. 평가 결과
   - F1-macro: 0.XX
   - Recall (정탐): 0.XX
   - Precision (오탐): 0.XX
   - PR-AUC (모니터링): 0.XX

5. 한계 및 권고
   - 소스코드 파일에서 성능 저하 (F1: 0.XX)
   - 새로운 오탐 패턴 출현 시 재학습 필요
   - 6개월 주기 성능 모니터링 권장

6. 윤리적 고려사항
   - 개인정보 보호 목적의 모델
   - 오분류로 인한 개인정보 유출 위험 존재
   - 인간 검토 병행 권장
```

---

## 8. v1.2 운영 자동화 검증 항목

### 8.1 Self-Validation Loop

docs/Architecture/ §11.3 기준으로, 현업 수동 피드백 대신 RULE↔ML 합의율과 시간 일관성으로 자동 검증한다.

| 지표 | 계산 방식 | 해석 |
|------|-----------|------|
| `overall_agreement` | RULE/ML 동시 분류 건 중 동일 클래스 비율 | 자동 검증 precision 프록시 |
| `temporal_consistency` | 동일 `pk_file`의 월간 분류 일치율 | 드리프트 탐지 지표 |

**판정 기준:**

- `temporal_consistency < 0.85` -> drift 경보
- `temporal_consistency < 0.75` -> 재학습 트리거 후보

### 8.2 Auto-Precision-Estimator

docs/Architecture/ §11.4 기준으로 자동 처리된 FP의 품질을 다음 식으로 평가한다.

```text
precision_est = (RULE과 ML이 동일 FP 클래스로 합의한 건수) / (자동 처리 FP 전체 건수)
```

| 상태 | 조건 | 자동 조치 |
|------|------|----------|
| `GOOD` | `precision_est >= 0.90` | 조치 없음 |
| `DEGRADED` | `precision_est < 0.90` | `Auto-Tuner` 보수화 + 재학습 검토 |

### 8.3 Auto-Remediation Playbook 검증

docs/Architecture/ §11.5 기준으로 KPI 알람과 자동 조치 매핑이 동작하는지 검증한다.

| KPI | 알람 조건 | 기대 자동 조치 |
|-----|----------|----------------|
| `parse_success_rate` | `< 0.95` | 파서 폴백 확장 + 로그 수집 |
| `quarantine_count` | `전월 대비 > 3배` | `trigger_auto_schema_detector` |
| `review_rate` | `> 0.35` | `auto_tune_tau` 또는 재학습 트리거 |
| `ood_rate` | `> 0.05` | UNKNOWN 클러스터링 + 재학습 트리거 |
| `auto_fp_precision_est` | `< 0.90` | TAU 보수화 + 재학습 트리거 |

### 8.4 Auto-Retrainer 검증

docs/Architecture/ §18.5 기준으로 자동 재학습 파이프라인을 다음 규칙으로 검증한다.

- 트리거: `oov_rate_raw > 0.30`, `confidence_p10 < 0.40`, `ood_rate > 0.05`, `auto_fp_precision_est < 0.90`, `temporal_consistency < 0.75`
- 안전장치 1: 새 모델 `tp_recall` 하락 시 즉시 롤백
- 안전장치 2: `macro_f1`이 기존 대비 `2%p` 이상 하락 시 롤백

### 8.5 라벨 품질 감사 (Confident Learning)

docs/Architecture/ §22 기준으로 **Confident Learning** 자동 감사를 평가 항목에 포함한다.

| 항목 | 기준 |
|------|------|
| 라벨 오류 후보 | OOF 예측과 라벨 불일치 + `confidence >= 0.90` |
| 노이즈율 | `error_count / total_samples` |
| 정제 수행 조건 | `noise_rate > 0.10` |
| 정제 방식(기본) | 오류 후보 학습 제외 (보수적) |

추가로 `pii_type_raw` vs `pii_type_inferred` 불일치율을 함께 보고하여 라벨 체계 품질을 모니터링한다.

---

## 부록 A: 평가 지표 수식 정리

### 이진 분류 지표

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall (Sensitivity, TPR) = TP / (TP + FN)

Specificity (TNR) = TN / (TN + FP)

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

F-beta = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

### 확률 기반 지표

```
Log Loss = -1/N × Σ[y_i × log(p_i) + (1-y_i) × log(1-p_i)]

Brier Score = 1/N × Σ(p_i - y_i)²
```

### 다중 클래스 지표

```
Macro Precision = 1/K × Σ Precision_k

Micro Precision = Σ TP_k / Σ (TP_k + FP_k)

Weighted Precision = Σ (n_k / N) × Precision_k
```

## 부록 B: 참고 문헌

1. Davis, J., & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. ICML.
2. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE TKDE.
3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
4. Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
5. Mitchell, M., et al. (2019). Model Cards for Model Reporting. FAT*.
6. Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. NeurIPS.
