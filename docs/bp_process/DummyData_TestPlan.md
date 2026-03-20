# Dummy Data 파이프라인 테스트 계획서

> **목적:** 더미 데이터를 생성하여 전체 ML 파이프라인(`data → features → training → evaluation`)이
> 정상 동작하는지 End-to-End로 검증한다.

---

## 1. 개요

| 항목 | 내용 |
|------|------|
| **테스트 대상** | `src/` 전체 모듈 + `scripts/run_training.py` + `scripts/run_evaluation.py` |
| **더미 데이터 규모** | 약 **1,600건** (클래스당 200건 × 8 클래스) |
| **생성 스크립트** | `scripts/generate_dummy_data.py` |
| **출력 파일** | `data/processed/merged_cleaned.csv` |

---

## 2. 데이터 스키마

파이프라인이 요구하는 최소 컬럼:

| # | 컬럼명 | 타입 | 설명 | 필수 |
|---|--------|------|------|------|
| 1 | `detection_id` | str | 더미 테스트용 PK(실운영은 복합 PK + fallback detection_id) | O |
| 2 | `detected_text_with_context` | str | DLP가 탐지한 텍스트 + 전후 컨텍스트 | O |
| 3 | `label` | str | 8종 클래스 레이블 | O |
| 4 | `file_path` | str | 탐지된 파일 경로 | O (tabular features용) |

---

## 3. 클래스별 더미 텍스트 설계

### 3.1 TP-실제개인정보 (200건)

실제 주민번호/전화번호/이메일 **패턴**을 포함하는 텍스트.

```
예시:
- "고객 홍길동의 연락처는 010-1234-5678 입니다"
- "주민등록번호: 850101-1234567 확인 바랍니다"
- "이메일 주소: user@company.com 으로 발송해주세요"
```

파일 경로 패턴: `/data/customers/`, `/home/user/documents/`

### 3.2 FP-더미데이터 (200건)

테스트용 키워드가 포함된 텍스트.

```
예시:
- "test case: 010-0000-0000 검증용 데이터"
- "테스트 계정 test_user@test.com 입니다"
- "testing phone number 000-0000-0000"
```

파일 경로 패턴: `/test/`, `/tests/`, `/src/test/`

### 3.3 FP-숫자나열/코드 (200건)

버전/식별자/숫자 코드 패턴이 포함된 텍스트.

```
예시:
- "version: 1.3.3.32-2087-1512"
- "제품코드 AB-1234567-CD 재고 확인"
- "주문번호 2024-0101-5678 배송완료"
```

파일 경로 패턴: `/src/`, `/dev/`, `/misc/`, `/archive/`

### 3.4 FP-패턴맥락 (200건)

파일 경로/로그 문맥으로 오탐 판단이 가능한 텍스트.

```
예시:
- "hadoop-cmf-hdfs-DATANODE-server01.log.out"
- "/var/lib/docker/overlay2/.../merged/app.log"
- "system log context: non-user generated token"
```

파일 경로 패턴: `/docker/`, `/overlay2/`, `/var/log/`, `/hadoop/`

### 3.5 FP-내부도메인 (200건)

내부 이메일 도메인이 포함된 텍스트.

```
예시:
- "담당자: user@lguplus.co.kr"
- "알림 수신: admin@bdp.lguplus.co.kr"
- "운영계정: svc@map.lguplus.co.kr"
```

파일 경로 패턴: `/mail/`, `/auth/`, `/internal/`

### 3.6 FP-타임스탬프 (200건)

시간/날짜/타임스탬프 패턴.

```
예시:
- "timestamp: 1704067200"
- "xpiryDate=1706031234 duration: 148400"
- "generated timestamp: 20240315143022"
```

파일 경로 패턴: `/var/log/`, `/system/`, `/logs/`, 확장자 `.log`

### 3.7 FP-OS저작권 (200건)

오픈소스/OS 저작권 도메인이 포함된 텍스트.

```
예시:
- "Copyright by user@redhat.com"
- "Maintainer: author@apache.org"
- "License contact: dev@fedoraproject.org"
```

파일 경로 패턴: `/licenses/`, `/opensource/`, `/system/docs/`

### 3.8 FP-bytes크기 (200건)

파일 크기/용량 표현이 포함된 텍스트.

```
예시:
- "size: 45 bytes 141022"
- "content-length: 256bytes"
- "file size=1024 KB"
```

파일 경로 패턴: `/logs/`, `/storage/`, `/system/`

---

## 4. 생성 스크립트 사양

### 4.1 스크립트 위치

```
scripts/generate_dummy_data.py
```

### 4.2 실행 방법

```bash
# 기본 실행 (1,600건)
python scripts/generate_dummy_data.py

# 건수 조정
python scripts/generate_dummy_data.py --samples-per-class 500
```

### 4.3 생성 로직

1. 클래스별 **텍스트 템플릿** 풀(pool)을 정의 (20~30개/클래스)
2. 템플릿에서 랜덤 선택 + **변수 치환** (이름, 숫자, 날짜 등)
3. `file_path`를 클래스별 패턴에서 랜덤 생성
4. `detection_id`는 `DET-{순번:06d}` 형식
5. 모든 랜덤 연산에 `RANDOM_SEED=42` 고정
6. 결과를 `data/processed/merged_cleaned.csv`로 저장

---

## 5. End-to-End 테스트 절차

### Phase 1: 더미 데이터 생성

```bash
cd /path/to/project
python scripts/generate_dummy_data.py
```

**검증:**
- `data/processed/merged_cleaned.csv` 생성 확인
- 행 수: 1,600건
- 컬럼: `detection_id`, `detected_text_with_context`, `label`, `file_path`
- 클래스 분포: 8개 클래스 × 200건

### Phase 2: 학습 파이프라인 실행

```bash
python scripts/run_training.py
# 선택: 필터 동시 적용
python scripts/run_training.py --use-filter
```

**검증:**
- Feature Engineering 정상 완료 (TF-IDF + 확장 텍스트/경로 Feature + 합성 상호작용)
- 5개 모델 학습 완료 (Baseline, XGB×2, LGB×2)
- `models/final/best_model_v1.joblib` 저장 확인
- `data/features/` 아티팩트 저장 확인

### Phase 3: 평가 파이프라인 실행

```bash
python scripts/run_evaluation.py
```

**검증:**
- Classification Report 출력
- Confusion Matrix 생성
- PoC 기준 체크 실행 (F1-macro, TP Recall, FP Precision)
- `outputs/reports/` 산출물 확인

### Phase 4: 노트북 실행 (선택)

Jupyter 노트북 01~05를 순서대로 실행하여 시각화/분석 기능 검증.

---

## 6. 예상 결과

| 지표 | 기대값 | 비고 |
|------|--------|------|
| 모델 학습 | 오류 없이 완료 | 5개 모델 모두 |
| F1-macro | 0.80+ | 더미 데이터는 패턴이 명확하므로 높게 예상 |
| TP Recall | 0.85+ | 실제 PII 패턴이 뚜렷 |
| FP Precision | 0.90+ | 키워드/경로 특성이 강함 |

> **참고:** 더미 데이터는 의도적으로 클래스별 특성을 명확하게 설계하므로,
> 실제 데이터 대비 높은 성능이 나올 수 있습니다. 이 테스트의 목적은
> **파이프라인 동작 검증**이지 모델 성능 벤치마크가 아닙니다.

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError` | 가상환경 미활성화 | `source venv/bin/activate` |
| `FileNotFoundError: merged_cleaned.csv` | 더미 데이터 미생성 | `python scripts/generate_dummy_data.py` 먼저 실행 |
| `ValueError: min_df > corpus size` | 데이터 건수 부족 | `--samples-per-class 100` 이상 권장 |
| XGBoost/LightGBM import error | 패키지 미설치 | `pip install --no-index --find-links=./offline_packages/wheels xgboost lightgbm` |

---

## 8. 파일 체크리스트

테스트 완료 후 생성되어야 하는 파일들:

```
✅ data/processed/merged_cleaned.csv          (더미 데이터)
✅ models/tfidf_vectorizer.joblib              (TF-IDF 모델)
✅ data/features/X_train.npz                   (학습 Feature)
✅ data/features/X_test.npz                    (테스트 Feature)
✅ models/final/best_model_v1.joblib           (최종 모델)
✅ outputs/reports/classification_report.txt   (분류 리포트)
✅ outputs/figures/confusion_matrix.png        (혼동 행렬)
```

