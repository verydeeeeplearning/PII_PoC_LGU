# PII False Positive Reduction — 폐쇄망 E2E 실행 가이드

## 문서 정보

| 항목 | 내용 |
|------|------|
| 프로젝트 | Server-i 오탐 개선 AI 모델링 (PoC) |
| 아키텍처 기준 | `docs/Architecture/` v1.3 |
| 목적 | 폐쇄망 반입부터 PoC 리포트 생성까지 E2E 명령어 런북 |
| 최종 수정 | 2026-03-16 |

---

## 환경 정보

| 항목 | 내용 |
|------|------|
| 서버 OS | Linux x86_64 |
| 접속 도구 | PuTTY (SSH 터미널) |
| 파일 전송 | FileZilla (SFTP) |
| Python | 3.12 (Miniconda3) |
| 제약 | 인터넷 차단, GPU 없음 (CPU 전용) |

### 서버 사양

| 구분 | 사양 |
|------|------|
| CPU | Intel Xeon E5-2620 v4 × 2 (32 Threads) |
| Memory | 128 GB |
| Storage | 3.6 TB RAID1 |
| GPU | 없음 |

---

## 반입 경로

```text
Google Drive
  → LGU+ PC 로컬 다운로드
  → 자료전송 시스템 (업무 클라우드)
  → 자료전송 시스템 (보안 클라우드)
  → FileZilla (SFTP) → 개발서버
  → PuTTY (SSH) → 명령 실행
```

---

## Phase 0 — 외부(인터넷) 사전 준비

> **로컬 Windows PC에서 실행** (인터넷 연결 상태)

### Step 0-1: 번들 구조 확인

서버로 전송할 파일 목록:

```text
PII_False_Positive_Reduction_v3/
├── offline_packages/
│   ├── Miniconda3-py312-Linux-x86_64.sh   ← Python 인스톨러
│   └── wheels/                             ← 오프라인 pip wheel (117개+)
├── config/
├── src/
├── scripts/
├── tests/
├── docs/
├── data/
│   └── raw/
│       ├── label/                          ← (빈 폴더, 데이터는 별도 반입)
│       └── dataset_a/
│           └── sumologic_sample_202506.csv ← 참조 샘플 (포맷 확인용)
├── requirements.txt
├── CLAUDE.md
├── README.md
├── PROJECT_GUIDE.md
└── Env_scenario.md
```

### Step 0-2: 무결성 파일 생성 (선택)

```bash
# Windows PowerShell (로컬)
Get-ChildItem -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv checksums.csv
```

또는 리눅스 환경에서:

```bash
find . -type f -not -path "./.git/*" -exec sha256sum {} \; > checksums.sha256
```

---

## Phase 1 — 파일 전송 (FileZilla)

> **FileZilla** (SFTP) 로 프로젝트 전체 업로드

### Step 1-1: 연결 설정

```
호스트: <서버 IP>
포트: 22
프로토콜: SFTP
사용자: <계정>
```

### Step 1-2: 전송 대상 → 서버 경로

```
로컬: C:\Users\aquap\Desktop\PII False Positive Reduction_v3\
서버: /home/<계정>/pii_fpr/
```

**전송 필수 항목 (우선순위 순):**

| 우선순위 | 항목 | 크기 예상 |
|---------|------|----------|
| 1 | `offline_packages/` | ~3 GB (wheel + Miniconda) |
| 2 | `src/`, `scripts/`, `config/`, `tests/` | ~수 MB |
| 3 | `requirements.txt`, `README.md`, `Env_scenario.md` 등 루트 파일 | KB |
| 4 | `docs/` | ~수 MB |
| 5 | 실제 데이터 (`data/raw/label/`, `data/raw/dataset_a/`) | 별도 반입 |

> **주의:** `data/raw/` 아래 실제 레이블/검출 데이터는 보안 정책에 따라 별도 경로로 반입

### Step 1-3: 실제 데이터 반입 (별도)

```
data/raw/label/        ← 레이블 Excel (정탐/오탐 × 월 × 조직)
                          파일명 예: fp_label_202506_orgA.xlsx
data/raw/dataset_a/    ← Sumologic 검출 xlsx
                          파일명 예: sumologic_202506.xlsx
```

포맷 참조: `data/raw/dataset_a/sumologic_sample_202506.csv` (실제 Sumologic 컬럼 구조)

---

## Phase 2 — 서버 환경 구성 (PuTTY)

> **PuTTY** (SSH) 로 서버 접속 후 실행

### Step 2-1: SSH 접속

```bash
ssh <계정>@<서버IP>
cd ~/pii_fpr
```

### Step 2-2: 서버 환경 확인

```bash
# OS/아키텍처 확인
uname -m          # x86_64 이어야 함
ldd --version     # glibc 버전 확인

# 디스크 여유 공간 확인 (최소 20GB 권장)
df -h ~
```

**glibc 버전별 참고:**

| glibc 버전 | LightGBM wheel | 비고 |
|-----------|---------------|------|
| >= 2.28 | 4.5.0 (`manylinux_2_28`) | 권장 |
| 2.17 ~ 2.27 | 3.3.5 (`manylinux1`) | `offline_packages/wheels/`에 포함 |
| < 2.17 | 설치 불가 | OS 업그레이드 필요 |

### Step 2-3: 환경 자동 설치

```bash
cd ~/pii_fpr
chmod +x scripts/setup_env.sh
bash scripts/setup_env.sh
```

스크립트가 자동으로 수행하는 작업:
1. Python 3.12 확인 → 없으면 `offline_packages/Miniconda3-py312-Linux-x86_64.sh` 설치
2. glibc 버전 확인 및 경고 출력
3. `venv/` 가상환경 생성
4. `offline_packages/wheels/`에서 오프라인 패키지 설치
5. numpy/pandas/lightgbm/openpyxl/pyarrow 등 설치 확인

### Step 2-4: 수동 설치 (자동 설치 실패 시 대안)

```bash
# Miniconda 설치
bash offline_packages/Miniconda3-py312-Linux-x86_64.sh -b -p ~/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install --no-index --find-links=offline_packages/wheels -r requirements.txt

# glibc < 2.28인 경우 LightGBM 다운그레이드
pip install --no-index --find-links=offline_packages/wheels lightgbm==3.3.5
```

### Step 2-5: 환경 검증

```bash
source venv/bin/activate
python scripts/verify_env.py
```

정상 출력 예시:
```
[OK] numpy          1.26.4
[OK] pandas         2.2.3
[OK] lightgbm       4.5.0
[OK] openpyxl       3.1.5
[OK] pyarrow        20.0.0
...
전체 N개 패키지 정상 확인
```

---

## Phase 3 — 데이터 검증 및 전처리 (S0~S2)

> 이후 모든 명령은 **가상환경 활성화 상태**에서 실행

```bash
source ~/pii_fpr/venv/bin/activate
cd ~/pii_fpr
```

### Step 3-1: 데이터 배치 확인

```bash
ls data/raw/label/        # 레이블 Excel 파일 확인
ls data/raw/dataset_a/    # Sumologic xlsx 확인
```

### Step 3-2: Phase 0 데이터 품질 검증 (Go/No-Go)

```bash
# 레이블 Excel → silver_label.parquet 생성
python scripts/run_data_pipeline.py --source label

# 데이터 품질 검증 + Go/No-Go 판정
python scripts/run_phase0_validation.py
```

출력 확인 포인트:
- 충돌률 (동일 pk_file에 TP/FP 혼재): 임계값 이하인지
- Bayes Error 추정
- `Go` 판정이면 다음 단계 진행, `No-Go`이면 데이터 정합성 점검

### Step 3-3: Sumologic JOIN 전처리 (Phase 2용, 선택)

```bash
# Sumologic xlsx → silver_detections.parquet
python scripts/run_data_pipeline.py --source detection

# pk_file 기준 inner JOIN → silver_joined.parquet
python scripts/run_data_pipeline.py --source joined
```

산출물:

| 파일 | 경로 | 내용 |
|------|------|------|
| `silver_label.parquet` | `data/processed/` | 레이블 파이프라인 산출물 |
| `silver_detections.parquet` | `data/processed/` | Sumologic 표준화 산출물 |
| `silver_joined.parquet` | `data/processed/` | JOIN 산출물 (full_context_raw 포함) |

---

## Phase 4 — 모델 학습 및 평가 (S3a~S6)

### [Phase 1 기본] 레이블 단독 학습

```bash
# 전처리 + 학습 + 평가 한 번에
python scripts/run_pipeline.py --mode label-only
```

단계별 개별 실행 (디버그용):

```bash
# 전처리만 (S0-S2)
python scripts/run_pipeline.py --mode label-only --dry-run

# 학습만 (silver_label.parquet 준비된 경우)
python scripts/run_training.py --source label

# RULE Labeler(S3a) 포함 학습
python scripts/run_training.py --source label --use-filter

# 합성 변수 확장 포함 (Tier 1)
python scripts/run_training.py --source label --use-extended-features

# Repeated GroupShuffleSplit 5회 (분산 추정 — 표본 안정성 확인)
python scripts/run_training.py --source label --n-splits 5

# 확률 보정 활성화 (isotonic, cv=3 — Coverage-Precision Curve 신뢰도 향상)
python scripts/run_training.py --source label --calibrate

# 평가만
python scripts/run_evaluation.py
```

### [Phase 2] Sumologic JOIN 학습

```bash
# 전처리(JOIN 포함) + 학습 + 평가
python scripts/run_pipeline.py --mode full
```

단계별:

```bash
python scripts/run_training.py --source detection   # silver_joined.parquet 기반
python scripts/run_evaluation.py --include-filtered
```

---

## Phase 5 — PoC 리포트 생성

```bash
# Phase 1 기본 (레이블 단독)
python scripts/run_poc_report.py

# 출력 경로 지정
python scripts/run_poc_report.py --output outputs/poc_phase1_$(date +%Y%m%d).xlsx

# Rule 분석만 (ML 없이)
python scripts/run_poc_report.py --skip-ml

# Phase 2 리포트
python scripts/run_poc_report.py --phase 2
```

6-sheet Excel 산출물 (`outputs/poc_report.xlsx`):
1. `poc_metrics` — 전체 성능 요약
2. `rule_analyzer` — Rule별 기여도
3. `split_comparison` — Split 전략 비교
4. `coverage_curve` — FP Coverage-Precision 커브
5. `error_analysis` — 오분류 케이스
6. `class_stats` — 클래스별 통계

---

## Phase 6 — 산출물 반출 (FileZilla)

### Step 6-1: Parquet → 사람이 읽을 수 있는 형식 변환

```bash
# CSV 변환 (기본, utf-8-sig — Windows Excel 한글 호환)
python scripts/run_export.py

# Excel 변환
python scripts/run_export.py --format xlsx

# 스키마/행수 확인 (변환 없이)
python scripts/run_export.py --info
```

### Step 6-2: 반출 가능 파일 목록

| 파일 | 경로 | 비고 |
|------|------|------|
| PoC 리포트 | `outputs/poc_report.xlsx` | 성능/Rule/Coverage 분석 |
| 평가 리포트 | `outputs/classification_report.txt` | sklearn 분류 리포트 |
| Feature 중요도 | `outputs/feature_importance.csv` | 피처 기여도 |
| 예측 결과 (익명화) | `outputs/predictions/` | pk_event + 예측 클래스 |
| 학습된 모델 | `models/` | joblib 파일 |

> **반출 제한:** `data/raw/` 아래 원본 PII 데이터는 반출 금지

### Step 6-3: FileZilla 반출

```
서버: ~/pii_fpr/outputs/
로컬: C:\Users\aquap\Desktop\PII False Positive Reduction_v3\outputs\
```

---

## 일상 접속 패턴

```bash
# 1. SSH 접속
ssh <계정>@<서버IP>

# 2. 가상환경 활성화
cd ~/pii_fpr
source venv/bin/activate

# 3. 실행
python scripts/run_pipeline.py --mode label-only
```

---

## 운영 체크리스트

### 최초 반입 시

- [ ] glibc 버전 확인 (`ldd --version`)
- [ ] `bash scripts/setup_env.sh` 정상 완료
- [ ] `python scripts/verify_env.py` 전체 OK
- [ ] `data/raw/label/` 레이블 Excel 배치 완료
- [ ] `python scripts/run_phase0_validation.py` Go 판정

### 월 배치 실행 시

- [ ] 신규 레이블 Excel → `data/raw/label/` 배치
- [ ] 신규 Sumologic xlsx → `data/raw/dataset_a/` 배치 (Phase 2)
- [ ] `python scripts/run_pipeline.py --mode label-only` 정상 완료
- [ ] `python scripts/run_poc_report.py` 리포트 생성
- [ ] 성능 KPI 확인 (Macro F1 ≥ 0.70 / TP Recall ≥ 0.75 / FP Precision ≥ 0.85)
- [ ] 반출 파일 FileZilla 수령

### 문제 발생 시

| 증상 | 확인 사항 |
|------|----------|
| `ImportError: No module named 'openpyxl'` | `pip install --no-index --find-links=offline_packages/wheels openpyxl` |
| `ImportError: No module named 'pyarrow'` | `pip install --no-index --find-links=offline_packages/wheels pyarrow` |
| LightGBM 설치 실패 | glibc < 2.28이면 `lightgbm==3.3.5` wheel 수동 설치 |
| `silver_label.parquet` 없음 | `python scripts/run_data_pipeline.py --source label` 먼저 실행 |
| JOIN 결과 0행 | label/detection이 같은 데이터 소스인지 확인 (pk_file 기준) |
| 평가 시 클래스 1개 오류 | 데이터 샘플 수 부족 — 실데이터 전체로 재실행 |

---

## 보안 운영 원칙

1. 원본 PII 데이터는 `data/raw/` 외 경로로 복제 금지
2. 로그/리포트에 원문 PII 출력 금지 (마스킹 데이터만 허용)
3. 반출 파일은 집계/익명화 결과 중심으로 제한
4. 룰/모델 버전 및 실행 로그를 결과와 함께 보관

---

## 파일명/인코딩 규칙

### 파일명 규칙 (소문자 + 언더스코어 전용)

| 규칙 | 올바른 예 | 잘못된 예 |
|------|----------|----------|
| 소문자만 | `dataset_a/` | `Dataset_A/` |
| 공백 없음 | `silver_label.parquet` | `silver label.parquet` |
| 하이픈 금지 | `run_export.py` | `run-export.py` |

> 이유: PuTTY(Linux SSH)에서 공백/대소문자 혼용은 명령어 오류 유발

### 인코딩 정책

| 상황 | 인코딩 |
|------|--------|
| 서버(Linux) 원본 | `utf-8` |
| Windows 공유 CSV | `utf-8-sig` (BOM) |
| 한글 Windows 파일 fallback | `cp949` |
| Parquet | 바이너리 (무관) |
| 내보내기 CSV | `utf-8-sig` (Windows Excel 한글 호환) |

---

**문서 끝** | 기준일: 2026-03-16
