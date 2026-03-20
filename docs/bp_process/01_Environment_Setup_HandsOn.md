# 01. 환경 구성 Hands-On 가이드 (A to Z)

> **문서 버전**: 3.0
> **최종 수정일**: 2026-02-23
> **문서 목적**: 개발 PC 사전 준비 → 폐쇄망 반입 → 서버 설치 → 파이프라인 실행까지 **전체 절차**
> **WSL 검증 완료**: Ubuntu 24.04 / Python 3.12.8 / 116 wheels 완전 오프라인 설치 성공

---

## 전체 흐름

```
┌────────────────────────────────────────────────────────────────────┐
│                         전체 작업 흐름                              │
│                                                                    │
│  [개발 PC (Windows, 인터넷)]                                        │
│    Phase 1. Miniconda py312 다운로드                                │
│    Phase 2. Wheel 파일 확인 (116개, 이미 준비됨)                     │
│    Phase 3. 반입 준비 (번들 생성 + 분할)                             │
│           ↓                                                        │
│    Google Drive                                                    │
│           ↓                                                        │
│    LGU PC 로컬 다운로드                                             │
│           ↓                                                        │
│    자료전송 시스템 (업무 클라우드)                                    │
│           ↓                                                        │
│    자료전송 시스템 (보안 클라우드)                                    │
│           ↓                                                        │
│    SSH 전송                                                        │
│           ↓                                                        │
│  [폐쇄망 서버 (Linux x86_64)]                                       │
│    Phase 4. 서버 환경 정보 수집                                      │
│    Phase 5. 프로젝트 반입 (수신 → 검증 → 배치)                       │
│    Phase 6. setup_env.sh 실행 (자동 설치)                            │
│    Phase 7. 환경 검증                                               │
│    Phase 8. 데이터 배치 & 파이프라인 실행                             │
└────────────────────────────────────────────────────────────────────┘
```

### 전제 조건

| 항목 | 내용 |
|------|------|
| 개발 PC | Windows, 인터넷 연결 가능 |
| 서버 OS | Linux x86_64 (Ubuntu, CentOS, RHEL 등) |
| 서버 Python | **없어도 됨** (Miniconda로 설치) |
| 서버 glibc | 2.17+ 필수, 2.28+ 권장 (LightGBM 4.x) |
| 반입 수단 | Google Drive → LGU PC 로컬 다운로드 → 자료전송(업무 클라우드) → 자료전송(보안 클라우드) → **FileZilla**(SFTP 파일 전송) |
| 서버 접속 | **PuTTY** (Windows → Linux SSH 터미널) |
| 서버 관리자 권한 | 불필요 (사용자 홈 디렉토리에 설치) |

이론적 배경은 [01_Environment_Setup.md](01_Environment_Setup.md)를 참고하세요.

---

## 반입 경로

파일 반입은 다음 단일 경로를 통해 수행합니다:

```
Google Drive -> LGU PC 로컬환경 다운로드 -> 자료전송 시스템(업무 클라우드) -> 자료전송 시스템(보안 클라우드) -> SSH를 통해 파일 전송
```

자료전송 시스템의 파일 크기 제한에 따라 번들을 분할하여 전송합니다 (Phase 3 참조).

---

## Phase 1. Miniconda 설치파일 다운로드

### 왜 Miniconda인가

- 서버에 Python이 없거나 시스템 Python에 `venv`/`ensurepip` 모듈이 없을 수 있음
- Miniconda는 관리자 권한 없이 사용자 홈에 설치 가능
- 자체 Python + pip + venv 지원

### 다운로드

**반드시 Python 3.12 버전**을 다운로드하세요. 최신(latest)은 Python 3.13이므로 wheel 호환이 안 됩니다.

```
https://repo.anaconda.com/miniconda/Miniconda3-py312_24.11.1-0-Linux-x86_64.sh
```

> 약 140MB. **Linux x86_64용**입니다 (Windows용 아님).

다운로드 후 프로젝트 루트에 배치:
```
PII False Positive Reduction_v2/
└── Miniconda3-py312-Linux-x86_64.sh     ← 이 이름으로 저장
```

> **주의**: `Miniconda3-latest-Linux-x86_64.sh` (Python 3.13)을 받으면 wheel이 설치되지 않습니다.
> setup_env.sh는 `Miniconda3-py312-Linux-x86_64.sh`를 우선 탐색하고, 없으면 latest를 fallback으로 시도합니다.

---

## Phase 2. Wheel 파일 확인

`offline_packages/wheels/` 디렉토리에 **116개**의 `.whl` 파일이 이미 준비되어 있습니다.

### 포함된 패키지

| 카테고리 | 패키지 | 버전 |
|----------|--------|------|
| 데이터 처리 | numpy | 1.26.4 |
| | pandas | 2.2.3 |
| | scipy | 1.13.1 |
| ML 모델링 | scikit-learn | 1.5.2 |
| | xgboost | 2.1.4 |
| | lightgbm | 4.5.0 (+ 3.3.5 fallback) |
| | imbalanced-learn | 0.12.4 |
| | joblib | 1.4.2 |
| 시각화 | matplotlib | 3.9.4 |
| | seaborn | 0.13.2 |
| 유틸리티 | pyyaml | 6.0.1 |
| | tqdm | 4.67.1 |
| Jupyter | notebook | 7.0.2 |
| | ipykernel | 6.29.5 |
| | jupyterlab | 4.5.3 |
| 전이 의존성 | setuptools, requests, urllib3, tinycss2 등 | 총 100+ |

### 플랫폼 호환성

| 플랫폼 태그 | glibc 요구 | 해당 패키지 |
|-------------|-----------|------------|
| `manylinux2014` | 2.17+ (CentOS 7+) | numpy, pandas, scipy, sklearn 등 대부분 |
| `manylinux_2_28` | 2.28+ (CentOS 8+ / Ubuntu 20.04+) | lightgbm 4.5.0 |
| `manylinux1` | 2.5+ | lightgbm 3.3.5 (fallback) |
| `py3-none-any` | 무관 | 순수 Python 패키지들 |

> pip이 서버의 glibc 버전에 맞는 wheel을 자동 선택합니다.

---

## Phase 3. 반입 준비

### 3.1 번들 생성 및 분할

자료전송 시스템의 파일 크기 제한에 맞춰 번들을 분할합니다.

```bash
# Linux/WSL
bash scripts/bundle_release.sh --split-size 50M
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/bundle_release.ps1 -SplitSizeMB 50
```

> `--split-size`는 자료전송 시스템의 파일 크기 제한에 맞춰 조정하세요 (기본 50MB).

생성 결과:
```
outputs/releases/PII_FPR_bundle_YYYYMMDD_HHMMSS_split/
├── PII_FPR_bundle_*.tar.gz.00.part    # 분할 파일 1
├── PII_FPR_bundle_*.tar.gz.01.part    # 분할 파일 2
├── ...
├── checksums.sha256                    # SHA256 해시 (무결성 검증용)
└── REASSEMBLE_README.txt              # 재조립 안내
```

### 3.2 파일 크기 확인

```bash
# 분할 전 전체 번들 크기 확인
ls -lh outputs/releases/PII_FPR_bundle_*.tar.gz

# 분할 파일 목록 확인
ls -lh outputs/releases/*_split/
```

### 3.3 Google Drive 업로드

1. `_split/` 디렉토리 내의 **모든 파일**을 Google Drive에 업로드:
   - `.part` 파일들 (분할된 번들)
   - `checksums.sha256` (무결성 검증용)
   - `REASSEMBLE_README.txt` (재조립 안내)
2. LGU PC에서 Google Drive의 파일을 로컬에 다운로드
3. 자료전송 시스템(업무 클라우드)을 통해 보안 클라우드로 전송

### 3.4 주의사항

- 자료전송 시스템에서 `.sh` 확장자가 차단될 수 있음 → 확장자를 `.txt`로 변경 후 전송, 수신 후 복원
- `.whl` 파일은 ZIP 포맷이므로 대부분의 보안 스캐너와 호환
- 분할 파일 수가 많으면 전송 시간이 길어질 수 있으므로 `--split-size`를 크게 설정 (시스템 허용 범위 내)
- Miniconda 설치파일 (~140MB)은 **별도로** 전송해야 할 수 있음 (분할 필요 시 `split -b 50M`)

---

## Phase 4. 서버 환경 정보 수집 (첫 접속) ⚠️ 필수

서버에 처음 접속하면 **반드시** 아래 명령으로 환경 정보를 수집하세요.
이 정보가 이후 설치 성패를 결정합니다.

```bash
echo "========== OS =========="
cat /etc/os-release 2>/dev/null || cat /etc/redhat-release 2>/dev/null

echo "========== glibc =========="
ldd --version 2>&1 | head -1

echo "========== CPU =========="
lscpu | grep "Model name"
nproc

echo "========== Memory =========="
free -h

echo "========== Disk =========="
df -h ~

echo "========== GPU =========="
lspci 2>/dev/null | grep -i nvidia || echo "NVIDIA GPU 없음"
nvidia-smi 2>/dev/null || echo "nvidia-smi 사용 불가"

echo "========== Python =========="
python3 --version 2>/dev/null || echo "python3 없음"

echo "========== 사용자 =========="
whoami
echo $HOME
```

**확인 포인트:**

| 항목 | 기준 | 영향 |
|------|------|------|
| glibc 버전 | 2.17+ 필수, 2.28+ 권장 | < 2.17이면 wheel 설치 불가, < 2.28이면 LightGBM 3.3.5로 fallback |
| 디스크 여유 | 최소 3 GB | Miniconda ~500MB + venv ~2GB + 데이터/모델 |
| 메모리 | 최소 8 GB | TF-IDF 행렬 + 모델 학습 |
| 홈 디렉토리 쓰기 권한 | 필수 | Miniconda, venv 설치 위치 |

### 서버 OS별 호환성 매트릭스

> **현재 미확인 상태** — 서버 접속 후 아래 표에서 해당 OS를 찾아 조치하세요.

| OS | glibc | 전체 패키지 | LightGBM | 조치 |
|----|-------|------------|----------|------|
| **Ubuntu 20.04+** | 2.31+ | 전부 OK | 4.5.0 OK | 조치 불필요 |
| **Ubuntu 18.04** | 2.27 | 대부분 OK | 4.5.0 실패 | `scripts/setup_env.sh`가 자동으로 LightGBM 3.3.5를 선택 (또는 아래 수동 조치) |
| **RHEL / CentOS 8+** | 2.28+ | 전부 OK | 4.5.0 OK | 조치 불필요 |
| **RHEL / CentOS 7** | 2.17 | 대부분 OK | 4.5.0 실패 | `scripts/setup_env.sh`가 자동으로 LightGBM 3.3.5를 선택 (또는 아래 수동 조치) |
| **CentOS 6 이하** | < 2.17 | **설치 불가** | — | OS 업그레이드 필요 |

> **참고:** LightGBM 3.3.5 fallback wheel이 `offline_packages/wheels/`에 포함되어 있습니다.
> `scripts/setup_env.sh`는 glibc 버전을 확인해 `requirements_glibc228.txt` / `requirements_glibc217.txt` 중 하나를 자동 선택합니다.
> XGBoost 2.1.4는 `manylinux2014` (glibc 2.17+)이므로 CentOS 7에서도 동작합니다.
> 단, XGBoost도 glibc 2.28 미만에서 FutureWarning을 출력합니다 (동작에는 문제 없음).

### 실제 데이터 컬럼 매핑 확인

파이프라인이 기대하는 CSV 컬럼:

| 컬럼명 | 용도 | 필수 여부 |
|--------|------|-----------|
| `detected_text_with_context` | DLP 탐지 텍스트 + 전후 컨텍스트 | **필수** |
| `label` | 분류 레이블 (8종) | **필수** |
| `file_path` | 탐지된 파일 경로 | 권장 (tabular feature용) |
| `detection_id` | fallback 식별자 (복합 PK 보조) | 권장 |

**실제 DLP 추출 데이터의 컬럼명이 다른 경우:**
- `config/feature_config.yaml`의 `data.text_column`, `data.label_column`, `data.file_path_column`을 실제 컬럼명으로 수정하세요.
- 예: 실제 컬럼이 `detected_value`이면 → `data.text_column: detected_value`

**레이블 체계가 다른 경우:**
- 권장: 전처리 단계에서 실제 레이블을 표준 8개 레이블로 매핑하세요(코드 수정 최소화).
- 예외: 표준화가 어려우면 `src/utils/constants.py`의 `LABEL_TP`, `LABEL_FP_*` 상수를 프로젝트 규칙에 맞게 조정합니다.
- 현재 기본값(8클래스): `TP-실제개인정보`, `FP-숫자나열/코드`, `FP-더미데이터`, `FP-타임스탬프`, `FP-내부도메인`, `FP-bytes크기`, `FP-OS저작권`, `FP-패턴맥락`

### 한글 시각화 (matplotlib)

서버에 한글 폰트가 없으면 차트의 한글이 깨집니다 (□□□ 표시).

**해결 방법:**
1. `NanumGothic.ttf` 파일을 반입 패키지에 포함
2. 프로젝트 루트에 `fonts/` 디렉토리를 만들어 배치
3. 코드에서 폰트 설정:
```python
import matplotlib
matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.font_manager.fontManager.addfont('fonts/NanumGothic.ttf')
```

> 또는 한글 없이 영문으로만 출력해도 분석에는 지장 없습니다.

---

## Phase 5. 프로젝트 반입

### 5.1 보안 클라우드에서 파일 다운로드

1. 자료전송 시스템(보안 클라우드)에서 전송된 파일들을 다운로드합니다:
   - 모든 `.part` 파일
   - `checksums.sha256`
   - `REASSEMBLE_README.txt`
   - Miniconda 설치파일 (별도 전송한 경우)

### 5.2 무결성 검증 및 재조립

**Linux 환경인 경우:**

```bash
# 수신한 파일을 한 디렉토리에 모음
cd ~/received_files/

# 무결성 검증
sha256sum -c checksums.sha256

# 재조립
cat *.part > PII_FPR_bundle.tar.gz

# 압축 해제
mkdir -p ~/pii_project
tar xzf PII_FPR_bundle.tar.gz -C ~/pii_project
```

**Windows 환경인 경우:**

```powershell
cd C:\received_files\

# 무결성 검증
Get-Content checksums.sha256 | ForEach-Object {
  $parts = $_ -split '  '
  $hash = (Get-FileHash $parts[1] -Algorithm SHA256).Hash.ToLower()
  if ($hash -eq $parts[0]) { Write-Host "[OK] $($parts[1])" }
  else { Write-Host "[FAIL] $($parts[1])" -ForegroundColor Red }
}

# 재조립
$out = [System.IO.File]::Create("PII_FPR_bundle.zip")
Get-ChildItem *.part | Sort-Object Name | ForEach-Object {
  $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
  $out.Write($bytes, 0, $bytes.Length)
}
$out.Close()

# 압축 해제
Expand-Archive PII_FPR_bundle.zip -DestinationPath C:\pii_project
```

### 5.3 SSH로 폐쇄망 서버에 전송

**보안 클라우드에서 폐쇄망 서버로 SCP 전송:**

```bash
# Linux 환경
scp -r ~/pii_project/ user@server-ip:~/pii_project/

# 또는 tar.gz 자체를 전송 후 서버에서 압축 해제
scp PII_FPR_bundle.tar.gz user@server-ip:~/
ssh user@server-ip "mkdir -p ~/pii_project && tar xzf ~/PII_FPR_bundle.tar.gz -C ~/pii_project"
```

```powershell
# Windows 환경 (PowerShell에서 scp 사용)
scp -r C:\pii_project\ user@server-ip:~/pii_project/
```

> Miniconda 설치파일을 별도로 전송한 경우, 프로젝트 루트에 배치하세요:
> `scp Miniconda3-py312-Linux-x86_64.sh user@server-ip:~/pii_project/`

### 5.4 서버에서 확인

```bash
# 파일 구조 확인
ls ~/pii_project/
ls ~/pii_project/offline_packages/wheels/ | wc -l    # 116이면 정상
ls -lh ~/pii_project/Miniconda3-py312-Linux-x86_64.sh
```

---

## Phase 6. 환경 설치

### 6.1 자동 설치 (권장)

```bash
cd ~/pii_project
chmod +x scripts/setup_env.sh
bash scripts/setup_env.sh
```

**스크립트가 자동으로 수행하는 작업:**

```
[Step 1] Python 3.12 확인
         → 시스템 Python에 venv+ensurepip 있으면 사용
         → 없으면 Miniconda3-py312 자동 설치 (~/.miniconda3)

[Step 2] 가상환경 생성
         → ~/pii_project/venv/

[Step 3] pip 확인

[Step 4] 패키지 오프라인 설치
         → offline_packages/wheels/ 에서 116개 wheel 설치

[Step 5] Jupyter Kernel 등록
         → "PII FP Reduction (Python 3.12)" 커널

[Step 6] 환경 검증
         → 14개 핵심 패키지 import + 버전 확인
```

### 6.2 수동 설치 (스크립트 사용 불가 시)

```bash
# 1. Miniconda 설치
cd ~/pii_project
bash Miniconda3-py312-Linux-x86_64.sh -b -p ~/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"

# 2. PATH 영구 설정
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 3. 가상환경 생성
cd ~/pii_project
python3 -m venv venv
source venv/bin/activate

# 4. 패키지 오프라인 설치
pip install --no-index --find-links=offline_packages/wheels/ -r requirements.txt

# 5. Jupyter Kernel 등록
python -m ipykernel install --user --name pii-fp-reduction \
    --display-name "PII FP Reduction (Python 3.12)"

# 6. 환경 검증
python scripts/verify_env.py
```

---

## Phase 7. 환경 검증

### 7.1 검증 스크립트

```bash
cd ~/pii_project
source venv/bin/activate
python scripts/verify_env.py
```

정상 출력:
```
============================================================
PII False Positive Reduction - 환경 검증
============================================================

Python: 3.12.8 | packaged by Anaconda, Inc. | ...

패키지                버전              상태
--------------------------------------------------
  [OK] numpy                1.26.4
  [OK] pandas               2.2.3
  [OK] scipy                1.13.1
  [OK] scikit-learn         1.5.2
  [OK] xgboost              2.1.4
  [OK] lightgbm             4.5.0
  [OK] imbalanced-learn     0.12.4
  [OK] joblib               1.4.2
  [OK] matplotlib           3.9.4
  [OK] seaborn              0.13.2
  [OK] pyyaml               6.0.1
  [OK] tqdm                 4.67.1
  [OK] notebook             7.0.2
  [OK] ipykernel            6.29.5

============================================================
  전체 14개 패키지 정상 확인
  환경이 정상적으로 구성되었습니다.
============================================================
```

### 7.2 Jupyter 실행 확인

```bash
source venv/bin/activate
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

- 출력되는 URL의 토큰을 복사
- 브라우저에서 `http://<서버IP>:8888/?token=...` 접속
- Kernel 선택에서 **"PII FP Reduction (Python 3.12)"** 확인

### 7.3 src 모듈 import 확인

```bash
source venv/bin/activate
cd ~/pii_project
python -c "
from src.utils.constants import RANDOM_SEED, LABEL_TP
from src.data.loader import load_raw_data
from src.features.pipeline import build_features
from src.models.trainer import train_xgboost
from src.evaluation.evaluator import full_evaluation
print('모든 src 모듈 import 성공')
"
```

---

## Phase 8. 데이터 배치 & 파이프라인 실행

### 8.1 데이터 배치

**FileZilla로 데이터 업로드 후 배치 확인:**

```bash
# data/raw/ 에 원본 데이터 배치 (FileZilla로 전송)
ls ~/pii_project/data/raw/
# → dataset_a/*.xlsx, dataset_b/*.xlsx
```

**파일명 규칙 확인 (PuTTY 터미널에서):**

```bash
# 파일명에 공백/대문자/하이픈이 없는지 확인
ls data/raw/dataset_a/
# 올바름:  오탐_취합_6월.xlsx, server_i_raw.xlsx
# 잘못됨:  오탐 취합 6월.xlsx (공백), Server-i-Raw.xlsx (대문자+하이픈)
```

> **파일명 규칙**: 소문자 + 언더스코어만 사용. 공백·대문자·하이픈 금지.
> Linux(PuTTY) 환경에서 공백이 포함된 경로는 명령어 실행 오류를 유발합니다.

**지원 형식 (자동 인식 순서)**: `.xlsx` → `.xls` → `.csv` → `.parquet`

### 8.2 스크립트 실행 (PuTTY 터미널)

> **주의**: Jupyter 노트북은 사용하지 않습니다. 모든 작업은 스크립트로 실행합니다.

```bash
source venv/bin/activate
cd ~/pii_project

# 1단계: 데이터 파이프라인 (xlsx 로드 → Silver Parquet 출력)
python scripts/run_data_pipeline.py

# 2단계: 학습
python scripts/run_training.py
python scripts/run_training.py --use-filter        # 선택: RULE/ML Labeler 체인 포함

# 3단계: 평가
python scripts/run_evaluation.py

# 4단계: 산출물 내보내기 (FileZilla로 내려받기 전 CSV/Excel 변환)
python scripts/run_export.py                       # Parquet → CSV (utf-8-sig)
python scripts/run_export.py --format xlsx         # Parquet → Excel
```

### 8.3 결과 확인 및 반출 (FileZilla)

```bash
# 학습된 모델
ls -lh models/final/best_model_v1.joblib

# 평가 리포트
cat outputs/reports/classification_report.txt

# 내보내기 결과 확인
ls data/processed/*.csv
ls outputs/exports/
```

**FileZilla로 반출할 파일:**

| 경로 | 내용 |
|------|------|
| `outputs/reports/` | 성능 리포트, 분류 보고서 |
| `outputs/figures/` | 시각화 이미지 |
| `outputs/exports/` | `run_export.py` 생성 CSV/Excel |

---

## 트러블슈팅

### Q: setup_env.sh 실행 시 "permission denied"
```bash
chmod +x scripts/setup_env.sh
bash scripts/setup_env.sh     # sh 대신 bash로 실행
```

### Q: "No matching distribution found for numpy==1.26.4"
Miniconda 버전이 Python 3.13일 가능성. 확인:
```bash
python3 --version   # 3.12.x 여야 함
```
Python 3.13이면 `Miniconda3-py312-Linux-x86_64.sh`로 재설치.

### Q: "GLIBC_2.28 not found" (LightGBM)
서버 glibc가 2.28 미만. fallback 버전 설치:
```bash
source venv/bin/activate
pip install --no-index --find-links=offline_packages/wheels/ lightgbm==3.3.5 --force-reinstall
```

### Q: Miniconda 설치 후 python3 명령이 안 됨
```bash
export PATH="$HOME/miniconda3/bin:$PATH"
# 영구 설정
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
```

### Q: Jupyter에서 커널이 안 보임
```bash
source venv/bin/activate
python -m ipykernel install --user --name pii-fp-reduction \
    --display-name "PII FP Reduction (Python 3.12)"
```

### Q: "ensurepip is not available"
시스템 Python에 venv 모듈이 없음. 정상입니다 — setup_env.sh가 자동으로 Miniconda로 fallback합니다.

### Q: matplotlib 한글 깨짐
서버에 한글 폰트가 없는 경우. 프로젝트는 `FONT_FAMILY = "DejaVu Sans"`를 기본으로 사용하므로 영문 출력은 정상. 한글 필요 시 폰트 파일 반입 필요.

### Q: SCP 전송 실패

**연결 거부 시:**
```bash
# 서버에서 SSH 서비스 확인
systemctl status sshd

# 방화벽 포트 확인 (22번)
ss -tlnp | grep 22
```

**키 인증 실패 시:**
```bash
# 비밀번호 인증으로 시도
scp -o PreferredAuthentications=password -r ~/pii_project/ user@server-ip:~/

# 또는 관리자에게 SSH 키 등록 요청
```

**대용량 전송 중 끊어질 때:**
```bash
# rsync 사용 (재시작 가능)
rsync -avz --progress ~/pii_project/ user@server-ip:~/pii_project/
```

---

## 검증 완료 이력

| 환경 | Python | glibc | 패키지 수 | 결과 |
|------|--------|-------|-----------|------|
| WSL Ubuntu 24.04 (x86_64) | 3.12.8 (Miniconda) | 2.39 | 116 wheels | **PASS** — 전체 오프라인 설치 성공 |
| Windows 11 (x86_64) | 3.12 (MS Store) | N/A | pip install | **PASS** — E2E 파이프라인 동작 확인 (더미 400건) |

### E2E 파이프라인 검증 결과 (더미 데이터)

| 단계 | 결과 |
|------|------|
| 더미 데이터 생성 (400건, 8클래스) | OK |
| Feature Engineering (TF-IDF 230 + Dense 23 = 253) | OK |
| 모델 학습 (Baseline + XGB×2 + LGB×2) | OK |
| 모델 저장 (`models/final/best_model_v1.joblib`) | OK |
| 평가 리포트 + Confusion Matrix + Feature Importance | OK |
| PoC 기준 판정 (F1/Recall/Precision) | OK |

---

## 현장 이식 전 체크리스트

> 서버 환경 확인 전까지 아래 항목은 **미확인 상태**입니다.
> 첫 서버 접속 시 Phase 4의 명령어를 실행하여 확인하세요.

```
[ ] 반입 경로 확인 (Google Drive -> LGU PC 로컬환경 다운로드 -> 자료전송 시스템(업무 클라우드) -> 자료전송 시스템(보안 클라우드) -> SSH를 통해 파일 전송)

[ ] 서버 OS 및 glibc 버전 확인 (ldd --version)
    → glibc 2.28+ : 그대로 진행
    → glibc 2.17~2.27 : LightGBM 3.3.5 fallback 필요
    → glibc < 2.17 : OS 업그레이드 필요

[ ] 실제 DLP 데이터 컬럼명 확인
    → `config/feature_config.yaml`의 `data.text_column`, `data.label_column` 수정

[ ] 레이블 체계 확인
    → 가능하면 전처리에서 표준 8종으로 매핑, 불가 시 `src/utils/constants.py` 조정

[ ] 한글 폰트 필요 여부 판단
    → 차트에 한글이 필요하면 NanumGothic.ttf 반입

[ ] 디스크 여유 공간 3GB+ 확인
[ ] 홈 디렉토리 쓰기 권한 확인
```

---

**문서 끝**

