#!/usr/bin/env bash
# =============================================================================
# PII False Positive Reduction - 폐쇄망 환경 자동 설정 스크립트
# =============================================================================
#
# 사용법:
#   chmod +x scripts/setup_env.sh
#   bash scripts/setup_env.sh                  # 기본 실행
#
# 전달 경로:
#   Google Drive → LGU PC → 자료전송 시스템(업무 클라우드) → 자료전송 시스템(보안 클라우드) → SSH
#
# 사전 준비:
#   1. Miniconda3-py312-Linux-x86_64.sh  -> 프로젝트 루트
#   2. offline_packages/wheels/           -> Linux x86_64 wheel 파일들
#   3. 프로젝트 폴더 전체
#
# 실행 결과:
#   - ~/miniconda3  에 Python 3.12 설치 (이미 있으면 건너뜀)
#   - 프로젝트 루트/venv  가상환경 생성 + 패키지 설치
#   - ipykernel 등록 (Jupyter에서 사용 가능)
# =============================================================================

set -euo pipefail

# ── 경로 설정 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WHEELS_DIR="$PROJECT_DIR/offline_packages/wheels"
VENV_DIR="$PROJECT_DIR/venv"
REQ_FILE="$PROJECT_DIR/requirements.txt"
# Miniconda 인스톨러: offline_packages/ 우선, 루트 fallback
if [[ -f "$PROJECT_DIR/offline_packages/Miniconda3-py312-Linux-x86_64.sh" ]]; then
    MINICONDA_INSTALLER="$PROJECT_DIR/offline_packages/Miniconda3-py312-Linux-x86_64.sh"
elif [[ -f "$PROJECT_DIR/Miniconda3-py312-Linux-x86_64.sh" ]]; then
    MINICONDA_INSTALLER="$PROJECT_DIR/Miniconda3-py312-Linux-x86_64.sh"
elif [[ -f "$PROJECT_DIR/Miniconda3-latest-Linux-x86_64.sh" ]]; then
    MINICONDA_INSTALLER="$PROJECT_DIR/Miniconda3-latest-Linux-x86_64.sh"
else
    MINICONDA_INSTALLER="$PROJECT_DIR/offline_packages/Miniconda3-py312-Linux-x86_64.sh"  # 오류 메시지용
fi
MINICONDA_HOME="$HOME/miniconda3"

_ver_lt() {
    # Usage: _ver_lt 2.17 2.28  -> true if first < second
    local a="$1"
    local b="$2"
    [[ "$(printf '%s\n' "$a" "$b" | sort -V | head -n1)" == "$a" && "$a" != "$b" ]]
}

_glibc_version() {
    # Extract the numeric version from `ldd --version` first line.
    # Examples:
    #   ldd (Ubuntu GLIBC 2.35-0ubuntu3.4) 2.35
    #   ldd (GNU libc) 2.17
    ldd --version 2>/dev/null | head -1 | awk '{print $NF}'
}

echo "============================================================"
echo " PII False Positive Reduction - 환경 설정"
echo "============================================================"
echo "  프로젝트 경로: $PROJECT_DIR"
echo "  Wheels 경로:   $WHEELS_DIR"
echo "  가상환경 경로: $VENV_DIR"
echo "============================================================"
echo ""

# ── Step 1: Python 확인 / Miniconda 설치 ──
echo "[Step 1] Python 환경 확인"
echo "------------------------------------------------------------"

PYTHON_CMD=""

# venv + ensurepip 모듈 사용 가능 여부를 검사하는 함수
_check_venv() {
    "$1" -c "import venv; import ensurepip" 2>/dev/null
}

# 1-1. 시스템에 Python 3.12 + venv 있는지 확인
if command -v python3.12 &>/dev/null && _check_venv python3.12; then
    PYTHON_CMD="python3.12"
    echo "  [OK] python3.12 발견 (venv 사용 가능): $($PYTHON_CMD --version)"
elif command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PY_VER" == "3.12" ]] && _check_venv python3; then
        PYTHON_CMD="python3"
        echo "  [OK] python3 (3.12, venv 사용 가능) 발견"
    else
        echo "  [!] python3 발견 ($PY_VER) - venv 모듈 없거나 버전 불일치"
    fi
fi

# 1-2. Miniconda로 Python 확보 (시스템 Python이 없거나 venv 불가)
if [[ -z "$PYTHON_CMD" ]]; then
    echo "  사용 가능한 Python 3.12 + venv가 없습니다. Miniconda를 사용합니다."

    if [[ -f "$MINICONDA_HOME/bin/python3" ]]; then
        echo "  [OK] 기존 Miniconda 발견: $MINICONDA_HOME"
        PYTHON_CMD="$MINICONDA_HOME/bin/python3"
    elif [[ -f "$MINICONDA_INSTALLER" ]]; then
        echo "  Miniconda 설치 중..."
        bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_HOME"
        PYTHON_CMD="$MINICONDA_HOME/bin/python3"
        echo "  [OK] Miniconda 설치 완료: $MINICONDA_HOME"
    else
        echo "  [오류] Miniconda 설치파일을 찾을 수 없습니다."
        echo "         다음 위치에 설치파일을 배치하세요:"
        echo "         $MINICONDA_INSTALLER"
        echo ""
        echo "  [힌트] 자료전송 시스템을 통해 번들 파일을 전송 후 SCP로 배포했는지 확인하세요."
        exit 1
    fi

    # PATH에 추가 (현재 세션)
    export PATH="$MINICONDA_HOME/bin:$PATH"
fi

echo "  사용할 Python: $($PYTHON_CMD --version 2>&1)"
echo ""

# ── Step 1-b: glibc 버전 확인 + requirements 선택 ──
echo "[Step 1-b] 서버 glibc 확인"
echo "------------------------------------------------------------"

GLIBC_VER="$(_glibc_version || true)"

if [[ -z "$GLIBC_VER" ]]; then
    echo "  [경고] glibc 버전을 확인할 수 없습니다 (ldd 없음?)."
else
    echo "  glibc 버전: $GLIBC_VER"

    # 최소 요구(manylinux2014 기준): glibc 2.17+
    if _ver_lt "$GLIBC_VER" "2.17"; then
        echo "  [오류] glibc $GLIBC_VER 는 너무 낮습니다. (최소 2.17 필요)"
        echo "         OS 업그레이드 또는 별도 빌드가 필요합니다."
        exit 1
    fi

    # LightGBM 4.5.0 wheel은 manylinux_2_28 기준 → glibc < 2.28이면 3.3.5 wheel 사용 가능
    if _ver_lt "$GLIBC_VER" "2.28"; then
        echo "  [주의] glibc $GLIBC_VER < 2.28: LightGBM 4.5.0 대신 3.3.5 wheel이 선택될 수 있습니다."
        echo "         (offline_packages/wheels/에 lightgbm-3.3.5 wheel 포함됨)"
    fi
fi

echo "  requirements: $REQ_FILE"
echo ""

# ── Step 2: 가상환경 생성 ──
echo "[Step 2] 가상환경 생성"
echo "------------------------------------------------------------"

if [[ -d "$VENV_DIR" ]]; then
    echo "  [!] 기존 가상환경 발견: $VENV_DIR"
    if [[ -t 0 ]]; then
        # 대화형 모드: 사용자에게 물어봄
        read -r -p "  삭제 후 재생성하시겠습니까? (y/N): " REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            echo "  기존 가상환경 삭제 완료"
        else
            echo "  기존 가상환경을 유지합니다."
        fi
    else
        # 비대화형 모드: 자동 재생성
        echo "  비대화형 모드 - 기존 가상환경 삭제 후 재생성"
        rm -rf "$VENV_DIR"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "  [OK] 가상환경 생성 완료: $VENV_DIR"
fi

# 가상환경 활성화
source "$VENV_DIR/bin/activate"
echo "  [OK] 가상환경 활성화됨: $(which python)"
echo ""

# ── Step 3: pip 업그레이드 (wheel에서) ──
echo "[Step 3] pip 업그레이드"
echo "------------------------------------------------------------"

# wheels에 pip가 있으면 사용, 없으면 내장 pip 사용
PIP_WHL=$(find "$WHEELS_DIR" -name "pip-*.whl" 2>/dev/null | head -1)
if [[ -n "$PIP_WHL" ]]; then
    python -m pip install --no-index --find-links="$WHEELS_DIR" pip 2>/dev/null || true
fi
echo "  pip 버전: $(python -m pip --version)"
echo ""

# ── Step 4: 패키지 설치 (오프라인) ──
echo "[Step 4] 패키지 설치 (오프라인, wheels)"
echo "------------------------------------------------------------"

if [[ ! -d "$WHEELS_DIR" ]] || [[ -z "$(ls -A "$WHEELS_DIR"/*.whl 2>/dev/null)" ]]; then
    echo "  [오류] Wheel 파일이 없습니다: $WHEELS_DIR"
    echo "         offline_packages/wheels/ 에 .whl 파일을 배치하세요."
    echo ""
    echo "  [힌트] 번들 분할 파일을 수신 후 재조립했는지 확인하세요."
    echo "         cat PII_FPR_bundle_*.tar.gz.part* > bundle.tar.gz && tar xzf bundle.tar.gz"
    exit 1
fi

WHL_COUNT=$(ls "$WHEELS_DIR"/*.whl 2>/dev/null | wc -l)
echo "  발견된 wheel 파일: ${WHL_COUNT}개"

# requirements.txt 기반 설치
echo ""
echo "  [4-1] requirements 기반 패키지 설치..."
echo "    requirements: $REQ_FILE"
python -m pip install \
    --no-index \
    --find-links="$WHEELS_DIR" \
    -r "$REQ_FILE" \
    2>&1 | while IFS= read -r line; do echo "    $line"; done

echo ""
echo "  [OK] 패키지 설치 완료"
echo ""

# ── Step 5: ipykernel 등록 ──
echo "[Step 5] Jupyter Kernel 등록"
echo "------------------------------------------------------------"

python -m ipykernel install \
    --user \
    --name pii-fp-reduction \
    --display-name "PII FP Reduction (Python 3.12)" \
    2>&1 | while IFS= read -r line; do echo "    $line"; done

echo "  [OK] Jupyter Kernel 등록 완료"
echo ""

# ── Step 6: 환경 검증 ──
echo "[Step 6] 환경 검증"
echo "------------------------------------------------------------"

python -c "
import sys
print(f'  Python: {sys.version}')

packages = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('scipy', 'scipy'),
    ('sklearn', 'scikit-learn'),
    ('xgboost', 'xgboost'),
    ('lightgbm', 'lightgbm'),
    ('imblearn', 'imbalanced-learn'),
    ('joblib', 'joblib'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('yaml', 'pyyaml'),
    ('tqdm', 'tqdm'),
    ('openpyxl', 'openpyxl'),
    ('pyarrow', 'pyarrow'),
    ('pytest', 'pytest'),
]

ok_count = 0
fail_count = 0

for import_name, display_name in packages:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', 'OK')
        print(f'  [OK] {display_name:20s} {ver}')
        ok_count += 1
    except ImportError as e:
        print(f'  [X]  {display_name:20s} FAILED ({e})')
        fail_count += 1

print()
if fail_count == 0:
    print(f'  전체 {ok_count}개 패키지 정상 확인')
else:
    print(f'  성공: {ok_count}개 / 실패: {fail_count}개')
    print('  실패한 패키지를 확인하세요.')
"

echo ""
echo "============================================================"
echo " 환경 설정 완료"
echo "============================================================"
echo ""
echo " 사용법:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "   # Jupyter 실행"
echo "   jupyter notebook --no-browser --ip=0.0.0.0 --port=8888"
echo ""
echo "   # 학습 실행"
echo "   python scripts/run_training.py"
echo ""
echo "   # 평가 실행"
echo "   python scripts/run_evaluation.py"
echo ""
echo " [다음 단계]"
echo "   - 데이터 파일을 자료전송 시스템에서 수신 후 SCP로 data/raw/에 전송하세요."
echo "   - 추가 패키지가 필요하면 동일 경로로 wheel을 반입하세요."
echo "============================================================"
