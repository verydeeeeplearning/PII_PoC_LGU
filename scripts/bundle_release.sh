#!/usr/bin/env bash
# Creates a release tar.gz for air-gapped deployment (Linux/WSL).
# Generates a split bundle by default for 자료전송 시스템 upload.
#
# Transfer path:
#   Google Drive → LGU PC → 자료전송 시스템(업무 클라우드) → 자료전송 시스템(보안 클라우드) → SSH
#
# Usage:
#   bash scripts/bundle_release.sh                    # 기본 (분할 + checksums)
#   bash scripts/bundle_release.sh --split-size 50M   # 분할 크기 지정
#   bash scripts/bundle_release.sh --no-split          # 단일 아카이브만 생성
#
# Output:
#   Default:  outputs/releases/PII_FPR_bundle_YYYYMMDD_HHMMSS_split/ (분할 파일 + checksums + README)
#   --no-split: outputs/releases/PII_FPR_bundle_YYYYMMDD_HHMMSS.tar.gz (단일 아카이브)

set -euo pipefail

# ── 인자 파싱 ──
DO_SPLIT=true
SPLIT_SIZE="50M"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-split)
            DO_SPLIT=false
            shift
            ;;
        --split-size)
            SPLIT_SIZE="$2"
            shift 2
            ;;
        --split-size=*)
            SPLIT_SIZE="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_DIR/outputs/releases"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="PII_FPR_bundle_${STAMP}"
STAGE="$OUT_DIR/staging_${STAMP}"
ARCHIVE="$OUT_DIR/${NAME}.tar.gz"

mkdir -p "$OUT_DIR"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# Copy required directories
cp -a "$PROJECT_DIR/src" "$STAGE/src"
cp -a "$PROJECT_DIR/scripts" "$STAGE/scripts"
cp -a "$PROJECT_DIR/notebooks" "$STAGE/notebooks"
cp -a "$PROJECT_DIR/Detail_BP_Process" "$STAGE/Detail_BP_Process"
cp -a "$PROJECT_DIR/config" "$STAGE/config"
cp -a "$PROJECT_DIR/offline_packages" "$STAGE/offline_packages"

# Copy required files
for f in README.md PROJECT_GUIDE.md requirements.txt requirements_glibc228.txt requirements_glibc217.txt Miniconda3-py312-Linux-x86_64.sh CLAUDE.md; do
  if [[ -f "$PROJECT_DIR/$f" ]]; then
    cp -a "$PROJECT_DIR/$f" "$STAGE/$f"
  fi
done

# Create empty placeholders (do not bundle real data/models/outputs)
mkdir -p "$STAGE/data/raw" "$STAGE/data/processed" "$STAGE/data/features"
mkdir -p "$STAGE/models/baseline" "$STAGE/models/experiments" "$STAGE/models/final"
mkdir -p "$STAGE/outputs/reports" "$STAGE/outputs/figures"

# Remove local artifacts that shouldn't be shipped
rm -rf "$STAGE/venv" 2>/dev/null || true
find "$STAGE" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} + 2>/dev/null || true

# Create archive
tar -C "$STAGE" -czf "$ARCHIVE" .
rm -rf "$STAGE"

echo "Created: $ARCHIVE"

# ── 망연계용 분할 ──
if [[ "$DO_SPLIT" == "true" ]]; then
    SPLIT_DIR="$OUT_DIR/${NAME}_split"
    mkdir -p "$SPLIT_DIR"

    echo ""
    echo "[분할 모드] 망연계 전송용 파일 분할 (크기: $SPLIT_SIZE)"
    echo "------------------------------------------------------------"

    # 분할
    split -b "$SPLIT_SIZE" -d --additional-suffix=".part" "$ARCHIVE" "$SPLIT_DIR/${NAME}.tar.gz."

    # 원본 아카이브 크기
    ARCHIVE_SIZE=$(du -h "$ARCHIVE" | cut -f1)
    PART_COUNT=$(ls "$SPLIT_DIR"/*.part 2>/dev/null | wc -l)
    echo "  원본 크기: $ARCHIVE_SIZE"
    echo "  분할 파일: ${PART_COUNT}개"

    # checksums 생성
    (cd "$SPLIT_DIR" && sha256sum *.part > checksums.sha256)
    echo "  [OK] checksums.sha256 생성"

    # REASSEMBLE_README.txt 생성
    cat > "$SPLIT_DIR/REASSEMBLE_README.txt" << 'READMEEOF'
# 번들 재조립 안내 (망연계 수신 후)
#
# 1. 모든 .part 파일과 checksums.sha256을 같은 디렉토리에 배치
#
# 2. 무결성 검증:
#    sha256sum -c checksums.sha256
#
# 3. 파일 재조립:
#    cat *.part > PII_FPR_bundle.tar.gz
#
# 4. 압축 해제:
#    mkdir -p pii_project && tar xzf PII_FPR_bundle.tar.gz -C pii_project
#
# 5. 환경 설치:
#    cd pii_project
#    bash scripts/setup_env.sh
READMEEOF
    echo "  [OK] REASSEMBLE_README.txt 생성"

    echo ""
    echo "분할 결과: $SPLIT_DIR/"
    ls -lh "$SPLIT_DIR/"
fi

echo ""
echo "Done."
