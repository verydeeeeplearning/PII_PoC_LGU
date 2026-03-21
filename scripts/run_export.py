"""Parquet -> 사람이 읽을 수 있는 형식 변환 유틸리티

Parquet 파일은 ML 파이프라인 내부 산출물로 최적화되어 있지만,
감사·검토·업무 공유 목적으로 사람이 읽을 수 있는 형식이 필요할 때 사용합니다.

지원 출력 형식:
    csv   - UTF-8 BOM (utf-8-sig): Windows Excel에서 한글 깨짐 방지
    xlsx  - Excel 파일 (openpyxl)

사용법:
    # 기본: 모든 Parquet 산출물을 CSV로 변환
    python scripts/run_export.py

    # 특정 파일만 변환
    python scripts/run_export.py --input data/processed/silver_detections.parquet

    # Excel로 변환
    python scripts/run_export.py --format xlsx

    # 대용량 파일: 행 수 제한 (검토용)
    python scripts/run_export.py --max-rows 10000

    # 출력 디렉토리 지정
    python scripts/run_export.py --output-dir outputs/exports

기본 변환 대상 (--input 미지정 시):
    data/processed/silver_detections.parquet
    data/processed/silver_quarantine.parquet
    data/features/df_train.parquet
    data/features/df_test.parquet
    outputs/predictions_main.parquet
    outputs/prediction_evidence.parquet
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


# ── 기본 변환 대상 ───────────────────────────────────────────────────────────

DEFAULT_TARGETS = [
    "data/processed/silver_detections.parquet",
    "data/processed/silver_quarantine.parquet",
    "data/features/df_train.parquet",
    "data/features/df_test.parquet",
    "outputs/predictions_main.parquet",
    "outputs/prediction_evidence.parquet",
]


# ── 변환 함수 ────────────────────────────────────────────────────────────────

def parquet_to_csv(src: Path, dst: Path, max_rows: int = 0) -> None:
    """Parquet -> CSV (UTF-8 BOM)

    utf-8-sig 인코딩:
        - Linux 서버: 정상 UTF-8로 읽힘
        - Windows Excel: BOM 덕분에 한글 자동 인식
    """
    df = pd.read_parquet(src)
    if max_rows > 0:
        df = df.head(max_rows)
        print(f"  [제한] 상위 {max_rows:,}행만 출력")

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"  -> {dst}  ({len(df):,}행 x {len(df.columns)}열)")


def parquet_to_excel(src: Path, dst: Path, max_rows: int = 0) -> None:
    """Parquet -> Excel (.xlsx, openpyxl)

    Excel 행 제한(1,048,576행) 초과 시 자동으로 여러 시트로 분할합니다.
    """
    df = pd.read_parquet(src)
    if max_rows > 0:
        df = df.head(max_rows)
        print(f"  [제한] 상위 {max_rows:,}행만 출력")

    dst.parent.mkdir(parents=True, exist_ok=True)

    EXCEL_MAX_ROWS = 1_048_575  # 헤더 제외
    n_total = len(df)

    if n_total <= EXCEL_MAX_ROWS:
        df.to_excel(dst, index=False, engine="openpyxl")
        print(f"  -> {dst}  ({n_total:,}행 x {len(df.columns)}열)")
    else:
        # 시트 분할
        with pd.ExcelWriter(dst, engine="openpyxl") as writer:
            chunk_num = 0
            for start in range(0, n_total, EXCEL_MAX_ROWS):
                chunk = df.iloc[start : start + EXCEL_MAX_ROWS]
                sheet_name = f"part_{chunk_num + 1}"
                chunk.to_excel(writer, sheet_name=sheet_name, index=False)
                chunk_num += 1
        print(f"  -> {dst}  ({n_total:,}행 / {chunk_num}개 시트)")


def export_file(
    src: Path,
    output_dir: Path,
    fmt: str,
    max_rows: int = 0,
) -> bool:
    """단일 Parquet 파일 변환

    Returns:
        True: 성공, False: 실패 또는 파일 없음
    """
    if not src.exists():
        print(f"  [건너뜀] 파일 없음: {src}")
        return False

    suffix = ".csv" if fmt == "csv" else ".xlsx"
    dst = output_dir / (src.stem + suffix)

    print(f"\n[변환] {src.name}  ->  {dst.name}")

    try:
        if fmt == "csv":
            parquet_to_csv(src, dst, max_rows)
        else:
            parquet_to_excel(src, dst, max_rows)
        return True
    except Exception as exc:
        print(f"  [오류] {exc}")
        return False


# ── 메타정보 함수 ─────────────────────────────────────────────────────────────

def print_parquet_info(src: Path) -> None:
    """Parquet 파일 스키마 및 기본 통계 출력"""
    if not src.exists():
        print(f"  파일 없음: {src}")
        return

    df = pd.read_parquet(src)
    print(f"\n[{src.name}]")
    print(f"  행: {len(df):,}  |  열: {len(df.columns)}")
    print(f"  컬럼: {list(df.columns)}")

    # 용량 추정
    size_bytes = src.stat().st_size
    size_mb = size_bytes / 1024 / 1024
    print(f"  파일 크기: {size_mb:.1f} MB")

    # 결측 확인
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if not miss.empty:
        print(f"  결측 컬럼: {miss.to_dict()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parquet -> CSV/Excel 변환 유틸리티",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="변환할 Parquet 파일 경로 (없으면 기본 대상 전체 변환)",
    )
    parser.add_argument(
        "--format", choices=["csv", "xlsx"], default="csv",
        help="출력 형식 (기본: csv)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "outputs" / "exports"),
        help="출력 디렉토리 (기본: outputs/exports/)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="출력 최대 행 수 (0=전체, 기본: 0)",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="변환 없이 Parquet 스키마 정보만 출력",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # 변환 대상 결정
    if args.input:
        targets = [Path(args.input)]
    else:
        targets = [PROJECT_ROOT / t for t in DEFAULT_TARGETS]

    print("=" * 60)
    print(f"Parquet -> {args.format.upper()} 변환")
    print(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"출력 디렉토리: {output_dir}")
    if args.max_rows > 0:
        print(f"행 제한: {args.max_rows:,}행")
    print("=" * 60)

    if args.info:
        for t in targets:
            print_parquet_info(t)
        return

    success, skip = 0, 0
    for t in targets:
        ok = export_file(t, output_dir, fmt=args.format, max_rows=args.max_rows)
        if ok:
            success += 1
        else:
            skip += 1

    print("\n" + "=" * 60)
    print(f"변환 완료: {success}개 성공  /  {skip}개 건너뜀")
    print(f"출력 위치: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
