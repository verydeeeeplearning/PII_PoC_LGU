"""파일 기반 Mock E2E 파이프라인 검증 스크립트

실제 파일 시스템의 mock Excel -> 전체 파이프라인 실행 -> 출력 파일 검증

사용법:
    python scripts/run_mock_e2e.py                    # 기본: label-only, mock 재생성 포함
    python scripts/run_mock_e2e.py --mode label-only  # 레이블 Excel만
    python scripts/run_mock_e2e.py --mode full        # Sumologic + 레이블 Excel
    python scripts/run_mock_e2e.py --no-generate      # mock 재생성 건너뜀 (실 데이터 보존)
    python scripts/run_mock_e2e.py --dry-run          # 전처리(S0-S2)까지만 검증

주의:
    --no-generate 없이 실행 시 data/raw/label/ 디렉토리가 삭제 후 재생성됩니다.
    실 데이터가 있는 경우 반드시 --no-generate 옵션을 사용하세요.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

SILVER_LABEL = PROJECT_ROOT / "data" / "processed" / "silver_label.parquet"
SILVER_DETECTIONS = PROJECT_ROOT / "data" / "processed" / "silver_detections.parquet"
SILVER_JOINED = PROJECT_ROOT / "data" / "processed" / "silver_joined.parquet"
LABEL_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "label"
DATASET_A_DIR = PROJECT_ROOT / "data" / "raw" / "dataset_a"
MODEL_PATHS = [
    PROJECT_ROOT / "models" / "phase1_label_lgb.joblib",
    PROJECT_ROOT / "models" / "final" / "best_model_v1.joblib",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PII FPR Mock E2E 파이프라인 검증")
    parser.add_argument(
        "--mode",
        choices=["label-only", "full"],
        default="label-only",
        help="파이프라인 모드 (기본: label-only)",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="mock 데이터 재생성 건너뜀 (실 데이터 보존 시 사용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="전처리(S0-S2)까지만 실행 (학습/평가 없음)",
    )
    return parser.parse_args()


def _run_script(script_name: str, extra_args: list[str] | None = None) -> bool:
    """서브스크립트를 실행하고 성공 여부를 반환한다."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + (extra_args or [])
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def _check_parquet(path: Path) -> tuple[bool, int]:
    """parquet 파일 존재 여부와 행 수를 반환한다."""
    if not path.exists():
        return False, 0
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return True, len(df)
    except Exception:
        return False, 0


def _check_columns(path: Path, required: list[str]) -> bool:
    """parquet 파일에 필수 컬럼이 모두 있는지 확인한다."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return all(c in df.columns for c in required)
    except Exception:
        return False


def _count_label_xlsx() -> int:
    if not LABEL_RAW_DIR.exists():
        return 0
    return len(list(LABEL_RAW_DIR.rglob("*.xlsx")))


def _count_sumologic_xlsx() -> int:
    if not DATASET_A_DIR.exists():
        return 0
    return len(list(DATASET_A_DIR.glob("sumologic_mock_*.xlsx")))


def _has_model() -> bool:
    return any(p.exists() for p in MODEL_PATHS)


def _has_eval_output() -> bool:
    reports = PROJECT_ROOT / "outputs" / "reports"
    figures = PROJECT_ROOT / "outputs" / "figures"
    has_report = reports.exists() and any(reports.iterdir()) if reports.exists() else False
    has_figure = figures.exists() and any(figures.iterdir()) if figures.exists() else False
    return has_report or has_figure


def _warn_real_data() -> bool:
    """실 데이터가 존재하면 경고 후 확인을 요청한다. True = 계속 진행."""
    label_xlsx = _count_label_xlsx()
    if label_xlsx == 0:
        return True
    print()
    print("  [경고] data/raw/label/ 에 기존 파일이 있습니다.")
    print(f"         현재 파일 수: {label_xlsx}개")
    print("         --no-generate 없이 계속하면 해당 디렉토리가 삭제됩니다.")
    try:
        answer = input("  계속하시겠습니까? (y/N): ").strip().lower()
    except EOFError:
        print("  (non-interactive - 자동 진행)")
        return True
    return answer == "y"


def main() -> None:
    args = parse_args()
    start_time = datetime.now()

    results: list[tuple[str, str, str]] = []  # (step_id, status, detail)

    print()
    print("=" * 60)
    print("  PII FPR Mock E2E 검증 리포트")
    print(f"  모드    : {args.mode}")
    print(f"  시작    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # Step 1: Mock 데이터 생성
    # ──────────────────────────────────────────────────────────────
    step_id = "Step 1: Mock 데이터 생성"
    if args.no_generate:
        label_count = _count_label_xlsx()
        sulo_count = _count_sumologic_xlsx()
        if label_count == 0:
            results.append((step_id, "FAIL", "label xlsx 없음 - --no-generate 사용 중이나 파일 미존재"))
        else:
            results.append((step_id, "SKIP", f"--no-generate (label {label_count}개, sumologic {sulo_count}개)"))
    else:
        if not _warn_real_data():
            print("  중단합니다.")
            sys.exit(0)

        extra = ["--csv"] if args.mode == "full" else []
        ok = _run_script("generate_mock_raw_data.py", extra)
        if ok:
            label_count = _count_label_xlsx()
            sulo_count = _count_sumologic_xlsx()
            results.append((step_id, "PASS", f"label {label_count}개 + sumologic {sulo_count}개"))
        else:
            results.append((step_id, "FAIL", "generate_mock_raw_data.py 실패"))

    _print_step(results[-1])
    prev_failed = results[-1][1] == "FAIL"

    # ──────────────────────────────────────────────────────────────
    # Step 2A: Label 파이프라인
    # ──────────────────────────────────────────────────────────────
    step_id = "Step 2A: Label 파이프라인"
    if prev_failed:
        results.append((step_id, "SKIP", "Step 1 실패로 건너뜀"))
    else:
        ok = _run_script("run_data_pipeline.py", ["--source", "label"])
        if ok:
            exists, row_count = _check_parquet(SILVER_LABEL)
            has_cols = _check_columns(SILVER_LABEL, ["label_raw", "pk_event", "pk_file"])
            if exists and row_count > 0 and has_cols:
                results.append((step_id, "PASS", f"silver_label.parquet, {row_count:,}행"))
            elif not exists:
                results.append((step_id, "FAIL", "silver_label.parquet 미생성"))
            elif row_count == 0:
                results.append((step_id, "FAIL", "silver_label.parquet 0행"))
            else:
                results.append((step_id, "FAIL", "필수 컬럼 누락 (label_raw/pk_event/pk_file)"))
        else:
            results.append((step_id, "FAIL", "run_data_pipeline.py --source label 실패"))

    _print_step(results[-1])
    if results[-1][1] == "FAIL":
        prev_failed = True

    # ──────────────────────────────────────────────────────────────
    # Step 2B: Detection 파이프라인 (full 모드만)
    # ──────────────────────────────────────────────────────────────
    step_id = "Step 2B: Detection 파이프라인"
    if args.mode != "full":
        results.append((step_id, "SKIP", "label-only 모드"))
    elif prev_failed:
        results.append((step_id, "SKIP", "이전 스텝 실패로 건너뜀"))
    else:
        ok = _run_script("run_data_pipeline.py", ["--source", "detection"])
        if ok:
            exists, row_count = _check_parquet(SILVER_DETECTIONS)
            if exists and row_count > 0:
                results.append((step_id, "PASS", f"silver_detections.parquet, {row_count:,}행"))
            elif not exists:
                results.append((step_id, "FAIL", "silver_detections.parquet 미생성"))
            else:
                results.append((step_id, "FAIL", "silver_detections.parquet 0행"))
        else:
            results.append((step_id, "FAIL", "run_data_pipeline.py --source detection 실패"))

    _print_step(results[-1])
    if results[-1][1] == "FAIL":
        prev_failed = True

    # ──────────────────────────────────────────────────────────────
    # Step 2C: JOIN (full 모드만) - silver_label + silver_detections -> silver_joined
    # ──────────────────────────────────────────────────────────────
    step_id = "Step 2C: JOIN (label + detection)"
    if args.mode != "full":
        results.append((step_id, "SKIP", "label-only 모드"))
    elif prev_failed:
        results.append((step_id, "SKIP", "이전 스텝 실패로 건너뜀"))
    else:
        ok = _run_script("run_data_pipeline.py", ["--source", "joined"])
        if ok:
            exists, row_count = _check_parquet(SILVER_JOINED)
            has_label = _check_columns(SILVER_JOINED, ["label_raw", "pk_file"])
            if exists and row_count > 0 and has_label:
                results.append((step_id, "PASS", f"silver_joined.parquet, {row_count:,}행"))
            elif not exists:
                results.append((step_id, "FAIL", "silver_joined.parquet 미생성"))
            elif row_count == 0:
                results.append((step_id, "FAIL", "silver_joined.parquet 0행 - pk_file 매핑 불일치"))
            else:
                results.append((step_id, "FAIL", "label_raw/pk_file 컬럼 누락"))
        else:
            results.append((step_id, "FAIL", "run_data_pipeline.py --source joined 실패"))

    _print_step(results[-1])
    if results[-1][1] == "FAIL":
        prev_failed = True

    # ──────────────────────────────────────────────────────────────
    # Step 3: 학습 (dry-run 아닌 경우)
    # ──────────────────────────────────────────────────────────────
    step_id = "Step 3: 학습"
    train_source = "detection" if args.mode == "full" else "label"
    if args.dry_run:
        results.append((step_id, "SKIP", "--dry-run 모드"))
    elif prev_failed:
        results.append((step_id, "SKIP", "이전 스텝 실패로 건너뜀"))
    else:
        ok = _run_script("run_training.py", ["--source", train_source])
        if ok and _has_model():
            found = next((p for p in MODEL_PATHS if p.exists()), None)
            results.append((step_id, "PASS", found.name if found else "모델 파일 생성"))
        elif ok:
            results.append((step_id, "FAIL", "스크립트 성공이나 모델 파일 미생성"))
        else:
            results.append((step_id, "FAIL", f"run_training.py --source {train_source} 실패"))

    _print_step(results[-1])
    if results[-1][1] == "FAIL":
        prev_failed = True

    # ──────────────────────────────────────────────────────────────
    # Step 4: 평가 (dry-run 아닌 경우)
    # ──────────────────────────────────────────────────────────────
    step_id = "Step 4: 평가"
    if args.dry_run:
        results.append((step_id, "SKIP", "--dry-run 모드"))
    elif prev_failed:
        results.append((step_id, "SKIP", "이전 스텝 실패로 건너뜀"))
    else:
        ok = _run_script("run_evaluation.py")
        if ok:
            results.append((step_id, "PASS", "평가 완료"))
        else:
            results.append((step_id, "FAIL", "run_evaluation.py 실패"))

    _print_step(results[-1])

    # ──────────────────────────────────────────────────────────────
    # 최종 요약
    # ──────────────────────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")

    print()
    print(f"전체: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
    print(f"소요 시간: {str(elapsed).split('.')[0]}")
    print("=" * 60)

    sys.exit(1 if n_fail > 0 else 0)


def _print_step(result: tuple[str, str, str]) -> None:
    step_id, status, detail = result
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"  {icon} {step_id}: {detail}")


if __name__ == "__main__":
    main()
