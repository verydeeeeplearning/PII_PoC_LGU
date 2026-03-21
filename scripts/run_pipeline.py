"""PII 오탐 개선 전체 파이프라인 실행 스크립트 (S0 -> S6)

데이터 전처리(S0->S2)를 반드시 먼저 수행한 후
학습(S3a/S3b), 출력(S4/S5), 평가(S6) 순서로 실행한다.

사용법:
    python scripts/run_pipeline.py --mode label-only
    python scripts/run_pipeline.py --mode full
    python scripts/run_pipeline.py --mode label-only --skip-eval
    python scripts/run_pipeline.py --mode full --use-filter --dry-run

모드:
    label-only  레이블 Excel(data/raw/label/) 단독 사용
                -> silver_label.parquet -> ML 학습
    full        Sumologic Excel(data/raw/dataset_a/) + 레이블 Excel 모두 처리
                -> silver_detections.parquet + silver_label.parquet -> ML 학습

실행 단계:
    ─────────────────────────────────────────────────────────────
    [전처리]
    S0  Raw ingest & schema canonicalization
        - label-only : data/raw/label/**/*.xlsx  -> silver_label.parquet
        - full       : data/raw/label/**/*.xlsx  -> silver_label.parquet
                       data/raw/dataset_a/*.xlsx -> silver_detections.parquet
    S1  Normalize & parse  (column_normalizer, label_loader)
    S2  Feature prep       (path / tabular 피처 빌드)
    ─────────────────────────────────────────────────────────────
    [학습]
    S3a RULE Labeler   (rules.yaml 기반 경로/패턴 룰)
    S3b ML Labeler     (LightGBM / XGBoost 학습)
    S4  Decision combiner  (RULE ↔ ML 결합, TP 안전장치)
    S5  Output writer      (predictions_main + prediction_evidence)
    ─────────────────────────────────────────────────────────────
    [평가]
    S6  Monitoring & KPI   (monthly_metrics, 오분류 분석)
    ─────────────────────────────────────────────────────────────

참고:
    - Sumologic + 레이블 완전 JOIN은 Phase 2 (후속).
      full 모드에서는 두 소스를 각각 전처리한 뒤 현재 구현된
      레이블 기반 학습을 사용한다.
    - 각 스텝은 독립 실행도 가능:
        python scripts/run_data_pipeline.py --source label
        python scripts/run_training.py --source label
        python scripts/run_report.py --source label
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PII 오탐 개선 전체 파이프라인 (S0->S6)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "label-only"],
        required=True,
        help=(
            "데이터 소스 모드\n"
            "  label-only : 레이블 Excel만 사용 (data/raw/label/)\n"
            "  full       : Sumologic Excel + 레이블 Excel 모두 처리"
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="S6 평가 단계 건너뜀 (기본: 평가 포함)",
    )
    parser.add_argument(
        "--use-filter",
        action="store_true",
        default=False,
        help="S3a RULE Labeler 활성화 (기본: ML만)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="전처리(S0-S2)까지만 실행 - 모델 학습/저장 없음",
    )
    return parser.parse_args()


def _run_step(label: str, cmd: list[str]) -> None:
    """단계 실행: 실패 시 즉시 중단"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")
    subprocess.run([str(c) for c in cmd], check=True)


def stage_preprocess(mode: str) -> None:
    """S0-S2: 데이터 로드 + 정규화 + 전처리 (항상 가장 먼저 실행)"""
    python = sys.executable
    pipeline = SCRIPTS_DIR / "run_data_pipeline.py"

    # 레이블 Excel -> silver_label.parquet  (label-only / full 공통)
    _run_step(
        "[S0-S1] 전처리 ① - 레이블 Excel -> silver_label.parquet",
        [python, pipeline, "--source", "label"],
    )

    if mode == "full":
        # Sumologic Excel -> silver_detections.parquet (pk_file 계산 포함)
        _run_step(
            "[S0-S1] 전처리 ② - Sumologic Excel -> silver_detections.parquet",
            [python, pipeline, "--source", "detection"],
        )
        # silver_label + silver_detections -> silver_joined.parquet (pk_file JOIN)
        _run_step(
            "[S1-S2] JOIN - silver_label + silver_detections -> silver_joined.parquet",
            [python, pipeline, "--source", "joined"],
        )


def stage_train(mode: str, use_filter: bool = False) -> None:
    """S2-S5: Feature prep + RULE Labeler + ML 학습 + Decision + Output"""
    python = sys.executable
    training = SCRIPTS_DIR / "run_training.py"

    # label-only : silver_label.parquet 기반 학습
    # full       : silver_joined.parquet 기반 학습 (Sumologic + 레이블 JOIN)
    source = "detection" if mode == "full" else "label"

    cmd = [python, training, "--source", source]
    if use_filter:
        cmd.append("--use-filter")

    _run_step(
        "[S2-S5] Feature prep -> S3a RULE -> S3b ML -> S4 Decision -> S5 Output",
        cmd,
    )


def stage_report(mode: str, include_diagnosis: bool = False) -> None:
    """S6: 통합 리포트 생성 (평가 + 분석 + 진단)"""
    python = sys.executable
    report = SCRIPTS_DIR / "run_report.py"

    source = "detection" if mode == "full" else "label"
    cmd = [python, report, "--source", source]
    if include_diagnosis:
        cmd.append("--include-diagnosis")

    _run_step(
        "[S6] 통합 리포트 (평가 + 분석 + Feature Importance + 진단)",
        cmd,
    )


def main() -> None:
    args = parse_args()

    start_time = datetime.now()
    print("=" * 60)
    print("  PII 오탐 개선 파이프라인")
    print(f"  모드    : {args.mode}")
    print(f"  시작    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  dry-run : {args.dry_run}")
    print("=" * 60)

    # ── [S0-S2] 전처리 - 항상 가장 먼저 ──────────────────────────
    stage_preprocess(args.mode)

    if args.dry_run:
        print("\n[dry-run] 전처리 완료 - 학습/평가 건너뜀")
        return

    # ── [S3a/S3b/S4/S5] 학습 ─────────────────────────────────────
    stage_train(args.mode, use_filter=args.use_filter)

    # ── [S6] 평가 + 리포트 ────────────────────────────────────────
    if not args.skip_eval:
        stage_report(args.mode, include_diagnosis=False)

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 60)
    print("  파이프라인 완료")
    print(f"  소요 시간: {elapsed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
