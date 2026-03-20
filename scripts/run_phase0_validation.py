"""Phase 0 데이터 품질 검증 스크립트.

전략 리포트 §4의 5종 사전 계산을 수행하고 Go/No-Go 판정 리포트를 생성한다.

사용법:
    python scripts/run_phase0_validation.py
    python scripts/run_phase0_validation.py --dry-run
    python scripts/run_phase0_validation.py --input data/processed/silver_label.parquet

입력:
    data/processed/silver_label.parquet
      (run_data_pipeline.py --source label 출력)

출력:
    outputs/go_no_go_report.md
    outputs/fp_description_unique_list.csv
    outputs/label_conflict_report.txt
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.merger import detect_cross_label_duplicates
from src.evaluation.data_quality import (
    analyze_fp_description,
    compute_bayes_error_lower_bound,
    compute_label_conflict_rate,
    compute_org_consistency,
    make_go_no_go_decision,
)
from src.utils.constants import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────

# 충돌률/Bayes Error 계산에 사용할 피처 컬럼 (검출 통계)
_FEATURE_COLS = ["pattern_count", "ssn_count", "phone_count", "email_count"]

# Phase 0 출력 디렉토리
_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 기본 입력 Parquet 경로
_DEFAULT_INPUT = PROCESSED_DATA_DIR / "silver_label.parquet"


def _safe_console_text(text: str) -> str:
    """Windows 콘솔(cp949)에서 출력 가능한 형태로 정리."""
    return str(text).replace("-", "-")


# ─────────────────────────────────────────────────────────────────────────────
# argparse
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="Phase 0 데이터 품질 검증 + Go/No-Go 판정 리포트 생성",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="파일 저장 없이 결과를 콘솔에 출력만 함 (기본: False)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="입력 Parquet 파일 경로 (기본: data/processed/silver_label.parquet)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 입력 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_silver_label(input_path: Path) -> pd.DataFrame:
    """silver_label.parquet를 로드한다.

    Args:
        input_path: Parquet 파일 경로

    Returns:
        로드된 DataFrame

    Raises:
        FileNotFoundError: 파일이 없을 때
        RuntimeError: 파일 로드 실패 시
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 파일 없음: {input_path}\n"
            "먼저 'python scripts/run_data_pipeline.py --source label'을 실행하세요."
        )

    try:
        df = pd.read_parquet(input_path)
        logger.info("silver_label 로드 완료: %d건 (파일: %s)", len(df), input_path)
        return df
    except Exception as exc:
        raise RuntimeError(f"Parquet 로드 실패: {input_path} - {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Go/No-Go 리포트 Markdown 생성
# ─────────────────────────────────────────────────────────────────────────────

def _build_go_no_go_report(
    decision: dict,
    conflict_result: dict,
    bayes_result: dict,
    df: pd.DataFrame,
    now_str: str,
    cross_dup_df: pd.DataFrame = None,
) -> str:
    """Go/No-Go 판정 결과를 Markdown 문자열로 변환한다."""
    if cross_dup_df is None:
        cross_dup_df = pd.DataFrame()

    n_total = len(df)
    n_tp = int((df["label_raw"] == "TP").sum()) if "label_raw" in df.columns else 0
    n_fp = int((df["label_raw"] == "FP").sum()) if "label_raw" in df.columns else 0
    n_cross = len(cross_dup_df)

    lines = [
        "# Phase 0 Go/No-Go 판정 리포트",
        f"생성 시각: {now_str}",
        "",
        f"## 판정 결과: {decision['verdict']}",
        "",
        "### 근거",
        f"- 라벨 충돌률: {decision['conflict_rate']:.2%}",
        f"- Bayes Error 하한: {decision['bayes_error_lb']:.4f}",
        f"- 판정 이유: {decision['reason']}",
        "",
        "## 세부 통계",
        "",
        "### 데이터 규모",
        f"- 전체 건수: {n_total:,}건",
        f"- 정탐(TP): {n_tp:,}건",
        f"- 오탐(FP): {n_fp:,}건",
        "",
        "### 정탐/오탐 교차 중복 (pk_event 기준)",
        f"- 교차 중복 pk_event: {n_cross}건",
    ]

    if n_cross > 0:
        lines.append("- ⚠️ 동일 파일이 정탐/오탐 양쪽에 존재 - 레이블 신뢰도 검토 필요")

    # 조직별 분포 (있는 경우)
    if "organization" in df.columns:
        lines.append("")
        lines.append("### 조직별 건수")
        org_counts = df.groupby("organization")["label_raw"].value_counts().unstack(fill_value=0)
        for org in sorted(org_counts.index):
            tp_cnt = int(org_counts.loc[org, "TP"]) if "TP" in org_counts.columns else 0
            fp_cnt = int(org_counts.loc[org, "FP"]) if "FP" in org_counts.columns else 0
            lines.append(f"- {org}: TP={tp_cnt:,}건, FP={fp_cnt:,}건")

    lines += [
        "",
        "### 피처 공간 충돌 분석",
        f"- 전체 피처 그룹 수: {conflict_result['total_groups']:,}",
        f"- 충돌 그룹 수: {conflict_result['conflicted_groups']:,}",
        f"- 충돌률: {conflict_result['conflict_rate']:.2%}",
        "",
        "### Bayes Error 하한",
        f"- bayes_error_lb: {bayes_result['bayes_error_lb']:.4f}",
        "  (0.0 = 완전 분리 가능, 0.5 = 완전 혼재)",
        "",
        "---",
        "_Phase 0-B 자동 생성 - run_phase0_validation.py_",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 충돌 리포트 텍스트 생성
# ─────────────────────────────────────────────────────────────────────────────

def _build_conflict_report(conflict_result: dict, bayes_result: dict, now_str: str) -> str:
    """라벨 충돌 상세 리포트를 텍스트 문자열로 반환한다."""
    lines = [
        "=== Phase 0 라벨 충돌 상세 리포트 ===",
        f"생성 시각: {now_str}",
        "",
        "[라벨 충돌률]",
        f"  전체 피처 그룹: {conflict_result['total_groups']:,}",
        f"  충돌 그룹:       {conflict_result['conflicted_groups']:,}",
        f"  충돌률:          {conflict_result['conflict_rate']:.4f} ({conflict_result['conflict_rate']:.2%})",
        "",
        "[Bayes Error 하한]",
        f"  bayes_error_lb: {bayes_result['bayes_error_lb']:.4f}",
        "  해석: 0.0=완전분리, 0.5=완전혼재",
        "",
        "[사용된 피처 컬럼]",
        "  pattern_count, ssn_count, phone_count, email_count",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 출력 저장
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    go_no_go_md: str,
    conflict_txt: str,
    fp_desc_df: pd.DataFrame,
    output_dir: Path,
    cross_dup_df: pd.DataFrame = None,
) -> None:
    """분석 결과 파일을 output_dir에 저장한다.

    Args:
        go_no_go_md:   Go/No-Go 리포트 Markdown 문자열
        conflict_txt:  충돌 리포트 텍스트 문자열
        fp_desc_df:    fp_description 분석 DataFrame
        output_dir:    출력 디렉토리 (없으면 자동 생성)
        cross_dup_df:  정탐/오탐 교차 중복 pk_event DataFrame (없으면 저장 생략)
    """
    if cross_dup_df is None:
        cross_dup_df = pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Go/No-Go 리포트
    go_no_go_path = output_dir / "go_no_go_report.md"
    go_no_go_path.write_text(go_no_go_md, encoding="utf-8")
    logger.info("Go/No-Go 리포트 저장: %s", go_no_go_path)

    # 충돌 리포트
    conflict_path = output_dir / "label_conflict_report.txt"
    conflict_path.write_text(conflict_txt, encoding="utf-8")
    logger.info("충돌 리포트 저장: %s", conflict_path)

    # fp_description unique 목록
    fp_desc_path = output_dir / "fp_description_unique_list.csv"
    if not fp_desc_df.empty:
        fp_desc_df.to_csv(fp_desc_path, index=False, encoding="utf-8-sig")
        logger.info("fp_description 목록 저장: %s (%d건)", fp_desc_path, len(fp_desc_df))
    else:
        # 빈 파일이라도 생성 (후속 스크립트가 파일 존재 여부를 확인할 수 있도록)
        pd.DataFrame(columns=["fp_description", "count", "orgs"]).to_csv(
            fp_desc_path, index=False, encoding="utf-8-sig"
        )
        logger.info("fp_description 목록 없음 - 빈 파일 생성: %s", fp_desc_path)

    # 정탐/오탐 교차 중복 pk_event 목록 (중복 있을 때만 저장)
    if not cross_dup_df.empty:
        cross_dup_path = output_dir / "cross_label_duplicates.csv"
        cross_dup_df.to_csv(cross_dup_path, index=False, encoding="utf-8-sig")
        logger.info("교차 중복 목록 저장: %s (%d건)", cross_dup_path, len(cross_dup_df))


# ─────────────────────────────────────────────────────────────────────────────
# 콘솔 출력 (--dry-run 모드)
# ─────────────────────────────────────────────────────────────────────────────

def print_results(
    decision: dict,
    conflict_result: dict,
    bayes_result: dict,
    fp_desc_df: pd.DataFrame,
    org_consistency_df: pd.DataFrame,
    cross_dup_df: pd.DataFrame = None,
) -> None:
    """분석 결과를 콘솔에 출력한다 (--dry-run 모드용)."""
    if cross_dup_df is None:
        cross_dup_df = pd.DataFrame()

    separator = "=" * 60
    print(f"\n{separator}")
    print("  Phase 0 데이터 품질 검증 결과 (dry-run)")
    print(separator)

    print("\n[Go/No-Go 판정]")
    print(f"  판정: {decision['verdict']}")
    print(f"  근거: {decision['reason']}")

    print("\n[라벨 충돌률]")
    print(f"  충돌률:     {conflict_result['conflict_rate']:.4f} ({conflict_result['conflict_rate']:.2%})")
    print(f"  충돌 그룹:  {conflict_result['conflicted_groups']:,} / {conflict_result['total_groups']:,}")

    print("\n[Bayes Error 하한]")
    print(f"  bayes_error_lb: {bayes_result['bayes_error_lb']:.4f}")

    print("\n[조직 간 일관성]")
    if org_consistency_df.empty:
        print("  분석 불가 (organization 컬럼 없음)")
    else:
        print(f"  분석 행 수: {len(org_consistency_df):,}")
        fp_rate_std = org_consistency_df.groupby("feature_group")["fp_rate"].std()
        high_var = (fp_rate_std > 0.2).sum()
        print(f"  고분산 패턴 그룹 (std > 0.2): {high_var:,}개")

    print("\n[fp_description 분석]")
    if fp_desc_df.empty:
        print("  FP 행 없음 또는 fp_description 컬럼 없음")
    else:
        print(f"  unique fp_description 수: {len(fp_desc_df):,}")
        print("  상위 5개:")
        for _, row in fp_desc_df.head(5).iterrows():
            print(f"    [{row['count']:4d}건] {row['fp_description']} (조직: {row['orgs']})")

    print("\n[정탐/오탐 교차 중복]")
    n_cross = len(cross_dup_df)
    if n_cross > 0:
        print(f"  ⚠️ 교차 중복 pk_event: {n_cross}건")
        print(cross_dup_df.to_string(index=False))
    else:
        print("  교차 중복 없음")

    print(f"\n{separator}\n")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Phase 0 검증 메인 실행 함수."""
    args = parse_args()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Phase 0 데이터 품질 검증 시작 (dry_run=%s)", args.dry_run)

    # 1. 입력 로딩
    df = load_silver_label(args.input)

    # 2. 정탐/오탐 교차 중복 검증 (pk_event 기반)
    logger.info("[0/5] 정탐/오탐 교차 중복 검증 중...")
    cross_dup_df = detect_cross_label_duplicates(df)
    n_cross = len(cross_dup_df)
    if n_cross > 0:
        logger.warning("교차 중복 pk_event 발견: %d건 - label 충돌 존재", n_cross)
    else:
        logger.info("교차 중복 없음 (정탐/오탐 레이블 충돌 없음)")

    # 3. 라벨 충돌률 계산
    logger.info("[1/5] 라벨 충돌률 계산 중...")
    conflict_result = compute_label_conflict_rate(df, _FEATURE_COLS)

    # 4. Bayes Error 하한 계산
    logger.info("[2/5] Bayes Error 하한 계산 중...")
    bayes_result = compute_bayes_error_lower_bound(df, _FEATURE_COLS)

    # 5. 조직 간 일관성 계산
    logger.info("[3/5] 조직 간 라벨링 일관성 계산 중...")
    org_consistency_df = compute_org_consistency(df, _FEATURE_COLS)

    # 6. fp_description 분석
    logger.info("[4/5] fp_description 분석 중...")
    fp_desc_df = analyze_fp_description(df)

    # 7. Go/No-Go 판정
    logger.info("[5/5] Go/No-Go 판정 중...")
    decision = make_go_no_go_decision(
        conflict_rate=conflict_result["conflict_rate"],
        bayes_error_lb=bayes_result["bayes_error_lb"],
    )

    # 리포트 문자열 생성 (항상 생성 - dry-run에서도 콘솔 출력용)
    go_no_go_md = _build_go_no_go_report(decision, conflict_result, bayes_result, df, now_str, cross_dup_df)
    conflict_txt = _build_conflict_report(conflict_result, bayes_result, now_str)

    if args.dry_run:
        print_results(decision, conflict_result, bayes_result, fp_desc_df, org_consistency_df, cross_dup_df)
        print("[dry-run] 파일 저장 생략")
    else:
        save_outputs(go_no_go_md, conflict_txt, fp_desc_df, _OUTPUT_DIR, cross_dup_df)
        print(f"\nPhase 0 완료. 출력 디렉토리: {_OUTPUT_DIR}")
        print("  - go_no_go_report.md")
        print("  - fp_description_unique_list.csv")
        print("  - label_conflict_report.txt")
        print(f"\n판정 결과: {decision['verdict']}")
        print(f"  {_safe_console_text(decision['reason'])}")

    logger.info("Phase 0 데이터 품질 검증 완료")


if __name__ == "__main__":
    main()
