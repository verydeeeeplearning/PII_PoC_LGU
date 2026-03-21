"""탐색적 데이터 분석 실행 스크립트

Architecture.md v1.2 반영:
- Silver Parquet 우선 로드, CSV fallback
- parse_status 분포 + silver_quarantine.parquet 분석
- pk_file GroupShuffleSplit Leakage 검증 (§14)
- Email domain 분포 (FP-내부도메인 vs FP-OS저작권 구분 근거)
- 신규 Figure: tp_fp_distribution / class_by_extension / path_depth_by_class / correlation_matrix

사용법:
    python scripts/run_eda.py
    python scripts/run_eda.py --input data/processed/silver_detections.parquet
    python scripts/run_eda.py --skip-leakage
    python scripts/run_eda.py --top-n 30

출력:
    outputs/class_distribution.png
    outputs/tp_fp_distribution.png
    outputs/text_length_by_class.png
    outputs/text_ratio_by_class.png
    outputs/keyword_heatmap_by_class.png
    outputs/class_by_extension.png
    outputs/path_depth_by_class.png
    outputs/correlation_matrix.png
    outputs/eda_report.txt
"""
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 Linux 서버에서 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from src.utils.common import ensure_dirs
from src.utils.constants import (
    PROCESSED_DATA_DIR,
    MERGED_CLEANED_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    FILE_PATH_COLUMN,
    FIGURES_DIR,
    REPORT_DIR,
    LABEL_NAMES,
)
from src.evaluation.eda import (
    plot_class_distribution,
    analyze_text_column,
    analyze_text_ratios,
    analyze_keyword_frequency,
    analyze_categorical_column,
    check_data_leakage,
)


# ── 폰트 설정 (Linux UTF-8 환경) ─────────────────────────────────────────────

def _set_font():
    """Linux/Windows 공용 폰트 자동 설정"""
    candidates = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    selected = next((n for n in candidates if n in available), "DejaVu Sans")
    plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"[{title}]")
    print("=" * 60)


def _safe_savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [저장] {path.name}")


# ── 분석 함수 ─────────────────────────────────────────────────────────────────

def basic_stats(df: pd.DataFrame, label_column: str) -> dict:
    """기본 통계 출력"""
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    n_duplicates = int(df.duplicated().sum())

    print(f"  행 수:      {n_rows:,}")
    print(f"  컬럼 수:    {n_cols}")
    print(f"  전체 결측:  {n_missing:,}개")
    print(f"  중복 행:    {n_duplicates:,}개")

    print("\n  [컬럼별 결측]")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("    결측 없음")
    else:
        for col, cnt in miss.items():
            print(f"    {col:40s}  {cnt:>6,}개  ({cnt/n_rows:.1%})")

    label_dist = {}
    if label_column in df.columns:
        print(f"\n  [레이블 분포] ({label_column})")
        label_dist = df[label_column].value_counts()
        for label, cnt in label_dist.items():
            bar = "█" * int(cnt / label_dist.max() * 20)
            print(f"    {str(label):30s}  {cnt:>6,}건  ({cnt/n_rows:.1%})  {bar}")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_missing": n_missing,
        "n_duplicates": n_duplicates,
        "label_distribution": label_dist.to_dict() if hasattr(label_dist, "to_dict") else {},
    }


def analyze_parse_status(df: pd.DataFrame, processed_dir: Path) -> dict:
    """parse_status 분포 + silver_quarantine.parquet 분석"""
    result = {}

    if "parse_status" in df.columns:
        print("  [parse_status 분포]")
        ps = df["parse_status"].value_counts(dropna=False)
        for status, cnt in ps.items():
            print(f"    {str(status):20s}  {cnt:>6,}건")
        result["parse_status_dist"] = ps.to_dict()

    quarantine_path = processed_dir / "silver_quarantine.parquet"
    if quarantine_path.exists():
        df_q = pd.read_parquet(quarantine_path)
        n_silver = len(df)
        n_quarantine = len(df_q)
        n_total = n_silver + n_quarantine
        rate = n_silver / n_total if n_total > 0 else 1.0
        kpi_pass = rate >= 0.95

        print(f"\n  [Quarantine 분석]")
        print(f"    silver_detections:   {n_silver:,}행")
        print(f"    silver_quarantine:   {n_quarantine:,}행")
        print(f"    parse_success_rate:  {rate:.4f}  "
              f"({'PASS ≥0.95' if kpi_pass else 'FAIL <0.95'})")

        if not df_q.empty and "quarantine_reason" in df_q.columns:
            print("    Quarantine 사유:")
            for reason, cnt in df_q["quarantine_reason"].value_counts().items():
                print(f"      {str(reason):30s}  {cnt:,}건")

        result.update({
            "n_silver": n_silver,
            "n_quarantine": n_quarantine,
            "parse_success_rate": round(rate, 4),
            "parse_kpi_pass": kpi_pass,
        })
    else:
        print(f"  [참고] silver_quarantine.parquet 없음 ({quarantine_path})")
        print("         -> run_data_pipeline.py 실행 후 확인하세요.")

    return result


def plot_tp_fp_pie(df: pd.DataFrame, label_column: str, save_path: Path) -> None:
    """TP vs FP 이진 분류 파이차트"""
    df = df.copy()
    df["_binary"] = df[label_column].apply(
        lambda x: "정탐(TP)" if "TP" in str(x) else "오탐(FP)"
    )
    counts = df["_binary"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    counts.plot(
        kind="pie", ax=ax, autopct="%1.1f%%", startangle=90,
        colors=["#e74c3c", "#2ecc71"],
    )
    ax.set_title("TP vs FP Distribution", fontsize=14)
    ax.set_ylabel("")
    _safe_savefig(save_path)

    for label, cnt in counts.items():
        print(f"  {label}: {cnt:,}건 ({cnt/len(df):.1%})")


def plot_class_by_extension(
    df: pd.DataFrame, label_column: str, save_path: Path, top_n: int = 10
) -> None:
    """파일 확장자별 클래스 분포 Stacked Bar"""
    if "file_extension" not in df.columns:
        print("  [건너뜀] file_extension 컬럼 없음")
        return

    top_exts = df["file_extension"].value_counts().head(top_n).index
    df_top = df[df["file_extension"].isin(top_exts)]
    cross = pd.crosstab(df_top["file_extension"], df_top[label_column])
    cross = cross.loc[top_exts]

    cross.plot(kind="barh", stacked=True, figsize=(14, 6), colormap="Set2")
    plt.title(f"Class Distribution by File Extension (Top {top_n})", fontsize=14)
    plt.xlabel("Count")
    plt.ylabel("File Extension")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    _safe_savefig(save_path)


def plot_path_depth(df: pd.DataFrame, label_column: str, save_path: Path) -> None:
    """경로 depth 클래스별 Box Plot"""
    if "path_depth" not in df.columns:
        print("  [건너뜀] path_depth 컬럼 없음")
        return

    print("  [클래스별 path_depth 평균]")
    for cls, mean_d in df.groupby(label_column)["path_depth"].mean().sort_values(ascending=False).items():
        print(f"    {str(cls):30s}  {mean_d:.1f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    df.boxplot(column="path_depth", by=label_column, ax=ax, vert=False)
    ax.set_title("Path Depth by Class", fontsize=13)
    ax.set_xlabel("Path Depth")
    plt.suptitle("")
    _safe_savefig(save_path)


def plot_correlation_matrix(df: pd.DataFrame, save_path: Path) -> None:
    """수치형 Feature 상관관계 히트맵"""
    try:
        import seaborn as sns
    except ImportError:
        print("  [건너뜀] seaborn 미설치")
        return

    analysis_cols = [
        c for c in df.columns
        if df[c].dtype in [np.float64, np.int64, float, int]
        and not c.startswith("_")
        and df[c].nunique() > 1
    ]
    # 너무 많으면 기본 통계 Feature만 선택
    if len(analysis_cols) > 20:
        priority = ["text_length", "word_count", "digit_ratio", "special_char_ratio",
                    "uppercase_ratio", "path_depth", "digit_count"]
        analysis_cols = [c for c in priority if c in df.columns]

    if len(analysis_cols) < 2:
        print("  [건너뜀] 상관관계 분석에 필요한 수치형 컬럼 부족")
        return

    corr = df[analysis_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(analysis_cols)), max(6, len(analysis_cols) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    _safe_savefig(save_path)


def analyze_email_domains(df: pd.DataFrame, text_column: str, top_n: int = 20) -> None:
    """Email domain 분포 (FP-내부도메인 vs FP-OS저작권 구분 근거)"""
    if text_column not in df.columns:
        return

    pattern = r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
    domains = df[text_column].str.extract(pattern, expand=False).dropna()
    total_with_email = len(domains)

    print(f"  이메일 도메인 포함 행: {total_with_email:,}건 ({total_with_email/len(df):.1%})")
    if total_with_email == 0:
        return

    top = domains.str.lower().value_counts().head(top_n)
    print(f"\n  상위 {top_n}개 이메일 도메인:")
    for domain, cnt in top.items():
        pct = cnt / total_with_email * 100
        print(f"    @{domain:35s}  {cnt:>5,}건  ({pct:4.1f}%)")


def check_group_split_leakage(df: pd.DataFrame, label_column: str, test_size: float = 0.2) -> dict:
    """pk_file GroupShuffleSplit Leakage 검증 (Architecture §14)"""
    if "pk_file" not in df.columns:
        print("  [건너뜀] pk_file 컬럼 없음")
        return {}

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(df[label_column])
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=df["pk_file"]))

    train_pks = set(df.iloc[train_idx]["pk_file"])
    test_pks  = set(df.iloc[test_idx]["pk_file"])
    overlap   = train_pks & test_pks

    print(f"  Split 전략: pk_file GroupShuffleSplit (test_size={test_size})")
    print(f"  Train: {len(train_idx):,}행  ({len(train_pks):,} files)")
    print(f"  Test:  {len(test_idx):,}행  ({len(test_pks):,} files)")
    print(f"  pk_file 중복(Leakage): {len(overlap)}건  "
          f"({'OK - Leakage 없음' if len(overlap) == 0 else 'WARN - Leakage 존재'})")

    return {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_pkfiles": len(train_pks),
        "n_test_pkfiles": len(test_pks),
        "leakage_count": len(overlap),
        "leakage_ok": len(overlap) == 0,
    }


def save_eda_report(
    stats: dict,
    args,
    sections: list,
    parse_result: dict,
    leakage_result: dict,
    path: Path,
) -> None:
    """EDA 요약 리포트 저장 (UTF-8)"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EDA 요약 리포트\n")
        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"입력 파일: {args.input}\n")
        f.write("=" * 60 + "\n\n")

        f.write("[기본 통계]\n")
        f.write(f"  행 수:        {stats['n_rows']:,}\n")
        f.write(f"  컬럼 수:      {stats['n_cols']}\n")
        f.write(f"  전체 결측:    {stats['n_missing']:,}개\n")
        f.write(f"  중복 행:      {stats['n_duplicates']:,}개\n\n")

        if stats["label_distribution"]:
            f.write("[레이블 분포]\n")
            total = sum(stats["label_distribution"].values())
            for label, cnt in stats["label_distribution"].items():
                f.write(f"  {str(label):30s}  {cnt:>6,}건  ({cnt/total:.1%})\n")
            f.write("\n")

        if parse_result:
            f.write("[parse_status / Quarantine]\n")
            for k, v in parse_result.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        if leakage_result:
            f.write("[pk_file GroupShuffleSplit Leakage 검증]\n")
            for k, v in leakage_result.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("[실행된 분석 단계]\n")
        for s in sections:
            f.write(f"  - {s}\n")

    print(f"[저장] EDA 요약 리포트: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="탐색적 데이터 분석 실행")
    parser.add_argument(
        "--input", type=str,
        default=None,
        help="입력 파일 경로 (기본: silver_detections.parquet -> merged_cleaned.csv 순으로 자동 탐색)",
    )
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR))
    parser.add_argument("--report-dir",  type=str, default=str(REPORT_DIR))
    parser.add_argument("--skip-leakage", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def _load_input(args) -> pd.DataFrame:
    """입력 파일 자동 탐색 및 로드 (Parquet -> CSV 순)"""
    if args.input:
        p = Path(args.input)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p, encoding="utf-8", low_memory=False)

    # 자동 탐색 순서
    candidates = [
        PROCESSED_DATA_DIR / "silver_detections.parquet",
        PROCESSED_DATA_DIR / MERGED_CLEANED_FILE,
    ]
    for p in candidates:
        if p.exists():
            print(f"  [자동 로드] {p}")
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, encoding="utf-8", low_memory=False)

    print("[오류] 입력 파일을 찾을 수 없습니다.")
    print("  먼저 다음을 실행하세요: python scripts/run_data_pipeline.py")
    sys.exit(1)


def main():
    args = parse_args()
    font = _set_font()

    figures_dir = Path(args.figures_dir)
    report_dir  = Path(args.report_dir)
    ensure_dirs(figures_dir, report_dir)

    sections = []

    # ── Step 1: 데이터 로드 ─────────────────────────────────────────────────
    _section("Step 1: 데이터 로드")
    df = _load_input(args)
    print(f"  shape: {df.shape}  |  font: {font}")

    if "parse_status" not in df.columns:
        df["parse_status"] = "ok"

    # ── Step 2: 기본 통계 ───────────────────────────────────────────────────
    _section("Step 2: 기본 통계")
    stats = basic_stats(df, LABEL_COLUMN)
    sections.append("기본 통계 (shape, dtype, 결측, 중복, 레이블 분포)")

    # ── Step 3: parse_status + Quarantine 분석 ────────────────────────────
    _section("Step 3: parse_status / Quarantine 분석")
    parse_result = analyze_parse_status(df, PROCESSED_DATA_DIR)
    sections.append("parse_status 분포 + Quarantine 분석")

    # ── Step 4: 클래스 분포 시각화 ─────────────────────────────────────────
    if LABEL_COLUMN in df.columns:
        _section("Step 4: 클래스 분포 시각화")
        plot_class_distribution(
            df, label_column=LABEL_COLUMN,
            save_path=str(figures_dir / "class_distribution.png"),
        )
        sections.append("클래스 분포 -> class_distribution.png")

        # TP vs FP 파이차트
        plot_tp_fp_pie(df, LABEL_COLUMN, figures_dir / "tp_fp_distribution.png")
        sections.append("TP vs FP 파이차트 -> tp_fp_distribution.png")

    # ── Step 5: 텍스트 길이 분석 ───────────────────────────────────────────
    text_col = TEXT_COLUMN if TEXT_COLUMN in df.columns else next(
        (c for c in df.columns if "text" in c.lower() or "context" in c.lower()), None
    )
    if text_col:
        _section("Step 5: 텍스트 길이 분석")
        analyze_text_column(df, text_column=text_col, label_column=LABEL_COLUMN,
                            save_dir=str(figures_dir))
        sections.append(f"텍스트 길이 분석 ({text_col}) -> text_length_by_class.png")

        # ── Step 6: 텍스트 특성 비율 분석 ─────────────────────────────────
        _section("Step 6: 텍스트 특성 비율 분석")
        analyze_text_ratios(df, text_column=text_col, label_column=LABEL_COLUMN,
                            save_dir=str(figures_dir))
        sections.append("텍스트 특성 비율 분석 -> text_ratio_by_class.png")

        # ── Step 7: 키워드 빈도 + 히트맵 ──────────────────────────────────
        _section("Step 7: 키워드 출현 빈도 분석")
        analyze_keyword_frequency(df, text_column=text_col, label_column=LABEL_COLUMN,
                                  save_dir=str(figures_dir))
        sections.append("키워드 빈도 분석 -> keyword_heatmap_by_class.png")

        # ── Step 8: Email domain 분포 ──────────────────────────────────────
        _section("Step 8: Email Domain 분포")
        analyze_email_domains(df, text_col, top_n=args.top_n)
        sections.append("Email domain 분포 분석")
    else:
        print("\n[건너뜀] Step 5-8: 텍스트 컬럼을 찾을 수 없음")

    # ── Step 9: 파일 확장자별 클래스 분포 ─────────────────────────────────
    _section("Step 9: 파일 확장자별 클래스 분포")
    plot_class_by_extension(df, LABEL_COLUMN, figures_dir / "class_by_extension.png",
                            top_n=args.top_n)
    sections.append("파일 확장자별 클래스 stacked bar -> class_by_extension.png")

    # ── Step 10: 경로 depth 분석 ───────────────────────────────────────────
    _section("Step 10: 경로 Depth 분석")
    plot_path_depth(df, LABEL_COLUMN, figures_dir / "path_depth_by_class.png")
    sections.append("경로 depth 분포 -> path_depth_by_class.png")

    # ── Step 11: 범주형 컬럼 분석 ─────────────────────────────────────────
    _section("Step 11: 범주형 컬럼 분석")
    explicit_cols = [FILE_PATH_COLUMN, "pattern_type", "pii_type",
                     "server_name", "agent_ip", "parse_status"]
    cat_cols = [c for c in explicit_cols if c in df.columns]
    str_cols = [c for c in df.select_dtypes(include="object").columns
                if c not in cat_cols + [LABEL_COLUMN, text_col or ""]
                and df[c].nunique() <= 500]
    for col in cat_cols + str_cols:
        analyze_categorical_column(df, col, label_column=LABEL_COLUMN, top_n=args.top_n)
    sections.append(f"범주형 컬럼 분석 ({len(cat_cols + str_cols)}개)")

    # ── Step 12: 상관관계 히트맵 ───────────────────────────────────────────
    _section("Step 12: 상관관계 히트맵")
    plot_correlation_matrix(df, figures_dir / "correlation_matrix.png")
    sections.append("상관관계 히트맵 -> correlation_matrix.png")

    # ── Step 13: Data Leakage 점검 ────────────────────────────────────────
    leakage_result = {}
    if not args.skip_leakage and LABEL_COLUMN in df.columns:
        _section("Step 13: Data Leakage 점검")
        check_data_leakage(df, label_column=LABEL_COLUMN)

        _section("Step 13b: pk_file GroupShuffleSplit Leakage 검증")
        leakage_result = check_group_split_leakage(df, LABEL_COLUMN)
        sections.append("Data Leakage 점검 + pk_file GroupShuffleSplit 검증")
    else:
        print("\n[건너뜀] Step 13: --skip-leakage 또는 레이블 컬럼 없음")

    # ── Step 14: EDA 리포트 저장 ──────────────────────────────────────────
    _section("Step 14: EDA 요약 리포트 저장")
    save_eda_report(stats, args, sections, parse_result, leakage_result,
                    report_dir / "eda_report.txt")

    print("\n" + "=" * 60)
    print("EDA 완료")
    print("=" * 60)
    print(f"  시각화:  {figures_dir}/")
    print(f"  리포트:  {report_dir}/eda_report.txt")


if __name__ == "__main__":
    main()
