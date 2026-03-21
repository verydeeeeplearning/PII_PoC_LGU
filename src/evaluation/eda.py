"""탐색적 데이터 분석 모듈"""
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
from sklearn.preprocessing import LabelEncoder

from src.utils.constants import KEYWORD_GROUPS, FIGURE_DPI
from src.utils.common import ensure_dirs
from src.utils.plot_utils import setup_plot


def plot_class_distribution(
    df: pd.DataFrame,
    label_column: str = "label",
    save_path: str = "outputs/figures/class_distribution.png",
) -> None:
    """
    클래스 분포 시각화 (막대 그래프 + 비율 그래프)

    Args:
        df: DataFrame
        label_column: 레이블 컬럼명
        save_path: 이미지 저장 경로
    """
    setup_plot()

    label_counts = df[label_column].value_counts()
    label_pcts = df[label_column].value_counts(normalize=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = sns.color_palette("Set2", n_colors=len(label_counts))

    # 건수
    label_counts.plot(kind="barh", ax=axes[0], color=colors)
    axes[0].set_title("Class Distribution (Count)", fontsize=14)
    axes[0].set_xlabel("Count")
    for i, (val, name) in enumerate(zip(label_counts.values, label_counts.index)):
        axes[0].text(val + max(label_counts) * 0.01, i, f"{val:,}", va="center", fontsize=9)

    # 비율
    label_pcts.plot(kind="barh", ax=axes[1], color=colors)
    axes[1].set_title("Class Distribution (%)", fontsize=14)
    axes[1].set_xlabel("Percentage (%)")
    for i, (val, name) in enumerate(zip(label_pcts.values, label_pcts.index)):
        axes[1].text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    ensure_dirs(Path(save_path).parent)
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[저장] {save_path}")


def analyze_text_column(
    df: pd.DataFrame,
    text_column: str,
    label_column: str = "label",
    save_dir: str = "outputs/figures",
) -> pd.DataFrame:
    """
    텍스트 컬럼 분석 (길이 통계 + Box Plot)

    Args:
        df: DataFrame
        text_column: 텍스트 컬럼명
        label_column: 레이블 컬럼명
        save_dir: 시각화 저장 디렉토리

    Returns:
        클래스별 텍스트 길이 통계 DataFrame
    """
    setup_plot()

    if text_column not in df.columns:
        print(f"[경고] '{text_column}' 컬럼이 존재하지 않습니다.")
        return pd.DataFrame()

    df = df.copy()
    df["_text_len"] = df[text_column].fillna("").str.len()
    df["_word_count"] = df[text_column].fillna("").str.split().str.len().fillna(0).astype(int)

    print("[전체 텍스트 길이 통계]")
    print(df["_text_len"].describe().to_string())

    stats = df.groupby(label_column)["_text_len"].describe()
    print("\n[클래스별 텍스트 길이 통계]")
    print(stats.to_string())

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df.boxplot(column="_text_len", by=label_column, ax=axes[0], vert=False)
    axes[0].set_title("Text Length by Class (Box Plot)", fontsize=13)
    axes[0].set_xlabel("Text Length (chars)")
    plt.suptitle("")

    df["_text_len"].hist(bins=50, ax=axes[1], edgecolor="black", alpha=0.7)
    axes[1].set_title("Text Length Distribution (All)", fontsize=13)
    axes[1].set_xlabel("Text Length (chars)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    save_path = Path(save_dir) / "text_length_by_class.png"
    ensure_dirs(save_path.parent)
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[저장] {save_path}")

    return stats


def analyze_text_ratios(
    df: pd.DataFrame,
    text_column: str,
    label_column: str = "label",
    save_dir: str = "outputs/figures",
) -> pd.DataFrame:
    """
    텍스트 숫자/특수문자/대문자 비율 분석

    Args:
        df: DataFrame
        text_column: 텍스트 컬럼명
        label_column: 레이블 컬럼명
        save_dir: 시각화 저장 디렉토리

    Returns:
        클래스별 비율 통계 DataFrame
    """
    setup_plot()

    if text_column not in df.columns:
        print(f"[경고] '{text_column}' 컬럼이 존재하지 않습니다.")
        return pd.DataFrame()

    df = df.copy()
    texts = df[text_column].fillna("")

    df["_digit_ratio"] = texts.apply(
        lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)
    )
    df["_special_char_ratio"] = texts.apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / max(len(x), 1)
    )
    df["_uppercase_ratio"] = texts.apply(
        lambda x: sum(c.isupper() for c in x) / max(len(x), 1)
    )

    ratio_cols = ["_digit_ratio", "_special_char_ratio", "_uppercase_ratio"]
    ratio_stats = df.groupby(label_column)[ratio_cols].mean()

    print("[클래스별 텍스트 특성 비율 (평균)]")
    print(ratio_stats.round(4).to_string())

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Digit Ratio", "Special Char Ratio", "Uppercase Ratio"]

    for ax, col, title in zip(axes, ratio_cols, titles):
        df.boxplot(column=col, by=label_column, ax=ax, vert=False)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Ratio")

    plt.suptitle("")
    plt.tight_layout()
    save_path = Path(save_dir) / "text_ratio_by_class.png"
    ensure_dirs(save_path.parent)
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[저장] {save_path}")

    return ratio_stats


def analyze_keyword_frequency(
    df: pd.DataFrame,
    text_column: str,
    label_column: str = "label",
    keyword_groups: Optional[dict] = None,
    save_dir: str = "outputs/figures",
) -> None:
    """
    키워드 출현 빈도 분석 + 클래스별 히트맵

    Args:
        df: DataFrame
        text_column: 텍스트 컬럼명
        label_column: 레이블 컬럼명
        keyword_groups: {그룹명: [키워드, ...]} (기본값: constants.KEYWORD_GROUPS)
        save_dir: 시각화 저장 디렉토리
    """
    setup_plot()

    if keyword_groups is None:
        keyword_groups = KEYWORD_GROUPS

    if text_column not in df.columns:
        print(f"[경고] '{text_column}' 컬럼이 존재하지 않습니다.")
        return

    df = df.copy()
    lower_texts = df[text_column].fillna("").str.lower()

    print("[키워드 출현 빈도]")
    print(f"{'키워드 그룹':25s}  {'출현 건수':>10s}  {'출현 비율':>10s}")
    print("-" * 50)

    kw_cols = []
    for group_name, keywords in keyword_groups.items():
        pattern = "|".join(re.escape(kw) for kw in keywords)
        col_name = f"_kw_{group_name}"
        df[col_name] = lower_texts.str.contains(pattern, regex=True).astype(int)
        kw_cols.append(col_name)

        matches = df[col_name].sum()
        pct = matches / len(df) * 100
        print(f"  {group_name:23s}  {matches:>8,}건  {pct:>8.1f}%")

    # 클래스별 히트맵
    kw_by_class = df.groupby(label_column)[kw_cols].mean()
    kw_by_class.columns = [c.replace("_kw_", "") for c in kw_by_class.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(kw_by_class, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Keyword Occurrence Rate by Class", fontsize=14)
    ax.set_ylabel("Class")
    ax.set_xlabel("Keyword Group")
    plt.tight_layout()
    save_path = Path(save_dir) / "keyword_heatmap_by_class.png"
    ensure_dirs(save_path.parent)
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[저장] {save_path}")


def analyze_categorical_column(
    df: pd.DataFrame,
    column: str,
    label_column: str = "label",
    top_n: int = 20,
) -> None:
    """
    범주형 변수 분석 (고유값 수, 상위 N개, 클래스별 교차 분석)

    Args:
        df: DataFrame
        column: 분석할 컬럼명
        label_column: 레이블 컬럼명
        top_n: 상위 출력 개수
    """
    if column not in df.columns:
        print(f"  [건너뜀] '{column}' 컬럼 없음")
        return

    print(f"\n{'=' * 50}")
    print(f"[{column} 분석]")
    print(f"{'=' * 50}")
    print(f"  고유값 수: {df[column].nunique()}")
    print(f"  결측 수: {df[column].isnull().sum()}")

    print(f"\n  상위 {top_n}개:")
    top_vals = df[column].value_counts().head(top_n)
    for val, cnt in top_vals.items():
        pct = cnt / len(df) * 100
        print(f"    {str(val):30s}  {cnt:>6,}건  ({pct:5.1f}%)")


def check_data_leakage(
    df: pd.DataFrame,
    label_column: str = "label",
    suspect_columns: Optional[List[str]] = None,
) -> None:
    """
    Data Leakage 의심 컬럼 점검 (|상관계수| > 0.9 경고)

    Args:
        df: DataFrame
        label_column: 레이블 컬럼명
        suspect_columns: 점검 대상 컬럼 (None이면 수치형 전체)
    """
    print("[Data Leakage 점검]")
    print("레이블과 비정상적으로 높은 상관관계(|r| > 0.9)를 가진 컬럼을 찾습니다.")
    print("-" * 60)

    le = LabelEncoder()
    y = le.fit_transform(df[label_column])

    if suspect_columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        suspect_columns = [c for c in numeric_cols if not c.startswith("_")]

    suspect_found = False
    for col in suspect_columns:
        try:
            corr = np.corrcoef(df[col].fillna(0), y)[0, 1]
            flag = ""
            if abs(corr) > 0.9:
                flag = " *** LEAKAGE 의심 ***"
                suspect_found = True
            elif abs(corr) > 0.7:
                flag = " (높은 상관)"
            print(f"  {col:40s}  corr = {corr:+.4f}{flag}")
        except Exception:
            pass

    if not suspect_found:
        print("\n  [OK] Data Leakage 의심 컬럼 없음")
    else:
        print("\n  [경고] 위 컬럼들을 Feature에서 제외하거나 원인을 확인하세요.")
