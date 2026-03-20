"""데이터 전처리 모듈"""
import re
import pandas as pd
import numpy as np
from typing import Optional

from src.utils.constants import FILE_PATH_COLUMN


def preprocess_text(text: str) -> str:
    """
    텍스트 전처리 (정규화)

    처리:
        - NaN -> 빈 문자열
        - 연속 공백 -> 단일 공백
        - 앞뒤 공백 제거

    Args:
        text: 원본 텍스트

    Returns:
        정규화된 텍스트
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str,
    label_column: str = "label",
    file_path_column: str = FILE_PATH_COLUMN,
    drop_duplicates: bool = True,
    drop_na_label: bool = True,
) -> pd.DataFrame:
    """
    DataFrame 전체 전처리 파이프라인

    처리 순서:
        1. 레이블 결측 행 제거
        2. 중복 행 제거
        3. 텍스트 정규화 (preprocess_text 적용)
        4. 파일 경로 기본 Feature 추출 (file_extension, path_depth)
        5. 인덱스 리셋

    Args:
        df: 원본 DataFrame
        text_column: 텍스트 컬럼명
        label_column: 레이블 컬럼명
        file_path_column: 파일 경로 컬럼명
        drop_duplicates: 중복 제거 여부
        drop_na_label: 레이블 결측 제거 여부

    Returns:
        전처리된 DataFrame (새 복사본)
    """
    df = df.copy()
    n_before = len(df)

    print("=" * 60)
    print("전처리 시작")
    print("=" * 60)
    print(f"  원본 데이터: {n_before:,}건")

    # 1. 레이블 결측 제거
    if drop_na_label and label_column in df.columns:
        n = len(df)
        df = df.dropna(subset=[label_column])
        removed = n - len(df)
        if removed > 0:
            print(f"  [Step 1] 레이블 결측 제거: {removed}건")
        else:
            print(f"  [Step 1] 레이블 결측 없음")

    # 2. 중복 제거
    if drop_duplicates:
        n = len(df)
        df = df.drop_duplicates()
        removed = n - len(df)
        if removed > 0:
            print(f"  [Step 2] 중복 제거: {removed}건")
        else:
            print(f"  [Step 2] 중복 없음")

    # 3. 텍스트 전처리
    if text_column in df.columns:
        df[text_column] = df[text_column].apply(preprocess_text)
        print(f"  [Step 3] 텍스트 정규화 완료 ({text_column})")

    # 4. 파일 경로 기본 Feature 추출 (있는 경우)
    if file_path_column in df.columns:
        df["file_extension"] = df[file_path_column].str.extract(
            r'\.([a-zA-Z0-9]+)$'
        )[0].str.lower().fillna("unknown")
        df["path_depth"] = df[file_path_column].str.count(r'[/\\]')
        print(f"  [Step 4] 파일 경로 Feature 추출 (extension, depth)")

    # 5. 인덱스 리셋
    df = df.reset_index(drop=True)

    print(f"\n  전처리 완료: {n_before:,}건 -> {len(df):,}건")
    print("=" * 60)

    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    전처리된 데이터를 CSV로 저장합니다.

    Args:
        df: 저장할 DataFrame
        path: 저장 경로 (.csv)
    """
    from pathlib import Path as PathLib

    output_path = PathLib(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[저장 완료]")
    print(f"  경로: {output_path}")
    print(f"  크기: {file_size_mb:.1f} MB")
    print(f"  행 수: {len(df):,}")
    print(f"  열 수: {len(df.columns)}")
