"""
Stage S2-6: File-level Aggregation - Architecture.md §6.6

pk_file 단위 집계를 train fold에서만 계산하고,
test fold에는 join(merge)으로 적용한다.

누수 차단 원칙:
  compute_file_aggregates(df_train) - 반드시 train 데이터만 전달
  merge_file_aggregates(df_test, agg) - test에 join 후 NaN은 0으로 채움

'10자 컨텍스트' 한계를 구조적으로 보완하는 가장 강력한 방법.
"""

from __future__ import annotations

import pandas as pd


def compute_file_aggregates(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    pk_file 단위 검출 통계 집계.

    반드시 train fold만 전달해야 한다.
    test fold 정보가 포함되면 데이터 누수가 발생한다.

    Parameters
    ----------
    df_train : train fold DataFrame (pk_file, pk_event, email_domain, ... 포함)

    Returns
    -------
    pk_file 단위 집계 DataFrame:
        pk_file                          : 파일 PK
        file_event_count                 : 해당 파일의 총 이벤트 수
        file_unique_domains              : 고유 이메일 도메인 수
        file_has_timestamp_kw_ratio      : timestamp 키워드 보유 비율
        file_has_bytes_kw_ratio          : bytes 키워드 보유 비율
        file_pii_type_diversity          : PII 유형 다양성 (nunique)
    """
    agg = df_train.groupby("pk_file").agg(
        file_event_count=("pk_event", "count"),
        file_unique_domains=("email_domain", lambda s: s.dropna().nunique()),
        file_has_timestamp_kw_ratio=(
            "has_timestamp_kw", "mean"
        ) if "has_timestamp_kw" in df_train.columns else ("pk_event", lambda s: 0),
        file_has_bytes_kw_ratio=(
            "has_byte_kw", "mean"
        ) if "has_byte_kw" in df_train.columns else ("pk_event", lambda s: 0),
        file_pii_type_diversity=(
            "pii_type_inferred", "nunique"
        ) if "pii_type_inferred" in df_train.columns else ("pk_event", lambda s: 1),
    ).reset_index()

    return agg


def merge_file_aggregates(
    df: pd.DataFrame,
    file_agg: pd.DataFrame,
    how: str = "left",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    df에 file_agg를 pk_file 기준으로 join.

    test fold에서는 학습에 없는 pk_file이 있을 수 있으므로
    left join 후 NaN을 fill_value(기본 0.0)로 채운다.

    Parameters
    ----------
    df        : 이벤트 레벨 DataFrame (pk_file 컬럼 필요)
    file_agg  : compute_file_aggregates 결과
    how       : join 방식 (기본 'left')
    fill_value: NaN 채움 값 (기본 0.0)

    Returns
    -------
    agg 컬럼이 추가된 DataFrame (행 수 = len(df))
    """
    agg_cols = [c for c in file_agg.columns if c != "pk_file"]
    merged = df.merge(file_agg, on="pk_file", how=how)
    merged[agg_cols] = merged[agg_cols].fillna(fill_value)
    return merged
