"""S6 Split Strategies - Architecture v1.2 §19.2

Group+Time Split (기본값): pk_file 그룹 보존 + 시간 순서 분리
Server Group Split: 서버 단위 보수적 분리
Event Random Split: 디버깅 전용 (공식 평가에서 제외)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def group_time_split(
    df: pd.DataFrame,
    group_col: str = "pk_file",
    time_col: str = "detection_time",
    test_months: int = 1,
) -> tuple[list, list]:
    """pk_file Group + Time Split (Architecture §19.2 기본값).

    각 pk_file 내 최신 이벤트 기준으로 파일을 정렬하여, 가장 최신 파일들을
    test 로 분리한다. train/test 간 pk_file 누수 없음.

    Args:
        df          : Silver DataFrame (group_col, time_col 포함)
        group_col   : 그룹 컬럼 (기본: pk_file)
        time_col    : 시간 컬럼 (기본: detection_time)
        test_months : 마지막 N개월을 test로 분리

    Returns:
        (train_indices, test_indices) - DataFrame 행 인덱스 리스트
    """
    df_reset = df.reset_index(drop=True)
    ts = pd.to_datetime(df_reset[time_col])

    # 파일별 최신 타임스탬프
    file_max_time = (
        df_reset.groupby(group_col, sort=False)[time_col]
        .transform("max")
    )
    file_max_time = pd.to_datetime(file_max_time)

    # 전체 최신 시간 기준 cutoff
    global_max = pd.to_datetime(df_reset[time_col]).max()
    cutoff = global_max - pd.DateOffset(months=test_months)

    # 파일별 max_time이 cutoff 이후면 test
    file_last = (
        df_reset.assign(_ts=pd.to_datetime(df_reset[time_col]))
        .groupby(group_col)["_ts"]
        .max()
    )
    test_files = set(file_last[file_last > cutoff].index)

    test_idx = df_reset.index[df_reset[group_col].isin(test_files)].tolist()
    train_idx = df_reset.index[~df_reset[group_col].isin(test_files)].tolist()

    return train_idx, test_idx


def server_group_split(
    df: pd.DataFrame,
    server_col: str = "server_name",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """서버 그룹 기준 보수적 Split.

    서버 단위로 train/test 분리하여 서버별 데이터 특성 누수 방지.

    Args:
        df         : DataFrame (server_col 포함)
        server_col : 서버 컬럼명 (기본: server_name)
        test_ratio : test 서버 비율 (기본: 0.2)
        seed       : 랜덤 시드

    Returns:
        (train_indices, test_indices)
    """
    df_reset = df.reset_index(drop=True)
    servers = sorted(df_reset[server_col].unique())
    rng = np.random.default_rng(seed)
    n_test_servers = max(1, int(len(servers) * test_ratio))
    test_servers = set(rng.choice(servers, size=n_test_servers, replace=False))

    test_idx = df_reset.index[df_reset[server_col].isin(test_servers)].tolist()
    train_idx = df_reset.index[~df_reset[server_col].isin(test_servers)].tolist()

    return train_idx, test_idx


def event_random_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[list, list]:
    """이벤트 랜덤 Split.

    Warning:
        공식 평가에서 제외 (pk_file 누수 가능). 디버깅 전용.

    Args:
        df        : DataFrame
        test_size : test 비율 (기본: 0.1)
        seed      : 랜덤 시드

    Returns:
        (train_indices, test_indices)
    """
    from sklearn.model_selection import train_test_split

    idx = list(range(len(df)))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    return train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# Secondary Split - label_work_month 기반 시간 분리
# ─────────────────────────────────────────────────────────────────────────────

def _parse_month_label(month_str: str) -> int:
    """한글 월 문자열을 정수로 변환.

    Examples:
        "3월" -> 3, "10월" -> 10, "3" -> 3

    Raises:
        ValueError: 변환 불가능한 문자열
    """
    if isinstance(month_str, (int, float)):
        return int(month_str)
    s = str(month_str).strip().replace("월", "")
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"month_str 변환 실패: {month_str!r}")


def _extract_org_from_filename(filename: str) -> str | None:
    """파일명에서 조직명(CTO/NW/품질혁신센터) 패턴 추출.

    Returns:
        조직명 문자열 또는 None (매칭 실패 시)
    """
    _ORG_PATTERNS = [
        (r"품질혁신센터", "품질혁신센터"),
        (r"CTO", "CTO"),
        (r"NW", "NW"),
    ]
    for pattern, org_name in _ORG_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return org_name
    return None


def work_month_time_split(
    df: pd.DataFrame,
    month_col: str = "label_work_month",
    group_col: str = "pk_file",
    test_months: int = 3,
) -> tuple[list, list]:
    """label_work_month 기준 엄격한 시간 분할 (행 단위).

    한글 월("3월") -> 정수 변환 후, 마지막 test_months개 월에 해당하는
    **행**을 test로 분리한다. 동일 pk_file이 train/test 양쪽에 나타날 수
    있으나, 시간적 분리가 깨지지 않는다.

    예: unique months = [3,4,5,6,7,8,9,10,11,12], test_months=3
        → train: 3~9월 행, test: 10~12월 행 (엄격 분리)

    Args:
        df          : DataFrame (month_col 포함)
        month_col   : 월 컬럼 (기본: label_work_month)
        group_col   : (사용하지 않음, 호환성 유지)
        test_months : 마지막 N개월을 test로 (기본: 3)

    Returns:
        (train_indices, test_indices)
    """
    df_reset = df.reset_index(drop=True).copy()
    df_reset["_month_int"] = df_reset[month_col].apply(_parse_month_label)

    sorted_months = sorted(df_reset["_month_int"].unique())
    if len(sorted_months) <= test_months:
        test_month_set = set(sorted_months)
    else:
        test_month_set = set(sorted_months[-test_months:])

    test_mask = df_reset["_month_int"].isin(test_month_set)
    test_idx = df_reset.index[test_mask].tolist()
    train_idx = df_reset.index[~test_mask].tolist()

    # 진단 로그
    train_months = sorted(df_reset.loc[~test_mask, "_month_int"].unique())
    test_months_actual = sorted(df_reset.loc[test_mask, "_month_int"].unique())
    print(f"  [Temporal] train 월: {[f'{m}월' for m in train_months]}")
    print(f"  [Temporal] test 월:  {[f'{m}월' for m in test_months_actual]}")
    print(f"  [Temporal] train 행: {len(train_idx):,} / test 행: {len(test_idx):,}")

    if group_col in df_reset.columns:
        train_files = set(df_reset.loc[~test_mask, group_col])
        test_files = set(df_reset.loc[test_mask, group_col])
        overlap = train_files & test_files
        print(f"  [Temporal] pk_file 중복(train∩test): {len(overlap):,}건"
              f" (train: {len(train_files):,}, test: {len(test_files):,})")

    return train_idx, test_idx


def org_subset_split(
    df: pd.DataFrame,
    target_org: str,
    org_col: str = "organization",
    source_file_col: str = "_source_file",
) -> tuple[list, list]:
    """조직별 서브셋 Tertiary Split.

    org_col 없으면 source_file_col 파일명에서 CTO/NW/품질혁신센터 패턴 추출.
    추출 실패 시 ValueError.

    Args:
        df             : DataFrame
        target_org     : test로 분리할 조직명
        org_col        : 조직 컬럼명 (기본: organization)
        source_file_col: 파일명 컬럼 (org_col 없을 때 fallback)

    Returns:
        (train_indices, test_indices)

    Raises:
        ValueError: org_col도 없고 source_file_col에서도 추출 실패
    """
    df_reset = df.reset_index(drop=True)

    if org_col in df_reset.columns:
        org_series = df_reset[org_col]
    elif source_file_col in df_reset.columns:
        org_series = df_reset[source_file_col].apply(
            lambda x: _extract_org_from_filename(str(x)) if pd.notna(x) else None
        )
        if org_series.isna().all():
            raise ValueError(
                f"org_col='{org_col}' 없음, source_file_col='{source_file_col}'에서도 "
                "CTO/NW/품질혁신센터 패턴 추출 실패"
            )
    else:
        raise ValueError(
            f"org_col='{org_col}'도 source_file_col='{source_file_col}'도 없음"
        )

    test_mask = org_series == target_org
    test_idx = df_reset.index[test_mask].tolist()
    train_idx = df_reset.index[~test_mask].tolist()

    return train_idx, test_idx
