"""PK 기반 데이터 병합 모듈

회의록 2026-01 반영:
- 복합 PK 지원 (서버명 + IP + 파일경로)
- 데이터셋 A/B/C 컬럼 정규화
- Fallback PK 지원
"""
import pandas as pd
from typing import List, Optional, Dict, Union

from src.utils.constants import PK_COLUMNS


def normalize_columns(
    df: pd.DataFrame,
    columns_mapping: Dict[str, str],
    inplace: bool = False,
) -> pd.DataFrame:
    """
    데이터셋별 컬럼명을 표준 컬럼명으로 정규화합니다.

    Args:
        df: 원본 DataFrame
        columns_mapping: {표준컬럼명: 원본컬럼명} 매핑 딕셔너리
        inplace: True면 원본 수정, False면 복사본 반환

    Returns:
        컬럼명이 정규화된 DataFrame
    """
    if not inplace:
        df = df.copy()

    # 역매핑 생성: {원본컬럼명: 표준컬럼명}
    reverse_mapping = {v: k for k, v in columns_mapping.items() if v in df.columns}

    if reverse_mapping:
        df = df.rename(columns=reverse_mapping)
        print(f"  [컬럼 정규화] {len(reverse_mapping)}개 컬럼 매핑 완료")

    return df


def validate_pk_columns(
    df: pd.DataFrame,
    pk_columns: List[str],
    dataset_name: str = "데이터",
) -> bool:
    """
    PK 컬럼 존재 여부를 검증합니다.

    Args:
        df: 검증할 DataFrame
        pk_columns: PK 컬럼 목록
        dataset_name: 로그에 표시할 데이터셋 이름

    Returns:
        모든 PK 컬럼이 존재하면 True
    """
    missing = [col for col in pk_columns if col not in df.columns]

    if missing:
        print(f"  [경고] {dataset_name}에 PK 컬럼 누락: {missing}")
        print(f"         사용 가능한 컬럼: {list(df.columns)}")
        return False

    return True


def get_available_pk(
    df: pd.DataFrame,
    pk_config: Union[List[str], Dict[str, List[str]]],
) -> List[str]:
    """
    DataFrame에서 사용 가능한 PK 컬럼을 결정합니다.

    Args:
        df: 검사할 DataFrame
        pk_config: PK 설정 (리스트 또는 primary/fallback 딕셔너리)

    Returns:
        사용할 PK 컬럼 리스트
    """
    # 기존 방식 (단순 리스트)
    if isinstance(pk_config, list):
        if all(col in df.columns for col in pk_config):
            return pk_config
        return []

    # 새 방식 (primary/fallback 딕셔너리)
    if isinstance(pk_config, dict):
        primary = pk_config.get("primary", [])
        fallback = pk_config.get("fallback", [])

        # primary PK 시도
        if primary and all(col in df.columns for col in primary):
            print(f"  [PK] 복합 PK 사용: {primary}")
            return primary

        # fallback PK 시도
        if fallback and all(col in df.columns for col in fallback):
            print(f"  [PK] Fallback PK 사용: {fallback}")
            return fallback

    return []


def merge_detection_with_labels(
    df_detection: pd.DataFrame,
    df_label: pd.DataFrame,
    pk_columns: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    detection_columns_mapping: Optional[Dict[str, str]] = None,
    label_columns_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    검출 결과와 레이블 데이터를 PK 기반으로 병합합니다.

    Args:
        df_detection: Server-i 검출 결과 DataFrame (데이터셋 A)
        df_label: 레이블(정탐/오탐) DataFrame (데이터셋 B)
        pk_columns: Primary Key 설정
            - 리스트: ["col1", "col2"] 형태
            - 딕셔너리: {"primary": [...], "fallback": [...]} 형태
        detection_columns_mapping: 검출 데이터 컬럼 매핑 (표준->원본)
        label_columns_mapping: 레이블 데이터 컬럼 매핑 (표준->원본)

    Returns:
        inner join으로 병합된 DataFrame
    """
    if pk_columns is None:
        pk_columns = PK_COLUMNS

    print("\n[데이터 병합 시작]")

    # 컬럼 정규화 (원본 컬럼명 -> 표준 컬럼명)
    if detection_columns_mapping:
        df_detection = normalize_columns(df_detection, detection_columns_mapping)

    if label_columns_mapping:
        df_label = normalize_columns(df_label, label_columns_mapping)

    # 사용 가능한 PK 결정
    pk_det = get_available_pk(df_detection, pk_columns)
    pk_lbl = get_available_pk(df_label, pk_columns)

    if not pk_det:
        raise ValueError(
            f"검출 데이터에서 사용 가능한 PK 컬럼을 찾을 수 없습니다.\n"
            f"설정된 PK: {pk_columns}\n"
            f"데이터 컬럼: {list(df_detection.columns)}"
        )

    if not pk_lbl:
        raise ValueError(
            f"레이블 데이터에서 사용 가능한 PK 컬럼을 찾을 수 없습니다.\n"
            f"설정된 PK: {pk_columns}\n"
            f"데이터 컬럼: {list(df_label.columns)}"
        )

    # PK가 일치하는지 확인
    if set(pk_det) != set(pk_lbl):
        print(f"  [경고] 검출 PK({pk_det})와 레이블 PK({pk_lbl})가 다릅니다.")
        # 공통 PK 사용
        common_pk = list(set(pk_det) & set(pk_lbl))
        if not common_pk:
            raise ValueError("공통 PK 컬럼이 없습니다.")
        print(f"  [PK] 공통 PK 사용: {common_pk}")
        pk_columns_final = common_pk
    else:
        pk_columns_final = pk_det

    # 병합 수행
    df_merged = df_detection.merge(df_label, on=pk_columns_final, how="inner")

    # 통계 출력
    n_det = len(df_detection)
    n_lbl = len(df_label)
    n_merged = len(df_merged)

    print("\n[병합 결과]")
    print(f"  검출 데이터:  {n_det:,}건")
    print(f"  레이블 데이터: {n_lbl:,}건")
    print(f"  병합 결과:    {n_merged:,}건")
    print(f"  사용된 PK:    {pk_columns_final}")

    if n_merged < n_lbl:
        n_unmatched = n_lbl - n_merged
        print(f"  [주의] 레이블 {n_unmatched:,}건이 매핑되지 않음 ({n_unmatched / n_lbl * 100:.1f}%)")

    if n_merged < n_det:
        n_unlabeled = n_det - n_merged
        print(f"  [참고] 검출 {n_unlabeled:,}건에 레이블 없음 ({n_unlabeled / n_det * 100:.1f}%)")

    return df_merged


def merge_multiple_datasets(
    dataframes: Dict[str, pd.DataFrame],
    pk_columns: Union[List[str], Dict[str, List[str]]],
    columns_mappings: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    여러 데이터셋을 순차적으로 병합합니다.

    Args:
        dataframes: {데이터셋명: DataFrame} 딕셔너리
        pk_columns: PK 설정
        columns_mappings: {데이터셋명: 컬럼매핑} 딕셔너리

    Returns:
        모든 데이터셋이 병합된 DataFrame
    """
    if columns_mappings is None:
        columns_mappings = {}

    print("\n[다중 데이터셋 병합]")
    print(f"  대상 데이터셋: {list(dataframes.keys())}")

    # 각 데이터셋 컬럼 정규화
    normalized = {}
    for name, df in dataframes.items():
        mapping = columns_mappings.get(name, {})
        if mapping:
            normalized[name] = normalize_columns(df, mapping)
        else:
            normalized[name] = df.copy()
        print(f"  [{name}] {len(df):,}건 로드")

    # 순차 병합 (첫 번째를 기준으로)
    names = list(normalized.keys())
    result = normalized[names[0]]

    for name in names[1:]:
        pk = get_available_pk(result, pk_columns)
        pk_other = get_available_pk(normalized[name], pk_columns)
        common_pk = list(set(pk) & set(pk_other))

        if common_pk:
            before = len(result)
            result = result.merge(normalized[name], on=common_pk, how="inner")
            print(f"  [{names[0]}] + [{name}] 병합: {before:,} -> {len(result):,}건")
        else:
            print(f"  [경고] [{name}]와 공통 PK 없음, 병합 스킵")

    print(f"\n  최종 병합 결과: {len(result):,}건")
    return result


def create_composite_pk(
    df: pd.DataFrame,
    pk_columns: List[str],
    pk_name: str = "composite_pk",
    separator: str = "||",
) -> pd.DataFrame:
    """
    여러 컬럼을 결합하여 단일 복합 PK 컬럼을 생성합니다.

    Args:
        df: 원본 DataFrame
        pk_columns: 결합할 컬럼 목록
        pk_name: 생성할 PK 컬럼명
        separator: 결합 시 사용할 구분자

    Returns:
        복합 PK 컬럼이 추가된 DataFrame
    """
    df = df.copy()

    # 결측값을 빈 문자열로 처리하고 문자열로 변환
    pk_values = df[pk_columns].fillna("").astype(str)
    df[pk_name] = pk_values.apply(lambda row: separator.join(row), axis=1)

    print(f"  [복합 PK] '{pk_name}' 생성 완료 ({len(pk_columns)}개 컬럼 결합)")
    return df


def detect_cross_label_duplicates(
    df: pd.DataFrame,
    pk_col: str = "pk_event",
    label_col: str = "label_raw",
    tp_value: str = "TP",
    fp_value: str = "FP",
) -> pd.DataFrame:
    """정탐/오탐 파일 교차 중복 감지.

    동일 pk_event가 TP와 FP 양쪽에 존재하는 건을 찾아 반환한다.
    이런 건은 레이블 충돌(동일 파일에 대해 정탐 판정 + 오탐 판정)이므로
    Phase 0 품질 검증에서 우선 처리해야 한다.

    Args:
        df:        label_raw 컬럼이 포함된 합산 DataFrame
        pk_col:    PK 컬럼명 (기본: "pk_event")
        label_col: 레이블 컬럼명 (기본: "label_raw")
        tp_value:  정탐 레이블 값 (기본: "TP")
        fp_value:  오탐 레이블 값 (기본: "FP")

    Returns:
        교차 중복 pk 목록 DataFrame - 컬럼: [pk_col, "tp_count", "fp_count", "conflict_count"]
        교차 중복 없으면 빈 DataFrame 반환
    """
    if pk_col not in df.columns or label_col not in df.columns:
        return pd.DataFrame()

    # pk별 TP/FP 건수 집계
    tp_counts = (
        df[df[label_col] == tp_value].groupby(pk_col).size().rename("tp_count")
    )
    fp_counts = (
        df[df[label_col] == fp_value].groupby(pk_col).size().rename("fp_count")
    )

    # 양쪽에 모두 존재하는 pk만 추출
    cross = pd.concat([tp_counts, fp_counts], axis=1).dropna()

    if cross.empty:
        return pd.DataFrame()

    cross = cross.reset_index()
    cross["tp_count"] = cross["tp_count"].astype(int)
    cross["fp_count"] = cross["fp_count"].astype(int)
    cross["conflict_count"] = cross["tp_count"] + cross["fp_count"]

    return cross[[pk_col, "tp_count", "fp_count", "conflict_count"]]
