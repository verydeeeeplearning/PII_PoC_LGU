"""Tabular Feature 생성 모듈

회의록 2026-01 반영:
- Docker overlay 경로 Feature
- Hadoop/HDFS 경로 Feature
- 레거시 경로 Feature
"""
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Optional, Tuple, List
import joblib

from src.utils.constants import FILE_PATH_COLUMN


# ============================================================
# 회의록 확정 경로 패턴
# ============================================================

# Docker 관련 경로 패턴
DOCKER_PATH_PATTERNS = [
    r"/docker/",
    r"/overlay/",
    r"/overlay2/",
    r"/containers/",
    r"docker\.io",
]

# Hadoop/HDFS 관련 경로 패턴
HADOOP_PATH_PATTERNS = [
    r"hadoop",
    r"hdfs",
    r"DATANODE",
    r"NAMENODE",
    r"hive",
    r"spark",
    r"yarn",
]

# 레거시 날짜 패턴 (2012년 등 오래된 데이터)
LEGACY_DATE_PATTERNS = [
    r"/201[0-5]\d{4}/",      # 2010-2015년
    r"/20[0-1]\d[-/]\d{2}[-/]",
]

# 시스템/로그 경로 패턴
SYSTEM_PATH_PATTERNS = [
    r"/var/log/",
    r"/var/lib/",
    r"/etc/",
    r"/usr/",
    r"/opt/",
    r"/tmp/",
    r"\.log$",
    r"\.log\.\d+$",
]

# 개발/테스트 경로 패턴 (기존 + 확장)
DEV_PATH_PATTERNS = [
    r"/test/",
    r"/tests/",
    r"/testing/",
    r"/dev/",
    r"/develop/",
    r"/debug/",
    r"/mock/",
    r"/stub/",
    r"/fixture/",
]


# ============================================================
# 기존 Feature 함수
# ============================================================

def create_file_path_features(
    df: pd.DataFrame,
    path_column: str = FILE_PATH_COLUMN,
) -> pd.DataFrame:
    """
    파일 경로 기반 Feature 생성

    생성 Feature (8개):
        - file_extension: 파일 확장자 (문자열, 인코딩 전)
        - path_depth: 경로 구분자 수
        - path_has_test: test/tset/testing 포함 여부 (0/1)
        - path_has_dev: dev/develop/debug 포함 여부 (0/1)
        - path_has_sample: sample/example/demo 포함 여부 (0/1)
        - path_has_backup: backup/bak/old/archive 포함 여부 (0/1)
        - path_has_log: log/logs/logging 포함 여부 (0/1)
        - path_has_config: config/conf/cfg/setting 포함 여부 (0/1)

    Args:
        df: DataFrame (file_path 컬럼 필요)
        path_column: 파일 경로 컬럼명

    Returns:
        파일 경로 Feature DataFrame (8개 컬럼)
    """
    if path_column not in df.columns:
        print(f"  [건너뜀] '{path_column}' 컬럼 없음")
        return pd.DataFrame(index=df.index)

    paths = df[path_column].fillna("")

    features = {
        "file_extension": paths.str.extract(r'\.([a-zA-Z0-9]+)$')[0].fillna("none").str.lower(),
        "path_depth": paths.str.count(r'[/\\]'),
        "path_has_test": paths.str.lower().str.contains(
            r'test|tset|testing', regex=True
        ).astype(int),
        "path_has_dev": paths.str.lower().str.contains(
            r'dev|develop|debug', regex=True
        ).astype(int),
        "path_has_sample": paths.str.lower().str.contains(
            r'sample|example|demo', regex=True
        ).astype(int),
        "path_has_backup": paths.str.lower().str.contains(
            r'backup|bak|old|archive', regex=True
        ).astype(int),
        "path_has_log": paths.str.lower().str.contains(
            r'log|logs|logging', regex=True
        ).astype(int),
        "path_has_config": paths.str.lower().str.contains(
            r'config|conf|cfg|setting', regex=True
        ).astype(int),
    }

    df_features = pd.DataFrame(features, index=df.index)
    print(f"[파일 경로 Feature] {df_features.shape[1]}개 생성")
    return df_features


def encode_categorical(
    train_df: pd.DataFrame,
    columns: list,
    test_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> Tuple:
    """
    범주형 변수 Label Encoding

    학습셋에 없는 테스트 값은 "unknown"으로 매핑합니다.

    Args:
        train_df: 학습 DataFrame
        columns: 인코딩할 컬럼 목록
        test_df: 테스트 DataFrame (선택)
        save_path: 인코더 저장 경로 (.joblib)

    Returns:
        test_df 있음: (encoded_train, encoded_test, encoders_dict)
        test_df 없음: (encoded_train, encoders_dict)
    """
    encoders = {}
    encoded_train = train_df.copy()
    encoded_test = test_df.copy() if test_df is not None else None

    for col in columns:
        if col not in train_df.columns:
            print(f"  [건너뜀] '{col}' 컬럼 없음")
            continue

        le = LabelEncoder()
        encoded_train[col] = le.fit_transform(train_df[col].astype(str))

        if encoded_test is not None and col in test_df.columns:
            known_classes = set(le.classes_)
            test_values = test_df[col].astype(str).apply(
                lambda x: x if x in known_classes else "unknown"
            )
            if "unknown" not in known_classes:
                le.classes_ = np.append(le.classes_, "unknown")
            encoded_test[col] = le.transform(test_values)

        encoders[col] = le
        print(f"[인코딩] {col}: {len(le.classes_)}개 카테고리")

    if save_path is not None:
        joblib.dump(encoders, save_path)
        print(f"[저장] {save_path}")

    if encoded_test is not None:
        return encoded_train, encoded_test, encoders

    return encoded_train, encoders


# ============================================================
# 회의록 반영 신규 Feature 함수
# ============================================================

def create_docker_features(
    df: pd.DataFrame,
    path_column: str = FILE_PATH_COLUMN,
) -> pd.DataFrame:
    """
    Docker 관련 경로 Feature 생성

    회의록 발견:
    - Docker 컨테이너 overlay 파일시스템에서 대량 검출 발생
    - Docker layer ID(해시값)가 파일 경로에 포함

    생성 Feature (2개):
        - path_has_docker: Docker 경로 포함 여부 (0/1)
        - path_has_overlay: overlay 경로 포함 여부 (0/1)

    Args:
        df: DataFrame
        path_column: 파일 경로 컬럼명

    Returns:
        Docker Feature DataFrame
    """
    if path_column not in df.columns:
        print(f"  [건너뜀] '{path_column}' 컬럼 없음 (Docker Feature)")
        return pd.DataFrame(index=df.index)

    paths = df[path_column].fillna("").str.lower()

    features = {
        "path_has_docker": paths.str.contains(
            r'docker|container', regex=True
        ).astype(int),
        "path_has_overlay": paths.str.contains(
            r'overlay', regex=True
        ).astype(int),
    }

    df_docker = pd.DataFrame(features, index=df.index)

    print(f"[Docker Feature] {df_docker.shape[1]}개 생성")
    print(f"  Docker 경로: {features['path_has_docker'].sum():,}건")
    print(f"  Overlay 경로: {features['path_has_overlay'].sum():,}건")

    return df_docker


def create_hadoop_features(
    df: pd.DataFrame,
    path_column: str = FILE_PATH_COLUMN,
) -> pd.DataFrame:
    """
    Hadoop/HDFS 관련 경로 Feature 생성

    회의록 확인된 파일:
    - hadoop-cmf-hdfs-DATANODE-dbhotdat55.bdp.lguplus.co.kr.log.out.41

    생성 Feature (2개):
        - path_has_hadoop: Hadoop 관련 경로 포함 여부 (0/1)
        - path_has_hdfs: HDFS 관련 경로 포함 여부 (0/1)

    Args:
        df: DataFrame
        path_column: 파일 경로 컬럼명

    Returns:
        Hadoop Feature DataFrame
    """
    if path_column not in df.columns:
        print(f"  [건너뜀] '{path_column}' 컬럼 없음 (Hadoop Feature)")
        return pd.DataFrame(index=df.index)

    paths = df[path_column].fillna("").str.lower()

    features = {
        "path_has_hadoop": paths.str.contains(
            r'hadoop|hive|spark|yarn', regex=True
        ).astype(int),
        "path_has_hdfs": paths.str.contains(
            r'hdfs|datanode|namenode', regex=True
        ).astype(int),
    }

    df_hadoop = pd.DataFrame(features, index=df.index)

    print(f"[Hadoop Feature] {df_hadoop.shape[1]}개 생성")
    print(f"  Hadoop 경로: {features['path_has_hadoop'].sum():,}건")
    print(f"  HDFS 경로:   {features['path_has_hdfs'].sum():,}건")

    return df_hadoop


def create_legacy_features(
    df: pd.DataFrame,
    path_column: str = FILE_PATH_COLUMN,
) -> pd.DataFrame:
    """
    레거시 데이터 관련 Feature 생성

    회의록 확인:
    - 로그 날짜가 2012년인 레거시 서버 데이터 잔존
    - 경로: /JDINAS/R39/data1/mmu/CM/log/20120306/

    생성 Feature (2개):
        - path_has_legacy_date: 오래된 날짜 패턴 포함 여부 (0/1)
        - path_has_jdinas: JDINAS 경로 포함 여부 (0/1)

    Args:
        df: DataFrame
        path_column: 파일 경로 컬럼명

    Returns:
        레거시 Feature DataFrame
    """
    if path_column not in df.columns:
        print(f"  [건너뜀] '{path_column}' 컬럼 없음 (Legacy Feature)")
        return pd.DataFrame(index=df.index)

    paths = df[path_column].fillna("")

    features = {
        "path_has_legacy_date": paths.str.contains(
            r'/201[0-5]\d{4}/', regex=True
        ).astype(int),
        "path_has_jdinas": paths.str.upper().str.contains(
            r'JDINAS', regex=True
        ).astype(int),
    }

    df_legacy = pd.DataFrame(features, index=df.index)

    print(f"[레거시 Feature] {df_legacy.shape[1]}개 생성")
    print(f"  레거시 날짜: {features['path_has_legacy_date'].sum():,}건")
    print(f"  JDINAS 경로: {features['path_has_jdinas'].sum():,}건")

    return df_legacy


def create_system_path_features(
    df: pd.DataFrame,
    path_column: str = FILE_PATH_COLUMN,
) -> pd.DataFrame:
    """
    시스템 경로 관련 Feature 생성

    생성 Feature (3개):
        - path_is_system: 시스템 경로 여부 (/var, /etc, /usr 등)
        - path_is_temp: 임시 경로 여부 (/tmp 등)
        - file_is_log: 로그 파일 여부 (.log 확장자)

    Args:
        df: DataFrame
        path_column: 파일 경로 컬럼명

    Returns:
        시스템 경로 Feature DataFrame
    """
    if path_column not in df.columns:
        print(f"  [건너뜀] '{path_column}' 컬럼 없음 (System Feature)")
        return pd.DataFrame(index=df.index)

    paths = df[path_column].fillna("").str.lower()

    features = {
        "path_is_system": paths.str.contains(
            r'^/var/|^/etc/|^/usr/|^/opt/', regex=True
        ).astype(int),
        "path_is_temp": paths.str.contains(
            r'/tmp/|/temp/', regex=True
        ).astype(int),
        "file_is_log": paths.str.contains(
            r'\.log(?:$|\.\d+$)', regex=True
        ).astype(int),
    }

    df_system = pd.DataFrame(features, index=df.index)

    print(f"[시스템 경로 Feature] {df_system.shape[1]}개 생성")

    return df_system


def create_all_path_features(
    df: pd.DataFrame,
    path_column: str = FILE_PATH_COLUMN,
    include_docker: bool = True,
    include_hadoop: bool = True,
    include_legacy: bool = True,
    include_system: bool = True,
) -> pd.DataFrame:
    """
    모든 경로 기반 Feature를 통합 생성

    Args:
        df: DataFrame
        path_column: 파일 경로 컬럼명
        include_docker: Docker Feature 포함 여부
        include_hadoop: Hadoop Feature 포함 여부
        include_legacy: 레거시 Feature 포함 여부
        include_system: 시스템 경로 Feature 포함 여부

    Returns:
        통합 Feature DataFrame
    """
    print("\n[경로 Feature 통합 생성]")

    feature_dfs = []

    # 1. 기본 경로 Feature (8개)
    df_basic = create_file_path_features(df, path_column)
    if not df_basic.empty:
        feature_dfs.append(df_basic)

    # 2. Docker Feature (2개) - 회의록 반영
    if include_docker:
        df_docker = create_docker_features(df, path_column)
        if not df_docker.empty:
            feature_dfs.append(df_docker)

    # 3. Hadoop Feature (2개) - 회의록 반영
    if include_hadoop:
        df_hadoop = create_hadoop_features(df, path_column)
        if not df_hadoop.empty:
            feature_dfs.append(df_hadoop)

    # 4. 레거시 Feature (2개) - 회의록 반영
    if include_legacy:
        df_legacy = create_legacy_features(df, path_column)
        if not df_legacy.empty:
            feature_dfs.append(df_legacy)

    # 5. 시스템 경로 Feature (3개)
    if include_system:
        df_system = create_system_path_features(df, path_column)
        if not df_system.empty:
            feature_dfs.append(df_system)

    if not feature_dfs:
        return pd.DataFrame(index=df.index)

    # 통합
    df_all = pd.concat(feature_dfs, axis=1)

    print(f"\n[통합 결과] 총 {df_all.shape[1]}개 경로 Feature 생성")
    return df_all


# ============================================================
# S2-4: Tabular 피처 정규화 (Architecture.md §6.4)
# ============================================================

def extract_tabular_features(row: dict) -> dict:
    """
    단일 이벤트(row dict)에서 Tabular 피처 추출 (Architecture.md §6.4).

    생성 피처:
        inspect_count_raw    : 원본 검출 건수
        inspect_count_log1p  : log1p 변환 값 (극단값 보정)
        is_mass_detection    : count > 10,000 이면 1
        is_extreme_detection : count > 100,000 이면 1
        pii_type_inferred    : PII 유형 재추론 값 (그대로 전달)

    Parameters
    ----------
    row : dict 형태의 단일 이벤트 (silver_detections 한 행)

    Returns
    -------
    피처 dict
    """
    import math
    import pandas as _pd
    _count_raw = _pd.to_numeric(row.get("inspect_count", 0), errors="coerce")
    count = int(_count_raw) if not _pd.isna(_count_raw) else 0

    return {
        "inspect_count_raw": count,
        "inspect_count_log1p": math.log1p(count),
        "is_mass_detection": int(count > 10_000),
        "is_extreme_detection": int(count > 100_000),
        "pii_type_inferred": row.get("pii_type_inferred", "unknown"),
    }
