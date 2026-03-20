"""PK 병합 모듈 테스트

회의록 2026-01 반영:
- 복합 PK (server_name + agent_ip + file_path)
- 컬럼명 정규화
- Fallback PK
"""
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np

from src.data.merger import (
    merge_detection_with_labels,
    normalize_columns,
    get_available_pk,
    create_composite_pk,
    validate_pk_columns,
)


class TestNormalizeColumns:
    """컬럼명 정규화 테스트"""

    def test_basic_normalization(self):
        """기본 정규화 테스트"""
        df = pd.DataFrame({
            "dfile_computername": ["server1", "server2"],
            "dfile_agentip": ["10.0.0.1", "10.0.0.2"],
        })

        mapping = {
            "server_name": "dfile_computername",
            "agent_ip": "dfile_agentip",
        }

        result = normalize_columns(df, mapping)

        assert "server_name" in result.columns
        assert "agent_ip" in result.columns

    def test_partial_mapping(self):
        """일부만 매핑되는 경우"""
        df = pd.DataFrame({
            "dfile_computername": ["server1"],
            "other_column": ["value"],
        })

        mapping = {
            "server_name": "dfile_computername",
            "agent_ip": "dfile_agentip",  # 존재하지 않음
        }

        result = normalize_columns(df, mapping)

        assert "server_name" in result.columns
        assert "other_column" in result.columns

    def test_inplace_false_returns_copy(self):
        """inplace=False는 복사본 반환"""
        df = pd.DataFrame({
            "dfile_computername": ["server1"],
        })
        original_columns = list(df.columns)

        mapping = {"server_name": "dfile_computername"}
        result = normalize_columns(df, mapping, inplace=False)

        # 원본은 변경되지 않음
        assert list(df.columns) == original_columns
        # 결과에는 변경됨
        assert "server_name" in result.columns


class TestValidatePkColumns:
    """PK 컬럼 검증 테스트"""

    def test_all_columns_present(self):
        """모든 PK 컬럼이 있는 경우"""
        df = pd.DataFrame({
            "server_name": ["s1", "s2"],
            "agent_ip": ["10.0.0.1", "10.0.0.2"],
            "file_path": ["/a/b", "/c/d"],
        })

        pk_columns = ["server_name", "agent_ip", "file_path"]
        assert validate_pk_columns(df, pk_columns, "TestDataset") is True

    def test_missing_columns(self):
        """일부 PK 컬럼이 없는 경우"""
        df = pd.DataFrame({
            "server_name": ["s1", "s2"],
            "file_path": ["/a/b", "/c/d"],
        })

        pk_columns = ["server_name", "agent_ip", "file_path"]
        assert validate_pk_columns(df, pk_columns, "TestDataset") is False


class TestGetAvailablePk:
    """사용 가능한 PK 확인 테스트"""

    def test_primary_pk_available(self):
        """Primary PK가 모두 있는 경우"""
        df = pd.DataFrame({
            "server_name": ["s1"],
            "agent_ip": ["10.0.0.1"],
            "file_path": ["/a/b"],
        })

        pk_config = {
            "primary": ["server_name", "agent_ip", "file_path"],
            "fallback": ["detection_id"],
        }

        result = get_available_pk(df, pk_config)
        assert result == ["server_name", "agent_ip", "file_path"]

    def test_fallback_pk(self):
        """Primary PK가 없어서 Fallback 사용"""
        df = pd.DataFrame({
            "detection_id": ["id1", "id2"],
            "other_col": ["a", "b"],
        })

        pk_config = {
            "primary": ["server_name", "agent_ip", "file_path"],
            "fallback": ["detection_id"],
        }

        result = get_available_pk(df, pk_config)
        assert result == ["detection_id"]

    def test_no_pk_available(self):
        """PK가 전혀 없는 경우"""
        df = pd.DataFrame({
            "col1": ["a"],
            "col2": ["b"],
        })

        pk_config = {
            "primary": ["server_name"],
            "fallback": ["detection_id"],
        }

        result = get_available_pk(df, pk_config)
        assert result == []

    def test_list_pk_config(self):
        """리스트 형태 PK 설정"""
        df = pd.DataFrame({
            "detection_id": ["id1"],
            "other": ["x"],
        })

        pk_config = ["detection_id"]
        result = get_available_pk(df, pk_config)
        assert result == ["detection_id"]


class TestCreateCompositePk:
    """복합 PK 생성 테스트"""

    def test_composite_pk_creation(self):
        """복합 PK 생성"""
        df = pd.DataFrame({
            "server_name": ["server1", "server2"],
            "agent_ip": ["10.0.0.1", "10.0.0.2"],
            "file_path": ["/path/a", "/path/b"],
        })

        pk_columns = ["server_name", "agent_ip", "file_path"]
        result = create_composite_pk(df, pk_columns)

        assert "composite_pk" in result.columns
        assert "server1||10.0.0.1||/path/a" == result["composite_pk"].iloc[0]

    def test_composite_pk_custom_name(self):
        """커스텀 PK 이름"""
        df = pd.DataFrame({
            "a": ["x"],
            "b": ["y"],
        })

        result = create_composite_pk(df, ["a", "b"], pk_name="my_pk", separator="|")
        assert "my_pk" in result.columns
        assert result["my_pk"].iloc[0] == "x|y"

    def test_handle_null_values(self):
        """NULL 값 처리"""
        df = pd.DataFrame({
            "server_name": ["s1", None],
            "agent_ip": ["ip1", "ip2"],
        })

        pk_columns = ["server_name", "agent_ip"]
        result = create_composite_pk(df, pk_columns)

        # NULL은 빈 문자열로 처리
        assert "composite_pk" in result.columns
        assert result["composite_pk"].iloc[1] == "||ip2"


class TestMergeDetectionWithLabels:
    """검출-레이블 병합 테스트"""

    def test_basic_merge(self):
        """기본 병합 테스트"""
        df_det = pd.DataFrame({
            "detection_id": ["id1", "id2", "id3"],
            "content": ["a", "b", "c"],
        })

        df_lbl = pd.DataFrame({
            "detection_id": ["id1", "id2"],
            "label": ["TP", "FP"],
        })

        result = merge_detection_with_labels(
            df_det, df_lbl,
            pk_columns=["detection_id"],
        )

        assert len(result) == 2  # inner join
        assert "label" in result.columns

    def test_composite_pk_merge(self):
        """복합 PK로 병합"""
        df_det = pd.DataFrame({
            "server_name": ["s1", "s2"],
            "agent_ip": ["ip1", "ip2"],
            "content": ["x", "y"],
        })

        df_lbl = pd.DataFrame({
            "server_name": ["s1"],
            "agent_ip": ["ip1"],
            "label": ["TP"],
        })

        result = merge_detection_with_labels(
            df_det, df_lbl,
            pk_columns=["server_name", "agent_ip"],
        )

        assert len(result) == 1
        assert result["label"].iloc[0] == "TP"

    def test_dict_pk_config(self):
        """딕셔너리 형태 PK 설정으로 병합"""
        df_det = pd.DataFrame({
            "detection_id": ["id1", "id2"],
            "content": ["a", "b"],
        })

        df_lbl = pd.DataFrame({
            "detection_id": ["id1"],
            "label": ["TP"],
        })

        pk_config = {
            "primary": ["server_name", "agent_ip"],  # 없음
            "fallback": ["detection_id"],            # 이게 사용됨
        }

        result = merge_detection_with_labels(
            df_det, df_lbl,
            pk_columns=pk_config,
        )

        assert len(result) == 1

    def test_with_column_mapping(self):
        """컬럼 매핑과 함께 병합"""
        df_det = pd.DataFrame({
            "dfile_computername": ["s1"],
            "content": ["a"],
        })

        df_lbl = pd.DataFrame({
            "server_name": ["s1"],
            "label": ["TP"],
        })

        det_mapping = {"server_name": "dfile_computername"}

        result = merge_detection_with_labels(
            df_det, df_lbl,
            pk_columns=["server_name"],
            detection_columns_mapping=det_mapping,
        )

        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
