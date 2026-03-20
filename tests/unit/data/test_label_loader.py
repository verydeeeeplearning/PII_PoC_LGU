"""Unit tests for label_loader.py / column_normalizer.py / merger.py

폴더 구조 (2026-03-11 확인):
  data/raw/label/
    25년 정탐 (3월~12월)/{월}/파일들  → label_raw = TP
    25년 오탐 (3월~12월)/{월}/파일들  → label_raw = FP
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.column_normalizer import ColumnNormalizer
from src.data.label_loader import LabelLoader, compute_pk_event
from src.data.merger import detect_cross_label_duplicates


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

KOREAN_COLS = [
    "서버이름", "에이전트IP", "파일경로", "파일이름",
    "패턴개수", "주민등록번호개수", "핸드폰번호개수", "이메일주소개수",
    "파일생성일시", "파일크기",
]

ENGLISH_COLS = [
    "server_name", "agent_ip", "file_path", "file_name",
    "pattern_count", "ssn_count", "phone_count", "email_count",
    "file_created_at",  # file_size 는 drop=True 이므로 제외
]

_TP_FOLDER = "25년 정탐 (3월~12월)"
_FP_FOLDER = "25년 오탐 (3월~12월)"


def _make_label_csv(
    path: Path,
    n_rows: int = 3,
    use_alias: bool = False,
    created_at_fmt: str = "2025-07-01 10:00:00",
) -> Path:
    """한글 컬럼명 CSV 파일 생성 (테스트용)."""
    col_server = "서버명" if use_alias else "서버이름"
    df = pd.DataFrame({
        col_server: [f"srv{i}" for i in range(n_rows)],
        "에이전트IP": [f"10.0.0.{i}" for i in range(n_rows)],
        "파일경로": [f"/var/log/app/file_{i}.log" for i in range(n_rows)],
        "파일이름": [f"file_{i}.log" for i in range(n_rows)],
        "패턴개수": [100 * (i + 1) for i in range(n_rows)],
        "주민등록번호개수": [i for i in range(n_rows)],
        "핸드폰번호개수": [i for i in range(n_rows)],
        "이메일주소개수": [i for i in range(n_rows)],
        "파일생성일시": [created_at_fmt] * n_rows,
        "파일크기": [None] * n_rows,
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _make_tp_fp_structure(
    root: Path,
    month: str = "3월",
    tp_rows: int = 3,
    fp_rows: int = 2,
) -> None:
    """실제 폴더 구조 기반 테스트 데이터 생성.

    root/
      25년 정탐 (3월~12월)/{month}/정탐 취합 보고 자료_{month}_CTO.csv
      25년 오탐 (3월~12월)/{month}/오탐 취합 보고 자료_{month}_CTO.csv
    """
    if tp_rows > 0:
        tp_dir = root / _TP_FOLDER / month
        tp_dir.mkdir(parents=True, exist_ok=True)
        _make_label_csv(tp_dir / f"정탐 취합 보고 자료_{month}_CTO.csv", n_rows=tp_rows)
    if fp_rows > 0:
        fp_dir = root / _FP_FOLDER / month
        fp_dir.mkdir(parents=True, exist_ok=True)
        _make_label_csv(fp_dir / f"오탐 취합 보고 자료_{month}_CTO.csv", n_rows=fp_rows)


# ─────────────────────────────────────────────────────────────────────────────
# ColumnNormalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestColumnNormalizer:
    @pytest.fixture
    def normalizer(self):
        return ColumnNormalizer()

    @pytest.fixture
    def normalizer_from_tmp(self, tmp_path):
        mapping = {
            "mappings": {
                "server_name": {"primary": "서버이름", "aliases": ["서버명"]},
                "agent_ip": {"primary": "에이전트IP", "aliases": []},
                "file_size": {"primary": "파일크기", "aliases": [], "drop": True},
            }
        }
        p = tmp_path / "col_map.yaml"
        p.write_text(yaml.dump(mapping, allow_unicode=True), encoding="utf-8")
        return ColumnNormalizer(mapping_path=p)

    def test_primary_korean_converted_to_english(self, normalizer_from_tmp):
        df = pd.DataFrame({"서버이름": ["s1"], "에이전트IP": ["10.0.0.1"]})
        result = normalizer_from_tmp.normalize(df)
        assert "server_name" in result.columns
        assert "agent_ip" in result.columns
        assert "서버이름" not in result.columns

    def test_alias_variant_also_converted(self, normalizer_from_tmp):
        df = pd.DataFrame({"서버명": ["s1"], "에이전트IP": ["10.0.0.1"]})
        result = normalizer_from_tmp.normalize(df)
        assert "server_name" in result.columns
        assert "서버명" not in result.columns

    def test_drop_column_removed_after_normalize(self, normalizer_from_tmp):
        df = pd.DataFrame({"서버이름": ["s1"], "파일크기": [None]})
        result = normalizer_from_tmp.normalize(df)
        assert "file_size" not in result.columns
        assert "파일크기" not in result.columns

    def test_unknown_korean_column_kept_with_warning(self, normalizer_from_tmp, caplog):
        import logging
        df = pd.DataFrame({"서버이름": ["s1"], "미지의컬럼": ["x"]})
        with caplog.at_level(logging.WARNING, logger="src.data.column_normalizer"):
            result = normalizer_from_tmp.normalize(df)
        assert "미지의컬럼" in result.columns
        assert "미지의컬럼" in caplog.text

    def test_english_columns_not_touched(self, normalizer_from_tmp):
        df = pd.DataFrame({"server_name": ["s1"], "agent_ip": ["10.0.0.1"]})
        result = normalizer_from_tmp.normalize(df)
        assert list(result.columns) == ["server_name", "agent_ip"]

    def test_original_dataframe_not_modified(self, normalizer_from_tmp):
        df = pd.DataFrame({"서버이름": ["s1"]})
        _ = normalizer_from_tmp.normalize(df)
        assert "서버이름" in df.columns

    def test_all_9_label_columns_converted(self, normalizer):
        df = pd.DataFrame({col: ["x"] for col in KOREAN_COLS})
        result = normalizer.normalize(df)
        for eng_col in ENGLISH_COLS:
            assert eng_col in result.columns, f"'{eng_col}' 변환 실패"
        assert "file_size" not in result.columns
        assert "파일크기" not in result.columns

    def test_empty_dataframe_returns_empty(self, normalizer_from_tmp):
        df = pd.DataFrame()
        result = normalizer_from_tmp.normalize(df)
        assert result.empty

    def test_missing_mapping_file_graceful(self, tmp_path):
        normalizer = ColumnNormalizer(mapping_path=tmp_path / "nonexistent.yaml")
        df = pd.DataFrame({"서버이름": ["s1"]})
        result = normalizer.normalize(df)
        assert "서버이름" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# compute_pk_event
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePkEvent:
    def test_produces_sha256_hex_string(self):
        df = pd.DataFrame({
            "server_name": ["srv1"],
            "agent_ip": ["10.0.0.1"],
            "file_path": ["/var/log/app.log"],
            "file_name": ["app.log"],
            "file_created_at": ["2025-07-01"],
        })
        result = compute_pk_event(df, ["server_name", "agent_ip", "file_path", "file_name", "file_created_at"])
        assert len(result) == 1
        assert len(result.iloc[0]) == 64

    def test_deterministic_same_input_same_hash(self):
        df = pd.DataFrame({
            "server_name": ["srv1", "srv1"],
            "agent_ip": ["10.0.0.1", "10.0.0.1"],
            "file_path": ["/a/b.log", "/a/b.log"],
            "file_name": ["b.log", "b.log"],
            "file_created_at": ["2025-07-01", "2025-07-01"],
        })
        result = compute_pk_event(df, ["server_name", "agent_ip", "file_path", "file_name", "file_created_at"])
        assert result.iloc[0] == result.iloc[1]

    def test_different_input_different_hash(self):
        df = pd.DataFrame({
            "server_name": ["srv1", "srv2"],
            "agent_ip": ["10.0.0.1", "10.0.0.1"],
            "file_path": ["/a/b.log", "/a/b.log"],
            "file_name": ["b.log", "b.log"],
            "file_created_at": ["2025-07-01", "2025-07-01"],
        })
        result = compute_pk_event(df, ["server_name", "agent_ip", "file_path", "file_name", "file_created_at"])
        assert result.iloc[0] != result.iloc[1]

    def test_missing_field_uses_available_fields_with_warning(self, caplog):
        import logging
        df = pd.DataFrame({
            "server_name": ["srv1"],
            "file_path": ["/a/b.log"],
        })
        with caplog.at_level(logging.WARNING, logger="src.data.label_loader"):
            result = compute_pk_event(df, ["server_name", "agent_ip", "file_path", "file_name", "file_created_at"])
        assert len(result) == 1
        assert len(result.iloc[0]) == 64
        assert "agent_ip" in caplog.text

    def test_all_fields_missing_raises(self):
        df = pd.DataFrame({"other_col": ["x"]})
        with pytest.raises(ValueError, match="pk_event 생성 불가"):
            compute_pk_event(df, ["server_name", "agent_ip"])

    def test_null_values_handled_as_empty_string(self):
        df = pd.DataFrame({
            "server_name": ["srv1", None],
            "file_path": ["/a/b.log", "/a/b.log"],
        })
        result = compute_pk_event(df, ["server_name", "file_path"])
        assert len(result) == 2
        assert result.iloc[0] != result.iloc[1]


# ─────────────────────────────────────────────────────────────────────────────
# LabelLoader — 새 폴더 구조 기반 테스트
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelLoaderFindFiles:
    """find_all_files() — 새 폴더 구조(25년 정탐/오탐) 기반."""

    def test_finds_tp_and_fp_files(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월")
        loader = LabelLoader(label_root=tmp_path)
        files = loader.find_all_files()
        labels = {f[2] for f in files}  # (path, month, label)
        assert labels == {"TP", "FP"}

    def test_finds_multiple_files_per_month(self, tmp_path):
        """한 월에 파일이 여러 개여도 모두 탐색."""
        fp_dir = tmp_path / _FP_FOLDER / "7월"
        fp_dir.mkdir(parents=True)
        _make_label_csv(fp_dir / "오탐 취합 보고 자료_7월_CTO.csv", n_rows=2)
        _make_label_csv(fp_dir / "오탐 취합 보고 자료_7월_NW부문.csv", n_rows=3)
        loader = LabelLoader(label_root=tmp_path)
        files = loader.find_all_files()
        assert len(files) == 2

    def test_empty_root_returns_empty_list(self, tmp_path):
        loader = LabelLoader(label_root=tmp_path / "nonexistent")
        assert loader.find_all_files() == []

    def test_month_extracted_correctly(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "6월")
        loader = LabelLoader(label_root=tmp_path)
        files = loader.find_all_files()
        months = {f[1] for f in files}
        assert "6월" in months

    def test_label_raw_determined_by_folder(self, tmp_path):
        """TP/FP는 폴더명에서 결정 — 파일명 무관."""
        tp_dir = tmp_path / _TP_FOLDER / "3월"
        tp_dir.mkdir(parents=True)
        # 파일명에 "오탐"이 있어도 폴더가 정탐이면 TP
        _make_label_csv(tp_dir / "오탐처럼생긴파일.csv", n_rows=2)
        loader = LabelLoader(label_root=tmp_path)
        files = loader.find_all_files()
        assert all(f[2] == "TP" for f in files)

    def test_returns_three_tuple(self, tmp_path):
        """반환 값: (Path, month_str, label_raw)."""
        _make_tp_fp_structure(tmp_path, "3월", tp_rows=1, fp_rows=0)
        loader = LabelLoader(label_root=tmp_path)
        files = loader.find_all_files()
        assert len(files) == 1
        path, month, label = files[0]
        assert isinstance(path, Path)
        assert isinstance(month, str)
        assert label in ("TP", "FP")


class TestLabelLoaderLoad:
    """load_all() — 새 폴더 구조 로딩 테스트."""

    def test_label_raw_tp_from_tp_folder(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월", tp_rows=3, fp_rows=0)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert (df["label_raw"] == "TP").all()

    def test_label_raw_fp_from_fp_folder(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월", tp_rows=0, fp_rows=4)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert (df["label_raw"] == "FP").all()

    def test_mixed_tp_fp_rows(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월", tp_rows=3, fp_rows=2)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert set(df["label_raw"].unique()) == {"TP", "FP"}
        assert len(df) == 5

    def test_label_work_month_meta_column(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "5월")
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "label_work_month" in df.columns
        assert (df["label_work_month"] == "5월").all()

    def test_source_file_meta_column(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월")
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "_source_file" in df.columns

    def test_korean_columns_normalized(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월")
        df = LabelLoader(label_root=tmp_path).load_all()
        for eng_col in ENGLISH_COLS:
            assert eng_col in df.columns, f"'{eng_col}' 변환 실패"
        for kor_col in KOREAN_COLS:
            assert kor_col not in df.columns

    def test_file_size_column_dropped(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월")
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "file_size" not in df.columns

    def test_pk_event_generated(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월")
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "pk_event" in df.columns
        assert df["pk_event"].apply(lambda x: len(x) == 64).all()

    def test_multiple_months_combined(self, tmp_path):
        _make_tp_fp_structure(tmp_path, "3월", tp_rows=2, fp_rows=3)
        _make_tp_fp_structure(tmp_path, "4월", tp_rows=1, fp_rows=2)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert len(df) == 8
        months = set(df["label_work_month"].unique())
        assert "3월" in months and "4월" in months

    def test_multiple_files_per_month_concatenated(self, tmp_path):
        fp_dir = tmp_path / _FP_FOLDER / "7월"
        fp_dir.mkdir(parents=True)
        _make_label_csv(fp_dir / "오탐_CTO.csv", n_rows=3)
        _make_label_csv(fp_dir / "오탐_NW부문.csv", n_rows=4)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert len(df) == 7

    def test_empty_root_returns_empty_dataframe(self, tmp_path):
        df = LabelLoader(label_root=tmp_path / "empty").load_all()
        assert df.empty

    def test_alias_column_normalized(self, tmp_path):
        """alias 표기('서버명')도 server_name으로 정규화."""
        fp_dir = tmp_path / _FP_FOLDER / "3월"
        fp_dir.mkdir(parents=True)
        _make_label_csv(fp_dir / "오탐_CTO.csv", n_rows=2, use_alias=True)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "server_name" in df.columns
        assert "서버명" not in df.columns

    def test_pk_null_rows_removed(self, tmp_path):
        """pk_event 5개 필드 중 하나라도 NaN이면 제거."""
        fp_dir = tmp_path / _FP_FOLDER / "3월"
        fp_dir.mkdir(parents=True)
        df_raw = pd.DataFrame({
            "서버이름": ["srv1", None, "srv3"],   # 2번째 행 server_name 결측
            "에이전트IP": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            "파일경로": ["/a/b", "/c/d", "/e/f"],
            "파일이름": ["a.log", "b.log", "c.log"],
            "패턴개수": [1, 2, 3],
            "주민등록번호개수": [0, 0, 0],
            "핸드폰번호개수": [0, 0, 0],
            "이메일주소개수": [0, 0, 0],
            "파일생성일시": ["2025-03-01"] * 3,
            "파일크기": [None] * 3,
        })
        df_raw.to_csv(fp_dir / "오탐_CTO.csv", index=False, encoding="utf-8-sig")
        df = LabelLoader(label_root=tmp_path).load_all()
        # server_name이 NaN인 행 제거 → 2건만 남아야 함
        assert len(df) == 2
        assert df["server_name"].notna().all()

    def test_file_created_at_parsed_as_datetime(self, tmp_path):
        """file_created_at이 datetime 타입으로 파싱됨."""
        _make_tp_fp_structure(tmp_path, "3월", tp_rows=2, fp_rows=0)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "file_created_at" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["file_created_at"])

    def test_file_created_at_slash_format_parsed(self, tmp_path):
        """2025/03/15 14:23:01 포맷도 파싱됨."""
        tp_dir = tmp_path / _TP_FOLDER / "3월"
        tp_dir.mkdir(parents=True)
        _make_label_csv(
            tp_dir / "정탐_CTO.csv",
            n_rows=2,
            created_at_fmt="2025/03/15 14:23:01",
        )
        df = LabelLoader(label_root=tmp_path).load_all()
        assert pd.api.types.is_datetime64_any_dtype(df["file_created_at"])
        assert df["file_created_at"].notna().all()

    def test_file_created_at_unparseable_rows_dropped(self, tmp_path):
        """파싱 불가 file_created_at → NaT → pk_event 필드 결측으로 행 제거, 오류 없음."""
        tp_dir = tmp_path / _TP_FOLDER / "3월"
        tp_dir.mkdir(parents=True)
        _make_label_csv(
            tp_dir / "정탐_CTO.csv",
            n_rows=2,
            created_at_fmt="not-a-date",
        )
        # 예외 없이 실행되어야 함; file_created_at NaT → pk_event 결측 → 전체 행 제거
        df = LabelLoader(label_root=tmp_path).load_all()
        assert isinstance(df, pd.DataFrame)

    def test_no_organization_column(self, tmp_path):
        """organization 컬럼은 생성하지 않음 (제거됨)."""
        _make_tp_fp_structure(tmp_path, "3월")
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "organization" not in df.columns

    def test_pk_file_column_generated(self, tmp_path):
        """pk_file = SHA256(server_name|agent_ip|file_path|file_name)."""
        _make_tp_fp_structure(tmp_path, "3월", tp_rows=2, fp_rows=0)
        df = LabelLoader(label_root=tmp_path).load_all()
        assert "pk_file" in df.columns
        assert df["pk_file"].apply(lambda x: len(x) == 64).all()


# ─────────────────────────────────────────────────────────────────────────────
# detect_cross_label_duplicates (merger.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectCrossLabelDuplicates:
    def _make_df_with_pk(self, pks, label):
        return pd.DataFrame({
            "pk_event": pks,
            "label_raw": [label] * len(pks),
            "server_name": ["srv"] * len(pks),
        })

    def test_no_duplicates_returns_empty(self):
        combined = pd.concat([
            self._make_df_with_pk(["aaa", "bbb"], "TP"),
            self._make_df_with_pk(["ccc", "ddd"], "FP"),
        ], ignore_index=True)
        assert detect_cross_label_duplicates(combined).empty

    def test_cross_duplicate_detected(self):
        combined = pd.concat([
            self._make_df_with_pk(["aaa111", "bbb"], "TP"),
            self._make_df_with_pk(["aaa111", "ccc"], "FP"),
        ], ignore_index=True)
        result = detect_cross_label_duplicates(combined)
        assert len(result) == 1
        assert result.iloc[0]["pk_event"] == "aaa111"

    def test_multiple_cross_duplicates_all_detected(self):
        combined = pd.concat([
            self._make_df_with_pk(["pk1", "pk2", "pk3"], "TP"),
            self._make_df_with_pk(["pk1", "pk2", "pk_unique"], "FP"),
        ], ignore_index=True)
        result = detect_cross_label_duplicates(combined)
        assert len(result) == 2
        assert set(result["pk_event"]) == {"pk1", "pk2"}

    def test_same_label_duplicates_not_flagged(self):
        df = pd.DataFrame({
            "pk_event": ["aaa", "aaa", "bbb"],
            "label_raw": ["FP", "FP", "TP"],
        })
        assert detect_cross_label_duplicates(df).empty

    def test_result_includes_conflict_count(self):
        common_pk = "dupkey"
        combined = pd.concat([
            self._make_df_with_pk([common_pk, common_pk], "TP"),
            self._make_df_with_pk([common_pk], "FP"),
        ], ignore_index=True)
        result = detect_cross_label_duplicates(combined)
        assert "conflict_count" in result.columns
