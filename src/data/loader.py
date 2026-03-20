"""데이터 로딩 모듈

회의록 2026-01 반영:
- 데이터셋 A: Server-i 검출 원본
- 데이터셋 B: 소만사 오탐 레이블링 (✅ 활용 확정)
- 데이터셋 C: 현업 피드백 회신 (❓ 활용 미확정)
"""
import os
import glob
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from src.utils.constants import PROJECT_ROOT, ENCODING, load_yaml


def _get_csv_encoding_candidates(first_encoding: str = "utf-8") -> list:
    """preprocessing_config.yaml의 csv_encoding_candidates 로드. 실패 시 기본값."""
    try:
        import yaml as _yaml
        _cfg_path = PROJECT_ROOT / "config" / "preprocessing_config.yaml"
        if _cfg_path.exists():
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f) or {}
            candidates = _cfg.get("parsing", {}).get("loader", {}).get("csv_encoding_candidates", [])
            if candidates:
                # Ensure first_encoding is first
                result = [first_encoding] + [c for c in candidates if c != first_encoding]
                return result
    except Exception:
        pass
    return [first_encoding, "utf-8-sig", "cp949"]


def _find_file_any_format(base_path: Path, stem_pattern: str = "*") -> Optional[Path]:
    """xlsx -> xls -> csv -> parquet 순으로 파일 탐색

    Args:
        base_path: 탐색할 디렉토리
        stem_pattern: 파일명 패턴 (확장자 제외)

    Returns:
        발견된 첫 번째 파일 경로 또는 None
    """
    for ext in (".xlsx", ".xls", ".csv", ".parquet"):
        matches = list(base_path.glob(f"{stem_pattern}{ext}"))
        if matches:
            return matches[0]
    return None


def load_raw_data(
    file_path: str,
    encoding: str = "utf-8",
    sheet_name: Optional[Union[str, int]] = None,
) -> pd.DataFrame:
    """원본 데이터 파일을 로드합니다.

    지원 형식: .xlsx, .xls, .csv, .parquet
    Linux UTF-8 환경 기준. CSV는 utf-8 -> utf-8-sig 순으로 시도.

    Args:
        file_path: 데이터 파일 경로
        encoding: CSV 파일 인코딩 (기본 utf-8)
        sheet_name: Excel 파일의 시트명 또는 인덱스 (None이면 첫 번째 시트)

    Returns:
        로드된 DataFrame
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext in (".xlsx", ".xls"):
            # openpyxl 명시: air-gapped 환경에서 engine 누락 오류 방지
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        elif ext == ".csv":
            # 인코딩 fallback: 지정값 -> utf-8-sig (Windows BOM) -> cp949 (한글 Windows)
            _enc_candidates = _get_csv_encoding_candidates(encoding)
            _seen = set()
            df = None
            for _enc in _enc_candidates:
                if _enc in _seen:
                    continue
                _seen.add(_enc)
                try:
                    df = pd.read_csv(file_path, encoding=_enc, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise UnicodeDecodeError(
                    "utf-8", b"", 0, 1,
                    f"모든 인코딩 시도 실패: {_enc_candidates}"
                )
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}  (지원: xlsx, xls, csv, parquet)")
    except Exception as exc:
        print(f"  [로드 실패] {path.name}: {exc}")
        raise

    print(f"[로드 완료] {path.name}")
    print(f"  행: {len(df):,}  |  열: {len(df.columns)}")

    return df


def load_label_data(
    file_path: str,
    encoding: str = "utf-8",
    label_column: str = "label",
) -> pd.DataFrame:
    """
    레이블(정탐/오탐 분류) 데이터를 로드하고 분포를 출력합니다.

    Args:
        file_path: 레이블 파일 경로
        encoding: 파일 인코딩
        label_column: 레이블 컬럼명

    Returns:
        레이블 DataFrame
    """
    df = load_raw_data(file_path, encoding=encoding)

    if label_column in df.columns:
        print("\n[레이블 분포]")
        print(df[label_column].value_counts())

    return df


def load_config() -> Dict[str, Any]:
    """feature_config.yaml에서 데이터셋 설정을 로드합니다."""
    return load_yaml(PROJECT_ROOT / "config" / "feature_config.yaml")


class DatasetLoader:
    """데이터셋 A/B/C 통합 로더

    회의록 확정 데이터셋:
    - A: Server-i 검출 원본 (dfile_* 컬럼)
    - B: 소만사 오탐 레이블링 (7개 클래스)
    - C: 현업 피드백 회신 (비정형)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 데이터셋 설정 딕셔너리 (feature_config.yaml의 datasets 섹션)
        """
        if config is None:
            full_config = load_config()
            config = full_config.get("datasets", {})

        self.config = config
        self.data_dir = PROJECT_ROOT / "data" / "raw"

    def load_dataset_a(
        self,
        file_path: Optional[str] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        데이터셋 A (Server-i 검출 원본)를 로드합니다.

        Args:
            file_path: 파일 경로 (없으면 설정에서 검색)
            normalize: 컬럼명 정규화 여부

        Returns:
            DataFrame
        """
        print("\n" + "=" * 60)
        print("[데이터셋 A] Server-i 검출 원본")
        print("=" * 60)

        ds_config = self.config.get("dataset_a", {})

        if file_path is None:
            base_path = PROJECT_ROOT / ds_config.get("path", "data/raw/dataset_a/")
            # xlsx -> xls -> csv -> parquet 순으로 탐색 (실운영 파일은 xlsx)
            found = None
            for ext in (".xlsx", ".xls", ".csv", ".parquet"):
                matches = sorted(base_path.glob(f"*{ext}"))
                if matches:
                    found = matches
                    break
            if not found:
                print(f"  [경고] 파일 없음: {base_path}/  (xlsx/xls/csv/parquet 탐색)")
                return pd.DataFrame()

            file_path = str(found[0])
            if len(found) > 1:
                print(f"  [참고] {len(found)}개 파일 중 첫 번째 로드: {found[0].name}")

        df = load_raw_data(
            file_path,
            encoding=ds_config.get("encoding", "utf-8"),
        )

        if normalize:
            df = self._normalize_columns(df, ds_config.get("columns_mapping", {}))

        print(f"  컬럼: {list(df.columns)}")
        return df

    def load_dataset_b(
        self,
        file_path: Optional[str] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        데이터셋 B (소만사 오탐 레이블링)를 로드합니다.

        ✅ 활용 확정: 7개 오탐 클래스가 정의된 학습용 데이터

        Args:
            file_path: 파일 경로 (없으면 설정에서 검색)
            normalize: 컬럼명 정규화 여부

        Returns:
            DataFrame
        """
        print("\n" + "=" * 60)
        print("[데이터셋 B] 소만사 오탐 레이블링 (학습용)")
        print("=" * 60)

        ds_config = self.config.get("dataset_b", {})

        if file_path is None:
            base_path = PROJECT_ROOT / ds_config.get("path", "data/raw/dataset_b/")
            pattern = ds_config.get("file_pattern", "*.xlsx")
            files = list(base_path.glob(pattern))

            if not files:
                print(f"  [경고] 파일 없음: {base_path / pattern}")
                return pd.DataFrame()

            file_path = str(files[0])
            if len(files) > 1:
                print(f"  [참고] {len(files)}개 파일 중 첫 번째 로드: {files[0].name}")

        df = load_raw_data(
            file_path,
            encoding=ds_config.get("encoding", "utf-8"),
        )

        if normalize:
            df = self._normalize_columns(df, ds_config.get("columns_mapping", {}))

        # 레이블 분포 출력
        label_col = ds_config.get("columns_mapping", {}).get("label", "label")
        if label_col in df.columns or "label" in df.columns:
            actual_col = label_col if label_col in df.columns else "label"
            print(f"\n[레이블 분포 - {actual_col}]")
            print(df[actual_col].value_counts())

        print(f"  컬럼: {list(df.columns)}")
        return df

    def load_dataset_c(
        self,
        file_path: Optional[str] = None,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        데이터셋 C (현업 피드백 회신)를 로드합니다.

        ❓ 활용 미확정: 비정형 자유 텍스트, 그루핑 필요

        Args:
            file_path: 파일 경로 (없으면 설정에서 검색)
            normalize: 컬럼명 정규화 여부

        Returns:
            DataFrame
        """
        print("\n" + "=" * 60)
        print("[데이터셋 C] 현업 피드백 회신 (활용 미확정)")
        print("=" * 60)

        ds_config = self.config.get("dataset_c", {})

        if not ds_config.get("use_for_training", False):
            print("  [참고] 이 데이터셋은 현재 학습에 사용하지 않도록 설정됨")

        if file_path is None:
            base_path = PROJECT_ROOT / ds_config.get("path", "data/raw/dataset_c/")
            pattern = ds_config.get("file_pattern", "*.xlsx")
            files = list(base_path.glob(pattern))

            if not files:
                print(f"  [경고] 파일 없음: {base_path / pattern}")
                return pd.DataFrame()

            file_path = str(files[0])
            if len(files) > 1:
                print(f"  [참고] {len(files)}개 파일 발견")
                for f in files[:5]:
                    print(f"         - {f.name}")
                if len(files) > 5:
                    print(f"         ... 외 {len(files) - 5}개")

        df = load_raw_data(
            file_path,
            encoding=ds_config.get("encoding", "utf-8"),
        )

        if normalize:
            df = self._normalize_columns(df, ds_config.get("columns_mapping", {}))

        print(f"  컬럼: {list(df.columns)}")
        return df

    def load_multiple_files(
        self,
        directory: str,
        pattern: str = "*.xlsx",
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """디렉토리 내 여러 파일을 로드하여 병합합니다.

        기본 패턴: *.xlsx (실운영 파일 형식)
        패턴 없이 모든 지원 포맷 탐색 시: pattern="*"

        Args:
            directory: 디렉토리 경로
            pattern: 파일 패턴 (glob, 기본 *.xlsx)
            encoding: CSV 파일 인코딩

        Returns:
            병합된 DataFrame
        """
        dir_path = Path(directory)

        # 지정 패턴으로 먼저 탐색, 없으면 xlsx -> csv 순으로 재탐색
        files = sorted(dir_path.glob(pattern))
        if not files and pattern == "*.xlsx":
            files = sorted(dir_path.glob("*.xls")) or sorted(dir_path.glob("*.csv"))

        if not files:
            print(f"[경고] 파일 없음: {dir_path / pattern}")
            return pd.DataFrame()

        print(f"\n[다중 파일 로드] {len(files)}개 파일")

        dfs = []
        for f in files:
            try:
                df = load_raw_data(str(f), encoding=encoding)
                df["_source_file"] = f.name
                dfs.append(df)
            except Exception as e:
                print(f"  [에러] {f.name}: {e}")

        if not dfs:
            return pd.DataFrame()

        merged = pd.concat(dfs, ignore_index=True)
        print(f"\n[병합 완료] 총 {len(merged):,}건")

        return merged

    def _normalize_columns(
        self,
        df: pd.DataFrame,
        mapping: Dict[str, str],
    ) -> pd.DataFrame:
        """컬럼명을 정규화합니다."""
        if not mapping:
            return df

        # 역매핑: {원본컬럼명: 표준컬럼명}
        reverse_mapping = {v: k for k, v in mapping.items() if v in df.columns}

        if reverse_mapping:
            df = df.rename(columns=reverse_mapping)
            print(f"  [컬럼 정규화] {len(reverse_mapping)}개 매핑")

        return df

    def get_dataset_info(self) -> Dict[str, Any]:
        """데이터셋 설정 정보를 반환합니다."""
        info = {}
        for name in ["dataset_a", "dataset_b", "dataset_c"]:
            ds_config = self.config.get(name, {})
            base_path = PROJECT_ROOT / ds_config.get("path", f"data/raw/{name}/")
            pattern = ds_config.get("file_pattern", "*.*")
            files = list(base_path.glob(pattern)) if base_path.exists() else []

            info[name] = {
                "name": ds_config.get("name", name),
                "path": str(base_path),
                "file_count": len(files),
                "files": [f.name for f in files[:5]],
                "use_for_training": ds_config.get("use_for_training", True),
            }

        return info


def create_dataset_directories() -> None:
    """데이터셋 디렉토리 구조를 생성합니다."""
    dirs = [
        PROJECT_ROOT / "data" / "raw" / "dataset_a",
        PROJECT_ROOT / "data" / "raw" / "dataset_b",
        PROJECT_ROOT / "data" / "raw" / "dataset_c",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "merged",
        PROJECT_ROOT / "data" / "features",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [디렉토리] {d}")

    print("\n데이터셋 디렉토리 구조 생성 완료")
    print("\n사용법:")
    print("  - dataset_a/: Server-i 검출 원본 (12월_거래주_검출내역.csv)")
    print("  - dataset_b/: 소만사 오탐 레이블링 (오탐case정리_260121.xlsx)")
    print("  - dataset_c/: 현업 피드백 (오탐_취합_보고_자료_*.xlsx)")
