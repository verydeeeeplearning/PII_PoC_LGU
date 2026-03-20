"""레이블 Excel 다중 파일 로딩 모듈

실제 폴더 구조 (2026-03-11 확인):
  data/raw/label/
    25년 정탐 (3월~12월)/3월/, 4월/, ..., 12월/  -> label_raw = TP
    25년 오탐 (3월~12월)/3월/, 4월/, ..., 12월/  -> label_raw = FP

label_raw 부여 기준:
  최상위 폴더명에 "정탐" 포함 -> "TP"
  최상위 폴더명에 "오탐" 포함 -> "FP"
  (파일명이나 Excel 내 컬럼 값 기반 아님)

메타 컬럼 자동 추가:
  label_raw:         "TP" 또는 "FP" (폴더 출처 기반)
  label_work_month:  작업 월 (월 서브폴더명, 예: "3월")
  _source_file:      원본 파일명 (추적용)
  pk_event:          SHA256(server_name|agent_ip|file_path|file_name|file_created_at)
  pk_file:           SHA256(server_name|agent_ip|file_path|file_name)

제거된 컬럼:
  organization: 폴더 구조 변경으로 추출 불가, 피처 불필요 (확인됨)
"""
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yaml

from src.data.column_normalizer import ColumnNormalizer
from src.data.loader import load_raw_data
from src.utils.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DEFAULT_INGESTION_CONFIG_PATH = PROJECT_ROOT / "config" / "ingestion_config.yaml"

_DEFAULT_PK_EVENT_FIELDS: List[str] = [
    "server_name", "agent_ip", "file_path", "file_name", "file_created_at",
]
_DEFAULT_PK_FILE_FIELDS: List[str] = [
    "server_name", "agent_ip", "file_path", "file_name",
]

# 멀티시트 유효 시트 판별용 - 아래 중 하나라도 있으면 유효
_SHEET_REQUIRED_COLS_DEFAULT: List[str] = [
    "server_name", "서버이름", "서버명", "서버 이름",
]

# file_created_at 파싱 포맷 목록 (순서대로 시도)
_DATETIME_FORMATS: List[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M",
    "%y.%m.%d %H:%M:%S",
    "%y.%m.%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
]

# ─────────────────────────────────────────────────────────────────────────────
# PK Canonicalization
# ─────────────────────────────────────────────────────────────────────────────

# datetime 필드의 PK 직렬화 포맷 — 초 단위로 절사, 서브초/나노초 제거
# 두 소스의 datetime이 동일한 순간을 가리키더라도 서브초 표현이 다를 수 있으므로
# 정규 포맷으로 통일해야 SHA256 해시가 일치함
_PK_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# 필드별 문자열 정규화 함수 (pandas Series → Series)
# 새 PK 필드가 추가될 때 여기에만 항목을 추가하면 됨
# None: datetime 타입으로 처리 (아래 _canonicalize_pk_series 참고)
_PK_FIELD_NORMALIZERS: dict = {
    # 호스트명은 DNS 관례상 대소문자 무관 → 소문자 통일
    "server_name": lambda s: s.str.strip().str.lower(),
    # IP 주소는 공백만 제거 (IPv4는 대소문자 없음, IPv6는 소문자 권장이나 현재 미사용)
    "agent_ip": lambda s: s.str.strip(),
    # 경로 구분자 통일 (Windows '\' → Linux '/') + 우측 슬래시 제거
    "file_path": lambda s: s.str.strip().str.replace("\\", "/", regex=False).str.rstrip("/"),
    # 파일명은 Linux 서버 기준 대소문자 구분 → 공백 제거만
    "file_name": lambda s: s.str.strip(),
}


def _canonicalize_pk_series(series: pd.Series, field_name: str) -> pd.Series:
    """PK 계산용 필드를 정규화된 문자열 Series로 변환.

    - datetime64 타입 → _PK_DATETIME_FORMAT 으로 strftime (초 단위 절사)
    - 문자열 타입    → _PK_FIELD_NORMALIZERS 에 등록된 규칙 적용
    - 미등록 필드    → 공백 strip 만 수행

    결측/NaT 값은 항상 빈 문자열 "" 로 처리.

    새로운 PK 구성 필드가 생기면 _PK_FIELD_NORMALIZERS 에 항목을 추가하기만 하면 됨.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        # dt.strftime은 NaT → NaN(str) 반환 → fillna로 ""로 처리
        return series.dt.strftime(_PK_DATETIME_FORMAT).fillna("")

    str_series = series.fillna("").astype(str)
    normalizer = _PK_FIELD_NORMALIZERS.get(field_name)
    if normalizer is not None:
        return normalizer(str_series)
    return str_series.str.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public utility functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_pk_event(df: pd.DataFrame, pk_fields: List[str]) -> pd.Series:
    """SHA256(canonical(f1)|canonical(f2)|...) 기반 PK Series 생성.

    각 필드를 _canonicalize_pk_series()로 정규화한 뒤 해싱:
      - datetime64 컬럼 : "%Y-%m-%d %H:%M:%S" (초 단위 절사, 서브초 무시)
      - server_name     : strip + lowercase
      - agent_ip        : strip
      - file_path       : strip + 역슬래시→슬래시 + 우측 슬래시 제거
      - file_name       : strip
      - 기타 문자열    : strip

    소스 포맷이 달라도 (예: "2025/01/15 10:30" vs "2025-01-15 10:30:00")
    두 소스 모두 동일 Timestamp로 파싱된 뒤 이 함수에서 동일 문자열로 직렬화됨.

    결측값은 빈 문자열로 처리.
    일부 필드가 DataFrame에 없으면 가용 필드만 사용하고 경고.
    """
    available = [f for f in pk_fields if f in df.columns]
    missing = [f for f in pk_fields if f not in df.columns]

    if missing:
        logger.warning(
            "pk 필드 결측 - 가용 필드만 사용: missing=%s, available=%s",
            missing, available,
        )

    if not available:
        raise ValueError(
            f"pk 생성 불가: 지정된 필드가 DataFrame에 없음. 설정된 필드: {pk_fields}"
        )

    canonicalized = pd.DataFrame(
        {f: _canonicalize_pk_series(df[f], f) for f in available}
    )
    combined = canonicalized.apply(lambda row: "|".join(row), axis=1)
    return combined.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_ingestion_config(config_path: Optional[Path]) -> dict:
    path = config_path or _DEFAULT_INGESTION_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_datetime_series(series: pd.Series, formats: list = None) -> pd.Series:
    """여러 포맷을 순차 시도하는 datetime 파싱.

    pd.to_datetime()로 먼저 시도하고,
    실패한 값에 대해 명시적 포맷 목록을 순차 시도.
    최종 실패 -> NaT (오류 없음).
    """
    if formats is None:
        formats = _DATETIME_FORMATS

    # 1차: 자동 추론 (pandas 2.x: infer_datetime_format 제거됨)
    result = pd.to_datetime(series, errors="coerce")

    # NaT인 값에 대해 명시적 포맷 순차 시도
    for fmt in formats:
        still_nat = result.isna() & series.notna()
        if not still_nat.any():
            break
        parsed = pd.to_datetime(series[still_nat], format=fmt, errors="coerce")
        result.update(parsed)

    nat_count = result.isna().sum()
    if nat_count > 0:
        logger.warning("file_created_at 파싱 실패: %d건 -> NaT", nat_count)

    return result


def _is_valid_sheet(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """레이블 데이터 스키마(서버이름/server_name)를 가진 시트인지 확인.

    컬럼명 앞뒤 공백/탭을 strip한 후 비교하여 minor 표기 차이에 robust하게 대응.
    """
    stripped = {str(c).strip() for c in df.columns}
    return any(col in stripped for col in required_cols)


# ─────────────────────────────────────────────────────────────────────────────
# LabelLoader
# ─────────────────────────────────────────────────────────────────────────────

class LabelLoader:
    """정탐/오탐 월별 레이블 파일 다중 로딩기.

    폴더 구조:
      {label_root}/
        25년 정탐 (3월~12월)/3월/, ..., 12월/  -> label_raw = TP
        25년 오탐 (3월~12월)/3월/, ..., 12월/  -> label_raw = FP

    사용 예시:
        loader = LabelLoader()
        df = loader.load_all()           # 전 월 합산
        files = loader.find_all_files()  # 파일 구조 확인 (로딩 없이)
    """

    def __init__(
        self,
        label_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        normalizer: Optional[ColumnNormalizer] = None,
    ):
        cfg = _load_ingestion_config(config_path)
        label_cfg = cfg.get("label_data", {})

        self._label_root: Path = label_root or (
            PROJECT_ROOT / label_cfg.get("root", "data/raw/label")
        )
        self._tp_keyword: str = label_cfg.get("tp_folder_keyword", "정탐")
        self._fp_keyword: str = label_cfg.get("fp_folder_keyword", "오탐")
        self._pk_event_fields: List[str] = label_cfg.get(
            "pk_event_fields", _DEFAULT_PK_EVENT_FIELDS
        )
        self._sheet_required_cols: List[str] = label_cfg.get(
            "sheet_required_cols", _SHEET_REQUIRED_COLS_DEFAULT
        )
        _excel_read_cfg = label_cfg.get("excel_read", {})
        self._header_row: int = _excel_read_cfg.get("header_row", 0)
        self._skip_first_col: bool = _excel_read_cfg.get("skip_first_col", False)
        self._normalizer: ColumnNormalizer = normalizer or ColumnNormalizer()

        # Load preprocessing config for behavior settings
        _prep_cfg_path = PROJECT_ROOT / "config" / "preprocessing_config.yaml"
        _prep_cfg: dict = {}
        if _prep_cfg_path.exists():
            with open(_prep_cfg_path, "r", encoding="utf-8") as _f:
                _prep_cfg = yaml.safe_load(_f) or {}
        _label_loader_cfg = _prep_cfg.get("parsing", {}).get("label_loader", {})

        self._on_no_valid_sheet: str = _label_loader_cfg.get("on_no_valid_sheet", "warn_and_first_fallback")
        self._on_datetime_parse_fail: str = _label_loader_cfg.get("on_datetime_parse_fail", "warn")
        _datetime_formats_from_cfg = _label_loader_cfg.get("datetime_formats", None)
        if _datetime_formats_from_cfg:
            self._datetime_formats = _datetime_formats_from_cfg
        else:
            self._datetime_formats = _DATETIME_FORMATS

    # ── public API ────────────────────────────────────────────────────────────

    def load_all(self) -> pd.DataFrame:
        """전 월 × 정탐/오탐 파일 합산 + pk_event/pk_file 생성.

        Returns:
            합산된 DataFrame (pk_event, pk_file 포함)
        """
        if not self._label_root.exists():
            logger.warning("label_root 없음: %s", self._label_root)
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for root_dir, label_raw in self._find_label_roots():
            dfs.extend(self._load_root_dfs(root_dir, label_raw))

        if not dfs:
            logger.warning("로드된 파일 없음: %s", self._label_root)
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # pk_file (4-field: server_name|agent_ip|file_path|file_name)
        df["pk_file"] = compute_pk_event(df, _DEFAULT_PK_FILE_FIELDS)
        # pk_event (5-field: + file_created_at)
        df["pk_event"] = compute_pk_event(df, self._pk_event_fields)

        logger.info("load_all 완료: %d건", len(df))
        return df

    def find_all_files(self) -> List[Tuple[Path, str, str]]:
        """전체 파일 목록 탐색 (파일 로딩 없이 구조 확인).

        Returns:
            [(file_path, month_name, label_raw), ...]
        """
        result: List[Tuple[Path, str, str]] = []

        if not self._label_root.exists():
            return result

        for root_dir, label_raw in self._find_label_roots():
            for month_dir in sorted(root_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                month = month_dir.name
                for fp in self._find_excel_files(month_dir):
                    result.append((fp, month, label_raw))

        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _find_label_roots(self) -> List[Tuple[Path, str]]:
        """label_root 하위에서 정탐/오탐 폴더를 탐색.

        Returns:
            [(folder_path, "TP"), (folder_path, "FP")]
        """
        roots: List[Tuple[Path, str]] = []
        for d in sorted(self._label_root.iterdir()):
            if not d.is_dir():
                continue
            if self._tp_keyword in d.name:
                roots.append((d, "TP"))
            elif self._fp_keyword in d.name:
                roots.append((d, "FP"))
        return roots

    def _load_root_dfs(self, root_dir: Path, label_raw: str) -> List[pd.DataFrame]:
        dfs: List[pd.DataFrame] = []
        for month_dir in sorted(root_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            dfs.extend(self._load_month_dfs(month_dir, label_raw, month_dir.name))
        return dfs

    def _load_month_dfs(
        self, month_dir: Path, label_raw: str, month: str
    ) -> List[pd.DataFrame]:
        dfs: List[pd.DataFrame] = []
        for fp in self._find_excel_files(month_dir):
            df = self._load_single_file(fp, label_raw, month)
            if df is not None and not df.empty:
                dfs.append(df)
        return dfs

    def _find_excel_files(self, directory: Path) -> List[Path]:
        """xlsx -> xls -> csv 순으로 파일 탐색."""
        result: List[Path] = []
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            result.extend(sorted(directory.glob(ext)))
        return result

    def _load_single_file(
        self, file_path: Path, label_raw: str, month: str
    ) -> Optional[pd.DataFrame]:
        """단일 파일 로딩 - 멀티시트 지원 + 정규화 + 파싱 + 결측 제거."""
        try:
            sheets = self._read_excel_sheets(file_path)
        except Exception as exc:
            logger.error("파일 로드 실패 (skip): %s - %s", file_path, exc)
            return None

        if not sheets:
            logger.warning("유효 시트 없음 (skip): %s", file_path)
            return None

        dfs: List[pd.DataFrame] = []
        for sheet_df in sheets:
            sheet_df = self._normalizer.normalize(sheet_df)
            sheet_df = self._parse_file_created_at(sheet_df)
            sheet_df = self._drop_pk_null_rows(sheet_df)

            if sheet_df.empty:
                continue

            sheet_df["label_raw"] = label_raw
            sheet_df["label_work_month"] = month
            sheet_df["_source_file"] = file_path.name
            dfs.append(sheet_df)

        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def _read_excel_sheets(self, file_path: Path) -> List[pd.DataFrame]:
        """Excel 파일의 유효 시트만 반환.

        유효 시트: sheet_required_cols 중 하나라도 포함.
        CSV 파일: 단일 DataFrame으로 처리 (header_row / skip_first_col 미적용).

        Excel 읽기 옵션 (ingestion_config.yaml excel_read 섹션):
          header_row     : 헤더 행 위치 (0-indexed). 1 이면 첫 행을 타이틀로 건너뜀.
          skip_first_col : True 이면 읽은 후 첫 번째 컬럼(연번 등) 제거.
        """
        ext = file_path.suffix.lower()

        if ext == ".csv":
            df = load_raw_data(str(file_path))
            return [df] if _is_valid_sheet(df, self._sheet_required_cols) else []

        # Excel - header_row 적용하여 모든 시트 로드
        try:
            all_sheets: dict = pd.read_excel(
                file_path, sheet_name=None, engine="openpyxl",
                header=self._header_row,
            )
        except Exception as exc:
            logger.error("Excel 멀티시트 로드 실패: %s - %s", file_path, exc)
            raise

        def _apply_skip(df: pd.DataFrame) -> pd.DataFrame:
            """skip_first_col 설정에 따라 첫 번째 컬럼 제거."""
            if self._skip_first_col and len(df.columns) > 0:
                return df.iloc[:, 1:]
            return df

        valid: List[pd.DataFrame] = []
        for sheet_name, df in all_sheets.items():
            df = _apply_skip(df)
            if _is_valid_sheet(df, self._sheet_required_cols):
                valid.append(df)
            else:
                logger.debug(
                    "시트 skip (스키마 불일치): %s / %s", file_path.name, sheet_name
                )

        if not valid and all_sheets:
            if self._on_no_valid_sheet == "raise":
                raise ValueError(f"유효 시트 없음 (on_no_valid_sheet=raise): {file_path}")
            elif self._on_no_valid_sheet == "skip":
                logger.warning("유효 시트 없음 (skip): %s", file_path.name)
                return []
            else:  # "warn_and_first_fallback" (default)
                first_name = next(iter(all_sheets))
                logger.warning(
                    "유효 시트 없음 - 첫 번째 시트 fallback 사용: %s / %s",
                    file_path.name, first_name,
                )
                valid.append(_apply_skip(all_sheets[first_name]))

        return valid

    def _parse_file_created_at(self, df: pd.DataFrame) -> pd.DataFrame:
        """file_created_at 컬럼을 datetime으로 파싱 (이미 datetime이면 skip)."""
        if "file_created_at" not in df.columns:
            return df

        if pd.api.types.is_datetime64_any_dtype(df["file_created_at"]):
            return df

        df = df.copy()
        series = df["file_created_at"].astype(str).replace("nan", pd.NA)
        parsed = _parse_datetime_series(series, formats=self._datetime_formats)

        if self._on_datetime_parse_fail == "quarantine":
            fail_mask = parsed.isna() & df["file_created_at"].notna()
            if fail_mask.any():
                n_fail = int(fail_mask.sum())
                logger.warning(
                    "file_created_at 파싱 실패 %d건 -> quarantine",
                    n_fail,
                )
                _qpath = PROJECT_ROOT / "data" / "processed" / "silver_quarantine.parquet"
                _qpath.parent.mkdir(parents=True, exist_ok=True)
                df_q = df[fail_mask].copy()
                df_q["quarantine_reason"] = "datetime_parse_fail"
                if _qpath.exists():
                    try:
                        existing = pd.read_parquet(_qpath)
                        df_q = pd.concat([existing, df_q], ignore_index=True)
                    except Exception:
                        pass
                df_q.to_parquet(_qpath, index=False, engine="pyarrow")
                df = df[~fail_mask].reset_index(drop=True)
                parsed = parsed[~fail_mask].reset_index(drop=True)

        df["file_created_at"] = parsed
        return df

    def _drop_pk_null_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """pk_event 5개 필드 중 하나라도 NaN이면 행 제거 (EDA §2-4 규칙3)."""
        present_fields = [f for f in self._pk_event_fields if f in df.columns]
        if not present_fields:
            return df

        before = len(df)
        df = df.dropna(subset=present_fields).reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            logger.info("pk_event 필드 결측 행 제거: %d건", removed)
        return df
