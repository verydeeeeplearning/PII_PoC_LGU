"""데이터 파이프라인 실행 스크립트 (raw -> processed)

회의록 2026-01 반영:
- 데이터셋 A/B/C 통합 로드
- 복합 PK 기반 병합
- 마스킹 검증 & 패턴 타입 검증

사용법:
    python scripts/run_data_pipeline.py [--dataset a|b|c|all] [--validate-only]
    python scripts/run_data_pipeline.py --source label [--validate-only]

입력:
    data/raw/dataset_a/  (Server-i 검출 원본)
    data/raw/dataset_b/  (소만사 오탐 레이블링)
    data/raw/dataset_c/  (현업 피드백 회신)
    data/raw/label/      (조직별 정탐/오탐 레이블 파일 - --source label 모드)

출력:
    data/processed/merged_cleaned.csv
    outputs/data_validation_report.txt
    data/processed/silver_label.parquet  (--source label 모드)
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Make imports work even if the script is executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils.common import set_seed, ensure_dirs
from src.utils.constants import (
    RANDOM_SEED,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MERGED_CLEANED_FILE,
    ENCODING,
    TEXT_COLUMN,
    LABEL_COLUMN,
    PK_COLUMNS,
    REPORT_DIR,
)
from src.data.loader import DatasetLoader, load_config, load_raw_data
from src.data.merger import (
    merge_detection_with_labels,
    normalize_columns,
    get_available_pk,
    create_composite_pk,
)
from src.data.validator import (
    validate_data,
    validate_masking,
    validate_pattern_type,
    full_validation,
)
from src.data.preprocessor import preprocess_dataframe, save_processed


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="데이터 파이프라인 실행")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["a", "b", "c", "all"],
        help="로드할 데이터셋 (기본: all)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="검증만 수행 (저장 안 함)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="상세 검증 건너뛰기",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="detection",
        choices=["detection", "label", "joined"],
        help=(
            "데이터 소스 모드\n"
            "  detection : Sumologic xlsx -> pk_file 계산 -> silver_detections.parquet\n"
            "  label     : 레이블 다중 Excel -> silver_label.parquet\n"
            "  joined    : silver_label + silver_detections JOIN -> silver_joined.parquet"
        ),
    )
    parser.add_argument(
        "--datasource",
        type=str,
        default="sumologic_server_i",
        help="데이터 소스 이름 (preprocessing_config.yaml data_sources 섹션 키)",
    )
    return parser.parse_args()


def save_validation_report(report: dict, path: Path) -> None:
    """검증 리포트를 파일로 저장"""
    def _pick(obj, *keys, default=None):
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                return obj[key]
            if hasattr(obj, key):
                return getattr(obj, key)
        return default

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("데이터 검증 리포트\n")
        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        # 기본 검증
        if "basic" in report:
            basic = report["basic"]
            n_rows = _pick(basic, "n_rows", "total_rows", default=0)
            n_missing = _pick(basic, "missing_columns", default={})
            n_dup = _pick(basic, "n_duplicates", default=0)
            label_dist = _pick(basic, "label_distribution", default={})
            f.write("[기본 검증]\n")
            f.write(f"  총 행 수: {n_rows:,}\n")
            f.write(f"  결측 컬럼 수: {len(n_missing)}\n")
            f.write(f"  중복 행 수: {n_dup:,}\n")
            f.write(f"  레이블 분포:\n")
            for label, count in label_dist.items():
                f.write(f"    - {label}: {count:,}\n")
            f.write("\n")

        # 마스킹 검증
        if "masking" in report:
            mask = report["masking"]
            total_rows = _pick(mask, "total_rows", default=0)
            masked_rows = _pick(mask, "masked_rows", default=0)
            masking_rate = _pick(mask, "masking_rate", default=0.0)
            ctx_stats = _pick(mask, "context_length_stats", default={}) or {}
            exposed_phone = _pick(mask, "exposed_phone", default=0)
            exposed_email = _pick(mask, "exposed_email", default=0)
            exposed_jumin = _pick(mask, "exposed_jumin", default=0)
            f.write("[마스킹 검증]\n")
            f.write(f"  검사 샘플: {total_rows:,}\n")
            f.write(f"  마스킹 적용 건수: {masked_rows:,}\n")
            f.write(f"  마스킹 비율: {masking_rate:.2%}\n")
            f.write(f"  평균 컨텍스트 길이: {ctx_stats.get('mean', 0):.1f}\n")
            f.write(f"  최대 컨텍스트 길이: {ctx_stats.get('max', 0):.0f}\n")
            f.write("  PII 노출 건수:\n")
            f.write(f"    - 휴대폰: {exposed_phone:,}\n")
            f.write(f"    - 이메일: {exposed_email:,}\n")
            f.write(f"    - 주민번호: {exposed_jumin:,}\n")
            f.write("\n")

        # 패턴 타입 검증
        if "pattern_type" in report:
            pt = report["pattern_type"]
            total_rows = _pick(pt, "total_rows", default=0)
            mismatch_rate = _pick(pt, "mismatch_rate", default=0.0)
            mismatch_details = _pick(pt, "mismatch_details", "mismatch_types", default={})
            f.write("[패턴 타입 검증]\n")
            f.write(f"  검사 샘플: {total_rows:,}\n")
            f.write(f"  불일치 비율: {mismatch_rate:.2%}\n")
            f.write(f"  불일치 유형:\n")
            for mtype, count in mismatch_details.items():
                f.write(f"    - {mtype}: {count:,}\n")
            f.write("\n")

    print(f"[저장] 검증 리포트: {path}")


def _load_preprocessing_config() -> dict:
    """preprocessing_config.yaml 로드. 실패 시 빈 dict 반환."""
    config_path = PROJECT_ROOT / "config" / "preprocessing_config.yaml"
    if not config_path.exists():
        return {}
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_label_mode(args) -> None:
    """--source label 모드: LabelLoader -> silver_label.parquet"""
    from src.data.label_loader import LabelLoader

    print("=" * 60)
    print("[레이블 소스 모드] data/raw/label/ -> silver_label.parquet")
    print("=" * 60)

    ensure_dirs(PROCESSED_DATA_DIR)

    # --dataset 플래그는 label 모드에서 무시됨
    if hasattr(args, "dataset") and args.dataset != "all":
        print(f"  [정보] --dataset '{args.dataset}' 옵션은 label 모드에서 무시됩니다.")

    loader = LabelLoader()
    df = loader.load_all()

    if df.empty:
        print("\n[경고] 로드된 레이블 데이터가 없습니다.")
        print("data/raw/label/ 디렉토리 구조를 확인해주세요.")
        return

    # 통계 출력
    print(f"\n  총 행 수: {len(df):,}")

    if "organization" in df.columns:
        print("\n  [조직별 분포]")
        for org, cnt in df["organization"].value_counts().items():
            print(f"    {org}: {cnt:,}")

    if "label_work_month" in df.columns:
        print("\n  [월별 분포]")
        for month, cnt in df["label_work_month"].value_counts().items():
            print(f"    {month}: {cnt:,}")

    if "label_raw" in df.columns:
        print("\n  [TP/FP 분포]")
        for lbl, cnt in df["label_raw"].value_counts().items():
            print(f"    {lbl}: {cnt:,}")

    # --validate-only: 통계만 출력하고 저장 건너뜀
    if args.validate_only:
        print("\n[검증 전용 모드] 데이터 저장을 건너뜁니다.")
        return

    # silver_label.parquet 저장
    output_path = PROCESSED_DATA_DIR / "silver_label.parquet"
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\n[저장] silver_label.parquet: {len(df):,}건")
    print(f"  경로: {output_path}")


def _diagnose_join_mismatch(df_label: pd.DataFrame, df_det: pd.DataFrame) -> None:
    """JOIN 결과 0건 시 pk_file 불일치 원인 진단."""
    print("\n  [진단] pk_file 불일치 원인 분석:")

    # pk_file 길이 검증
    if "pk_file" in df_label.columns and not df_label["pk_file"].empty:
        sample_label_pk = df_label["pk_file"].dropna().iloc[0] if not df_label["pk_file"].dropna().empty else ""
        label_pk_len = len(sample_label_pk)
        print(f"    silver_label  pk_file 길이: {label_pk_len}  (기댓값: 64, SHA256)")
        if label_pk_len != 64:
            print(f"    [경고] silver_label pk_file 길이가 64(SHA256)가 아님 - MD5/구형 포맷 의심")

    if "pk_file" in df_det.columns and not df_det["pk_file"].empty:
        sample_det_pk = df_det["pk_file"].dropna().iloc[0] if not df_det["pk_file"].dropna().empty else ""
        det_pk_len = len(sample_det_pk)
        print(f"    silver_detections pk_file 길이: {det_pk_len}  (기댓값: 64, SHA256)")
        if det_pk_len != 64:
            print(f"    [경고] silver_detections pk_file 길이가 64(SHA256)가 아님")

    # pk_file 샘플 비교
    if "pk_file" in df_label.columns:
        label_samples = df_label["pk_file"].dropna().head(3).tolist()
        print(f"    silver_label pk_file 샘플: {label_samples}")
    if "pk_file" in df_det.columns:
        det_samples = df_det["pk_file"].dropna().head(3).tolist()
        print(f"    silver_detections pk_file 샘플: {det_samples}")

    print("    -> 두 파이프라인을 동일한 원본 데이터로 재실행하거나")
    print("      generate_mock_raw_data.py로 mock 데이터를 재생성하세요.")


def _run_detection_mode(args) -> None:
    """--source detection: Sumologic xlsx -> pk_file/pk_event 계산 -> silver_detections.parquet"""
    from src.data.label_loader import compute_pk_event as _compute_pk, _parse_datetime_series
    from src.data.datasource_registry import DataSourceRegistry

    datasource = getattr(args, "datasource", "sumologic_server_i")
    registry = DataSourceRegistry()

    print("=" * 60)
    print("[검출 소스 모드] data/raw/dataset_a/ -> silver_detections.parquet")
    print("=" * 60)

    ensure_dirs(PROCESSED_DATA_DIR)

    # 파일 탐색 (registry 패턴 사용, fallback: dataset_a dir)
    try:
        files = registry.find_files(datasource)
        if not files:
            dataset_a_dir = RAW_DATA_DIR / "dataset_a"
            files = sorted(dataset_a_dir.rglob("*.xlsx"))
            if not files:
                files = sorted(dataset_a_dir.rglob("*.csv"))
    except KeyError:
        dataset_a_dir = RAW_DATA_DIR / "dataset_a"
        files = sorted(dataset_a_dir.rglob("*.xlsx"))
        if not files:
            files = sorted(dataset_a_dir.rglob("*.csv"))

    if not files:
        dataset_a_dir = RAW_DATA_DIR / "dataset_a"
        print(f"  [오류] 파일 없음: {dataset_a_dir}/  (xlsx/csv 탐색)")
        return

    column_map = registry.get_column_map(datasource)
    pk_file_fields = registry.get_pk_fields(datasource)
    if not pk_file_fields:
        pk_file_fields = ["server_name", "agent_ip", "file_path", "file_name"]

    print(f"  파일 {len(files)}개 로드 중...")
    dfs = []
    for f in files:
        try:
            ext = f.suffix.lower()
            if ext in (".xlsx", ".xls"):
                df_f = pd.read_excel(str(f), sheet_name=0, engine="openpyxl")
            else:
                df_f = pd.read_csv(str(f), encoding="utf-8", low_memory=False)
            df_f["_source_file"] = f.name
            print(f"  [로드] {f.name}: {len(df_f):,}행 x {len(df_f.columns)}열")
            dfs.append(df_f)
        except Exception as e:
            print(f"  [경고] {f.name} 로드 실패: {e}")

    if not dfs:
        print("  [오류] 로드된 파일 없음")
        return

    df = pd.concat(dfs, ignore_index=True)

    # dfile_* -> 표준 컬럼명 변환 (있는 컬럼만)
    rename = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename)
    print(f"\n  총 행 수: {len(df):,}")
    print(f"  컬럼 매핑: {len(rename)}개")

    # file_created_at datetime 파싱 — label 파이프라인과 동일한 로직으로 정규화.
    # Excel이 이미 datetime으로 읽은 경우 재파싱 생략 (label_loader._parse_file_created_at과 동일한 guard).
    if "file_created_at" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["file_created_at"]):
            df["file_created_at"] = _parse_datetime_series(df["file_created_at"].astype(str))
        print("  file_created_at 파싱 완료")

    # pk_file 계산 (SHA256 4-field: server_name|agent_ip|file_path|file_name)
    try:
        df["pk_file"] = _compute_pk(df, pk_file_fields)
        print("  pk_file 계산 완료 (SHA256 4-field)")
    except Exception as e:
        print(f"  [경고] pk_file 계산 실패: {e}")

    # pk_event 계산 (SHA256 5-field: + file_created_at)
    pk_event_fields = ["server_name", "agent_ip", "file_path", "file_name", "file_created_at"]
    try:
        df["pk_event"] = _compute_pk(df, pk_event_fields)
        print("  pk_event 계산 완료 (SHA256 5-field)")
    except Exception as e:
        print(f"  [경고] pk_event 계산 실패: {e}")

    if args.validate_only:
        print("\n[검증 전용 모드] 저장 건너뜀")
        return

    # 수치 컬럼 명시적 변환 (object → numeric) — 피처 추출 시 타입 오류 방지
    _NUMERIC_COLS = ["pattern_count", "ssn_count", "phone_count", "email_count", "file_size"]
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    output_path = PROCESSED_DATA_DIR / "silver_detections.parquet"
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\n[저장] silver_detections.parquet: {len(df):,}건")
    print(f"  경로: {output_path}")


def _run_join_mode(args) -> None:
    """--source joined: silver_label + silver_detections -> silver_joined.parquet (pk_file 기준 JOIN)"""
    silver_label_path = PROCESSED_DATA_DIR / "silver_label.parquet"
    silver_detections_path = PROCESSED_DATA_DIR / "silver_detections.parquet"

    print("=" * 60)
    print("[JOIN 모드] silver_label + silver_detections -> silver_joined.parquet")
    print("=" * 60)

    if not silver_label_path.exists():
        print("  [오류] silver_label.parquet 없음 - 먼저 --source label 실행 필요")
        return
    if not silver_detections_path.exists():
        print("  [오류] silver_detections.parquet 없음 - 먼저 --source detection 실행 필요")
        return

    df_label = pd.read_parquet(silver_label_path)
    df_det = pd.read_parquet(silver_detections_path)

    print(f"  silver_label:      {len(df_label):,}행  (pk_event: {'pk_event' in df_label.columns})")
    print(f"  silver_detections: {len(df_det):,}행  (pk_event: {'pk_event' in df_det.columns})")

    if "pk_event" not in df_label.columns or "pk_event" not in df_det.columns:
        print("  [오류] pk_event 컬럼이 한 쪽에 없음 - detection pipeline 재실행 필요")
        return

    cfg = _load_preprocessing_config()
    join_label_cols = cfg.get("join", {}).get("label_cols", [
        "pk_file", "pk_event", "label_raw",
        "organization", "label_work_month",
        "ops_dept", "service",
    ])

    # pk_event 기준 inner join (5-field: server_name|agent_ip|file_path|file_name|file_created_at)
    # df_det에 이미 있는 컬럼(pk_file 등)을 label 측에서 제외해 suffix 충돌 방지
    label_only_cols = ["pk_event"] + [
        c for c in join_label_cols
        if c in df_label.columns and c not in df_det.columns and c != "pk_event"
    ]
    df_joined = df_det.merge(df_label[label_only_cols], on="pk_event", how="inner")

    n_joined = len(df_joined)
    print(f"\n  JOIN 결과: {n_joined:,}건")
    print(f"    label 매핑률: {n_joined / max(len(df_label), 1):.1%}")
    print(f"    detection 커버률: {n_joined / max(len(df_det), 1):.1%}")

    if n_joined == 0:
        on_zero = cfg.get("join", {}).get("on_zero_join", "warn_and_diagnose")
        if on_zero == "warn_and_diagnose":
            _diagnose_join_mismatch(df_label, df_det)
        else:
            print("  [경고] JOIN 결과 0건 - pk_event 매핑 불일치. server_name/agent_ip/file_path/file_name/file_created_at 확인 필요")
        return

    if "label_raw" in df_joined.columns:
        print("\n  [TP/FP 분포]")
        for lbl, cnt in df_joined["label_raw"].value_counts().items():
            print(f"    {lbl}: {cnt:,}")

    if args.validate_only:
        print("\n[검증 전용 모드] 저장 건너뜀")
        return

    ensure_dirs(PROCESSED_DATA_DIR)
    output_path = PROCESSED_DATA_DIR / "silver_joined.parquet"
    df_joined.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\n[저장] silver_joined.parquet: {n_joined:,}건")
    print(f"  경로: {output_path}")


def main() -> None:
    args = parse_args()

    # --source label 모드: 레이블 파일 로딩 -> silver_label.parquet
    if args.source == "label":
        _run_label_mode(args)
        return

    # --source detection 모드: Sumologic xlsx -> pk_file 계산 -> silver_detections.parquet
    if args.source == "detection":
        _run_detection_mode(args)
        return

    # --source joined 모드: silver_label + silver_detections JOIN -> silver_joined.parquet
    if args.source == "joined":
        _run_join_mode(args)
        return

    set_seed(RANDOM_SEED)

    ensure_dirs(RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORT_DIR)

    # 설정 로드
    config = load_config()
    loader = DatasetLoader(config.get("datasets", {}))
    pk_config = config.get("data", {}).get("pk_columns", {})

    print("=" * 60)
    print("[Step 1] 데이터셋 로드")
    print("=" * 60)

    dfs = {}

    # 데이터셋 A (Server-i 검출 원본)
    if args.dataset in ("a", "all"):
        try:
            dfs["a"] = loader.load_dataset_a()
        except Exception as e:
            print(f"  [경고] 데이터셋 A 로드 실패: {e}")

    # 데이터셋 B (소만사 오탐 레이블링) - 학습용
    if args.dataset in ("b", "all"):
        try:
            dfs["b"] = loader.load_dataset_b()
        except Exception as e:
            print(f"  [경고] 데이터셋 B 로드 실패: {e}")

    # 데이터셋 C (현업 피드백) - 참고용
    if args.dataset in ("c", "all"):
        try:
            dfs["c"] = loader.load_dataset_c()
        except Exception as e:
            print(f"  [경고] 데이터셋 C 로드 실패: {e}")

    if not dfs:
        print("\n[오류] 로드된 데이터셋이 없습니다.")
        print("data/raw/dataset_*/에 데이터 파일을 배치해주세요.")
        return

    print(f"\n로드된 데이터셋: {list(dfs.keys())}")

    # Step 2: PK 기반 병합
    print("\n" + "=" * 60)
    print("[Step 2] PK 기반 병합")
    print("=" * 60)

    # 주 데이터셋 결정 (B가 있으면 B, 없으면 A)
    if "b" in dfs and not dfs["b"].empty:
        df_main = dfs["b"]
        main_source = "B (소만사 레이블링)"
    elif "a" in dfs and not dfs["a"].empty:
        df_main = dfs["a"]
        main_source = "A (Server-i 원본)"
    else:
        print("[오류] 유효한 데이터셋이 없습니다.")
        return

    print(f"  주 데이터셋: {main_source}")
    print(f"  행: {len(df_main):,}")

    # 사용 가능한 PK 확인
    available_pk = get_available_pk(df_main, pk_config)
    print(f"  사용 PK: {available_pk}")

    # 복합 PK 생성
    if len(available_pk) > 1:
        df_main = create_composite_pk(df_main, available_pk)
        print(f"  복합 PK 생성 완료")

    # A와 B 병합 (둘 다 있는 경우)
    if "a" in dfs and "b" in dfs and not dfs["a"].empty and not dfs["b"].empty:
        df_a = dfs["a"]
        available_pk_a = get_available_pk(df_a, pk_config)

        if available_pk_a:
            print(f"\n  데이터셋 A와 병합 시도...")
            try:
                df_main = merge_detection_with_labels(
                    df_a, df_main,
                    pk_columns=available_pk_a,
                )
                print(f"  병합 완료: {len(df_main):,}행")
            except Exception as e:
                print(f"  [경고] 병합 실패: {e}")

    # Step 3: 데이터 품질 검증
    print("\n" + "=" * 60)
    print("[Step 3] 데이터 품질 검증")
    print("=" * 60)

    validation_report = {}

    # 기본 검증
    basic_result = validate_data(df_main, label_column=LABEL_COLUMN)
    validation_report["basic"] = basic_result

    # 상세 검증 (선택적)
    if not args.skip_validation:
        # 마스킹 검증
        if TEXT_COLUMN in df_main.columns:
            print(f"\n[마스킹 검증] 컬럼: {TEXT_COLUMN}")
            mask_result = validate_masking(df_main, TEXT_COLUMN)
            validation_report["masking"] = mask_result

            if getattr(mask_result, "has_exposure", False):
                print(
                    "  [주의] 마스킹 미적용 PII 노출 감지: "
                    f"휴대폰={mask_result.exposed_phone}, "
                    f"이메일={mask_result.exposed_email}, "
                    f"주민번호={mask_result.exposed_jumin}"
                )

        # 패턴 타입 검증
        pattern_type_col = config.get("data", {}).get("pattern_type_column", "pattern_type")
        if pattern_type_col in df_main.columns and TEXT_COLUMN in df_main.columns:
            print(f"\n[패턴 타입 검증] 컬럼: {pattern_type_col}")
            pt_result = validate_pattern_type(
                df_main, TEXT_COLUMN, pattern_type_col
            )
            validation_report["pattern_type"] = pt_result

            if pt_result.mismatch_ratio > 0.1:
                print(f"  [주의] 패턴 불일치 비율 높음: {pt_result.mismatch_ratio:.1%}")

    # 검증 리포트 저장
    report_path = REPORT_DIR / "data_validation_report.txt"
    save_validation_report(validation_report, report_path)

    # 검증만 수행 옵션
    if args.validate_only:
        print("\n[검증 전용 모드] 데이터 저장을 건너뜁니다.")
        return

    # Step 4: 전처리
    print("\n" + "=" * 60)
    print("[Step 4] 전처리")
    print("=" * 60)

    if TEXT_COLUMN not in df_main.columns:
        # 텍스트 컬럼 후보 검색
        text_candidates = ["masked_text", "content", "text", "검출내용"]
        found_col = None
        for col in text_candidates:
            if col in df_main.columns:
                found_col = col
                break

        if found_col:
            print(f"  [자동 매핑] {found_col} -> {TEXT_COLUMN}")
            df_main[TEXT_COLUMN] = df_main[found_col]
        else:
            raise KeyError(
                f"TEXT_COLUMN '{TEXT_COLUMN}' not found. "
                f"Available columns: {list(df_main.columns)}"
            )

    df_clean = preprocess_dataframe(
        df_main,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN
    )

    # Step 5: 저장
    print("\n" + "=" * 60)
    print("[Step 5] 저장")
    print("=" * 60)

    # ── 5a. CSV (기존 호환 유지) ─────────────────────────────────────────
    out_csv = PROCESSED_DATA_DIR / MERGED_CLEANED_FILE
    save_processed(df_clean, str(out_csv))

    # ── 5b. Silver Parquet (Architecture §4) ─────────────────────────────
    # 레이블/텍스트 컬럼이 있는 정상 행 -> silver_detections
    # 레이블 또는 텍스트 결측 행     -> silver_quarantine
    _text_ok  = TEXT_COLUMN in df_main.columns
    _label_ok = LABEL_COLUMN in df_main.columns

    if _text_ok and _label_ok:
        mask_valid = (
            df_main[TEXT_COLUMN].notna() & (df_main[TEXT_COLUMN].str.strip() != "")
            & df_main[LABEL_COLUMN].notna()
        )
        df_silver   = df_clean.copy()           # 전처리된 정상 행
        df_quarantine = df_main[~mask_valid].copy()
        if not df_quarantine.empty:
            df_quarantine["quarantine_reason"] = df_quarantine.apply(
                lambda r: (
                    "missing_label" if pd.isna(r.get(LABEL_COLUMN)) else
                    "missing_text"  if pd.isna(r.get(TEXT_COLUMN)) or str(r.get(TEXT_COLUMN, "")).strip() == "" else
                    "unknown"
                ),
                axis=1,
            )
    else:
        df_silver     = df_clean.copy()
        df_quarantine = pd.DataFrame()

    silver_path     = PROCESSED_DATA_DIR / "silver_detections.parquet"
    quarantine_path = PROCESSED_DATA_DIR / "silver_quarantine.parquet"

    _pq_ok = False
    try:
        df_silver.to_parquet(silver_path, index=False)
        df_quarantine.to_parquet(quarantine_path, index=False)
        _pq_ok = True
        print(f"  [Parquet] silver_detections:  {silver_path}  ({len(df_silver):,}행)")
        print(f"  [Parquet] silver_quarantine:  {quarantine_path}  ({len(df_quarantine):,}행)")
    except ImportError:
        print("  [경고] pyarrow/fastparquet 미설치 - Silver Parquet 저장 건너뜀 (CSV만 유지)")

    n_total = len(df_silver) + len(df_quarantine)
    if n_total > 0:
        rate = len(df_silver) / n_total
        kpi = "PASS" if rate >= 0.95 else "FAIL"
        print(f"  parse_success_rate: {rate:.4f}  [{kpi} ≥0.95]")

    print("\n" + "=" * 60)
    print("데이터 파이프라인 완료")
    print("=" * 60)
    print(f"  CSV:    {out_csv}")
    print(f"  Parquet silver: {silver_path}")
    print(f"  Parquet quarantine: {quarantine_path}")
    print(f"  최종 데이터: {len(df_clean):,}행")


if __name__ == "__main__":
    main()
