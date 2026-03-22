"""전체 학습 파이프라인 실행 스크립트

회의록 2026-01 반영:
- 기본 전략: 3-Layer Filter 미적용 (필요 시만 사용)
- 합성 변수 확장 Feature 기본 OFF (Tier 0), 필요 시 옵션으로 사용

사용법:
    python scripts/run_training.py [--use-filter] [--filter-only] [--use-extended-features]
    python scripts/run_training.py --source label

실행 순서:
    1. 데이터 로드 (data/processed/merged_cleaned.csv)
    2. (선택) 3-Layer Filter 적용 (Keyword -> Rule -> ML 대상 분류)
    3. Feature Engineering (기본: 합성 변수 확장 OFF)
    4. 레이블 인코딩
    5. Baseline 학습 (DummyClassifier)
    6. XGBoost 기본 학습
    7. LightGBM 기본 학습
    8. XGBoost + Class Weight
    9. LightGBM + Class Weight
    10. 모델 비교 & 최종 저장

출력:
    models/final/best_model_v1.joblib
    outputs/filter_statistics.txt (필터 사용 시)
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
    RANDOM_SEED, PROCESSED_DATA_DIR, MERGED_CLEANED_FILE,
    TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE, TFIDF_MAX_FEATURES,
    FEATURE_DIR, MODEL_DIR, FINAL_MODEL_DIR, CHECKPOINT_DIR,
    REPORT_DIR, FILE_PATH_COLUMN,
)
from src.features.pipeline import build_features, save_feature_artifacts
from src.models.trainer import (
    encode_labels, train_baseline,
    train_xgboost, train_lightgbm,
    save_model_with_meta,
)

# 3-Layer Filter 모듈
try:
    from src.filters import FilterPipeline
    FILTER_AVAILABLE = True
except ImportError:
    FILTER_AVAILABLE = False
    print("[경고] src.filters 모듈을 찾을 수 없습니다. 필터 기능이 비활성화됩니다.")


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="학습 파이프라인 실행")
    parser.add_argument(
        "--use-filter",
        action="store_true",
        help="3-Layer Filter 적용 (기본값: 미적용)",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="[하위 호환] 필터 미적용 (기본값과 동일)",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="필터만 적용하고 종료 (--use-filter 필요)",
    )
    parser.add_argument(
        "--use-extended-features",
        dest="use_synthetic_expansion",
        action="store_true",
        default=False,
        help="합성 변수 확장 Feature 사용 (기본: False)",
    )
    parser.add_argument(
        "--disable-extended-features",
        dest="use_synthetic_expansion",
        action="store_false",
        help="[하위 호환] 합성 변수 확장 Feature 비활성화 (기본값과 동일)",
    )
    parser.add_argument(
        "--stage",
        choices=["s2", "s3a", "s3b", "s4s5", "all"],
        default="all",
        help="실행 단계: s2(피처), s3a(RULE), s3b(ML), s4s5(판정), all(전체, 기본값)",
    )
    parser.add_argument(
        "--source",
        choices=["detection", "label"],
        default="detection",
        help="학습 데이터 소스 (detection: 기존 CSV, label: silver_label.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="데이터 로드 및 피처 생성까지만 실행 (모델 학습/저장 없음)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="Repeated GroupShuffleSplit split 횟수 (분산 추정용, 기본: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="체크포인트에서 재시작 (models/checkpoints/ 에서 최신 체크포인트 자동 감지)",
    )
    parser.add_argument(
        "--split",
        choices=["group", "temporal", "server"],
        default="group",
        help="Train/Test 분할 전략 (group: pk_file GroupShuffleSplit(기본), "
             "temporal: label_work_month 시간 분할, "
             "server: server_name 그룹 분할)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="temporal split 시 테스트 월 수 (기본: 3 -> 마지막 3개월)",
    )
    parser.add_argument(
        "--use-multiclass",
        action="store_true",
        default=False,
        help="fp_description 기반 7-class 학습 (추론 시 binary collapse)",
    )
    parser.add_argument(
        "--tp-weight",
        type=float,
        default=1.0,
        help="TP 샘플 가중치 배수 (기본 1.0=비활성, class_weight=balanced만 사용)",
    )
    return parser.parse_args()


def apply_filter_pipeline(
    df: pd.DataFrame,
    text_column: str,
    file_path_column: str,
) -> dict:
    """
    3-Layer Filter 파이프라인 적용

    Returns:
        dict: {
            'filtered_df': ML 대상 DataFrame,
            'keyword_filtered': 키워드 필터 결과,
            'rule_filtered': 룰 필터 결과,
            'statistics': 필터 통계
        }
    """
    print("\n" + "=" * 60)
    print("[3-Layer Filter] 적용 시작")
    print("=" * 60)

    if not FILTER_AVAILABLE:
        print("  [건너뜀] 필터 모듈 미설치")
        return {
            "filtered_df": df,
            "keyword_filtered": pd.DataFrame(),
            "rule_filtered": pd.DataFrame(),
            "statistics": {
                "total": len(df),
                "keyword_filtered": 0,
                "rule_filtered": 0,
                "ml_target": len(df),
            },
        }

    # 필터 적용 (DataFrame 단위 API)
    pipeline = FilterPipeline()
    path_col = file_path_column if file_path_column in df.columns else None
    pipeline_result = pipeline.apply(
        df=df,
        text_column=text_column,
        file_path_column=path_col,
    )

    df_keyword = (
        pipeline_result.layer1_result.filtered_df.copy()
        if pipeline_result.layer1_result is not None
        else pd.DataFrame()
    )
    df_rule = (
        pipeline_result.layer2_result.filtered_df.copy()
        if pipeline_result.layer2_result is not None
        else pd.DataFrame()
    )
    df_ml = pipeline_result.ml_input_df.copy()

    # 통계
    stats = {
        "total": pipeline_result.total_input,
        "keyword_filtered": len(df_keyword),
        "rule_filtered": len(df_rule),
        "ml_target": len(df_ml),
    }
    denom = max(stats["total"], 1)

    print(f"\n[필터 통계]")
    print(f"  전체 데이터:     {stats['total']:,}건")
    print(f"  키워드 필터:     {stats['keyword_filtered']:,}건 ({stats['keyword_filtered']/denom:.1%})")
    print(f"  룰 필터:         {stats['rule_filtered']:,}건 ({stats['rule_filtered']/denom:.1%})")
    print(f"  ML 대상:         {stats['ml_target']:,}건 ({stats['ml_target']/denom:.1%})")

    return {
        "filtered_df": df_ml,
        "keyword_filtered": df_keyword,
        "rule_filtered": df_rule,
        "statistics": stats,
    }


def save_filter_statistics(stats: dict, filter_results: dict, path: Path) -> None:
    """필터 통계를 파일로 저장"""
    denom = max(stats["total"], 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("3-Layer Filter 통계 리포트\n")
        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("[요약]\n")
        f.write(f"  전체 데이터:     {stats['total']:,}건\n")
        f.write(f"  키워드 필터:     {stats['keyword_filtered']:,}건 ({stats['keyword_filtered']/denom:.1%})\n")
        f.write(f"  룰 필터:         {stats['rule_filtered']:,}건 ({stats['rule_filtered']/denom:.1%})\n")
        f.write(f"  ML 대상:         {stats['ml_target']:,}건 ({stats['ml_target']/denom:.1%})\n")
        f.write("\n")

        # 키워드 필터 상세
        if not filter_results["keyword_filtered"].empty:
            df_kw = filter_results["keyword_filtered"]
            f.write("[키워드 필터 분류 결과]\n")
            if "filter_label" in df_kw.columns:
                label_counts = df_kw["filter_label"].value_counts()
                for label, count in label_counts.items():
                    f.write(f"  {label}: {count:,}건\n")
            f.write("\n")

        # 룰 필터 상세
        if not filter_results["rule_filtered"].empty:
            df_rl = filter_results["rule_filtered"]
            f.write("[룰 필터 분류 결과]\n")
            if "filter_label" in df_rl.columns:
                label_counts = df_rl["filter_label"].value_counts()
                for label, count in label_counts.items():
                    f.write(f"  {label}: {count:,}건\n")
            f.write("\n")

    print(f"[저장] 필터 통계: {path}")


def _run_label_mode(args, source_path: "Path | None" = None) -> None:
    """레이블 단독 학습 파이프라인 (Phase 1).

    silver_label.parquet (또는 source_path) -> 메타/경로 피처 -> RuleLabeler -> LightGBM.

    Args:
        source_path: 기본값 None이면 silver_label.parquet 사용.
                     silver_joined.parquet 등 대체 소스 전달 가능.
    """
    import joblib

    from src.features.pipeline import build_features
    from src.models.trainer import encode_labels, train_lightgbm, save_model_with_meta

    # ── 경로 상수 ──
    silver_label_path = source_path or (PROCESSED_DATA_DIR / "silver_label.parquet")
    rules_yaml_path   = PROJECT_ROOT / "config" / "rules.yaml"
    rule_stats_path   = PROJECT_ROOT / "config" / "rule_stats.json"

    # joined(detection) 소스인 경우 아티팩트 경로 분리
    _is_joined = (source_path is not None)
    if _is_joined:
        _artifact_prefix = "joined_"
        _model_name = "LightGBM_phase1_joined"
    else:
        _artifact_prefix = ""
        _model_name = "LightGBM_phase1_label"
    model_output_path = FINAL_MODEL_DIR / f"{_artifact_prefix}phase1_lgb.joblib"

    # ── 체크포인트 설정 ──
    # 소스 파일명으로 구분 (silver_label vs silver_joined 혼용 방지)
    _sfx      = silver_label_path.stem
    _ckpt_dir = CHECKPOINT_DIR
    _ckpt_dir.mkdir(parents=True, exist_ok=True)

    _ckpt1  = _ckpt_dir / f"step1_df_{_sfx}.pkl"          # Step 1: raw df
    _ckpt2  = _ckpt_dir / f"step2_df_{_sfx}.pkl"          # Step 2: df + 메타/경로 피처
    _ckpt3  = _ckpt_dir / f"step3_df_{_sfx}.pkl"          # Step 3: df + label_binary
    _ckpt4  = _ckpt_dir / f"step4_df_{_sfx}.pkl"          # Step 4: df + rule 컬럼
    _ckpt5  = _ckpt_dir / f"step5_features_{_sfx}.pkl"    # Step 5: feature matrix + le
    _ckpt6  = _ckpt_dir / f"step6_model_{_sfx}.pkl"       # Step 6: model + le + f1

    # ── Resume 자동 감지 (최신 체크포인트부터 역순 탐색) ──
    # _resume_from: 0=처음부터, 1~6=Step1~6 완료
    _resume = getattr(args, "resume", False)
    _resume_from = 0
    if _resume:
        for _lvl, _path in [(6, _ckpt6), (5, _ckpt5),
                            (4, _ckpt4), (3, _ckpt3), (2, _ckpt2), (1, _ckpt1)]:
            if _path.exists():
                _resume_from = _lvl
                break
        _step_name = "6b" if _resume_from == 7 else str(_resume_from)
        _msg = (f"Step {_step_name} 이후부터 재시작"
                if _resume_from > 0 else "저장된 체크포인트 없음 - 처음부터 시작")
        print(f"[체크포인트] {_msg}")

    source_label = silver_label_path.name
    print("=" * 60)
    print(f"[Phase 1 학습 파이프라인] 소스: {source_label}")
    print("=" * 60)

    # ── Steps 1-4: df 로드 또는 실행 ──
    # 최신 df 체크포인트 단일 로드 후, 미완료 단계만 이어서 실행
    df = None
    if _resume_from >= 4:
        print(f"\n[체크포인트 로드] Step 4 (df+rule): {_ckpt4.name}")
        df = joblib.load(_ckpt4)
        print(f"  df 로드 완료: {df.shape[0]:,}행 x {df.shape[1]}열")
    elif _resume_from >= 3:
        print(f"\n[체크포인트 로드] Step 3 (df+label): {_ckpt3.name}")
        df = joblib.load(_ckpt3)
        print(f"  df 로드 완료: {df.shape[0]:,}행 x {df.shape[1]}열")
    elif _resume_from >= 2:
        print(f"\n[체크포인트 로드] Step 2 (df+features): {_ckpt2.name}")
        df = joblib.load(_ckpt2)
        print(f"  df 로드 완료: {df.shape[0]:,}행 x {df.shape[1]}열")
    elif _resume_from >= 1:
        print(f"\n[체크포인트 로드] Step 1 (raw df): {_ckpt1.name}")
        df = joblib.load(_ckpt1)
        print(f"  df 로드 완료: {df.shape[0]:,}행 x {df.shape[1]}열")

    # Step 1: 데이터 로드
    if _resume_from < 1:
        print(f"\n[Step 1] {source_label} 로드")
        if not silver_label_path.exists():
            print(f"  [오류] 파일 없음: {silver_label_path}")
            if source_path is None:
                print("  먼저 python scripts/run_data_pipeline.py --source label 실행 필요")
            else:
                print("  먼저 python scripts/run_data_pipeline.py --source joined 실행 필요")
            return
        df = pd.read_parquet(silver_label_path)
        print(f"  로드 완료: {df.shape[0]:,}행 x {df.shape[1]}열")
        if not _ckpt1.exists():
            joblib.dump(df, _ckpt1)
            print(f"  [체크포인트 저장] {_ckpt1.name}")
        else:
            print(f"  [체크포인트 이미 존재 - 저장 생략] {_ckpt1.name}")

    # Step 2: 피처 준비 (meta + path + RuleLabeler)
    if _resume_from < 2:
        from src.features.feature_preparer import prepare_phase1_features

        print(f"\n[Step 2] 피처 준비 (meta + path + RuleLabeler)")
        df = prepare_phase1_features(
            df,
            rules_yaml_path=rules_yaml_path,
            rule_stats_path=rule_stats_path,
        )
        if not _ckpt2.exists():
            joblib.dump(df, _ckpt2)
            print(f"  [체크포인트 저장] {_ckpt2.name}")
        else:
            print(f"  [체크포인트 이미 존재 - 저장 생략] {_ckpt2.name}")

    # Step 3: 레이블 정규화
    if _resume_from < 3:
        if "label_raw" not in df.columns:
            print("  [오류] label_raw 컬럼 없음 - silver_label.parquet 재생성 필요")
            return

        print(f"\n[Step 3] 레이블 정규화")
        if getattr(args, "use_multiclass", False) and "fp_description" in df.columns:
            from src.features.fp_classifier import classify_fp_description as _classify_fp
            print(f"  Multi-class 레이블 생성 (fp_description → 7-class)")
            df["label_binary"] = df.apply(
                lambda row: (
                    "TP-실제개인정보" if row["label_raw"] == "TP"
                    else _classify_fp(str(row.get("fp_description", "")))
                ), axis=1,
            )
            df.loc[df["label_binary"] == "UNKNOWN", "label_binary"] = "FP"
        else:
            df["label_binary"] = df["label_raw"]

        if not _ckpt3.exists():
            joblib.dump(df, _ckpt3)
            print(f"  [체크포인트 저장] {_ckpt3.name}")
        else:
            print(f"  [체크포인트 이미 존재 - 저장 생략] {_ckpt3.name}")

    # Step 4: 피처 전처리 완료 확인
    if _resume_from < 4:
        print(f"\n[Step 4] 피처 + 레이블 준비 완료")
        print(f"  레이블 분포:\n{df['label_binary'].value_counts().to_string()}")
        if not _ckpt4.exists():
            joblib.dump(df, _ckpt4)
            print(f"  [체크포인트 저장] {_ckpt4.name}")
        else:
            print(f"  [체크포인트 이미 존재 - 저장 생략] {_ckpt4.name}")

    # ── 피처 전처리: 문자열 → 숫자 변환 ──
    if "exception_requested" in df.columns:
        df["exception_requested"] = (df["exception_requested"].astype(str).str.upper() == "Y").astype(int)
    if "rule_matched" in df.columns:
        df["rule_matched"] = df["rule_matched"].astype(int)
    if "rule_confidence_lb" in df.columns:
        df["rule_confidence_lb"] = pd.to_numeric(df["rule_confidence_lb"], errors="coerce").fillna(0.0)

    # ── Step 5: Feature matrix 생성 ──
    if _resume_from >= 5:
        print(f"\n[체크포인트 로드] Step 5: {_ckpt5.name}")
        _d5         = joblib.load(_ckpt5)
        result      = _d5["result"]
        y_train_enc = _d5["y_train_enc"]
        y_test_enc  = _d5["y_test_enc"]
        le          = _d5["le"]
        print(f"  feature matrix 로드 완료")
    else:
        print(f"\n[Step 5] Feature matrix 생성 (dense only, TF-IDF 비활성화)")
        # 불필요 컬럼 제거: build_features 내부 df_train/df_test copy 크기 감소
        _KEEP_COLS = [
            "label_binary",
            "file_name", "file_path",
            "pk_file",
            "server_name",
            "fname_has_date", "fname_has_hash", "fname_has_rotation_num",
            "pattern_count_log1p", "pattern_count_bin",
            "is_mass_detection", "is_extreme_detection", "pii_type_ratio",
            # created_hour, created_weekday, is_weekend, created_month 제거
            # — 시간대/요일/월은 운영 환경 패턴이지 FP/TP 본질 아님 (과적합 위험)
            "path_depth", "extension", "is_log_file", "is_docker_overlay",
            "has_license_path", "is_temp_or_dev", "is_system_device",
            "is_package_path", "has_cron_path", "has_date_in_path",
            "has_business_token", "has_system_token",
            "rule_matched", "rule_primary_class", "rule_id",
            # exception_requested — Sumologic에 없음, 추론 불가 → 제거
            # [Tier 2 B1] 범주형 피처 (Sumologic에서 사용 가능 확인됨)
            "service", "ops_dept", "organization", "retention_period",
            # [Tier 2 B7] 서버 의미 토큰 (server_freq 대체) — Sumologic server_name에서 파생 가능
            "server_env", "server_is_prod", "server_stack",
            # [Tier 2 B8] RULE 세부 신호 (rule_matched binary → 도메인 지식)
            "rule_confidence_lb",
            # [Tier 2 B9] file-level aggregation (파이프라인에서 계산)
            "file_event_count", "file_pii_diversity",
            # 파일 크기
            "file_size", "file_size_log1p",
        ]
        _keep = [c for c in _KEEP_COLS if c in df.columns]
        df_for_features = df[_keep]
        print(f"  컬럼 선택: {df.shape[1]}열 → {len(_keep)}열")
        # temporal split 시 label_work_month 컬럼 필요
        _split = getattr(args, "split", "group")
        _test_months = int(getattr(args, "test_months", 3))
        if _split == "temporal" and "label_work_month" not in df_for_features.columns:
            if "label_work_month" in df.columns:
                df_for_features = df_for_features.copy()
                df_for_features["label_work_month"] = df["label_work_month"].values

        # [Tier 2 B9] file-level aggregation — build_features 호출 전에 df에 추가
        # build_features 내부에서 train/test split 후 train fold에서만 aggregate 계산
        # 여기서는 전체 df에 빈 컬럼만 준비 (실제 계산은 split 후)
        # NOTE: build_features가 split을 수행하므로, aggregation은 split 후에 해야 누수 방지
        # → pipeline.py 내부에서 처리하지 않고, build_features 결과의 df_train에서 계산 후
        #   feature matrix에 직접 추가하는 방식으로 구현
        _do_file_agg = ("pk_file" in df_for_features.columns)

        result = build_features(
            df_for_features,
            label_column="label_binary",
            use_multiview_tfidf=False,
            use_phase1_tfidf=True,
            use_synthetic_expansion=False,
            use_group_split=True,
            use_variance_threshold=False,
            n_splits=int(getattr(args, "n_splits", 1)),
            split_strategy=_split,
            test_months=_test_months,
        )

        # [Tier 2 B9] file-level aggregation: train fold에서만 계산 → df에 추가 (참조용)
        # NOTE: X_train/X_test matrix에는 주입하지 않음 (temporal split 노이즈 방지)
        if _do_file_agg:
            from src.features.meta_features import (
                compute_file_aggregates_label, merge_file_aggregates_label,
            )
            _df_tr = result["df_train"]
            _df_te = result["df_test"]
            if "pk_file" in _df_tr.columns:
                _file_agg = compute_file_aggregates_label(_df_tr)
                _df_tr = merge_file_aggregates_label(_df_tr, _file_agg)
                _df_te = merge_file_aggregates_label(_df_te, _file_agg)
                result["df_train"] = _df_tr
                result["df_test"] = _df_te
                print(f"  [Tier 2 B9] file aggregation: df에 추가 (X_train 미주입)")

        save_feature_artifacts(result, str(FEATURE_DIR), str(MODEL_DIR))
        y_train_enc, y_test_enc, le = encode_labels(result["y_train"], result["y_test"])
        if not _ckpt5.exists():
            joblib.dump({
                "result": result,
                "y_train_enc": y_train_enc,
                "y_test_enc": y_test_enc,
                "le": le,
            }, _ckpt5)
            print(f"  [체크포인트 저장] {_ckpt5.name}")
        else:
            print(f"  [체크포인트 이미 존재 - 저장 생략] {_ckpt5.name}")

    X_train = result["X_train"]
    X_test  = result["X_test"]
    y_train = result["y_train"]
    y_test  = result["y_test"]
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    if args.dry_run:
        print("\n[--dry-run] 피처 생성까지 완료. 모델 학습/저장 건너뜀.")
        return

    # ── [Tier 3 C1] Easy FP Suppressor — 고확신 FP 선제 분리 ──
    import numpy as _np

    _df_tr = result.get("df_train", pd.DataFrame())
    _df_te = result.get("df_test", pd.DataFrame())

    def _easy_fp_mask(df_slice):
        """Sumologic에서 사용 가능한 컬럼만으로 고확신 FP 조건 판별."""
        mask = pd.Series(False, index=df_slice.index)
        if "is_system_device" in df_slice.columns:
            mask = mask | (df_slice["is_system_device"] == 1)
        if "is_package_path" in df_slice.columns and "is_mass_detection" in df_slice.columns:
            mask = mask | ((df_slice["is_package_path"] == 1) & (df_slice["is_mass_detection"] == 1))
        if "is_docker_overlay" in df_slice.columns:
            mask = mask | (df_slice["is_docker_overlay"] == 1)
        if "has_license_path" in df_slice.columns:
            mask = mask | (df_slice["has_license_path"] == 1)
        return mask

    _easy_fp_train_mask = _easy_fp_mask(_df_tr)
    _easy_fp_test_mask = _easy_fp_mask(_df_te)

    _y_train_str = le.inverse_transform(y_train_enc)
    _easy_fp_train_labels = _y_train_str[_easy_fp_train_mask.values]
    _easy_fp_count = _easy_fp_train_mask.sum()

    if _easy_fp_count > 0:
        _easy_fp_purity = (_easy_fp_train_labels == "FP").mean()
        _easy_tp_leak = (_easy_fp_train_labels == "TP").sum()
        print(f"\n[Tier 3 C1] Easy FP Suppressor")
        print(f"  Train easy FP: {_easy_fp_count:,}건 / {len(_df_tr):,}건 "
              f"({_easy_fp_count/len(_df_tr):.1%})")
        print(f"  Purity (실제 FP 비율): {_easy_fp_purity:.4f}")
        print(f"  TP 유출: {_easy_tp_leak:,}건")

        if _easy_fp_purity >= 0.95:
            print(f"  → Purity ≥ 0.95: Suppressor 활성화")
            _residual_train_mask = ~_easy_fp_train_mask.values
            _residual_test_mask = ~_easy_fp_test_mask.values

            _suppressed_test_y = le.inverse_transform(
                y_test_enc[_easy_fp_test_mask.values]
            ) if _easy_fp_test_mask.sum() > 0 else _np.array([])
            _suppressed_test_count = int(_easy_fp_test_mask.sum())

            X_train = X_train[_residual_train_mask]
            X_test = X_test[_residual_test_mask]
            y_train_enc = y_train_enc[_residual_train_mask]
            y_test_enc = y_test_enc[_residual_test_mask]

            _df_tr = _df_tr[_residual_train_mask].reset_index(drop=True)
            _df_te = _df_te[_residual_test_mask].reset_index(drop=True)

            print(f"  Residual train: {X_train.shape[0]:,}건  test: {X_test.shape[0]:,}건")
            print(f"  Suppressed test: {_suppressed_test_count:,}건")
        else:
            print(f"  → Purity < 0.95: Suppressor 비활성화")
            _suppressed_test_y = _np.array([])
            _suppressed_test_count = 0
    else:
        print(f"\n[Tier 3 C1] Easy FP Suppressor: 해당 조건 0건 → 비활성화")
        _suppressed_test_y = _np.array([])
        _suppressed_test_count = 0

    # ── [Tier 2 B2] 중복 샘플 가중치 계산 ──
    _sample_weight = None
    if _df_tr is not None and "file_path" in _df_tr.columns and "file_name" in _df_tr.columns:
        print(f"\n[Tier 2 B2] 중복 샘플 가중치 계산")
        _group_key = _df_tr["file_path"].fillna("") + "|" + _df_tr["file_name"].fillna("")
        _group_sizes = _group_key.map(_group_key.value_counts())
        _sample_weight = (1.0 / _np.sqrt(_group_sizes.values)).astype(_np.float64)
        # 정규화: mean=1로 스케일 (class_weight와 조합 시 총 가중치 보존)
        _sample_weight = _sample_weight / _sample_weight.mean()
        _n_unique = _group_key.nunique()
        _n_total = len(_group_key)
        print(f"  고유 (file_path, file_name) 그룹: {_n_unique:,} / 전체 행: {_n_total:,}")
        print(f"  weight range: [{_sample_weight.min():.4f}, {_sample_weight.max():.4f}], "
              f"mean={_sample_weight.mean():.4f}")

    # ── TP 가중치 (기본 비활성, --tp-weight로 조정) ──
    _tp_weight_multiplier = getattr(args, "tp_weight", 1.0)
    if _tp_weight_multiplier != 1.0 and _sample_weight is not None and le is not None:
        _y_train_labels = le.inverse_transform(y_train_enc)
        _tp_mask = _np.array(["TP" in str(c) for c in _y_train_labels])
        if _tp_mask.any():
            _sample_weight[_tp_mask] *= _tp_weight_multiplier
            _sample_weight = _sample_weight / _sample_weight.mean()  # re-normalize
            print(f"\n[TP Weight] multiplier={_tp_weight_multiplier}, "
                  f"TP samples={_tp_mask.sum():,}/{len(_tp_mask):,}")
    else:
        print(f"\n[TP Weight] 비활성 (multiplier=1.0, class_weight=balanced만 사용)")

    # ── Step 6: LightGBM 학습 ──
    if _resume_from >= 6:
        print(f"\n[체크포인트 로드] Step 6: {_ckpt6.name}")
        _d6         = joblib.load(_ckpt6)
        model       = _d6["model"]
        le          = _d6["le"]
        f1          = _d6["f1"]
        y_train_enc = _d6["y_train_enc"]
        y_test_enc  = _d6["y_test_enc"]
        print(f"  model 로드 완료  (F1={f1:.4f})")
    else:
        print(f"\n[Step 6] LightGBM 학습")
        model, f1, report = train_lightgbm(
            X_train, y_train_enc, X_test, y_test_enc, le,
            use_class_weight=True,
            sample_weight=_sample_weight,
        )
        print(f"\n  F1-macro: {f1:.4f}")
        print(report)
        if not _ckpt6.exists():
            joblib.dump({
                "model": model,
                "le": le,
                "f1": f1,
                "y_train_enc": y_train_enc,
                "y_test_enc": y_test_enc,
            }, _ckpt6)
            print(f"  [체크포인트 저장] {_ckpt6.name}")
        else:
            print(f"  [체크포인트 이미 존재 - 저장 생략] {_ckpt6.name}")

    # ── [Tier 3 C1] 합산 평가: Suppressed FP + ML Residual ──
    if _suppressed_test_count > 0:
        from sklearn.metrics import f1_score as _f1_fn, classification_report as _cr_fn
        _ml_pred_test = le.inverse_transform(model.predict(X_test))
        # suppressed는 전부 FP로 예측 (suppressor 판정)
        _suppressed_pred = _np.full(_suppressed_test_count, "FP")
        # 합산
        _combined_y_true = _np.concatenate([le.inverse_transform(y_test_enc), _suppressed_test_y])
        _combined_y_pred = _np.concatenate([_ml_pred_test, _suppressed_pred])
        # multi-class → binary collapse for evaluation
        if getattr(args, "use_multiclass", False):
            _y_true_bin = _np.array(["TP" if "TP" in str(c) else "FP" for c in _combined_y_true])
            _y_pred_bin = _np.array(["TP" if "TP" in str(c) else "FP" for c in _combined_y_pred])
            _combined_f1 = _f1_fn(_y_true_bin, _y_pred_bin, average="macro", zero_division=0)
            print(f"\n[Tier 3 C1] 합산 평가 (Suppressed + ML Residual, binary collapse)")
            print(f"  ML Residual F1: {f1:.4f} ({X_test.shape[0]:,}건)")
            print(f"  Suppressed: {_suppressed_test_count:,}건 (전부 FP 판정)")
            print(f"  합산 F1-macro (binary): {_combined_f1:.4f} ({len(_y_true_bin):,}건)")
            print(_cr_fn(_y_true_bin, _y_pred_bin, zero_division=0))
        else:
            _combined_f1 = _f1_fn(_combined_y_true, _combined_y_pred, average="macro", zero_division=0)
            print(f"\n[Tier 3 C1] 합산 평가 (Suppressed + ML Residual)")
            print(f"  ML Residual F1: {f1:.4f} ({X_test.shape[0]:,}건)")
            print(f"  Suppressed: {_suppressed_test_count:,}건 (전부 FP 판정)")
            print(f"  합산 F1-macro: {_combined_f1:.4f} ({len(_combined_y_true):,}건)")
            print(_cr_fn(_combined_y_true, _combined_y_pred, zero_division=0))

    # ── Step 6c: Coverage-Precision Curve (threshold 최적화) ──
    print(f"\n[Step 6c] Coverage-Precision Curve (τ 스윕)")
    try:
        from src.evaluation.poc_metrics import compute_coverage_precision_curve
        _proba = model.predict_proba(X_test)
        # FP 클래스 인덱스 (TP가 아닌 클래스들)
        _tp_cls = "TP"
        _tp_candidates = [c for c in le.classes_ if "TP" in str(c)]
        _tp_label_str = _tp_candidates[0] if _tp_candidates else le.classes_[0]

        # P(FP) 기준 curve — TP가 아닌 FP 클래스 확률 사용
        # (max_proba 사용 시 TP 고확신 예측도 auto_fp에 포함되어 precision 오염)
        _fp_candidates = [c for c in le.classes_ if c != _tp_label_str]
        if _fp_candidates:
            _fp_idx = list(le.classes_).index(_fp_candidates[0])
            _fp_proba = _proba[:, _fp_idx]
        else:
            _fp_proba = 1.0 - _proba[:, list(le.classes_).index(_tp_label_str)]
        _y_test_str = le.inverse_transform(y_test_enc)
        _cpc = compute_coverage_precision_curve(
            _y_test_str, _fp_proba,
            tau_range=(0.50, 1.00, 0.05),
            tp_label=_tp_label_str,
            precision_target=0.95,
        )
        _curve_df = _cpc["curve"]
        _rec_tau = _cpc["recommended_tau"]

        print(f"  권장 τ: {_rec_tau}  (FP Precision ≥ 0.95 기준)")
        print(f"  {'τ':>6}  {'Coverage':>10}  {'Precision':>10}  {'AutoFP건수':>10}")
        for _, _row in _curve_df.iterrows():
            if _row["tau"] in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
                print(f"  {_row['tau']:>6.2f}  {_row['coverage']:>10.3f}  "
                      f"{_row['precision']:>10.3f}  {_row['auto_fp_count']:>10,}")
        # [Tier 3 C5] threshold_policy.json 아티팩트 저장
        import json as _json_c5
        # [Tier 3 C2] Slice-aware threshold — server_env별 tau 계산
        _slice_thresholds = {}
        if _df_te is not None and "server_env" in _df_te.columns and len(_fp_proba) == len(_df_te):
            _y_test_str_c2 = le.inverse_transform(y_test_enc)
            for _env in _df_te["server_env"].unique():
                _env_mask = (_df_te["server_env"] == _env).values
                if _env_mask.sum() < 100:
                    continue
                try:
                    _env_cpc = compute_coverage_precision_curve(
                        _y_test_str_c2[_env_mask], _fp_proba[_env_mask],
                        tau_range=(0.50, 1.00, 0.05),
                        tp_label=_tp_label_str,
                        precision_target=0.95,
                    )
                    _env_tau = _env_cpc.get("recommended_tau")
                    _env_cov = 0.0
                    if _env_tau is not None and _env_cpc.get("curve") is not None:
                        _match = _env_cpc["curve"][_env_cpc["curve"]["tau"] == _env_tau]
                        if len(_match) > 0:
                            _env_cov = float(_match.iloc[0]["coverage"])
                    _slice_thresholds[str(_env)] = {
                        "tau": float(_env_tau) if _env_tau is not None else None,
                        "support": int(_env_mask.sum()),
                        "coverage": _env_cov,
                    }
                except Exception:
                    pass
            if _slice_thresholds:
                print(f"\n  [Tier 3 C2] Slice-aware threshold (server_env별):")
                for _env, _info in sorted(_slice_thresholds.items()):
                    print(f"    {_env:>10}: tau={_info['tau']}, support={_info['support']:,}, "
                          f"coverage={_info['coverage']:.3f}")

        _threshold_policy = {
            "recommended_fp_tau": float(_rec_tau) if _rec_tau is not None else None,
            "precision_target": 0.95,
            "curve_summary": [
                {"tau": float(r["tau"]), "coverage": float(r["coverage"]),
                 "precision": float(r["precision"]), "auto_fp_count": int(r["auto_fp_count"])}
                for _, r in _curve_df.iterrows()
            ],
            "slice_thresholds": _slice_thresholds,  # [Tier 3 C2]
            "easy_fp_suppressor": {  # [Tier 3 C1]
                "enabled": _suppressed_test_count > 0,
                "suppressed_test_count": int(_suppressed_test_count),
                "conditions": [
                    "is_system_device == 1",
                    "is_package_path == 1 AND is_mass_detection == 1",
                    "is_docker_overlay == 1",
                    "has_license_path == 1",
                ],
            },
            "split_strategy": result.get("split_meta", {}).get("split_strategy", "unknown"),
            "f1_macro": float(f1),
            "saved_at": datetime.now().isoformat(),
        }
        _tp_path = FINAL_MODEL_DIR / "threshold_policy.json"
        _tp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_tp_path, "w", encoding="utf-8") as _fp:
            _json_c5.dump(_threshold_policy, _fp, ensure_ascii=False, indent=2)
        print(f"  [Tier 3 C5] threshold_policy.json 저장: {_tp_path}")
    except Exception as _e:
        print(f"  [경고] Coverage-Precision Curve 계산 실패: {_e}")

    # ── Step 7: 모델 저장 ──
    print(f"\n[Step 7] 모델 저장")
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "f1_macro": f1}, model_output_path)
    print(f"  저장 완료: {model_output_path}")

    # ── Step 8: 표준 아티팩트 저장 (models/final/) ──
    # run_inference.py가 기대하는 표준 경로에 모든 아티팩트를 저장한다.
    print(f"\n[Step 8] 표준 아티팩트 저장 -> {FINAL_MODEL_DIR}")
    try:
        import json as _json
        import traceback as _tb
        from src.models.feature_builder_snapshot import FeatureBuilderSnapshot
        from src.models.ood_detector import OODDetector

        _final_dir = FINAL_MODEL_DIR
        _final_dir.mkdir(parents=True, exist_ok=True)

        # 1. best_model_v1.joblib (label) / joined_best_model_v1.joblib (joined)
        save_model_with_meta(
            model=model,
            path=str(_final_dir / f"{_artifact_prefix}best_model_v1.joblib"),
            label_encoder=le,
            f1_score_val=f1,
            model_name=_model_name,
            train_size=X_train.shape[0],
            test_size=X_test.shape[0],
            feature_count=X_train.shape[1],
        )

        # 2. label_encoder.joblib
        joblib.dump(le, _final_dir / f"{_artifact_prefix}label_encoder.joblib")

        # 3. feature_builder.joblib
        _snapshot = FeatureBuilderSnapshot.from_build_result(result)
        _snapshot.save(str(_final_dir / f"{_artifact_prefix}feature_builder.joblib"))
        _n_dense = len(_snapshot.dense_columns)

        # 4. ood_detector.joblib (dense 피처 부분에 적합)
        _ood_saved = False
        if _n_dense > 0:
            _n_tfidf = X_train.shape[1] - _n_dense
            try:
                _X_dense = X_train[:, _n_tfidf:].toarray()
                _ood = OODDetector()
                _ood.fit(_X_dense)
                _ood.save(str(_final_dir / f"{_artifact_prefix}ood_detector.joblib"))
                _ood_saved = True
            except Exception as _e_ood:
                print(f"  [경고] OOD Detector 학습 실패 (건너뜀): {_e_ood}")

        # 5. feature_schema.json
        _tfidf_views = list(result.get("tfidf_vectorizers", {}).keys())
        _schema = {
            "n_features": len(_snapshot.feature_names),
            "n_tfidf_features": len(_snapshot.feature_names) - _n_dense,
            "n_dense_features": _n_dense,
            "tfidf_views": _tfidf_views,
            "dense_columns": _snapshot.dense_columns,
            "saved_at": datetime.now().isoformat(),
        }
        with open(_final_dir / f"{_artifact_prefix}feature_schema.json", "w", encoding="utf-8") as _f:
            _json.dump(_schema, _f, indent=2, ensure_ascii=False)

        print(f"  [OK] {_artifact_prefix}best_model_v1.joblib  (F1={f1:.4f})")
        print(f"  [OK] {_artifact_prefix}label_encoder.joblib  (classes={list(le.classes_)})")
        print(f"  [OK] {_artifact_prefix}feature_builder.joblib  (TF-IDF views={_tfidf_views}, dense={_n_dense}개)")
        print(f"  {'[OK]' if _ood_saved else '[NG]'} {_artifact_prefix}ood_detector.joblib  ({_n_dense}차원 dense)")
        print(f"  [OK] {_artifact_prefix}feature_schema.json")

    except Exception as _e:
        print(f"  [경고] 표준 아티팩트 저장 실패: {_e}")
        _tb.print_exc()

    print("\nPhase 1 레이블 단독 학습 완료.")


def _run_detection_mode(args) -> None:
    """silver_joined.parquet + multiview TF-IDF (full_context_raw) 기반 학습 (Phase 1.5).

    Phase 1 label 모드와 달리 Sumologic 텍스트를 raw/shape/path 3-view TF-IDF로 활용.
    아티팩트는 models/final/detection_* 접두사로 저장.
    """
    import joblib

    from src.features.meta_features import build_meta_features
    from src.features.path_features import extract_path_features
    from src.filters.rule_labeler import RuleLabeler
    from src.features.pipeline import build_features, save_feature_artifacts
    from src.models.trainer import encode_labels, train_lightgbm, save_model_with_meta

    joined_path = PROCESSED_DATA_DIR / "silver_joined.parquet"
    if not joined_path.exists():
        print(f"[오류] silver_joined.parquet 없음")
        print("  먼저 실행: python scripts/run_data_pipeline.py --source joined")
        return

    rules_yaml_path  = PROJECT_ROOT / "config" / "rules.yaml"
    rule_stats_path  = PROJECT_ROOT / "config" / "rule_stats.json"
    model_output_path = FINAL_MODEL_DIR / "detection_lgb.joblib"

    _ckpt_dir = CHECKPOINT_DIR
    _ckpt_dir.mkdir(parents=True, exist_ok=True)

    _ckpt1 = _ckpt_dir / "step1_df_silver_joined_det.pkl"
    _ckpt2 = _ckpt_dir / "step2_df_silver_joined_det.pkl"
    _ckpt3 = _ckpt_dir / "step3_df_silver_joined_det.pkl"
    _ckpt4 = _ckpt_dir / "step4_df_silver_joined_det.pkl"
    _ckpt5 = _ckpt_dir / "step5_features_silver_joined_det.pkl"
    _ckpt6 = _ckpt_dir / "step6_model_silver_joined_det.pkl"

    _resume = getattr(args, "resume", False)
    _resume_from = 0
    if _resume:
        for _lvl, _path in [(6, _ckpt6), (5, _ckpt5), (4, _ckpt4),
                            (3, _ckpt3), (2, _ckpt2), (1, _ckpt1)]:
            if _path.exists():
                _resume_from = _lvl
                break
        _msg = (f"Step {_resume_from} 이후부터 재시작"
                if _resume_from > 0 else "저장된 체크포인트 없음 - 처음부터 시작")
        print(f"[체크포인트] {_msg}")

    print("=" * 60)
    print("[Phase 1.5 학습 파이프라인] 소스: silver_joined.parquet (Multiview TF-IDF)")
    print("=" * 60)

    df = None
    if _resume_from >= 4:
        print(f"\n[체크포인트 로드] Step 4: {_ckpt4.name}")
        df = joblib.load(_ckpt4)
    elif _resume_from >= 3:
        print(f"\n[체크포인트 로드] Step 3: {_ckpt3.name}")
        df = joblib.load(_ckpt3)
    elif _resume_from >= 2:
        print(f"\n[체크포인트 로드] Step 2: {_ckpt2.name}")
        df = joblib.load(_ckpt2)
    elif _resume_from >= 1:
        print(f"\n[체크포인트 로드] Step 1: {_ckpt1.name}")
        df = joblib.load(_ckpt1)

    # Step 1: 데이터 로드
    if _resume_from < 1:
        print(f"\n[Step 1] silver_joined.parquet 로드")
        df = pd.read_parquet(joined_path)
        print(f"  로드 완료: {df.shape[0]:,}행 x {df.shape[1]}열")
        if not _ckpt1.exists():
            joblib.dump(df, _ckpt1)
            print(f"  [체크포인트 저장] {_ckpt1.name}")

    # Step 2: 메타/경로 피처 추가
    _META_CHUNK = 500_000
    if _resume_from < 2:
        print(f"\n[Step 2] 메타/경로 피처 추출 (청크 크기: {_META_CHUNK:,})")
        _meta_chunks = []
        for _s in range(0, len(df), _META_CHUNK):
            _e = min(_s + _META_CHUNK, len(df))
            _c = build_meta_features(df.iloc[_s:_e].copy())
            if "file_path" in _c.columns:
                _path_feats = _c["file_path"].apply(extract_path_features)
                _path_df = pd.DataFrame(list(_path_feats), index=_c.index)
                for _col in _path_df.columns:
                    if _col not in _c.columns:
                        _c[_col] = _path_df[_col]
            _meta_chunks.append(_c)
            print(f"  [{_s:,}~{_e:,}] 완료", flush=True)
        df = pd.concat(_meta_chunks, ignore_index=True)
        del _meta_chunks
        print(f"  메타 피처 추가 완료. 전체 컬럼: {df.shape[1]}")
        if not _ckpt2.exists():
            joblib.dump(df, _ckpt2)
            print(f"  [체크포인트 저장] {_ckpt2.name}")

    # Step 3: 레이블 컬럼 정규화
    if _resume_from < 3:
        print(f"\n[Step 3] 레이블 컬럼 정규화")
        if "label_raw" not in df.columns:
            print("  [오류] label_raw 컬럼 없음")
            return
        df["label_binary"] = df["label_raw"]
        print(f"  레이블 분포:\n{df['label_binary'].value_counts().to_string()}")
        if not _ckpt3.exists():
            joblib.dump(df, _ckpt3)
            print(f"  [체크포인트 저장] {_ckpt3.name}")

    # Step 4: RuleLabeler 실행
    _RULE_CHUNK = 500_000
    if _resume_from < 4:
        print(f"\n[Step 4] RuleLabeler 적용 (청크 크기: {_RULE_CHUNK:,})")
        if rules_yaml_path.exists() and rule_stats_path.exists():
            labeler = RuleLabeler.from_config_files(str(rules_yaml_path), str(rule_stats_path))
            _label_chunks = []
            for _s in range(0, len(df), _RULE_CHUNK):
                _e = min(_s + _RULE_CHUNK, len(df))
                _chunk_labels, _ = labeler.label_batch(df.iloc[_s:_e])
                # [Tier 2 B8] rule_confidence_lb 포함하여 RULE 세부 신호 캡처
                _rule_cols = ["rule_matched", "rule_primary_class", "rule_id", "rule_confidence_lb"]
                _rule_cols_present = [c for c in _rule_cols if c in _chunk_labels.columns]
                _label_chunks.append(_chunk_labels[_rule_cols_present])
                print(f"  [{_s:,}~{_e:,}] 완료", flush=True)
            _rule_df = pd.concat(_label_chunks, ignore_index=True)
            df = df.reset_index(drop=True)
            df["rule_matched"]       = _rule_df["rule_matched"].values
            df["rule_primary_class"] = _rule_df["rule_primary_class"].values
            df["rule_id"]            = _rule_df["rule_id"].values
            del _label_chunks, _rule_df
            print(f"  룰 매칭 완료.")
        else:
            print(f"  [경고] 룰 설정 파일 없음 - RuleLabeler 건너뜀")
        if not _ckpt4.exists():
            joblib.dump(df, _ckpt4)
            print(f"  [체크포인트 저장] {_ckpt4.name}")

    label_counts = df["label_binary"].value_counts()
    print(f"\n  레이블 분포:\n{label_counts.to_string()}")

    # Step 5: Feature matrix 생성 (multiview TF-IDF 활성화)
    if _resume_from >= 5:
        print(f"\n[체크포인트 로드] Step 5: {_ckpt5.name}")
        _d5         = joblib.load(_ckpt5)
        result      = _d5["result"]
        y_train_enc = _d5["y_train_enc"]
        y_test_enc  = _d5["y_test_enc"]
        le          = _d5["le"]
        print(f"  feature matrix 로드 완료")
    else:
        print(f"\n[Step 5] Feature matrix 생성 (Multiview TF-IDF + dense)")
        _KEEP_COLS_DET = [
            "label_binary", "full_context_raw", "file_name", "file_path",
            "pk_file", "server_name",
            "fname_has_date", "fname_has_hash", "fname_has_rotation_num",
            "pattern_count_log1p", "pattern_count_bin",
            "is_mass_detection", "is_extreme_detection", "pii_type_ratio",
            # created_hour, created_weekday, is_weekend, created_month 제거
            "path_depth", "is_log_file", "is_docker_overlay",
            "has_license_path", "is_temp_or_dev", "is_system_device",
            "is_package_path", "has_cron_path", "has_date_in_path",
            "has_business_token", "has_system_token",
            "rule_matched", "rule_primary_class", "rule_id",
            # exception_requested — Sumologic에 없음 → 제거
        ]
        _keep = [c for c in _KEEP_COLS_DET if c in df.columns]
        df_for_features = df[_keep]
        print(f"  컬럼 선택: {df.shape[1]}열 → {len(_keep)}열")
        _det_feature_dir = str(FEATURE_DIR / "detection")
        result = build_features(
            df_for_features,
            text_column="full_context_raw",
            label_column="label_binary",
            use_multiview_tfidf=True,
            use_phase1_tfidf=False,
            use_synthetic_expansion=False,
            use_group_split=True,
            use_variance_threshold=False,
            n_splits=int(getattr(args, "n_splits", 1)),
        )
        print("[Multi-view TF-IDF] Detection 모드 피처 생성 완료")
        save_feature_artifacts(result, _det_feature_dir, str(MODEL_DIR))
        y_train_enc, y_test_enc, le = encode_labels(result["y_train"], result["y_test"])
        if not _ckpt5.exists():
            joblib.dump({
                "result": result,
                "y_train_enc": y_train_enc,
                "y_test_enc": y_test_enc,
                "le": le,
            }, _ckpt5)
            print(f"  [체크포인트 저장] {_ckpt5.name}")

    X_train = result["X_train"]
    X_test  = result["X_test"]
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    if args.dry_run:
        print("\n[--dry-run] 피처 생성까지 완료. 모델 학습/저장 건너뜀.")
        return

    # Step 6: LightGBM 학습
    if _resume_from >= 6:
        print(f"\n[체크포인트 로드] Step 6: {_ckpt6.name}")
        _d6         = joblib.load(_ckpt6)
        model       = _d6["model"]
        le          = _d6["le"]
        f1          = _d6["f1"]
        y_train_enc = _d6["y_train_enc"]
        y_test_enc  = _d6["y_test_enc"]
        print(f"  model 로드 완료  (F1={f1:.4f})")
    else:
        print(f"\n[Step 6] LightGBM 학습")
        model, f1, report = train_lightgbm(
            X_train, y_train_enc, X_test, y_test_enc, le,
            use_class_weight=True,
        )
        print(f"\n  F1-macro: {f1:.4f}")
        print(report)
        if not _ckpt6.exists():
            joblib.dump({
                "model": model, "le": le, "f1": f1,
                "y_train_enc": y_train_enc, "y_test_enc": y_test_enc,
            }, _ckpt6)
            print(f"  [체크포인트 저장] {_ckpt6.name}")

    # Step 7: 모델 저장
    print(f"\n[Step 7] 모델 저장")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "f1_macro": f1}, model_output_path)
    print(f"  저장 완료: {model_output_path}")

    # Step 8: 표준 아티팩트 저장 (models/final/detection_*)
    print(f"\n[Step 8] 표준 아티팩트 저장 -> {FINAL_MODEL_DIR} (detection_ 접두사)")
    try:
        import json as _json
        import traceback as _tb
        from src.models.feature_builder_snapshot import FeatureBuilderSnapshot
        from src.models.ood_detector import OODDetector
        from datetime import datetime as _dt

        _final_dir = FINAL_MODEL_DIR
        _final_dir.mkdir(parents=True, exist_ok=True)

        save_model_with_meta(
            model=model,
            path=str(_final_dir / "detection_best_model_v1.joblib"),
            label_encoder=le,
            f1_score_val=f1,
            model_name="LightGBM_phase1dot5_detection",
            train_size=X_train.shape[0],
            test_size=X_test.shape[0],
            feature_count=X_train.shape[1],
        )

        joblib.dump(le, _final_dir / "detection_label_encoder.joblib")

        _snapshot = FeatureBuilderSnapshot.from_build_result(result)
        _snapshot.save(str(_final_dir / "detection_feature_builder.joblib"))
        _n_dense = len(_snapshot.dense_columns)

        _ood_saved = False
        if _n_dense > 0:
            _n_tfidf = X_train.shape[1] - _n_dense
            try:
                _X_dense = X_train[:, _n_tfidf:].toarray()
                _ood = OODDetector()
                _ood.fit(_X_dense)
                _ood.save(str(_final_dir / "detection_ood_detector.joblib"))
                _ood_saved = True
            except Exception as _e_ood:
                print(f"  [경고] OOD Detector 학습 실패 (건너뜀): {_e_ood}")

        _tfidf_views = list(result.get("tfidf_vectorizers", {}).keys())
        _schema = {
            "n_features": len(_snapshot.feature_names),
            "n_tfidf_features": len(_snapshot.feature_names) - _n_dense,
            "n_dense_features": _n_dense,
            "tfidf_views": _tfidf_views,
            "dense_columns": _snapshot.dense_columns,
            "saved_at": _dt.now().isoformat(),
        }
        with open(_final_dir / "detection_feature_schema.json", "w", encoding="utf-8") as _f:
            _json.dump(_schema, _f, indent=2, ensure_ascii=False)

        print(f"  [OK] detection_best_model_v1.joblib  (F1={f1:.4f})")
        print(f"  [OK] detection_label_encoder.joblib  (classes={list(le.classes_)})")
        print(f"  [OK] detection_feature_builder.joblib  (TF-IDF views={_tfidf_views}, dense={_n_dense}개)")
        print(f"  {'[OK]' if _ood_saved else '[NG]'} detection_ood_detector.joblib")
        print(f"  [OK] detection_feature_schema.json  (총 피처={X_train.shape[1]:,})")

    except Exception as _e:
        print(f"  [경고] 표준 아티팩트 저장 실패: {_e}")
        import traceback as _tb
        _tb.print_exc()

    print("\nPhase 1.5 Detection 모드 학습 완료.")


def main():
    args = parse_args()

    if args.source == "label":
        _run_label_mode(args)
        return

    if args.source == "detection":
        # silver_joined.parquet 기반으로 label 모드와 동일한 파이프라인 사용
        _run_label_mode(args, source_path=PROCESSED_DATA_DIR / "silver_joined.parquet")
        return

    set_seed(RANDOM_SEED)

    use_filter = args.use_filter and not args.skip_filter
    if args.use_filter and args.skip_filter:
        print("[안내] --use-filter와 --skip-filter가 동시에 지정되어 필터를 미적용합니다.")
    if args.filter_only and not use_filter:
        print("[안내] --filter-only는 --use-filter와 함께 사용할 때만 동작합니다.")
        return

    # 디렉토리 생성
    ensure_dirs(
        MODEL_DIR / "baseline",
        MODEL_DIR / "experiments",
        FINAL_MODEL_DIR,
        FEATURE_DIR,
        REPORT_DIR,
    )

    # 1. 데이터 로드
    print("=" * 60)
    print("[Step 1] 데이터 로드")
    print("=" * 60)
    df = pd.read_csv(PROCESSED_DATA_DIR / MERGED_CLEANED_FILE)
    print(f"  데이터: {df.shape[0]:,}행 x {df.shape[1]}열")
    print(f"  전략: 필터={'ON' if use_filter else 'OFF'}, 합성변수확장={'ON' if args.use_synthetic_expansion else 'OFF'}")

    # 2. (선택) 3-Layer Filter 적용
    if use_filter and FILTER_AVAILABLE:
        filter_results = apply_filter_pipeline(df, TEXT_COLUMN, FILE_PATH_COLUMN)

        # 필터 통계 저장
        save_filter_statistics(
            filter_results["statistics"],
            filter_results,
            REPORT_DIR / "filter_statistics.txt"
        )

        # 필터링된 데이터 저장 (참고용)
        if not filter_results["keyword_filtered"].empty:
            filter_results["keyword_filtered"].to_csv(
                PROCESSED_DATA_DIR / "keyword_filtered.csv",
                index=False
            )

        if not filter_results["rule_filtered"].empty:
            filter_results["rule_filtered"].to_csv(
                PROCESSED_DATA_DIR / "rule_filtered.csv",
                index=False
            )

        # ML 대상 데이터로 교체
        df = filter_results["filtered_df"]
        print(f"\n  ML 학습 대상: {len(df):,}건")

        if args.filter_only:
            print("\n[필터 전용 모드] 학습을 건너뜁니다.")
            return
    elif use_filter and not FILTER_AVAILABLE:
        print("\n  [건너뜀] 필터 모듈을 찾을 수 없어 필터를 적용하지 않습니다.")
    else:
        print("\n  [기본 전략] 3-Layer Filter 미적용")

    if len(df) == 0:
        print("[오류] ML 학습 대상 데이터가 없습니다.")
        return

    # 3. Feature Engineering
    print("\n" + "=" * 60)
    print("[Step 2] Feature Engineering")
    print("=" * 60)
    result = build_features(
        df=df,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        test_size=TEST_SIZE,
        tfidf_max_features=TFIDF_MAX_FEATURES,
        use_synthetic_expansion=args.use_synthetic_expansion,
    )
    save_feature_artifacts(result, str(FEATURE_DIR), str(MODEL_DIR))

    X_train, X_test = result["X_train"], result["X_test"]
    y_train, y_test = result["y_train"], result["y_test"]

    # 4. 레이블 인코딩
    print("\n" + "=" * 60)
    print("[Step 3] 레이블 인코딩")
    print("=" * 60)
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    # 5~9. 모델 학습
    results = {}
    trained_models = {}

    # 5. Baseline
    model, f1 = train_baseline(X_train, y_train_enc, X_test, y_test_enc)
    results["Baseline (dummy)"] = f1
    trained_models["Baseline (dummy)"] = model

    # 6. XGBoost
    model, f1, _ = train_xgboost(X_train, y_train_enc, X_test, y_test_enc, le)
    results["XGBoost (default)"] = f1
    trained_models["XGBoost (default)"] = model

    # 7. LightGBM
    model, f1, _ = train_lightgbm(X_train, y_train_enc, X_test, y_test_enc, le)
    results["LightGBM (default)"] = f1
    trained_models["LightGBM (default)"] = model

    # 8. XGBoost + Class Weight
    model, f1, _ = train_xgboost(
        X_train, y_train_enc, X_test, y_test_enc, le, use_class_weight=True
    )
    results["XGBoost (Class Weight)"] = f1
    trained_models["XGBoost (Class Weight)"] = model

    # 9. LightGBM + Class Weight
    model, f1, _ = train_lightgbm(
        X_train, y_train_enc, X_test, y_test_enc, le, use_class_weight=True
    )
    results["LightGBM (Class Weight)"] = f1
    trained_models["LightGBM (Class Weight)"] = model

    # 10. 최고 모델 저장
    print("\n" + "=" * 60)
    print("[Step 9] 모델 비교 & 최종 저장")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    best_name, best_f1 = sorted_results[0]

    for name, score in sorted_results:
        marker = " <- BEST" if name == best_name else ""
        print(f"  {name:35s}  F1-macro: {score:.4f}{marker}")

    best_model = trained_models[best_name]
    save_model_with_meta(
        model=best_model,
        path=str(FINAL_MODEL_DIR / "best_model_v1.joblib"),
        label_encoder=le,
        f1_score_val=best_f1,
        model_name=best_name,
        all_results=results,
        train_size=X_train.shape[0],
        test_size=X_test.shape[0],
        feature_count=X_train.shape[1],
    )

    print("\n학습 파이프라인 완료.")


if __name__ == "__main__":
    main()
