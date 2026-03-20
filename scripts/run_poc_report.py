"""PoC 결과 Excel 리포트 자동 생성 스크립트.

Usage::

    python scripts/run_poc_report.py                       # 기본 (Phase 1, Label Only)
    python scripts/run_poc_report.py --phase 2             # Phase 2 (+ Sumologic)
    python scripts/run_poc_report.py --output my.xlsx
    python scripts/run_poc_report.py --precision-target 0.90
    python scripts/run_poc_report.py --skip-ml             # feature 없을 때 Rule 분석만

처리 흐름::

    Step 1: LabelLoader().load_all()          -> df_label
    Step 2: load_model_with_meta()            -> model, le, classes   (--skip-ml 시 생략)
    Step 3: load_feature_artifacts()          -> X_all                (--skip-ml 시 생략)
    Step 4: Primary Split (group_time_split)  -> train/test 인덱스
    Step 5: RuleLabeler.label_batch(df_test)  -> rule_labels_df
    Step 6: predict_with_uncertainty()        -> ml_predictions_df    (--skip-ml 시 생략)
    Step 7: 핵심 지표 계산
    Step 8: Secondary / Tertiary Split 성능 계산
    Step 9: PocReportData 조립
    Step 10: PocExcelWriter(data).write(output_path)
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── 조직명 패턴 (Tertiary Split 기준) ─────────────────────────────────────────
_TARGET_ORGS = ["CTO", "NW", "품질혁신센터"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="PoC 결과 Excel 리포트 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="분석 단계: 1=Label Only, 2=Label+Sumologic (기본: 1)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/poc_report.xlsx"),
        help="출력 Excel 파일 경로 (기본: outputs/poc_report.xlsx)",
    )
    parser.add_argument(
        "--precision-target", type=float, default=0.95,
        dest="precision_target",
        help="Coverage-Precision 곡선의 목표 Precision (기본: 0.95)",
    )
    parser.add_argument(
        "--skip-ml", action="store_true", default=False,
        dest="skip_ml",
        help="Feature Parquet 없을 때 Rule 분석만 실행",
    )
    parser.add_argument(
        "--label-dir", type=Path,
        default=Path("data/raw/label"),
        dest="label_dir",
        help="레이블 Excel 디렉토리 (기본: data/raw/label)",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("models"),
        dest="model_dir",
        help="학습된 모델 디렉토리 (기본: models)",
    )
    return parser.parse_args(args)


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼 함수 (단위 테스트에서 직접 임포트)
# ─────────────────────────────────────────────────────────────────────────────

def _get_data_condition(phase: int) -> str:
    """Phase 번호에 따른 데이터 조건 문자열."""
    return "Label Only" if phase == 1 else "Label + Sumologic"


def _make_split_summary(
    df: pd.DataFrame,
    train_idx: list,
    test_idx: list,
    split_method: str = "GroupTimeSplit",
) -> dict:
    """Split 결과 요약 딕셔너리 생성."""
    return {
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "split_method": split_method,
    }


def _extract_org_column(
    df: pd.DataFrame,
    org_col: str = "organization",
    source_file_col: str = "_source_file",
) -> pd.Series:
    """DataFrame에서 조직명 Series를 반환.

    org_col이 있으면 그것을 사용, 없으면 source_file_col 파일명에서 추출.
    """
    from src.evaluation.split_strategies import _extract_org_from_filename

    if org_col in df.columns:
        return df[org_col]

    if source_file_col in df.columns:
        return df[source_file_col].apply(
            lambda x: _extract_org_from_filename(str(x)) if pd.notna(x) else None
        )

    return pd.Series([None] * len(df), index=df.index)


def _build_tertiary_splits(
    df: pd.DataFrame,
    y_true,
    tp_label: str = "TP",
    org_col: str = "organization",
    source_file_col: str = "_source_file",
    y_pred_all: np.ndarray | None = None,
) -> list[dict]:
    """3개 조직별 Tertiary Split 결과 생성.

    조직 추출 실패 또는 해당 조직 데이터 없으면 해당 조직 건너뜀.

    Args:
        y_pred_all: 전체 데이터에 대한 ML 예측 (None이면 dummy all-TP 사용)

    Returns:
        list of split_result dicts (compute_split_comparison 입력 형식)
    """
    from src.evaluation.split_strategies import org_subset_split

    org_series = _extract_org_column(df, org_col=org_col, source_file_col=source_file_col)

    if org_series.isna().all():
        warnings.warn(
            "조직명 추출 실패 - Tertiary Split 건너뜀. "
            f"'{source_file_col}' 컬럼에 CTO/NW/품질혁신센터 패턴이 있는지 확인하세요.",
            UserWarning,
            stacklevel=2,
        )
        return []

    y_true_arr = np.asarray(y_true)
    results = []

    for target_org in _TARGET_ORGS:
        if (org_series == target_org).sum() == 0:
            logger.info("Tertiary Split: '%s' 데이터 없음 - 건너뜀", target_org)
            continue

        try:
            # org_col 있으면 org_col 사용, 없으면 _source_file에서 추출된 Series로 임시 컬럼 생성
            if org_col in df.columns:
                train_idx, test_idx = org_subset_split(df, target_org=target_org, org_col=org_col)
            else:
                df_tmp = df.copy()
                df_tmp["_org_tmp"] = org_series.values
                train_idx, test_idx = org_subset_split(
                    df_tmp, target_org=target_org, org_col="_org_tmp"
                )

            if len(test_idx) == 0:
                continue

            y_true_test = y_true_arr[test_idx]
            # ML 예측이 있으면 사용, 없으면 dummy all-TP
            if y_pred_all is not None:
                y_pred_test = y_pred_all[test_idx]
            else:
                y_pred_test = np.full_like(y_true_test, fill_value=tp_label)

            results.append({
                "split_name": f"Tertiary-{target_org}",
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "y_true": y_true_test,
                "y_pred": y_pred_test,
                "tp_label": tp_label,
                "coverage_at_target": 0.0,
            })

        except (ValueError, KeyError) as exc:
            warnings.warn(
                f"Tertiary Split '{target_org}' 실패: {exc} - 건너뜀",
                UserWarning,
                stacklevel=2,
            )

    return results


def _load_rules_config() -> tuple[list, dict]:
    """rules.yaml + rule_stats.json 로드.

    Returns:
        (rules_config list, rule_stats dict)
    """
    import yaml
    import json

    rules_path = PROJECT_ROOT / "config" / "rules.yaml"
    stats_path = PROJECT_ROOT / "config" / "rule_stats.json"

    rules_config = []
    if rules_path.exists():
        with open(rules_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rules_config = data.get("rules", [])

    rule_stats = {}
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            rule_stats = json.load(f)

    return rules_config, rule_stats


def _build_run_metadata(
    model_path,
    df: pd.DataFrame,
    month_col: str = "label_work_month",
) -> dict:
    """실행 메타데이터 딕셔너리 생성."""
    import datetime

    date_range = "N/A"
    if month_col in df.columns and not df.empty:
        date_range = f"{df[month_col].min()} ~ {df[month_col].max()}"

    return {
        "run_datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(model_path) if model_path is not None else "N/A",
        "data_date_range": date_range,
    }


def _build_business_impact(
    binary_stats: dict,
    coverage_curve: dict,
) -> dict:
    """비즈니스 임팩트 딕셔너리 생성."""
    total_fp = binary_stats.get("total", {}).get("fp", 0)

    recommended_tau = coverage_curve.get("recommended_tau")
    curve_df = coverage_curve.get("curve", pd.DataFrame())

    cov_at_tau = 0.0
    if recommended_tau is not None and not curve_df.empty and "tau" in curve_df.columns:
        row = curve_df[curve_df["tau"] == recommended_tau]
        if not row.empty and "coverage" in row.columns:
            cov_at_tau = float(row["coverage"].iloc[0])

    return {
        "total_fp": total_fp,
        "coverage_at_target": round(cov_at_tau, 4),
        "estimated_auto_fp": int(total_fp * cov_at_tau),
        "phase1_goal_40pct_met": cov_at_tau >= 0.40,
    }


def _build_error_risk_summary(
    y_test,
    y_pred_test,
    tp_label: str = "TP",
) -> dict:
    """오분류 위험도 요약 딕셔너리 생성."""
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred_test)

    fp_to_tp_count = int(((y_test_arr != tp_label) & (y_pred_arr == tp_label)).sum())
    tp_to_fp_count = int(((y_test_arr == tp_label) & (y_pred_arr != tp_label)).sum())
    total_errors = int((y_test_arr != y_pred_arr).sum())

    return {
        "fp_to_tp_count": fp_to_tp_count,
        "tp_to_fp_count": tp_to_fp_count,
        "total_errors": total_errors,
        "risk_rate": round(fp_to_tp_count / max(total_errors, 1), 4),
    }


def _build_primary_eval(
    df: pd.DataFrame,
    train_idx: list,
    test_idx: list,
    y_true_all: np.ndarray,
    y_pred_test: np.ndarray,
    tp_label: str = "TP",
    coverage_at_target: float = 0.0,
) -> dict:
    """Primary Split 평가 결과 dict 반환."""
    return {
        "split_name": "Primary",
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "y_true": y_true_all[test_idx],
        "y_pred": y_pred_test,
        "tp_label": tp_label,
        "coverage_at_target": coverage_at_target,
    }


def _build_secondary_eval(
    df: pd.DataFrame,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    tp_label: str = "TP",
    test_months: int = 2,
) -> dict | None:
    """Secondary Split (work_month_time_split) 평가 결과 dict 반환."""
    from src.evaluation.split_strategies import work_month_time_split

    if "label_work_month" not in df.columns:
        logger.info("Secondary Split: 'label_work_month' 컬럼 없음 - 건너뜀")
        return None

    try:
        train_idx, test_idx = work_month_time_split(df, test_months=test_months)
    except Exception as exc:
        logger.warning("Secondary Split 실패: %s - 건너뜀", exc)
        return None

    if len(test_idx) == 0:
        return None

    return {
        "split_name": "Secondary",
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "y_true": y_true_all[test_idx],
        "y_pred": y_pred_all[test_idx],
        "tp_label": tp_label,
        "coverage_at_target": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행 흐름
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(args)

    from src.report.excel_writer import PocExcelWriter, PocReportData
    from src.evaluation.poc_metrics import (
        compute_binary_stats,
        compute_class_imbalance,
        compute_class_metrics,
        compute_confidence_distribution,
        compute_coverage_precision_curve,
        compute_org_stats,
        compute_split_comparison,
    )
    from src.evaluation.rule_analyzer import (
        compute_rule_contribution,
        compute_class_rule_contribution,
        compute_rule_vs_ml_coverage,
    )
    from src.evaluation.evaluator import check_poc_criteria, analyze_errors
    from src.evaluation.split_strategies import group_time_split
    from src.data.label_loader import LabelLoader

    # ── Step 1: 레이블 데이터 로드 ────────────────────────────────────────────
    silver_label = PROJECT_ROOT / "data" / "processed" / "silver_label.parquet"
    if silver_label.exists():
        logger.info("[Step 1] silver_label.parquet 로드")
        df_label = pd.read_parquet(silver_label)
    else:
        logger.info("[Step 1] LabelLoader.load_all() 실행")
        loader = LabelLoader(label_root=args.label_dir)
        df_label = loader.load_all()

    if df_label.empty:
        logger.error("레이블 데이터가 비어 있습니다. data/raw/label/ 디렉토리를 확인하세요.")
        sys.exit(1)

    logger.info("  레이블 데이터: %d건", len(df_label))

    # label_raw -> TP/FP 이진 레이블 생성 (없으면 label_raw 그대로 사용)
    label_col = "label_raw" if "label_raw" in df_label.columns else df_label.columns[-1]
    y_true_all = df_label[label_col].values
    tp_label = "TP"

    # ── Step 2 & 3: ML 모델/피처 로드 (--skip-ml 시 생략) ────────────────────
    model = None
    le = None
    X_all = None
    feature_names = []

    if not args.skip_ml:
        try:
            from src.models.trainer import load_model_with_meta
            _model_dir = Path(args.model_dir)
            _model_path = _model_dir / "final" / "best_model_v1.joblib"
            if not _model_path.exists():
                raise FileNotFoundError(f"모델 없음: {_model_path}")
            logger.info("[Step 2] 모델 파일: %s", _model_path)
            _artifact = load_model_with_meta(str(_model_path))
            model = _artifact.get("model")
            le = _artifact.get("label_encoder")
            meta = _artifact
            logger.info("[Step 2] 모델 로드 완료")
        except Exception as exc:
            logger.warning("[Step 2] 모델 로드 실패 (%s) - Rule 분석만 실행", exc)
            model = None

        if model is not None:
            try:
                import scipy.sparse as _sp
                from src.features.meta_features import build_meta_features
                from src.features.path_features import extract_path_features as _epf
                from src.features.tabular_features import create_file_path_features
                from src.models.feature_builder_snapshot import FeatureBuilderSnapshot

                _snapshot_path = Path(args.model_dir) / "final" / "feature_builder.joblib"
                _df_feat = build_meta_features(df_label.copy())

                # extract_path_features 결과 추가 (FeatureBuilderSnapshot.dense_columns 충족)
                if "file_path" in _df_feat.columns:
                    _path_feats = _df_feat["file_path"].apply(_epf)
                    _path_df = pd.DataFrame(list(_path_feats), index=_df_feat.index)
                    for _col in _path_df.columns:
                        if _col not in _df_feat.columns:
                            _df_feat[_col] = _path_df[_col]

                # Dense 피처: create_file_path_features 결과를 df에 병합
                # (FeatureBuilderSnapshot이 column 이름으로 조회)
                _path_tab = create_file_path_features(_df_feat, path_column="file_path")
                for _c in _path_tab.select_dtypes(include=[np.number]).columns:
                    _df_feat[_c] = _path_tab[_c].values

                # server_freq: 전체 데이터 기준 빈도 (학습 train 기준과 소폭 차이 허용)
                if "server_name" in _df_feat.columns:
                    _sn_freq = _df_feat["server_name"].value_counts(normalize=True)
                    _df_feat["server_freq"] = _df_feat["server_name"].map(_sn_freq).fillna(0)

                if _snapshot_path.exists():
                    # ── 학습 시 TF-IDF vocab 그대로 사용 (vocab 불일치 방지) ──
                    builder = FeatureBuilderSnapshot.load(str(_snapshot_path))
                    X_all = builder.transform(_df_feat)
                    feature_names = builder.feature_names
                    logger.info(
                        "[Step 3] FeatureBuilderSnapshot 변환 완료: shape=%s, n_features=%d",
                        X_all.shape, len(feature_names),
                    )
                else:
                    # fallback: vocab 불일치 가능성 있음 (경고 출력)
                    logger.warning(
                        "[Step 3] feature_builder.joblib 없음 → 피처 재건 (vocab 불일치 가능)"
                    )
                    from src.features.pipeline import build_features
                    _label_col = "label_binary" if "label_binary" in _df_feat.columns else "label_raw"
                    _result = build_features(
                        _df_feat,
                        label_column=_label_col,
                        use_multiview_tfidf=False,
                        use_phase1_tfidf=True,
                        use_synthetic_expansion=False,
                        use_group_split=False,
                    )
                    X_all = _sp.vstack([_result["X_train"], _result["X_test"]])
                    feature_names = _result["feature_names"]
                    logger.info("[Step 3] 피처 재건 완료: %s", X_all.shape)
            except Exception as exc:
                logger.warning("[Step 3] 피처 재건 실패 (%s) - Rule 분석만 실행", exc)
                X_all = None

    # ── Step 4: Primary Split ─────────────────────────────────────────────────
    logger.info("[Step 4] Primary Split (group_time_split)")
    if "detection_time" in df_label.columns:
        train_idx, test_idx = group_time_split(df_label, time_col="detection_time")
    elif "label_work_month" in df_label.columns:
        from src.evaluation.split_strategies import work_month_time_split
        _n_months = df_label["label_work_month"].nunique()
        _test_months = min(2, max(1, _n_months - 1))
        train_idx, test_idx = work_month_time_split(df_label, test_months=_test_months)
    else:
        # fallback: 80/20 split
        n = len(df_label)
        train_idx = list(range(int(n * 0.8)))
        test_idx = list(range(int(n * 0.8), n))

    df_test = df_label.iloc[test_idx].reset_index(drop=True)
    y_test = y_true_all[test_idx]

    # ── Step 5: Rule Labeler ──────────────────────────────────────────────────
    logger.info("[Step 5] RuleLabeler.label_batch()")
    rule_labels_df = pd.DataFrame()
    try:
        from src.filters.rule_labeler import RuleLabeler
        rules_config, rule_stats = _load_rules_config()
        if rules_config:
            labeler = RuleLabeler(rules_config=rules_config, rule_stats=rule_stats)
            rule_result = labeler.label_batch(df_test)
            if isinstance(rule_result, tuple):
                rule_labels_df = rule_result[0]
            else:
                rule_labels_df = rule_result
    except Exception as exc:
        logger.warning("[Step 5] RuleLabeler 실패: %s - Rule 분석 생략", exc)

    # ── Step 6: ML 예측 ───────────────────────────────────────────────────────
    y_pred_test = np.full_like(y_test, fill_value=tp_label)
    y_pred_all = np.full_like(y_true_all, fill_value=tp_label)
    ml_proba_test = np.full(len(y_test), 0.5)

    if model is not None and X_all is not None:
        logger.info("[Step 6] ML 예측")
        try:
            from src.models.trainer import predict_with_uncertainty
            _label_names = list(le.classes_) if le is not None else None
            # 전체 데이터 예측 (Secondary/Tertiary Split에서 슬라이싱)
            ml_df_all = predict_with_uncertainty(model, X_all, label_names=_label_names)
            if "ml_top1_class_name" in ml_df_all.columns:
                y_pred_all = ml_df_all["ml_top1_class_name"].values
                y_pred_test = y_pred_all[test_idx]
                ml_proba_test = ml_df_all["ml_top1_proba"].values[test_idx]
                # Coverage-Precision 커브용 FP 클래스 확률 (P(FP) 전용)
                # top1_proba는 max(P(TP), P(FP))이므로 TP 고신뢰 예측도 포함돼 precision 왜곡
                # FP 클래스 인덱스를 특정하여 P(FP)만 추출
                if le is not None:
                    _fp_cls_list = [c for c in le.classes_ if c != tp_label]
                    if _fp_cls_list:
                        try:
                            _all_proba = model.predict_proba(X_all)
                            _fp_idx = list(le.classes_).index(_fp_cls_list[0])
                            ml_proba_test = _all_proba[test_idx, _fp_idx]
                        except Exception:
                            pass  # fallback: ml_proba_test 그대로 유지
        except Exception as exc:
            logger.warning("[Step 6] ML 예측 실패: %s - 기본 예측 사용", exc)

    # ── Step 7: 핵심 지표 ─────────────────────────────────────────────────────
    logger.info("[Step 7] 핵심 지표 계산")
    poc_criteria = check_poc_criteria(y_test, y_pred_test, tp_label=tp_label)

    binary_stats = compute_binary_stats(df_label, label_col=label_col)
    class_imbalance = compute_class_imbalance(df_label, label_col=label_col)

    coverage_curve = compute_coverage_precision_curve(
        y_test, ml_proba_test,
        precision_target=args.precision_target,
    )

    # 중복 제거 통계
    dedup_before = len(df_label)
    dedup_after = df_label.drop_duplicates(
        subset=[c for c in ["pk_event", "pk_file"] if c in df_label.columns]
    ).shape[0] if any(c in df_label.columns for c in ["pk_event", "pk_file"]) else dedup_before

    # fp_description 분포
    from src.evaluation.data_quality import analyze_fp_description
    fp_desc_stats = analyze_fp_description(df_label, label_col=label_col)

    # Rule 기여도
    rule_contribution = pd.DataFrame()
    class_rule_contribution = pd.DataFrame()
    if not rule_labels_df.empty:
        rule_contribution = compute_rule_contribution(rule_labels_df, y_test, tp_label=tp_label)
        class_rule_contribution = compute_class_rule_contribution(rule_labels_df)

    # 오분류 분석
    error_patterns: list = []
    error_samples = pd.DataFrame()
    try:
        from collections import Counter
        errors_mask = y_test != y_pred_test
        if errors_mask.sum() > 0:
            pairs = list(zip(y_test[errors_mask], y_pred_test[errors_mask]))
            error_patterns = [(a, p, c) for (a, p), c in Counter(pairs).most_common(15)]
            error_df = df_test[errors_mask].copy()
            error_df["actual_class"] = y_test[errors_mask]
            error_df["predicted_class"] = y_pred_test[errors_mask]
            error_samples = error_df.head(200)
    except Exception as exc:
        logger.warning("오분류 분석 실패: %s", exc)

    # 신규 계산 항목
    _model_path_for_meta = None
    if not args.skip_ml and model is not None:
        _model_path_for_meta = Path(args.model_dir) / "final" / "best_model_v1.joblib"
        if not _model_path_for_meta.exists():
            _model_path_for_meta = None

    run_metadata = _build_run_metadata(_model_path_for_meta, df_label)
    business_impact = _build_business_impact(binary_stats, coverage_curve)
    class_metrics = compute_class_metrics(y_test, y_pred_test, tp_label=tp_label)
    org_stats = compute_org_stats(df_label, label_col=label_col)
    confidence_dist = compute_confidence_distribution(ml_proba_test)
    error_risk_summary = _build_error_risk_summary(y_test, y_pred_test, tp_label=tp_label)

    rule_vs_ml = {}
    if not rule_labels_df.empty:
        rule_vs_ml = compute_rule_vs_ml_coverage(rule_labels_df, y_pred_test, y_test, tp_label=tp_label)

    # ── Step 8: Secondary / Tertiary Split ────────────────────────────────────
    logger.info("[Step 8] Secondary / Tertiary Split")
    split_results = [
        _build_primary_eval(
            df_label, train_idx, test_idx, y_true_all, y_pred_test,
            tp_label=tp_label,
            coverage_at_target=coverage_curve.get("recommended_tau") or 0.0,
        )
    ]

    secondary = _build_secondary_eval(df_label, y_true_all, y_pred_all, tp_label=tp_label)
    if secondary is not None:
        split_results.append(secondary)

    _y_pred_all_for_tertiary = y_pred_all if not np.all(y_pred_all == tp_label) else None
    tertiary_results = _build_tertiary_splits(
        df_label, y_true_all, tp_label=tp_label,
        y_pred_all=_y_pred_all_for_tertiary,
    )
    split_results.extend(tertiary_results)

    split_comparison = compute_split_comparison(split_results)

    # ── Step 9: PocReportData 조립 ────────────────────────────────────────────
    logger.info("[Step 9] PocReportData 조립")
    report_data = PocReportData(
        data_condition=_get_data_condition(args.phase),
        split_summary=_make_split_summary(df_label, train_idx, test_idx),
        poc_criteria=poc_criteria,
        binary_stats=binary_stats,
        class_imbalance=class_imbalance,
        dedup_before=dedup_before,
        dedup_after=dedup_after,
        fp_description_stats=fp_desc_stats,
        split_comparison=split_comparison,
        coverage_curve=coverage_curve,
        rule_contribution=rule_contribution,
        class_rule_contribution=class_rule_contribution,
        error_patterns=error_patterns,
        error_samples=error_samples,
        run_metadata=run_metadata,
        business_impact=business_impact,
        org_stats=org_stats,
        class_metrics=class_metrics,
        rule_vs_ml_coverage=rule_vs_ml,
        error_risk_summary=error_risk_summary,
        confidence_distribution=confidence_dist,
    )

    # ── Step 10: Excel 작성 ───────────────────────────────────────────────────
    logger.info("[Step 10] Excel 작성: %s", args.output)
    PocExcelWriter(report_data).write(args.output)
    logger.info("[완료] PoC 리포트 생성: %s", args.output)


if __name__ == "__main__":
    main()
