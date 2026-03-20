"""통합 리포트 생성 스크립트

run_evaluation.py + run_poc_report.py + diagnose_data_bias.py를 하나로 통합.
학습 완료 후 이 스크립트 하나로 모든 평가/분석/리포트를 생성한다.

사용법:
    # Label 모델 리포트 (기본)
    python scripts/run_report.py --source label

    # Joined 모델 리포트
    python scripts/run_report.py --source detection

    # 진단 포함
    python scripts/run_report.py --source label --include-diagnosis

    # 진단 제외 (빠른 실행)
    python scripts/run_report.py --source label --skip-diagnosis

산출물:
    outputs/
      poc_report.xlsx                 <- 통합 Excel (9-sheet)
      classification_report.txt      <- 텍스트 분류 리포트
      confusion_matrix.png           <- 혼동 행렬
      feature_importance.csv         <- 전체 피처 Importance
      feature_importance.png         <- Importance 바 차트
      error_analysis.csv             <- 오분류 샘플
      diagnosis/                     <- 진단 상세 (--include-diagnosis)
"""
import sys
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 모델/데이터 경로 결정
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_CONFIG = {
    "label": {
        "model_file": "best_model_v1.joblib",
        "feature_builder_file": "feature_builder.joblib",
        "silver_parquet": "silver_label.parquet",
        "output_file": "poc_report.xlsx",
        "data_condition": "Label Only",
    },
    "detection": {
        "model_file": "detection_best_model_v1.joblib",
        "feature_builder_file": "detection_feature_builder.joblib",
        "silver_parquet": "silver_joined.parquet",
        "output_file": "poc_report_detection.xlsx",
        "data_condition": "Label + Sumologic",
    },
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="통합 리포트 생성 (평가 + 분석 + 진단)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=["label", "detection"], default="label",
        help="데이터 소스 (label: silver_label, detection: silver_joined)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="출력 Excel 경로 (기본: source별 자동 결정)",
    )
    parser.add_argument(
        "--precision-target", type=float, default=0.95,
        dest="precision_target",
        help="Coverage-Precision 목표 Precision (기본: 0.95)",
    )
    parser.add_argument(
        "--skip-ml", action="store_true", default=False,
        dest="skip_ml",
        help="ML 모델 없을 때 Rule 분석만 실행",
    )
    parser.add_argument(
        "--include-diagnosis", action="store_true", default=False,
        dest="include_diagnosis",
        help="데이터 진단 포함 (Split Robustness, Ablation 등)",
    )
    parser.add_argument(
        "--skip-diagnosis", action="store_true", default=False,
        dest="skip_diagnosis",
        help="데이터 진단 제외 (빠른 실행)",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        dest="model_dir",
        help="모델 디렉토리 (기본: models)",
    )
    parser.add_argument(
        "--label-dir", type=Path, default=Path("data/raw/label"),
        dest="label_dir",
        help="레이블 Excel 디렉토리 (기본: data/raw/label)",
    )
    return parser.parse_args(args)


# ─────────────────────────────────────────────────────────────────────────────
# eval 고유 산출물 생성
# ─────────────────────────────────────────────────────────────────────────────

def _generate_eval_artifacts(
    model, y_test_enc, y_pred, le, feature_names, df_test,
    output_dir: Path,
):
    """confusion_matrix.png, feature_importance.csv/png, error_analysis.csv 생성."""
    from src.evaluation.evaluator import (
        full_evaluation, analyze_errors, feature_importance_analysis,
    )
    from src.utils.constants import TOP_N_FEATURES, TEXT_COLUMN
    from src.utils.common import ensure_dirs

    figures_dir = output_dir.parent / "figures"
    ensure_dirs(output_dir, figures_dir)

    # Classification Report + Confusion Matrix
    eval_results = full_evaluation(
        y_test_enc, y_pred, list(le.classes_), str(output_dir),
    )

    # Feature Importance
    fi_df = pd.DataFrame()
    if hasattr(model, "feature_importances_") and feature_names:
        fi_df = feature_importance_analysis(
            model, feature_names, TOP_N_FEATURES,
            str(figures_dir / "feature_importance.png"),
            str(output_dir / "feature_importance.csv"),
        )

    # Error Analysis
    analyze_errors(
        y_test_enc, y_pred, df_test, list(le.classes_),
        TEXT_COLUMN,
        save_path=str(output_dir / "error_analysis.csv"),
    )

    return eval_results, fi_df


def _build_feature_importance_data(model, feature_names):
    """Feature importance DataFrame + 그룹별 합산."""
    if not hasattr(model, "feature_importances_") or not feature_names:
        return pd.DataFrame(), {}

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # 그룹별 합산
    groups = {}
    for fname, imp in zip(feature_names, importances):
        if "server" in fname.lower():
            g = "Server"
        elif fname.startswith("tfidf_phase1path") or fname.startswith("tfidf_path"):
            g = "Path TF-IDF"
        elif fname.startswith("tfidf_fname") or fname.startswith("tfidf_raw") or fname.startswith("tfidf_shape"):
            g = "Text TF-IDF"
        elif fname in ("created_hour", "created_weekday", "is_weekend", "created_month"):
            g = "Time"
        elif fname in ("pattern_count_log1p", "pattern_count_bin",
                       "is_mass_detection", "is_extreme_detection", "pii_type_ratio"):
            g = "Detection Stats"
        elif fname.startswith("path_") or fname.startswith("is_log") or fname.startswith("has_") or fname.startswith("is_"):
            g = "Path Flags"
        elif fname.startswith("fname_"):
            g = "Filename"
        else:
            g = "Other"
        groups[g] = groups.get(g, 0) + imp

    return fi_df, groups


# ─────────────────────────────────────────────────────────────────────────────
# 진단 실행
# ─────────────────────────────────────────────────────────────────────────────

def _run_diagnosis(df_label, label_col, output_dir):
    """diagnose_data_bias.py의 핵심 로직을 호출."""
    from scripts.diagnose_data_bias import (
        diagnose_categorical,
        diagnose_rows,
        diagnose_split_robustness,
        diagnose_ablation,
        diagnose_column_registry,
        OUTPUT_DIR,
    )
    import io

    diag_dir = output_dir / "diagnosis"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 임시로 OUTPUT_DIR을 재지정
    import scripts.diagnose_data_bias as _diag_mod
    _orig_dir = _diag_mod.OUTPUT_DIR
    _diag_mod.OUTPUT_DIR = diag_dir

    results = {
        "column_risk_registry": pd.DataFrame(),
        "split_robustness": pd.DataFrame(),
        "ablation_results": pd.DataFrame(),
    }

    try:
        # Column Registry
        f = io.StringIO()
        diagnose_column_registry(df_label, f)

        registry_path = diag_dir / "column_risk_registry.csv"
        if registry_path.exists():
            results["column_risk_registry"] = pd.read_csv(
                registry_path, encoding="utf-8-sig"
            )

        # Split Robustness
        f = io.StringIO()
        diagnose_split_robustness(df_label, f)

        robustness_path = diag_dir / "split_robustness_report.csv"
        if robustness_path.exists():
            results["split_robustness"] = pd.read_csv(
                robustness_path, encoding="utf-8-sig"
            )

        # Ablation
        f = io.StringIO()
        diagnose_ablation(df_label, f)

        ablation_path = diag_dir / "ablation_report.csv"
        if ablation_path.exists():
            results["ablation_results"] = pd.read_csv(
                ablation_path, encoding="utf-8-sig"
            )

    except Exception as exc:
        logger.warning("진단 실행 중 오류: %s", exc)
    finally:
        _diag_mod.OUTPUT_DIR = _orig_dir

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(args)

    source_cfg = _SOURCE_CONFIG[args.source]
    output_path = args.output or (PROJECT_ROOT / "outputs" / source_cfg["output_file"])

    # ── 기존 run_poc_report.py 로직 재사용 ──
    # run_poc_report.py의 main()에 source 인식 기능을 추가하여 호출
    # 대신 직접 조립하여 eval 산출물 + 진단 결과도 포함

    from src.report.excel_writer import PocExcelWriter, PocReportData
    from src.evaluation.poc_metrics import (
        compute_binary_stats, compute_class_imbalance,
        compute_class_metrics, compute_confidence_distribution,
        compute_coverage_precision_curve, compute_org_stats,
        compute_split_comparison,
    )
    from src.evaluation.rule_analyzer import (
        compute_rule_contribution, compute_class_rule_contribution,
        compute_rule_vs_ml_coverage,
    )
    from src.evaluation.evaluator import check_poc_criteria
    from src.evaluation.split_strategies import group_time_split
    from src.utils.constants import PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR
    from src.utils.common import ensure_dirs

    ensure_dirs(REPORT_DIR)

    # ── Step 1: 데이터 로드 ──
    silver_path = PROCESSED_DATA_DIR / source_cfg["silver_parquet"]
    if silver_path.exists():
        logger.info("[Step 1] %s 로드", silver_path.name)
        df_label = pd.read_parquet(silver_path)
    else:
        logger.info("[Step 1] LabelLoader 실행 (parquet 없음)")
        from src.data.label_loader import LabelLoader
        df_label = LabelLoader(label_root=args.label_dir).load_all()

    if df_label.empty:
        logger.error("데이터가 비어 있습니다.")
        sys.exit(1)

    logger.info("  데이터: %d건", len(df_label))

    label_col = "label_raw" if "label_raw" in df_label.columns else df_label.columns[-1]
    y_true_all = df_label[label_col].values
    tp_label = "TP"

    # ── Step 2: 모델 로드 ──
    model = None
    le = None
    X_all = None
    feature_names = []

    if not args.skip_ml:
        try:
            from src.models.trainer import load_model_with_meta
            model_path = Path(args.model_dir) / "final" / source_cfg["model_file"]
            if not model_path.exists():
                # fallback: label 모델 시도
                model_path = Path(args.model_dir) / "final" / "best_model_v1.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"모델 없음: {model_path}")
            logger.info("[Step 2] 모델: %s", model_path.name)
            artifact = load_model_with_meta(str(model_path))
            model = artifact.get("model")
            le = artifact.get("label_encoder")
        except Exception as exc:
            logger.warning("[Step 2] 모델 로드 실패 (%s) - Rule 분석만 실행", exc)

    # ── Step 3: 피처 변환 ──
    if model is not None:
        try:
            import scipy.sparse as _sp
            from src.features.meta_features import build_meta_features
            from src.features.path_features import extract_path_features as _epf
            from src.features.tabular_features import create_file_path_features
            from src.models.feature_builder_snapshot import FeatureBuilderSnapshot

            snapshot_path = Path(args.model_dir) / "final" / source_cfg["feature_builder_file"]
            if not snapshot_path.exists():
                snapshot_path = Path(args.model_dir) / "final" / "feature_builder.joblib"

            _df_feat = build_meta_features(df_label.copy())

            if "file_path" in _df_feat.columns:
                _path_feats = _df_feat["file_path"].apply(_epf)
                _path_df = pd.DataFrame(list(_path_feats), index=_df_feat.index)
                for _col in _path_df.columns:
                    if _col not in _df_feat.columns:
                        _df_feat[_col] = _path_df[_col]

            _path_tab = create_file_path_features(_df_feat, path_column="file_path")
            for _c in _path_tab.select_dtypes(include=[np.number]).columns:
                _df_feat[_c] = _path_tab[_c].values

            # server_freq: train-only 통계 사용이 이상적이나,
            # 현재는 전체 데이터 기준 (B2 버그 - 후속 수정 대상)
            if "server_name" in _df_feat.columns:
                _sn_freq = _df_feat["server_name"].value_counts(normalize=True)
                _df_feat["server_freq"] = _df_feat["server_name"].map(_sn_freq).fillna(0)

            if snapshot_path.exists():
                builder = FeatureBuilderSnapshot.load(str(snapshot_path))
                X_all = builder.transform(_df_feat)
                feature_names = builder.feature_names
                logger.info("[Step 3] FeatureBuilderSnapshot 변환: %s", X_all.shape)
            else:
                logger.warning("[Step 3] snapshot 없음 - 피처 재건")
                from src.features.pipeline import build_features
                _result = build_features(
                    _df_feat,
                    label_column=label_col,
                    use_multiview_tfidf=False,
                    use_phase1_tfidf=True,
                    use_synthetic_expansion=False,
                    use_group_split=False,
                )
                X_all = _sp.vstack([_result["X_train"], _result["X_test"]])
                feature_names = _result["feature_names"]
        except Exception as exc:
            logger.warning("[Step 3] 피처 변환 실패 (%s)", exc)
            X_all = None

    # ── Step 4: Split ──
    logger.info("[Step 4] Primary Split")
    if "label_work_month" in df_label.columns:
        from src.evaluation.split_strategies import work_month_time_split
        _n_months = df_label["label_work_month"].nunique()
        _test_months = min(2, max(1, _n_months - 1))
        train_idx, test_idx = work_month_time_split(df_label, test_months=_test_months)
    elif "detection_time" in df_label.columns:
        train_idx, test_idx = group_time_split(df_label, time_col="detection_time")
    else:
        n = len(df_label)
        train_idx = list(range(int(n * 0.8)))
        test_idx = list(range(int(n * 0.8), n))

    df_test = df_label.iloc[test_idx].reset_index(drop=True)
    y_test = y_true_all[test_idx]
    logger.info("  train=%d, test=%d", len(train_idx), len(test_idx))

    # ── Step 5: Rule Labeler ──
    logger.info("[Step 5] RuleLabeler")
    rule_labels_df = pd.DataFrame()
    try:
        from src.filters.rule_labeler import RuleLabeler
        rules_yaml = PROJECT_ROOT / "config" / "rules.yaml"
        rule_stats = PROJECT_ROOT / "config" / "rule_stats.json"
        if rules_yaml.exists() and rule_stats.exists():
            labeler = RuleLabeler.from_config_files(str(rules_yaml), str(rule_stats))
            result = labeler.label_batch(df_test)
            rule_labels_df = result[0] if isinstance(result, tuple) else result
    except Exception as exc:
        logger.warning("[Step 5] RuleLabeler 실패: %s", exc)

    # ── Step 6: ML 예측 ──
    y_pred_test = np.full_like(y_test, fill_value=tp_label)
    y_pred_all = np.full_like(y_true_all, fill_value=tp_label)
    ml_proba_test = np.full(len(y_test), 0.5)

    if model is not None and X_all is not None:
        logger.info("[Step 6] ML 예측")
        try:
            from src.models.trainer import predict_with_uncertainty
            _label_names = list(le.classes_) if le is not None else None
            ml_df_all = predict_with_uncertainty(model, X_all, label_names=_label_names)
            if "ml_top1_class_name" in ml_df_all.columns:
                y_pred_all = ml_df_all["ml_top1_class_name"].values
                y_pred_test = y_pred_all[test_idx]
                ml_proba_test = ml_df_all["ml_top1_proba"].values[test_idx]
                if le is not None:
                    _fp_cls_list = [c for c in le.classes_ if c != tp_label]
                    if _fp_cls_list:
                        try:
                            _all_proba = model.predict_proba(X_all)
                            _fp_idx = list(le.classes_).index(_fp_cls_list[0])
                            ml_proba_test = _all_proba[test_idx, _fp_idx]
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("[Step 6] ML 예측 실패: %s", exc)

    # ── Step 7: 핵심 지표 ──
    logger.info("[Step 7] 핵심 지표")
    poc_criteria = check_poc_criteria(y_test, y_pred_test, tp_label=tp_label)
    binary_stats = compute_binary_stats(df_label, label_col=label_col)
    class_imbalance = compute_class_imbalance(df_label, label_col=label_col)
    coverage_curve = compute_coverage_precision_curve(
        y_test, ml_proba_test, precision_target=args.precision_target,
    )

    dedup_before = len(df_label)
    pk_cols = [c for c in ["pk_event", "pk_file"] if c in df_label.columns]
    dedup_after = df_label.drop_duplicates(subset=pk_cols).shape[0] if pk_cols else dedup_before

    from src.evaluation.data_quality import analyze_fp_description
    fp_desc_stats = analyze_fp_description(df_label, label_col=label_col)

    rule_contribution = pd.DataFrame()
    class_rule_contribution = pd.DataFrame()
    if not rule_labels_df.empty:
        rule_contribution = compute_rule_contribution(rule_labels_df, y_test, tp_label=tp_label)
        class_rule_contribution = compute_class_rule_contribution(rule_labels_df)

    # 오분류 분석
    error_patterns = []
    error_samples = pd.DataFrame()
    try:
        from collections import Counter
        errors_mask = y_test != y_pred_test
        if errors_mask.sum() > 0:
            pairs = list(zip(y_test[errors_mask], y_pred_test[errors_mask]))
            error_patterns = [(a, p, c) for (a, p), c in Counter(pairs).most_common(15)]
            edf = df_test[errors_mask].copy()
            edf["actual_class"] = y_test[errors_mask]
            edf["predicted_class"] = y_pred_test[errors_mask]
            error_samples = edf.head(200)
    except Exception:
        pass

    class_metrics = compute_class_metrics(y_test, y_pred_test, tp_label=tp_label)
    org_stats = compute_org_stats(df_label, label_col=label_col)
    confidence_dist = compute_confidence_distribution(ml_proba_test)

    rule_vs_ml = {}
    if not rule_labels_df.empty:
        rule_vs_ml = compute_rule_vs_ml_coverage(rule_labels_df, y_pred_test, y_test, tp_label=tp_label)

    # ── Step 7b: eval 고유 산출물 (confusion matrix, feature importance 등) ──
    logger.info("[Step 7b] 평가 산출물 생성")
    fi_df = pd.DataFrame()
    fi_groups = {}
    if model is not None and le is not None and X_all is not None:
        X_test_eval = X_all[test_idx] if hasattr(X_all, '__getitem__') else None
        if X_test_eval is not None:
            y_test_enc = le.transform(y_test)
            y_pred_enc = model.predict(X_test_eval)
            _generate_eval_artifacts(
                model, y_test_enc, y_pred_enc, le, feature_names,
                df_test, REPORT_DIR,
            )
        fi_df, fi_groups = _build_feature_importance_data(model, feature_names)

    # ── Step 8: Split 비교 ──
    logger.info("[Step 8] Split 비교")

    from scripts.run_poc_report import (
        _build_primary_eval, _build_secondary_eval,
        _build_tertiary_splits, _build_run_metadata,
        _build_business_impact, _build_error_risk_summary,
        _make_split_summary,
    )

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
    _y_pred_for_tert = y_pred_all if not np.all(y_pred_all == tp_label) else None
    split_results.extend(
        _build_tertiary_splits(df_label, y_true_all, tp_label=tp_label, y_pred_all=_y_pred_for_tert)
    )
    split_comparison = compute_split_comparison(split_results)

    model_path_for_meta = Path(args.model_dir) / "final" / source_cfg["model_file"]
    if not model_path_for_meta.exists():
        model_path_for_meta = None
    run_metadata = _build_run_metadata(model_path_for_meta, df_label)
    business_impact = _build_business_impact(binary_stats, coverage_curve)
    error_risk_summary = _build_error_risk_summary(y_test, y_pred_test, tp_label=tp_label)

    # ── Step 9: 진단 (선택) ──
    diag_results = {
        "column_risk_registry": pd.DataFrame(),
        "split_robustness": pd.DataFrame(),
        "ablation_results": pd.DataFrame(),
    }

    do_diagnosis = args.include_diagnosis and not args.skip_diagnosis
    if do_diagnosis:
        logger.info("[Step 9] 데이터 진단")
        diag_results = _run_diagnosis(df_label, label_col, REPORT_DIR)
    else:
        logger.info("[Step 9] 진단 생략 (--include-diagnosis로 활성화)")

    # ── Step 10: PocReportData 조립 + Excel 생성 ──
    logger.info("[Step 10] 리포트 조립")

    report_data = PocReportData(
        data_condition=source_cfg["data_condition"],
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
        # Sheet 8 & 9 (신규)
        feature_importance_df=fi_df,
        feature_group_importance=fi_groups,
        column_risk_registry=diag_results["column_risk_registry"],
        split_robustness=diag_results["split_robustness"],
        ablation_results=diag_results["ablation_results"],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PocExcelWriter(report_data).write(output_path)

    logger.info("[완료] 통합 리포트: %s", output_path)


if __name__ == "__main__":
    main()
