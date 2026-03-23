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
        "model_file": "joined_best_model_v1.joblib",
        "feature_builder_file": "joined_feature_builder.joblib",
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
    parser.add_argument(
        "--test-months", type=int, default=3,
        dest="test_months",
        help="temporal split 테스트 월 수 (기본: 3, run_training.py와 동일하게 설정)",
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

    figures_dir = output_dir / "figures"
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

    # ── Step 1: 학습 체크포인트에서 모델/피처/split 로드 ──
    # run_training.py가 저장한 체크포인트를 그대로 사용하여
    # 학습과 100% 동일한 피처/split/모델로 평가한다.
    import joblib as _jl

    _sfx = Path(source_cfg["silver_parquet"]).stem  # silver_label or silver_joined
    _ckpt_dir = Path(args.model_dir) / "checkpoints"
    _ckpt5 = _ckpt_dir / f"step5_features_{_sfx}.pkl"
    _ckpt6 = _ckpt_dir / f"step6_model_{_sfx}.pkl"

    # silver parquet은 통계/진단용으로만 로드 (전체 모집단 통계에 필요 — 체크포인트의
    # df_train/df_test는 split 후 일부이고, label_work_month/organization 등이 없을 수 있음)
    silver_path = PROCESSED_DATA_DIR / source_cfg["silver_parquet"]
    if silver_path.exists():
        logger.info("[Step 1a] %s 로드 (통계/진단용)", silver_path.name)
        df_label = pd.read_parquet(silver_path)
    else:
        logger.info("[Step 1a] LabelLoader 실행 (parquet 없음)")
        from src.data.label_loader import LabelLoader
        df_label = LabelLoader(label_root=args.label_dir).load_all()

    if df_label.empty:
        logger.error("데이터가 비어 있습니다.")
        sys.exit(1)

    label_col = "label_raw" if "label_raw" in df_label.columns else df_label.columns[-1]
    y_true_all = df_label[label_col].values
    tp_label = "TP"

    # 체크포인트 로드 (학습과 동일한 피처/split/모델)
    model = None
    le = None
    X_test = None
    X_train = None
    feature_names = []
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()
    y_test = np.array([])
    y_train = np.array([])
    y_train_enc = np.array([])
    y_test_enc = np.array([])
    train_idx = []
    test_idx = []

    if not args.skip_ml and _ckpt5.exists() and _ckpt6.exists():
        logger.info("[Step 1b] 학습 체크포인트 로드 (training과 동일)")
        _d5 = _jl.load(_ckpt5)
        _d6 = _jl.load(_ckpt6)

        result = _d5["result"]
        X_train = result["X_train"]
        X_test = result["X_test"]
        y_train_enc = _d5["y_train_enc"]
        y_test_enc = _d5["y_test_enc"]
        le = _d5["le"]
        feature_names = result.get("feature_names", [])
        df_train = result.get("df_train", pd.DataFrame())
        df_test = result.get("df_test", pd.DataFrame())

        model = _d6["model"]
        f1_from_training = _d6.get("f1", 0.0)

        # y_test/y_train을 문자열로 복원
        y_test = le.inverse_transform(y_test_enc)
        y_train = le.inverse_transform(y_train_enc)

        # train_idx/test_idx 구성 (df_label 기준 인덱스는 아니지만 크기 기반)
        train_idx = list(range(len(y_train)))
        test_idx = list(range(len(y_test)))

        logger.info("  모델: F1=%.4f (training 기록)", f1_from_training)
        logger.info("  X_train: %s  X_test: %s", X_train.shape, X_test.shape)
        logger.info("  피처 수: %d", len(feature_names))
    elif not args.skip_ml:
        # 체크포인트 없으면 models/final에서 모델만 로드 (fallback)
        logger.warning("[Step 1b] 체크포인트 없음 (%s) - models/final에서 모델만 로드", _ckpt5)
        try:
            from src.models.trainer import load_model_with_meta
            model_path = Path(args.model_dir) / "final" / source_cfg["model_file"]
            if not model_path.exists():
                model_path = Path(args.model_dir) / "final" / "best_model_v1.joblib"
            if model_path.exists():
                artifact = load_model_with_meta(str(model_path))
                model = artifact.get("model")
                le = artifact.get("label_encoder")
                logger.warning("  모델 로드됨. 단, 피처 불일치 가능 — 체크포인트 재생성 권장")
                logger.warning("  python scripts/run_training.py --source %s --split temporal --test-months 3", args.source)
        except Exception as exc:
            logger.warning("  모델 로드 실패: %s", exc)
    else:
        logger.info("[Step 1b] --skip-ml: ML 건너뜀")

    logger.info("  전체 데이터: %d건", len(df_label))

    # ── ML 예측 (체크포인트의 X_test 사용) ──
    y_pred_test = np.full(len(y_test), fill_value=tp_label)
    y_pred_all = np.full(len(y_true_all), fill_value=tp_label)
    y_pred_enc = None  # Step 7b에서 재사용
    ml_proba_test = np.full(len(y_test), 0.5)

    if model is not None and X_test is not None:
        logger.info("[Step 2] ML 예측 (체크포인트 X_test)")
        try:
            y_pred_enc = model.predict(X_test)
            y_pred_test = le.inverse_transform(y_pred_enc)

            _proba = model.predict_proba(X_test)
            _fp_cls_list = [c for c in le.classes_ if c != tp_label]
            if _fp_cls_list:
                _fp_idx = list(le.classes_).index(_fp_cls_list[0])
                ml_proba_test = _proba[:, _fp_idx]
            else:
                ml_proba_test = 1.0 - _proba[:, list(le.classes_).index(tp_label)]

            from sklearn.metrics import f1_score as _f1s, classification_report as _cr
            _f1 = _f1s(y_test_enc, y_pred_enc, average="macro")
            logger.info("  F1-macro: %.4f", _f1)
            logger.info("\n%s", _cr(y_test_enc, y_pred_enc, target_names=le.classes_))
        except Exception as exc:
            logger.warning("  ML 예측 실패: %s", exc)

    # ── Step 2b: Decision Combiner 시뮬레이션 ──
    dc_eval_result = {}
    _dc_final_pred = None        # DC 결합 예측 (binary TP/FP) — Step 7 PoC 판정 기준
    if model is not None and X_test is not None and not df_test.empty:
        logger.info("[Step 2b] Decision Combiner 시뮬레이션")
        try:
            from src.models.trainer import predict_with_uncertainty
            from src.models.decision_combiner import combine_decisions
            from sklearn.metrics import (
                confusion_matrix as _cm_fn, f1_score as _dc_f1,
                classification_report as _dc_cr,
            )

            _ml_pred_df = predict_with_uncertainty(
                model, X_test,
                pk_events=(df_test["pk_event"].tolist()
                           if "pk_event" in df_test.columns else None),
                label_names=list(le.classes_),
            )

            # 벡터화 Decision Combiner — decision_combiner.py 임계값과 동일
            from src.models.decision_combiner import _DEFAULT_THRESHOLDS as _DC_THR

            _rule_matched = df_test["rule_matched"].values if "rule_matched" in df_test.columns else np.zeros(len(df_test), dtype=bool)
            _rule_conf = df_test["rule_confidence_lb"].values if "rule_confidence_lb" in df_test.columns else np.zeros(len(df_test))
            _rule_class = df_test["rule_primary_class"].values if "rule_primary_class" in df_test.columns else np.full(len(df_test), "")
            _rule_id = df_test["rule_id"].values if "rule_id" in df_test.columns else np.full(len(df_test), "")

            _ml_class = _ml_pred_df["ml_top1_class_name"].values if "ml_top1_class_name" in _ml_pred_df.columns else np.full(len(df_test), "FP")
            _ml_tp_proba = _ml_pred_df["ml_tp_proba"].values if "ml_tp_proba" in _ml_pred_df.columns else np.zeros(len(df_test))
            _ml_top1_proba = _ml_pred_df["ml_top1_proba"].values if "ml_top1_proba" in _ml_pred_df.columns else np.zeros(len(df_test))
            _ml_margin = _ml_pred_df["ml_margin"].values if "ml_margin" in _ml_pred_df.columns else np.zeros(len(df_test))
            _ml_entropy = _ml_pred_df["ml_entropy"].values if "ml_entropy" in _ml_pred_df.columns else np.zeros(len(df_test))
            _ood_flag = _ml_pred_df["ood_flag"].values if "ood_flag" in _ml_pred_df.columns else np.zeros(len(df_test), dtype=bool)

            # 벡터화 판정: decision_combiner.py 4-Case 로직과 동일
            _n = len(df_test)
            _dc_primary = np.empty(_n, dtype=object)
            _dc_source = np.empty(_n, dtype=object)
            _dc_reason = np.empty(_n, dtype=object)
            # 기본값: Case 3 Fallback (TP 안전)
            _dc_primary[:] = "TP"
            _dc_source[:] = "FALLBACK"
            _dc_reason[:] = "TP_FALLBACK"

            _rule_hit = np.array(_rule_matched, dtype=bool)
            _rule_conf_arr = np.array(_rule_conf, dtype=float)
            _ood_arr = np.array(_ood_flag, dtype=bool)

            # Case 0: OOD → UNKNOWN
            _case0 = _ood_arr
            _dc_primary[_case0] = "UNKNOWN"
            _dc_source[_case0] = "OOD"
            _dc_reason[_case0] = "OOD"

            # Case 0b: 고엔트로피 + 룰 없음 → UNKNOWN
            _case0b = (~_case0) & (_ml_entropy >= _DC_THR["entropy_unknown"]) & (~_rule_hit)
            _dc_primary[_case0b] = "UNKNOWN"
            _dc_source[_case0b] = "OOD"
            _dc_reason[_case0b] = "HIGH_ENTROPY"

            # Case 1: RULE 매칭 + 고확신 → RULE 결과 사용
            _case1 = (~_case0) & (~_case0b) & _rule_hit & (_rule_conf_arr >= _DC_THR["rule_conf"])
            _dc_primary[_case1] = _rule_class[_case1]
            _dc_source[_case1] = "RULE"
            _dc_reason[_case1] = "RULE_HIGH_CONFIDENCE"

            # Case 1 TP override: RULE=FP인데 ML이 TP 강하게 주장 → TP 안전
            _is_rule_fp = np.array(["FP" in str(c) for c in _rule_class])
            _tp_override = _case1 & _is_rule_fp & (_ml_tp_proba >= _DC_THR["ml_tp_proba_override"])
            _dc_primary[_tp_override] = "TP"
            _dc_source[_tp_override] = "ML_OVERRIDE"
            _dc_reason[_tp_override] = "TP_SAFETY"

            # Case 2: ML 고확신 → ML 결과 사용
            _remaining = (~_case0) & (~_case0b) & (~_case1)
            _ml_confident = _remaining & (_ml_top1_proba >= _DC_THR["ml_conf"]) & (_ml_margin >= _DC_THR["ml_margin"])
            _dc_primary[_ml_confident] = _ml_class[_ml_confident]
            _dc_source[_ml_confident] = "ML"
            _dc_reason[_ml_confident] = "ML_CONFIDENT"

            # Case 2b: ML 애매 + TP 신호 → TP 안전
            _ambiguous = _remaining & (~_ml_confident) & (_ml_tp_proba >= _DC_THR["ml_tp_proba_ambiguous"])
            _dc_primary[_ambiguous] = "TP"
            _dc_source[_ambiguous] = "ML_TP_OVERRIDE"
            _dc_reason[_ambiguous] = "TP_SAFETY"

            # Case 3: 나머지 → 기본값(TP Fallback) 유지

            _dc_df = pd.DataFrame({
                "primary_class": _dc_primary,
                "decision_source": _dc_source,
                "reason_code": _dc_reason,
            })

            # binary collapse: TP vs FP
            _dc_pred_bin = np.array([
                "TP" if "TP" in str(c) else "FP"
                for c in _dc_df["primary_class"]
            ])
            _dc_true_bin = np.array([
                "TP" if "TP" in str(c) else "FP"
                for c in y_test
            ])

            _dc_f1_val = float(_dc_f1(_dc_true_bin, _dc_pred_bin, average="macro"))
            _dc_cm = _cm_fn(_dc_true_bin, _dc_pred_bin, labels=["FP", "TP"])

            # Case 분포
            _dc_source_counts = _dc_df["decision_source"].value_counts().to_dict()
            _dc_reason_counts = _dc_df["reason_code"].value_counts().to_dict()

            # ML 단독 F1 (binary)
            _ml_pred_bin = np.array([
                "TP" if "TP" in str(c) else "FP"
                for c in y_pred_test
            ])
            _ml_f1_val = float(_dc_f1(_dc_true_bin, _ml_pred_bin, average="macro"))

            # DC 결합 예측을 PoC 판정 기준으로 사용
            _dc_final_pred = _dc_pred_bin

            dc_eval_result = {
                "dc_f1": _dc_f1_val,
                "ml_f1": _ml_f1_val,
                "confusion_matrix": pd.DataFrame(
                    _dc_cm, index=["실제FP", "실제TP"], columns=["예측FP", "예측TP"]
                ),
                "decision_source_dist": _dc_source_counts,
                "reason_code_dist": _dc_reason_counts,
                "total_samples": len(_dc_df),
            }

            logger.info("  Decision Combiner F1-macro: %.4f (ML 단독: %.4f)",
                         _dc_f1_val, _ml_f1_val)
            logger.info("  Confusion Matrix:\n%s",
                         dc_eval_result["confusion_matrix"].to_string())
            logger.info("  Decision Source: %s", _dc_source_counts)
        except Exception as _dc_exc:
            logger.warning("  Decision Combiner 시뮬레이션 실패: %s", _dc_exc)

    # ── Rule 컬럼 확인 (training Step 4에서 이미 적용됨 → 재실행 불필요) ──
    logger.info("[Step 3] Rule 컬럼 확인 (체크포인트)")
    _rule_cols_needed = ["rule_matched", "rule_id", "rule_primary_class"]
    if not df_test.empty and all(c in df_test.columns for c in _rule_cols_needed):
        rule_labels_df = df_test[_rule_cols_needed].copy()
        _n_matched = int(rule_labels_df["rule_matched"].sum())
        logger.info("  체크포인트에서 Rule 컬럼 로드 완료 (matched=%d건)", _n_matched)
    else:
        rule_labels_df = pd.DataFrame()
        logger.info("  Rule 컬럼 없음 (rule_matched/rule_id/rule_primary_class)")

    # ── Step 7: 핵심 지표 ──
    # PoC 판정 기준: ML + Rule 결합(Decision Combiner) 결과
    # DC 시뮬레이션 성공 시 _dc_final_pred 사용, 실패 시 ML 단독 fallback
    if _dc_final_pred is not None:
        logger.info("[Step 7] 핵심 지표 (기준: ML + Rule 결합)")
        # DC 결과는 binary ("TP"/"FP"), y_test도 binary로 맞춤
        _y_test_bin = np.array([
            "TP" if "TP" in str(c) else "FP" for c in y_test
        ])
        poc_criteria = check_poc_criteria(
            _y_test_bin, _dc_final_pred, tp_label="TP",
        )
    else:
        logger.info("[Step 7] 핵심 지표 (기준: ML 단독 — DC 미적용)")
        poc_criteria = check_poc_criteria(y_test, y_pred_test, tp_label=tp_label)
    binary_stats = compute_binary_stats(df_label, label_col=label_col)
    class_imbalance = compute_class_imbalance(df_label, label_col=label_col)
    # Coverage-Precision Curve: training이 저장한 threshold_policy.json 우선 사용
    coverage_curve = {}
    _tp_path = Path(args.model_dir) / "final" / "threshold_policy.json"
    if _tp_path.exists():
        try:
            import json as _json_tp
            with open(_tp_path, "r", encoding="utf-8") as _f_tp:
                _tp_data = _json_tp.load(_f_tp)
            coverage_curve = {
                "recommended_tau": _tp_data.get("recommended_fp_tau"),
                "curve": pd.DataFrame(_tp_data.get("curve_summary", [])),
            }
            logger.info("  threshold_policy.json 로드 (tau=%.2f)",
                         coverage_curve.get("recommended_tau") or 0.0)
        except Exception as _e_tp:
            logger.warning("  threshold_policy.json 로드 실패: %s", _e_tp)
    if not coverage_curve:
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

    # 오분류 분석 — DC 결합 결과 기준 (없으면 ML 단독 fallback)
    error_patterns = []
    error_samples = pd.DataFrame()
    try:
        from collections import Counter
        if _dc_final_pred is not None:
            _y_test_bin_ea = np.array([
                "TP" if "TP" in str(c) else "FP" for c in y_test
            ])
            errors_mask = _y_test_bin_ea != _dc_final_pred
        else:
            errors_mask = y_test != y_pred_test
        if errors_mask.sum() > 0:
            _ea_true = _y_test_bin_ea if _dc_final_pred is not None else y_test
            _ea_pred = _dc_final_pred if _dc_final_pred is not None else y_pred_test
            pairs = list(zip(_ea_true[errors_mask], _ea_pred[errors_mask]))
            error_patterns = [(a, p, c) for (a, p), c in Counter(pairs).most_common(15)]
            edf = df_test[errors_mask].copy()
            edf["actual_class"] = _ea_true[errors_mask]
            edf["predicted_class"] = _ea_pred[errors_mask]
            error_samples = edf.head(200)
    except Exception:
        pass

    # class_metrics는 ML 단독 기준 유지 (DC 결합과 별도로 ML 성능 추적용)
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
    if model is not None and le is not None and y_pred_enc is not None:
        _generate_eval_artifacts(
            model, y_test_enc, y_pred_enc, le, feature_names,
            df_test, REPORT_DIR,
        )
        fi_df, fi_groups = _build_feature_importance_data(model, feature_names)

    # ── Step 8: Split 비교 ──
    logger.info("[Step 8] Split 비교")

    from scripts.run_poc_report import (
        _build_primary_eval,
        _build_tertiary_splits, _build_run_metadata,
        _build_business_impact, _build_error_risk_summary,
        _make_split_summary,
    )

    # 체크포인트 기반: train_idx/test_idx는 df_label 인덱스가 아닌 크기 기반
    # _build_primary_eval 등에 전달할 때 df_label 기준 인덱스가 필요한 경우 대비
    _n_train = len(y_train) if len(y_train) > 0 else int(len(df_label) * 0.8)
    _n_test = len(y_test) if len(y_test) > 0 else len(df_label) - _n_train
    _train_idx_for_report = list(range(_n_train))
    _test_idx_for_report = list(range(_n_train, _n_train + _n_test))

    # Primary split 평가: DC 결합 결과가 있으면 DC 기준, 없으면 ML 단독
    _primary_y_pred = y_pred_test
    _primary_tp_label = tp_label
    if _dc_final_pred is not None:
        # DC 결과(binary TP/FP)와 정답을 binary로 맞춤
        _primary_y_pred = _dc_final_pred
        _primary_tp_label = "TP"
        # y_true_all도 binary로 변환 (test 구간)
        y_true_all = np.array([
            "TP" if "TP" in str(c) else "FP" for c in y_true_all
        ])

    split_results = [
        _build_primary_eval(
            df_label, _train_idx_for_report, _test_idx_for_report,
            y_true_all, _primary_y_pred,
            tp_label=_primary_tp_label,
            coverage_at_target=coverage_curve.get("recommended_tau") or 0.0,
        )
    ]
    # secondary/tertiary splits: 전체 예측이 없으므로 test 기반만 사용
    split_results.extend(
        _build_tertiary_splits(df_label, y_true_all, tp_label=tp_label, y_pred_all=None)
    )
    split_comparison = compute_split_comparison(split_results)

    model_path_for_meta = Path(args.model_dir) / "final" / source_cfg["model_file"]
    if not model_path_for_meta.exists():
        model_path_for_meta = None
    run_metadata = _build_run_metadata(model_path_for_meta, df_label)
    run_metadata["note"] = "체크포인트 기반 평가 (run_training.py와 동일 피처/split)"
    business_impact = _build_business_impact(binary_stats, coverage_curve)
    # 오분류 위험도: DC 결합 결과 기준 (없으면 ML 단독 fallback)
    if _dc_final_pred is not None:
        _y_test_bin_err = np.array([
            "TP" if "TP" in str(c) else "FP" for c in y_test
        ])
        error_risk_summary = _build_error_risk_summary(
            _y_test_bin_err, _dc_final_pred, tp_label="TP",
        )
    else:
        error_risk_summary = _build_error_risk_summary(
            y_test, y_pred_test, tp_label=tp_label,
        )

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
        split_summary=_make_split_summary(df_label, _train_idx_for_report, _test_idx_for_report),
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
        # Sheet 10 - Decision Combiner
        dc_eval_result=dc_eval_result,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PocExcelWriter(report_data).write(output_path)

    logger.info("[완료] 통합 리포트: %s", output_path)


if __name__ == "__main__":
    main()
