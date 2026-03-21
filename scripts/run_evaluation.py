"""모델 평가 실행 스크립트

회의록 2026-01 반영:
- 3-Layer Filter 통합 평가
- Layer별 성능 분석
- PoC 성공 기준 판정

사용법:
    python scripts/run_evaluation.py [--include-filtered]

실행 순서:
    1. 최종 모델 로드 (models/final/best_model_v1.joblib)
    2. Feature & 레이블 로드 (data/features/)
    3. 예측 수행
    4. 전체 평가 (Classification Report, Confusion Matrix, PoC 판정)
    5. Feature Importance 분석
    6. 오분류 패턴 분석
    7. 3-Layer 통합 성능 분석 (선택)
    8. 결과 저장

출력:
    outputs/classification_report.txt
    outputs/error_analysis.csv
    outputs/feature_importance.csv
    outputs/layer_performance.txt
    outputs/confusion_matrix.png
    outputs/feature_importance.png
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Make imports work even if the script is executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.utils.common import ensure_dirs
from src.utils.constants import (
    MODEL_DIR, FINAL_MODEL_DIR, FEATURE_DIR, REPORT_DIR, FIGURES_DIR,
    TEXT_COLUMN, TOP_N_FEATURES, PROCESSED_DATA_DIR,
    POC_F1_MACRO_THRESHOLD, POC_TP_RECALL_THRESHOLD, POC_FP_PRECISION_THRESHOLD,
)
from src.models.trainer import load_model_with_meta
from src.features.pipeline import load_feature_artifacts
from src.evaluation.evaluator import (
    full_evaluation, analyze_errors, feature_importance_analysis,
)


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="모델 평가 실행")
    parser.add_argument(
        "--include-filtered",
        action="store_true",
        help="필터 결과까지 포함한 통합 평가",
    )
    parser.add_argument(
        "--stage",
        choices=["s6", "all"],
        default="all",
        help="실행 단계: s6(KPI+평가), all(전체, 기본값)",
    )
    return parser.parse_args()


def evaluate_filter_performance(
    df_keyword: pd.DataFrame,
    df_rule: pd.DataFrame,
    label_column: str = "label",
) -> dict:
    """
    필터 레이어별 성능 평가

    Args:
        df_keyword: 키워드 필터 결과 DataFrame
        df_rule: 룰 필터 결과 DataFrame
        label_column: 실제 레이블 컬럼

    Returns:
        dict: 레이어별 정확도 및 통계
    """
    results = {
        "keyword_filter": {},
        "rule_filter": {},
    }

    # 키워드 필터 평가
    if not df_keyword.empty and label_column in df_keyword.columns:
        if "filter_label" in df_keyword.columns:
            correct = (df_keyword["filter_label"] == df_keyword[label_column]).sum()
            total = len(df_keyword)
            results["keyword_filter"] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
                "label_distribution": df_keyword[label_column].value_counts().to_dict(),
            }

    # 룰 필터 평가
    if not df_rule.empty and label_column in df_rule.columns:
        if "filter_label" in df_rule.columns:
            correct = (df_rule["filter_label"] == df_rule[label_column]).sum()
            total = len(df_rule)
            results["rule_filter"] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
                "label_distribution": df_rule[label_column].value_counts().to_dict(),
            }

    return results


def calculate_combined_metrics(
    filter_results: dict,
    ml_y_true: np.ndarray,
    ml_y_pred: np.ndarray,
    label_encoder,
) -> dict:
    """
    3-Layer 통합 성능 지표 계산

    Args:
        filter_results: 필터 평가 결과
        ml_y_true: ML 실제 레이블
        ml_y_pred: ML 예측 레이블
        label_encoder: 레이블 인코더

    Returns:
        dict: 통합 성능 지표
    """
    # ML 정확도
    ml_correct = (ml_y_true == ml_y_pred).sum()
    ml_total = len(ml_y_true)

    # 전체 통계
    kw = filter_results.get("keyword_filter", {})
    rl = filter_results.get("rule_filter", {})

    total_samples = (
        kw.get("total", 0) +
        rl.get("total", 0) +
        ml_total
    )

    total_correct = (
        kw.get("correct", 0) +
        rl.get("correct", 0) +
        ml_correct
    )

    return {
        "total_samples": total_samples,
        "total_correct": total_correct,
        "combined_accuracy": total_correct / total_samples if total_samples > 0 else 0,
        "keyword_filter_contribution": kw.get("total", 0) / total_samples if total_samples > 0 else 0,
        "rule_filter_contribution": rl.get("total", 0) / total_samples if total_samples > 0 else 0,
        "ml_contribution": ml_total / total_samples if total_samples > 0 else 0,
    }


def save_layer_performance_report(
    filter_results: dict,
    combined_metrics: dict,
    ml_f1: float,
    path: Path,
) -> None:
    """Layer별 성능 리포트 저장"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("3-Layer 통합 성능 리포트\n")
        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        # 전체 요약
        f.write("[전체 요약]\n")
        f.write(f"  전체 샘플 수:       {combined_metrics['total_samples']:,}\n")
        f.write(f"  전체 정분류:        {combined_metrics['total_correct']:,}\n")
        f.write(f"  통합 정확도:        {combined_metrics['combined_accuracy']:.2%}\n")
        f.write("\n")

        # 레이어별 기여도
        f.write("[레이어별 처리 비율]\n")
        f.write(f"  키워드 필터:        {combined_metrics['keyword_filter_contribution']:.1%}\n")
        f.write(f"  룰 필터:            {combined_metrics['rule_filter_contribution']:.1%}\n")
        f.write(f"  ML 모델:            {combined_metrics['ml_contribution']:.1%}\n")
        f.write("\n")

        # 레이어별 상세
        kw = filter_results.get("keyword_filter", {})
        if kw:
            f.write("[Layer 1: 키워드 필터]\n")
            f.write(f"  처리 건수:    {kw.get('total', 0):,}\n")
            f.write(f"  정분류:       {kw.get('correct', 0):,}\n")
            f.write(f"  정확도:       {kw.get('accuracy', 0):.2%}\n")
            if "label_distribution" in kw:
                f.write("  실제 레이블 분포:\n")
                for label, count in kw["label_distribution"].items():
                    f.write(f"    - {label}: {count:,}\n")
            f.write("\n")

        rl = filter_results.get("rule_filter", {})
        if rl:
            f.write("[Layer 2: 룰 필터]\n")
            f.write(f"  처리 건수:    {rl.get('total', 0):,}\n")
            f.write(f"  정분류:       {rl.get('correct', 0):,}\n")
            f.write(f"  정확도:       {rl.get('accuracy', 0):.2%}\n")
            if "label_distribution" in rl:
                f.write("  실제 레이블 분포:\n")
                for label, count in rl["label_distribution"].items():
                    f.write(f"    - {label}: {count:,}\n")
            f.write("\n")

        # ML 모델
        f.write("[Layer 3: ML 모델]\n")
        f.write(f"  F1-macro:     {ml_f1:.4f}\n")
        f.write("\n")

        # PoC 판정
        f.write("[PoC 성공 기준 판정]\n")
        poc_pass = ml_f1 >= POC_F1_MACRO_THRESHOLD
        f.write(f"  F1-macro >= {POC_F1_MACRO_THRESHOLD}: {'PASS' if poc_pass else 'FAIL'} ({ml_f1:.4f})\n")

    print(f"[저장] 레이어 성능 리포트: {path}")


def main():
    args = parse_args()

    # 디렉토리 생성
    ensure_dirs(REPORT_DIR, FIGURES_DIR)

    # 1. 모델 로드
    print("=" * 60)
    print("[Step 1] 모델 로드")
    print("=" * 60)
    artifact = load_model_with_meta(str(FINAL_MODEL_DIR / "best_model_v1.joblib"))
    model = artifact["model"]
    le = artifact["label_encoder"]

    # 2. 데이터 로드
    print("\n" + "=" * 60)
    print("[Step 2] Feature & 레이블 로드")
    print("=" * 60)
    data = load_feature_artifacts(str(FEATURE_DIR))
    X_test = data["X_test"]
    y_test_enc = le.transform(data["y_test"])
    feature_names = data["feature_names"]
    df_test = data["df_test"]

    # 3. 예측
    print("\n" + "=" * 60)
    print("[Step 3] 예측 수행")
    print("=" * 60)
    y_pred = model.predict(X_test)
    print(f"  예측 완료: {len(y_pred):,}건")

    # 4. 전체 평가
    print()
    results = full_evaluation(
        y_test_enc, y_pred, list(le.classes_), str(REPORT_DIR)
    )

    ml_f1 = results.get("f1_macro", 0)

    # 5. Feature Importance
    print()
    feature_importance_analysis(
        model, feature_names, TOP_N_FEATURES,
        str(FIGURES_DIR / "feature_importance.png"),
        str(REPORT_DIR / "feature_importance.csv"),
    )

    # 6. 오분류 분석
    print()
    analyze_errors(
        y_test_enc, y_pred, df_test, list(le.classes_),
        TEXT_COLUMN,
        save_path=str(REPORT_DIR / "error_analysis.csv"),
    )

    # 7. 3-Layer 통합 평가 (선택)
    if args.include_filtered:
        print("\n" + "=" * 60)
        print("[Step 7] 3-Layer 통합 성능 분석")
        print("=" * 60)

        # 필터 결과 로드
        df_keyword = pd.DataFrame()
        df_rule = pd.DataFrame()

        kw_path = PROCESSED_DATA_DIR / "keyword_filtered.csv"
        rl_path = PROCESSED_DATA_DIR / "rule_filtered.csv"

        if kw_path.exists():
            df_keyword = pd.read_csv(kw_path)
            print(f"  키워드 필터 결과: {len(df_keyword):,}건")

        if rl_path.exists():
            df_rule = pd.read_csv(rl_path)
            print(f"  룰 필터 결과: {len(df_rule):,}건")

        # 필터 성능 평가
        filter_results = evaluate_filter_performance(df_keyword, df_rule)

        # 통합 지표 계산
        combined_metrics = calculate_combined_metrics(
            filter_results, y_test_enc, y_pred, le
        )

        print(f"\n  [통합 성능]")
        print(f"  통합 정확도: {combined_metrics['combined_accuracy']:.2%}")

        # 레이어 성능 리포트 저장
        save_layer_performance_report(
            filter_results,
            combined_metrics,
            ml_f1,
            REPORT_DIR / "layer_performance.txt",
        )

    # PoC 성공 기준 판정
    print("\n" + "=" * 60)
    print("[PoC 성공 기준 판정]")
    print("=" * 60)

    poc_criteria = {
        f"F1-macro >= {POC_F1_MACRO_THRESHOLD}": ml_f1 >= POC_F1_MACRO_THRESHOLD,
    }

    # TP Recall, FP Precision 계산 (가능한 경우)
    if "classification_report_dict" in results:
        report_dict = results["classification_report_dict"]

        # TP 클래스 찾기
        tp_labels = [l for l in le.classes_ if l.startswith("TP")]
        if tp_labels:
            tp_label = tp_labels[0]
            if tp_label in report_dict:
                tp_recall = report_dict[tp_label].get("recall", 0)
                poc_criteria[f"TP Recall >= {POC_TP_RECALL_THRESHOLD}"] = tp_recall >= POC_TP_RECALL_THRESHOLD
                print(f"  TP Recall: {tp_recall:.4f}")

        # FP 클래스들의 평균 Precision
        fp_labels = [l for l in le.classes_ if l.startswith("FP")]
        if fp_labels:
            fp_precisions = [
                report_dict[l].get("precision", 0)
                for l in fp_labels if l in report_dict
            ]
            if fp_precisions:
                avg_fp_precision = np.mean(fp_precisions)
                poc_criteria[f"FP Precision >= {POC_FP_PRECISION_THRESHOLD}"] = avg_fp_precision >= POC_FP_PRECISION_THRESHOLD
                print(f"  FP Avg Precision: {avg_fp_precision:.4f}")

    all_pass = all(poc_criteria.values())

    for criterion, passed in poc_criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {criterion}: {status}")

    print(f"\n  최종 판정: {'PoC 성공' if all_pass else 'PoC 미달'}")

    print("\n" + "=" * 60)
    print("평가 파이프라인 완료.")
    print("=" * 60)


if __name__ == "__main__":
    main()
