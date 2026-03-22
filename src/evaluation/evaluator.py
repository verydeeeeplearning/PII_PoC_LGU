"""모델 평가 모듈"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from typing import List, Optional, Dict

from src.utils.constants import (
    POC_F1_MACRO_THRESHOLD,
    POC_TP_RECALL_THRESHOLD,
    POC_FP_PRECISION_THRESHOLD,
    FIGURE_DPI,
    TOP_N_FEATURES,
)
from src.utils.common import ensure_dirs
from src.utils.plot_utils import setup_plot


def full_evaluation(
    y_true,
    y_pred,
    class_names: List[str],
    save_dir: str = "outputs",
) -> dict:
    """
    전체 평가 리포트 생성

    출력:
        - F1-macro, F1-weighted 콘솔 출력
        - Classification Report 콘솔 출력 + 파일 저장
        - Confusion Matrix 시각화 (절대값 + 정규화)
        - PoC 성공 기준 판정

    Args:
        y_true: 실제 레이블 (숫자 인코딩)
        y_pred: 예측 레이블 (숫자 인코딩)
        class_names: 클래스 이름 목록
        save_dir: 결과 저장 디렉토리

    Returns:
        dict: {
            "f1_macro": float,
            "f1_weighted": float,
            "classification_report": str,
            "confusion_matrix": np.ndarray,
            "poc_criteria": dict
        }
    """
    setup_plot()
    save_path = Path(save_dir)
    ensure_dirs(save_path)

    # 기본 메트릭
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print("=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"  F1-macro:    {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")

    # §6.2 PR-AUC (threshold-agnostic, 불균형에 강건)
    # 이진 레이블 여부 확인
    unique_labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    pr_auc = None
    if len(unique_labels) == 2:
        try:
            pr_auc = average_precision_score(
                (np.asarray(y_true) == unique_labels[0]).astype(int),
                (np.asarray(y_pred) == unique_labels[0]).astype(int),
            )
            print(f"  PR-AUC:      {pr_auc:.4f}  [참고: threshold-agnostic]")
        except Exception:
            pass

    # Classification Report
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(f"\n{report_str}")

    # Classification Report 파일 저장
    report_path = save_path / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"PII 오탐 개선 AI 모델 - 평가 결과\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"평가 시점: {pd.Timestamp.now().isoformat()}\n\n")
        f.write(f"F1-macro:    {f1_macro:.4f}\n")
        f.write(f"F1-weighted: {f1_weighted:.4f}\n\n")
        f.write(report_str)
    print(f"[저장] {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, save_dir)

    # PoC 판정
    poc_criteria = check_poc_criteria(y_true, y_pred, class_names)

    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report_str,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "poc_criteria": poc_criteria,
        "pr_auc": pr_auc,
    }


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_dir: str,
) -> None:
    """Confusion Matrix 시각화 (내부 함수)"""
    setup_plot()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 절대값
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    axes[0].set_title("Confusion Matrix (Count)", fontsize=14)

    # 정규화
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)
    axes[1].set_title("Confusion Matrix (Normalized by Row = Recall)", fontsize=14)

    plt.tight_layout()
    fig_dir = Path(save_dir) / "figures"
    ensure_dirs(fig_dir)
    save_path = fig_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[저장] {save_path}")


def check_poc_criteria(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    *,
    tp_label: Optional[str] = None,
) -> Dict:
    """
    PoC 성공 기준 판정

    두 가지 호출 방식 지원:
        1. 기존 방식 (class_names 사용): {기준명: {"value": float, "pass": bool}}
        2. 신규 방식 (tp_label 사용):    {"passes": bool, "f1_macro": float,
                                          "tp_recall": float, "fp_precision": float}

    기준:
        - F1-macro >= 0.70
        - 정탐 Recall >= 0.75 (TP 클래스)
        - 오탐 Precision >= 0.85 (FP 클래스)

    Args:
        y_true      : 실제 레이블
        y_pred      : 예측 레이블
        class_names : 클래스 이름 목록 (하위 호환)
        tp_label    : TP 클래스 문자열 (신규)

    Returns:
        dict (형식은 호출 방식에 따라 다름)
    """
    # ── 신규 방식: tp_label 키워드 ────────────────────────────────────────────
    if tp_label is not None:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        # F1-macro
        labels = sorted(set(y_true_arr) | set(y_pred_arr))
        f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro",
                                  labels=labels, zero_division=0))

        # TP recall
        tp_mask = y_true_arr == tp_label
        if tp_mask.sum() > 0:
            tp_recall = float((y_pred_arr[tp_mask] == tp_label).mean())
        else:
            tp_recall = 0.0

        # FP precision: FP로 예측한 것 중 실제 FP 비율
        fp_pred_mask = y_pred_arr != tp_label
        if fp_pred_mask.sum() > 0:
            fp_precision = float((y_true_arr[fp_pred_mask] != tp_label).mean())
        else:
            fp_precision = 1.0

        passes = (
            f1_macro >= POC_F1_MACRO_THRESHOLD
            and tp_recall >= POC_TP_RECALL_THRESHOLD
            and fp_precision >= POC_FP_PRECISION_THRESHOLD
        )

        return {
            "passes":       passes,
            "f1_macro":     f1_macro,
            "tp_recall":    tp_recall,
            "fp_precision": fp_precision,
        }

    # ── 기존 방식: class_names ────────────────────────────────────────────────
    class_names = class_names or []
    print("=" * 60)
    print("PoC 성공 기준 판정")
    print("=" * 60)

    criteria_results = {}

    # 1. F1-macro
    f1_macro = f1_score(y_true, y_pred, average="macro")
    criteria_results["F1-macro >= {:.2f}".format(POC_F1_MACRO_THRESHOLD)] = {
        "value": f1_macro,
        "pass": f1_macro >= POC_F1_MACRO_THRESHOLD,
    }

    # 2. 정탐 Recall
    tp_idx = None
    for i, name in enumerate(class_names):
        if "TP" in name or "정탐" in name or "실제" in name:
            tp_idx = i
            break

    if tp_idx is not None:
        tp_mask = (y_true == tp_idx)
        if tp_mask.sum() > 0:
            tp_recall = (y_pred[tp_mask] == tp_idx).mean()
            criteria_results["정탐 Recall >= {:.2f}".format(POC_TP_RECALL_THRESHOLD)] = {
                "value": tp_recall,
                "pass": tp_recall >= POC_TP_RECALL_THRESHOLD,
            }

        # 3. 오탐 Precision
        fp_pred_mask = (y_pred != tp_idx)
        if fp_pred_mask.sum() > 0:
            fp_precision = (y_true[fp_pred_mask] != tp_idx).mean()
            criteria_results["오탐 Precision >= {:.2f}".format(POC_FP_PRECISION_THRESHOLD)] = {
                "value": fp_precision,
                "pass": fp_precision >= POC_FP_PRECISION_THRESHOLD,
            }
    else:
        print("  [경고] 정탐(TP) 클래스를 찾을 수 없습니다.")

    # 결과 출력
    all_pass = True
    for criterion, result in criteria_results.items():
        status = "PASS" if result["pass"] else "FAIL"
        icon = "[O]" if result["pass"] else "[X]"
        print(f"  {icon} {criterion}: {result['value']:.4f}  -> {status}")
        if not result["pass"]:
            all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print("  종합 판정: PASS - PoC 성공 기준 충족")
    else:
        print("  종합 판정: FAIL - 추가 개선 필요")
    print(f"{'=' * 60}")

    return criteria_results


def analyze_errors(
    y_true,
    y_pred,
    df_test: pd.DataFrame,
    class_names: List[str],
    text_column: str = "detected_text_with_context",
    top_n: int = 15,
    save_path: str = "outputs/error_analysis.csv",   # run_report.py에서 REPORT_DIR 기준으로 오버라이드
) -> pd.DataFrame:
    """
    오분류 패턴 분석

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        df_test: 테스트 DataFrame
        class_names: 클래스 이름
        text_column: 텍스트 컬럼명
        top_n: 출력할 오분류 패턴 수
        save_path: 오분류 CSV 저장 경로

    Returns:
        오분류 샘플 DataFrame (actual_class, predicted_class 포함)
    """
    errors_mask = y_true != y_pred
    n_errors = errors_mask.sum()
    n_total = len(y_true)

    print(f"[오분류 분석]")
    print(f"  전체: {n_total:,}건")
    print(f"  정분류: {n_total - n_errors:,}건 ({(n_total - n_errors) / n_total * 100:.1f}%)")
    print(f"  오분류: {n_errors:,}건 ({n_errors / n_total * 100:.1f}%)")

    # 주요 오분류 패턴
    error_pairs = list(zip(y_true[errors_mask], y_pred[errors_mask]))
    pair_counts = Counter(error_pairs).most_common(top_n)

    print(f"\n[주요 오분류 패턴 (상위 {top_n})]")
    print(f"  {'실제 클래스':30s}  ->  {'예측 클래스':30s}  {'건수':>6s}")
    print("-" * 80)
    for (true_cls, pred_cls), cnt in pair_counts:
        true_name = class_names[true_cls]
        pred_name = class_names[pred_cls]
        print(f"  {true_name:30s}  ->  {pred_name:30s}  {cnt:>5,}건")

    # 오분류 DataFrame 생성 및 저장
    error_df = df_test[errors_mask].copy()
    error_df["actual_class"] = [class_names[i] for i in y_true[errors_mask]]
    error_df["predicted_class"] = [class_names[i] for i in y_pred[errors_mask]]

    out_path = Path(save_path)
    ensure_dirs(out_path.parent)
    error_df.head(200).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[저장] {out_path} ({min(200, len(error_df))}건)")

    return error_df


def feature_importance_analysis(
    model,
    feature_names: List[str],
    top_n: int = TOP_N_FEATURES,
    save_path: str = "outputs/figures/feature_importance.png",   # run_report.py에서 FIGURES_DIR 기준으로 오버라이드
    report_path: str = "outputs/feature_importance.csv",       # run_report.py에서 REPORT_DIR 기준으로 오버라이드
) -> pd.DataFrame:
    """
    Feature Importance 분석

    Args:
        model: 학습된 모델 (feature_importances_ 속성 필요)
        feature_names: Feature 이름 목록
        top_n: 상위 출력 개수
        save_path: 시각화 저장 경로
        report_path: CSV 저장 경로

    Returns:
        상위 N개 Feature Importance DataFrame
    """
    setup_plot()

    if not hasattr(model, "feature_importances_"):
        print("  모델이 feature_importances_를 지원하지 않습니다.")
        return pd.DataFrame()

    importances = model.feature_importances_

    # 상위 N개
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [(feature_names[i], importances[i]) for i in indices]
    df_imp = pd.DataFrame(top_features, columns=["feature", "importance"])

    print(f"[Feature Importance 상위 {top_n}]")
    for rank, (_, row) in enumerate(df_imp.iterrows(), 1):
        print(f"  {rank:2d}. {row['feature']:50s}  {row['importance']:.6f}")

    # 시각화
    df_imp_plot = df_imp.sort_values("importance")

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))

    colors = ["#3498db" if f.startswith("tfidf_") else "#e74c3c" for f in df_imp_plot["feature"]]
    ax.barh(df_imp_plot["feature"], df_imp_plot["importance"], color=colors)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    ax.set_xlabel("Importance")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="TF-IDF Feature"),
        Patch(facecolor="#e74c3c", label="Dense Feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig_path = Path(save_path)
    ensure_dirs(fig_path.parent)
    plt.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[저장] {fig_path}")

    # 그룹별 Importance 합산
    group_importance = {
        "TF-IDF": 0,
        "Keyword": 0,
        "Text Stat": 0,
        "File Path": 0,
        "Other": 0,
    }

    text_stat_features = {
        "text_length", "word_count", "digit_count", "digit_ratio",
        "special_char_ratio", "uppercase_ratio",
        "has_email_pattern", "has_phone_pattern",
    }

    for fname, imp in zip(feature_names, importances):
        if fname.startswith("tfidf_"):
            group_importance["TF-IDF"] += imp
        elif fname.startswith("has_") and "keyword" in fname:
            group_importance["Keyword"] += imp
        elif fname in text_stat_features:
            group_importance["Text Stat"] += imp
        elif fname.startswith("path_") or fname == "file_extension":
            group_importance["File Path"] += imp
        else:
            group_importance["Other"] += imp

    print("\n[Feature 그룹별 Importance]")
    total_imp = sum(group_importance.values())
    for group, imp in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
        pct = imp / total_imp * 100 if total_imp > 0 else 0
        print(f"  {group:15s}  {imp:.6f}  ({pct:.1f}%)")

    # 전체 Feature Importance CSV 저장
    rpt_path = Path(report_path)
    ensure_dirs(rpt_path.parent)
    fi_all = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    fi_all.to_csv(rpt_path, index=False)
    print(f"[저장] {rpt_path} ({len(fi_all)}건)")

    return df_imp


def bootstrap_metric(
    y_true,
    y_pred,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple:
    """Bootstrap Confidence Interval 산출.

    §6.4 수정: 단일 test set point estimate 대신 통계적 신뢰구간 제공.

    Args:
        y_true      : 실제 레이블 array
        y_pred      : 예측 레이블 array
        metric_fn   : (y_true, y_pred) -> float 형태의 지표 함수
        n_bootstrap : Bootstrap 반복 횟수 (기본: 1000)
        ci          : 신뢰구간 수준 (기본: 0.95)
        random_state: 재현성 시드

    Returns:
        (mean, lower, upper) - 지표 평균, 하한, 상한

    Example:
        from sklearn.metrics import recall_score
        mean, lo, hi = bootstrap_metric(
            y_test, y_pred,
            lambda y, yp: recall_score(y, yp, pos_label=0, zero_division=0)
        )
        print(f"TP Recall: {mean:.3f} ({lo:.3f}~{hi:.3f}, 95% CI)")
    """
    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n = len(y_true_arr)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true_arr[idx], y_pred_arr[idx])
            scores.append(float(score))
        except Exception:
            continue

    if not scores:
        return 0.0, 0.0, 0.0

    alpha = (1 - ci) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    mean  = float(np.mean(scores))
    return mean, lower, upper
