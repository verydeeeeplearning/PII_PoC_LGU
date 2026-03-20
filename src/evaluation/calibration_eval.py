"""S6 Calibration Evaluation - Architecture v1.2 §19.4

ECE/MCE + Reliability Diagram 생성.

ECE 목표: ≤ 0.05 (보정 후)
"""

from __future__ import annotations

import numpy as np


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        y_true  : 실제 레이블 (0/1)
        y_proba : 양성 클래스 확률 (0~1)
        n_bins  : 구간 수

    Returns:
        ECE (낮을수록 좋음)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins[1:-1])  # 0 ~ n_bins-1
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        n_bin = mask.sum()
        conf = y_proba[mask].mean()
        acc = y_true[mask].mean()
        ece += (n_bin / n) * abs(conf - acc)

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (MCE).

    Args:
        y_true  : 실제 레이블 (0/1)
        y_proba : 양성 클래스 확률 (0~1)
        n_bins  : 구간 수

    Returns:
        MCE (낮을수록 좋음)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins[1:-1])  # 0 ~ n_bins-1
    mce = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        conf = y_proba[mask].mean()
        acc = y_true[mask].mean()
        mce = max(mce, abs(conf - acc))

    return float(mce)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: str,
    n_bins: int = 10,
) -> None:
    """Reliability Diagram (보정 곡선) 저장.

    Args:
        y_true      : 실제 레이블 (0/1)
        y_proba     : 양성 클래스 확률 (0~1)
        output_path : 저장 경로 (.png)
        n_bins      : 구간 수
    """
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs, bin_accs = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.any():
            bin_confs.append(y_proba[mask].mean())
            bin_accs.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="완벽 보정")
    ax.plot(bin_confs, bin_accs, "s-", label="모델 보정")
    ax.set_xlabel("예측 신뢰도")
    ax.set_ylabel("실제 정확도")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
