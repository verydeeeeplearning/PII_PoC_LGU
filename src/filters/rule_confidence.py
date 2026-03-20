"""Bayesian lower bound for rule confidence estimation.

Architecture v1.2 §7.5:
    Beta posterior 95% lower bound을 사용해 룰의 precision을 보수적으로 추정한다.
    소용량 샘플(N=15)에서도 과신하지 않는 신뢰도를 산출한다.
"""

try:
    from scipy import stats as sp_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def compute_rule_precision_lb(N: int, M: int, alpha: float = 0.05) -> float:
    """Beta posterior (1-alpha) lower bound.

    Args:
        N: 총 적용 횟수
        M: 정탐(FP 분류 정확) 횟수  (0 <= M <= N)
        alpha: 유의수준 (기본 0.05 -> 95% lower bound)

    Returns:
        Bayesian lower bound [0.0, 1.0], 소수점 4자리 반올림

    Examples:
        >>> compute_rule_precision_lb(N=5000, M=4985)   # ~0.994
        >>> compute_rule_precision_lb(N=15, M=15)        # ~0.814
        >>> compute_rule_precision_lb(N=0, M=0)          # 0.5 (uninformative)
    """
    if N == 0:
        return 0.5  # uninformative prior

    if _SCIPY_AVAILABLE:
        # Beta(1+M, 1+(N-M)) posterior의 alpha 분위수
        a = 1 + M
        b = 1 + (N - M)
        return round(float(sp_stats.beta.ppf(alpha, a, b)), 4)

    # scipy 미설치 시: Wilson score interval (근사)
    p_hat = M / N
    z = 1.645  # 95% one-sided z-score
    denom = 1 + z ** 2 / N
    center = p_hat + z ** 2 / (2 * N)
    margin = z * (p_hat * (1 - p_hat) / N + z ** 2 / (4 * N ** 2)) ** 0.5
    lb = (center - margin) / denom
    return round(max(0.0, min(1.0, lb)), 4)
