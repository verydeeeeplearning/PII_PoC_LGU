"""S3b OOD Detector - Architecture v1.2 §8.8

Mahalanobis Distance 기반 Out-of-Distribution 탐지.
학습 데이터 분포에서 벗어난 입력을 UNKNOWN으로 라우팅한다.

주의: 고차원 sparse 피처가 아닌 dense 피처(수동 피처, SVD 축소 결과 등)에 적용.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.covariance import EmpiricalCovariance


class OODDetector:
    """Mahalanobis Distance 기반 OOD 탐지기.

    Args:
        threshold_percentile: 학습 데이터 기준 임계값 백분위수 (기본 95%)
    """

    def __init__(self, threshold_percentile: float = 95.0) -> None:
        self.threshold_percentile = threshold_percentile
        self._ec: EmpiricalCovariance | None = None
        self.threshold_: float | None = None

    def fit(self, X_dense: np.ndarray) -> "OODDetector":
        """학습 데이터 분포 적합.

        Args:
            X_dense: 밀집 행렬 (n_samples, n_features)
        """
        self._ec = EmpiricalCovariance()
        self._ec.fit(X_dense)
        train_distances = self._ec.mahalanobis(X_dense)
        self.threshold_ = float(np.percentile(train_distances, self.threshold_percentile))
        return self

    def score(self, X_dense: np.ndarray) -> np.ndarray:
        """Mahalanobis distance 계산 (높을수록 OOD 가능성 높음).

        Returns:
            distances: shape (n_samples,)
        """
        if self._ec is None:
            raise RuntimeError("OODDetector must be fitted first")
        return self._ec.mahalanobis(X_dense)

    def predict(
        self,
        X_dense: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """OOD 여부 예측.

        Returns:
            bool array: True if OOD
        """
        scores = self.score(X_dense)
        thr = threshold if threshold is not None else self.threshold_
        return scores > thr

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "OODDetector":
        return joblib.load(path)
