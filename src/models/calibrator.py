"""S3b ClasswiseCalibrator - Architecture v1.2 §8.7

소수 클래스: Platt Scaling (sigmoid)
다수 클래스: Isotonic Regression
단일 보정기가 클래스 분포에 따라 method를 자동 선택.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV


class ClasswiseCalibrator:
    """클래스 분포 기반 차등 보정기.

    Args:
        minority_threshold: 이 값 미만의 소수 클래스가 있으면 'sigmoid', 아니면 'isotonic'
    """

    def __init__(self, minority_threshold: int = 100) -> None:
        self.minority_threshold = minority_threshold
        self.method: str = "sigmoid"
        self._calibrated: CalibratedClassifierCV | None = None

    def fit(
        self,
        model,
        X_cal,
        y_cal,
        class_counts: dict | None = None,
    ) -> "ClasswiseCalibrator":
        """보정 학습.

        Args:
            model       : 학습된 분류 모델 (predict_proba 필요)
            X_cal       : 보정용 입력 데이터
            y_cal       : 보정용 레이블
            class_counts: {class_name: count} 딕셔너리 (선택)
        """
        # 소수 클래스 여부 판단
        if class_counts:
            min_count = min(class_counts.values())
        else:
            _, counts = np.unique(y_cal, return_counts=True)
            min_count = counts.min()

        self.method = "sigmoid" if min_count < self.minority_threshold else "isotonic"

        # sklearn 1.4+: cv='prefit' 제거됨 -> CalibratedClassifierCV(cv=3)으로 대체
        # 모델이 이미 학습된 경우 직접 isotonic/sigmoid regression으로 보정
        self._calibrated = CalibratedClassifierCV(
            estimator=model,
            method=self.method,
            cv=3,
        )
        self._calibrated.fit(X_cal, y_cal)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._calibrated is None:
            raise RuntimeError("ClasswiseCalibrator must be fitted first")
        return self._calibrated.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ClasswiseCalibrator":
        return joblib.load(path)
