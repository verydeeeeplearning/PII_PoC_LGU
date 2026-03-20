"""S6 Confident Learning Auditor - Architecture v1.2 §22 (Step E Tier 1)

K-Fold OOF 예측으로 라벨 오류 후보 탐지.
탐지된 후보를 제거하여 학습 데이터 품질 개선.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


class ConfidentLearningAuditor:
    """K-Fold OOF 기반 라벨 노이즈 탐지기 (Step E Tier 1).

    Args:
        noise_threshold     : 노이즈 판단 임계값 (예측 신뢰도 기준)
        confidence_threshold: OOF 최고 확률 임계값
    """

    NOISE_THRESHOLD: float = 0.10
    CONFIDENCE_THRESHOLD: float = 0.90

    def __init__(
        self,
        noise_threshold: float | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self.noise_threshold = noise_threshold or self.NOISE_THRESHOLD
        self.confidence_threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD

    def audit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_cls,
        n_splits: int = 5,
        **model_kwargs,
    ) -> dict:
        """K-Fold OOF로 라벨 오류 후보 탐지.

        Args:
            X          : 학습 피처 행렬
            y          : 레이블 배열
            model_cls  : sklearn 호환 분류기 클래스 (미학습)
            n_splits   : K-Fold 분할 수
            **model_kwargs: 모델 생성 파라미터

        Returns:
            {
                "noise_indices": List[int],  # 오류 후보 행 인덱스
                "noise_rate": float,         # 노이즈 비율
                "oof_proba": np.ndarray,     # OOF 확률 (n_samples, n_classes)
            }
        """
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        oof_proba = np.zeros((len(y), n_classes))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_train, fold_val in skf.split(X, y):
            model = model_cls(**model_kwargs)
            model.fit(X[fold_train], y[fold_train])
            proba = model.predict_proba(X[fold_val])
            oof_proba[fold_val] = proba

        # 노이즈 후보: OOF 최고 확률 클래스 ≠ 레이블 AND 확률 ≥ confidence_threshold
        oof_pred_idx = np.argmax(oof_proba, axis=1)
        oof_pred_class = classes[oof_pred_idx]
        oof_max_proba = oof_proba.max(axis=1)

        noise_mask = (oof_pred_class != y) & (oof_max_proba >= self.confidence_threshold)
        noise_indices = np.where(noise_mask)[0].tolist()
        noise_rate = len(noise_indices) / len(y)

        return {
            "noise_indices": noise_indices,
            "noise_rate": noise_rate,
            "oof_proba": oof_proba,
        }

    def clean_labels(
        self,
        X: np.ndarray,
        y: np.ndarray,
        audit_result: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """라벨 오류 후보 제외 (보수적 클리닝).

        Args:
            X            : 학습 피처 행렬
            y            : 레이블 배열
            audit_result : audit() 반환 dict

        Returns:
            (X_clean, y_clean)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        noise_set = set(audit_result.get("noise_indices", []))
        keep_mask = np.array([i not in noise_set for i in range(len(y))])
        return X[keep_mask], y[keep_mask]
