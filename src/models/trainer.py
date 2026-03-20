"""모델 학습 모듈"""
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# XGBoost wheels may emit a glibc FutureWarning even on modern distros; it's noise for operators.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^xgboost(\.|$)")

from src.utils.constants import (
    RANDOM_SEED, XGB_DEFAULT_PARAMS, LGB_DEFAULT_PARAMS,
    XGB_PARAM_GRID, TUNING_N_ITER, TUNING_CV_FOLDS,
    EARLY_STOPPING_ROUNDS,
)


def encode_labels(
    y_train: pd.Series,
    y_test: Optional[pd.Series] = None,
) -> Tuple:
    """
    레이블 문자열 -> 숫자 인코딩

    Args:
        y_train: 학습 레이블 (문자열)
        y_test: 테스트 레이블 (문자열, 선택)

    Returns:
        (y_train_enc, y_test_enc_or_None, LabelEncoder)
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    y_test_enc = None
    if y_test is not None:
        y_test_enc = le.transform(y_test)

    print(f"[레이블 인코딩]")
    print(f"  클래스 수: {len(le.classes_)}")
    for i, cls_name in enumerate(le.classes_):
        n_train = (y_train_enc == i).sum()
        n_test_count = (y_test_enc == i).sum() if y_test_enc is not None else 0
        print(f"  {i}: {cls_name:30s}  학습: {n_train:>5,}  테스트: {n_test_count:>5,}")

    return y_train_enc, y_test_enc, le


def train_baseline(
    X_train, y_train_enc,
    X_test, y_test_enc,
) -> Tuple:
    """
    Baseline 모델 학습 (DummyClassifier, most_frequent)

    Returns:
        (model, f1_macro)
    """
    print("=" * 60)
    print("[Baseline] DummyClassifier (most_frequent)")
    print("=" * 60)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train_enc)

    y_pred = baseline.predict(X_test)
    f1 = f1_score(y_test_enc, y_pred, average="macro")

    print(f"\n  F1-macro (Baseline): {f1:.4f}")
    print(f"  -> 이 점수가 하한선입니다. 모든 모델은 이를 초과해야 합니다.")

    return baseline, f1


def train_xgboost(
    X_train, y_train_enc,
    X_test, y_test_enc,
    label_encoder: LabelEncoder,
    params: Optional[Dict] = None,
    use_class_weight: bool = False,
) -> Tuple:
    """
    XGBoost 모델 학습

    Args:
        X_train, y_train_enc: 학습 데이터 (y는 숫자 인코딩)
        X_test, y_test_enc: 테스트 데이터
        label_encoder: 클래스 이름 복원용 LabelEncoder
        params: 하이퍼파라미터 (None이면 constants.XGB_DEFAULT_PARAMS)
        use_class_weight: True면 compute_sample_weight("balanced") 적용

    Returns:
        (model, f1_macro, classification_report_str)
    """
    if params is None:
        params = XGB_DEFAULT_PARAMS.copy()

    n_classes = len(label_encoder.classes_)
    cw_label = " + Class Weight" if use_class_weight else ""

    print("=" * 60)
    print(f"[XGBoost{cw_label}]")
    print("=" * 60)

    model = XGBClassifier(
        num_class=n_classes,
        random_state=RANDOM_SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if (EARLY_STOPPING_ROUNDS and EARLY_STOPPING_ROUNDS > 0) else None,
        **params,
    )

    fit_kwargs = {"verbose": False}

    # §5.2 수정: eval_set으로 최종 test set 사용 금지 - 정보 누출(information leakage) 방지
    # Early stopping이 활성화된 경우, train에서 내부 validation(20%)을 분리하여 사용
    if EARLY_STOPPING_ROUNDS and EARLY_STOPPING_ROUNDS > 0:
        from sklearn.model_selection import train_test_split as _tts
        try:
            X_inner, X_val_es, y_inner, y_val_es = _tts(
                X_train, y_train_enc,
                test_size=0.2, stratify=y_train_enc, random_state=RANDOM_SEED,
            )
        except ValueError:
            X_inner, X_val_es, y_inner, y_val_es = _tts(
                X_train, y_train_enc, test_size=0.2, random_state=RANDOM_SEED,
            )
        fit_X, fit_y = X_inner, y_inner
        fit_kwargs["eval_set"] = [(X_val_es, y_val_es)]
        print(f"  Early stopping: inner_train={fit_X.shape[0]:,}, val={X_val_es.shape[0]:,}")
    else:
        fit_X, fit_y = X_train, y_train_enc

    if use_class_weight:
        sample_weights = compute_sample_weight("balanced", fit_y)
        fit_kwargs["sample_weight"] = sample_weights

    model.fit(fit_X, fit_y, **fit_kwargs)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test_enc, y_pred, average="macro")
    report = classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_)

    print(f"\n  F1-macro: {f1:.4f}")
    print("\n" + report)

    return model, f1, report


def train_lightgbm(
    X_train, y_train_enc,
    X_test, y_test_enc,
    label_encoder: LabelEncoder,
    params: Optional[Dict] = None,
    use_class_weight: bool = False,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple:
    """
    LightGBM 모델 학습

    Args:
        (train_xgboost와 동일)
        use_class_weight: True면 class_weight="balanced" 파라미터 적용
        sample_weight: 외부 제공 샘플 가중치 (None이면 미적용)

    Returns:
        (model, f1_macro, classification_report_str)
    """
    if params is None:
        params = LGB_DEFAULT_PARAMS.copy()

    n_classes = len(label_encoder.classes_)
    cw_label = " + Class Weight" if use_class_weight else ""

    print("=" * 60)
    print(f"[LightGBM{cw_label}]")
    print("=" * 60)

    if use_class_weight:
        params["class_weight"] = "balanced"

    model = LGBMClassifier(
        num_class=n_classes,
        random_state=RANDOM_SEED,
        **params,
    )

    if sample_weight is not None:
        print(f"  Sample weight: min={sample_weight.min():.4f}, max={sample_weight.max():.4f}, "
              f"mean={sample_weight.mean():.4f}")

    # §5.2 수정: eval_set으로 최종 test set 사용 금지 - 정보 누출(information leakage) 방지
    if EARLY_STOPPING_ROUNDS and EARLY_STOPPING_ROUNDS > 0:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split as _tts
        if sample_weight is not None:
            try:
                X_inner, X_val_es, y_inner, y_val_es, sw_inner, sw_val = _tts(
                    X_train, y_train_enc, sample_weight,
                    test_size=0.2, stratify=y_train_enc, random_state=RANDOM_SEED,
                )
            except ValueError:
                X_inner, X_val_es, y_inner, y_val_es, sw_inner, sw_val = _tts(
                    X_train, y_train_enc, sample_weight,
                    test_size=0.2, random_state=RANDOM_SEED,
                )
        else:
            try:
                X_inner, X_val_es, y_inner, y_val_es = _tts(
                    X_train, y_train_enc,
                    test_size=0.2, stratify=y_train_enc, random_state=RANDOM_SEED,
                )
            except ValueError:
                X_inner, X_val_es, y_inner, y_val_es = _tts(
                    X_train, y_train_enc, test_size=0.2, random_state=RANDOM_SEED,
                )
            sw_inner = None
        fit_X, fit_y = X_inner, y_inner
        print(f"  Early stopping: inner_train={fit_X.shape[0]:,}, val={X_val_es.shape[0]:,}")
        _fit_kwargs = {}
        if sw_inner is not None:
            _fit_kwargs["sample_weight"] = sw_inner
        model.fit(
            fit_X, fit_y,
            eval_set=[(X_val_es, y_val_es)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            **_fit_kwargs,
        )
    else:
        _fit_kwargs = {}
        if sample_weight is not None:
            _fit_kwargs["sample_weight"] = sample_weight
        model.fit(X_train, y_train_enc, **_fit_kwargs)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test_enc, y_pred, average="macro")
    report = classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_)

    print(f"\n  F1-macro: {f1:.4f}")
    print("\n" + report)

    return model, f1, report


def tune_model(
    model_type: str,
    X_train, y_train_enc,
    param_grid: Optional[Dict] = None,
    cv: int = TUNING_CV_FOLDS,
    n_iter: int = TUNING_N_ITER,
    scoring: str = "f1_macro",
) -> Tuple:
    """
    RandomizedSearchCV 기반 하이퍼파라미터 튜닝

    Args:
        model_type: "xgboost" 또는 "lightgbm"
        X_train: 학습 Feature
        y_train_enc: 학습 레이블 (숫자)
        param_grid: 탐색 파라미터 딕셔너리
        cv: CV fold 수
        n_iter: 랜덤 탐색 반복 횟수
        scoring: 평가 지표

    Returns:
        (best_model, best_params, best_cv_score)
    """
    if param_grid is None:
        param_grid = XGB_PARAM_GRID

    n_classes = len(np.unique(y_train_enc))

    print("=" * 60)
    print(f"[튜닝] {model_type} - RandomizedSearchCV")
    print("=" * 60)

    if model_type.lower() == "xgboost":
        base_model = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
    elif model_type.lower() == "lightgbm":
        base_model = LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    search = RandomizedSearchCV(
        base_model, param_grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED),
        scoring=scoring,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train_enc)

    print(f"\n  Best CV Score: {search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def calibrate_model(
    model,
    X_train,
    y_train_enc,
    label_encoder: LabelEncoder,
    method: str = "isotonic",
    cv: int = 3,
) -> object:
    """모델 확률 보정 (Probability Calibration).

    GBDT의 raw predict_proba는 과도한 확신(overconfident)을 보이는 경향.
    Calibration으로 predict_proba가 실제 확률에 근접하도록 보정.

    Note:
        sklearn 1.8.0에서 cv='prefit' 제거됨 -> cv=3 기본값 사용.

    Args:
        model         : 학습된 분류기 (predict_proba 지원)
        X_train       : 보정용 학습 데이터
        y_train_enc   : 학습 레이블 (숫자 인코딩)
        label_encoder : LabelEncoder (ECE 계산용)
        method        : 'isotonic' 또는 'sigmoid'
        cv            : 교차검증 fold 수

    Returns:
        CalibratedClassifierCV 적합 객체
    """
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve

    print("=" * 60)
    print(f"[Calibration] method={method}, cv={cv}")
    print("=" * 60)

    calibrated = CalibratedClassifierCV(model, cv=cv, method=method)
    calibrated.fit(X_train, y_train_enc)

    # ECE (Expected Calibration Error) - binary case만 계산
    n_classes = len(label_encoder.classes_)
    if n_classes == 2:
        try:
            proba = calibrated.predict_proba(X_train)
            prob_true, prob_pred = calibration_curve(
                (y_train_enc == 1).astype(int), proba[:, 1], n_bins=10,
            )
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
            print(f"  ECE (학습셋): {ece:.4f}  (< 0.05면 양호)")
        except Exception:
            pass

    print("  Calibration 완료")
    return calibrated


def focal_loss_lgb_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
):
    """Focal Loss - LightGBM custom objective.

    Hard example(경계 샘플)에 집중하고 easy negative(명백한 오탐) loss 기여를 감소.
    class_weight='balanced' 대비 precision 저하 없이 recall 향상 기대.

    사용법:
        model = LGBMClassifier(objective=lambda y, p: focal_loss_lgb_objective(y, p, gamma=2.0))

    Args:
        y_true : 실제 레이블 array
        y_pred : raw score array (sigmoid 변환 전)
        gamma  : focusing parameter (클수록 hard example에 집중)
        alpha  : TP 클래스 가중치

    Returns:
        (grad, hess) - LightGBM custom objective 반환 형식
    """
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)

    # Focal weight
    focal_weight = np.where(
        y_true == 1,
        alpha * (1 - p) ** gamma,
        (1 - alpha) * p ** gamma,
    )

    grad = np.where(
        y_true == 1,
        focal_weight * (p - 1),
        focal_weight * p,
    )
    hess = np.where(
        y_true == 1,
        focal_weight * p * (1 - p) * (1 + gamma * (1 - p) * np.log(p + 1e-10)),
        focal_weight * p * (1 - p) * (1 + gamma * p * np.log(1 - p + 1e-10)),
    )
    return grad, hess


def save_model_with_meta(
    model,
    path: str,
    label_encoder: LabelEncoder,
    f1_score_val: float,
    model_name: str = "",
    all_results: Optional[Dict] = None,
    train_size: int = 0,
    test_size: int = 0,
    feature_count: int = 0,
) -> None:
    """
    모델을 메타데이터와 함께 저장

    저장 형식 (joblib dict):
        model, label_encoder, f1_macro, model_name,
        all_results, feature_count, train_size, test_size,
        classes, saved_at

    Args:
        model: 학습된 모델 객체
        path: 저장 경로 (.joblib)
        label_encoder: LabelEncoder
        f1_score_val: F1-macro 점수
        model_name: 모델 설명 문자열
        all_results: 전체 실험 결과 dict
        train_size: 학습 데이터 건수
        test_size: 테스트 데이터 건수
        feature_count: Feature 수
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "f1_macro": f1_score_val,
        "model_name": model_name,
        "all_results": all_results or {},
        "feature_count": feature_count,
        "train_size": train_size,
        "test_size": test_size,
        "classes": list(label_encoder.classes_),
        "saved_at": datetime.now().isoformat(),
    }

    joblib.dump(artifact, save_path)

    print(f"[최종 모델 저장 완료]")
    print(f"  경로: {save_path}")
    print(f"  모델: {model_name}")
    print(f"  F1-macro: {f1_score_val:.4f}")
    print(f"  클래스: {list(label_encoder.classes_)}")
    print(f"  저장 시각: {artifact['saved_at']}")


def load_model_with_meta(path: str) -> dict:
    """
    저장된 모델과 메타데이터 로드

    Args:
        path: 모델 파일 경로 (.joblib)

    Returns:
        dict: {model, label_encoder, f1_macro, model_name, ...}
    """
    artifact = joblib.load(path)

    print(f"[모델 로드 완료]")
    print(f"  경로: {path}")
    print(f"  모델: {artifact.get('model_name', 'N/A')}")
    print(f"  F1-macro: {artifact.get('f1_macro', 'N/A')}")
    print(f"  클래스: {artifact.get('classes', [])}")

    return artifact


# ──────────────────────────────────────────────────────────────────────────────
# Architecture v1.2 §8 - predict_with_uncertainty (S3b 신규)
# ──────────────────────────────────────────────────────────────────────────────

def predict_with_uncertainty(
    model,
    X,
    pk_events: Optional[List[str]] = None,
    calibrator=None,
    ood_detector=None,
    X_dense: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """ML 예측 + 불확실성 지표 계산.

    Args:
        model       : 학습된 분류기 (predict_proba 지원)
        X           : 입력 피처 (sparse or dense)
        pk_events   : 이벤트 PK 목록 (None이면 자동 생성)
        calibrator  : ClasswiseCalibrator (선택)
        ood_detector: OODDetector (선택)
        X_dense     : OOD 탐지용 dense 피처 (선택)
        label_names : 클래스 이름 목록 (선택)

    Returns:
        ml_predictions DataFrame - Architecture §8 스키마
    """
    if calibrator is not None:
        proba = calibrator.predict_proba(X)
    else:
        proba = model.predict_proba(X)

    n_samples, n_classes = proba.shape

    # Top-1, Top-2 인덱스 및 확률
    sorted_idx = np.argsort(proba, axis=1)[:, ::-1]
    top1_idx = sorted_idx[:, 0]
    top2_idx = sorted_idx[:, 1] if n_classes > 1 else sorted_idx[:, 0]

    top1_proba = proba[np.arange(n_samples), top1_idx]
    top2_proba = proba[np.arange(n_samples), top2_idx]

    if label_names is not None:
        top1_names = [label_names[i] if i < len(label_names) else str(i) for i in top1_idx]
        top2_names = [label_names[i] if i < len(label_names) else str(i) for i in top2_idx]
    else:
        top1_names = [str(i) for i in top1_idx]
        top2_names = [str(i) for i in top2_idx]

    # 불확실성 지표
    margin = top1_proba - top2_proba
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)

    # TP 확률 — LabelEncoder 알파벳순이므로 TP 인덱스를 동적으로 탐색
    _tp_idx = n_classes - 1  # 기본: 마지막 클래스
    if label_names is not None:
        for _i, _n in enumerate(label_names):
            if "TP" in str(_n):
                _tp_idx = _i
                break
    tp_proba = proba[:, _tp_idx] if n_classes > 0 else np.zeros(n_samples)

    # OOD
    if ood_detector is not None and X_dense is not None:
        ood_distances = ood_detector.score(X_dense)
        ood_flags = ood_detector.predict(X_dense)
    else:
        ood_distances = np.zeros(n_samples)
        ood_flags = np.zeros(n_samples, dtype=bool)

    return pd.DataFrame({
        "pk_event":          pk_events if pk_events else [f"evt_{i}" for i in range(n_samples)],
        "ml_top1_class":     top1_idx.tolist(),
        "ml_top1_class_name": top1_names,
        "ml_top1_proba":     top1_proba,
        "ml_top2_class":     top2_idx.tolist(),
        "ml_top2_class_name": top2_names,
        "ml_top2_proba":     top2_proba,
        "ml_margin":         margin,
        "ml_entropy":        entropy,
        "ml_tp_proba":       tp_proba,
        "ood_mahalanobis":   ood_distances,
        "ood_leaf_support":  None,        # Tier 1 stub
        "ood_flag":          ood_flags,
    })
