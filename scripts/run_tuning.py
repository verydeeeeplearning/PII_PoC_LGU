"""Hyperparameter tuning script for XGBoost (can be slow).

Usage:
    python scripts/run_tuning.py
    python scripts/run_tuning.py --n-iter 50 --cv 5
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.metrics import f1_score

# Make imports work even if the script is executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import set_seed, ensure_dirs
from src.utils.constants import (
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
    MERGED_CLEANED_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    FEATURE_DIR,
    MODEL_DIR,
    XGB_PARAM_GRID,
    TUNING_N_ITER,
    TUNING_CV_FOLDS,
)
from src.features.pipeline import build_features, save_feature_artifacts
from src.models.trainer import encode_labels, tune_model, save_model_with_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost hyperparameter tuning")
    parser.add_argument("--n-iter", type=int, default=TUNING_N_ITER, help="RandomizedSearch n_iter")
    parser.add_argument("--cv", type=int, default=TUNING_CV_FOLDS, help="CV folds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    ensure_dirs(
        MODEL_DIR / "experiments",
        FEATURE_DIR,
    )

    data_path = PROCESSED_DATA_DIR / MERGED_CLEANED_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed data: {data_path}")

    print("=" * 60)
    print("[Step 1] Load processed data")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"  Data: {df.shape[0]:,} rows x {df.shape[1]} cols")

    print("\n" + "=" * 60)
    print("[Step 2] Feature engineering")
    print("=" * 60)
    result = build_features(
        df=df,
        text_column=TEXT_COLUMN,
        label_column=LABEL_COLUMN,
        test_size=TEST_SIZE,
        tfidf_max_features=TFIDF_MAX_FEATURES,
        use_synthetic_expansion=True,
    )
    save_feature_artifacts(result, str(FEATURE_DIR), str(MODEL_DIR))

    X_train, X_test = result["X_train"], result["X_test"]
    y_train, y_test = result["y_train"], result["y_test"]

    print("\n" + "=" * 60)
    print("[Step 3] Label encoding")
    print("=" * 60)
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    print("\n" + "=" * 60)
    print("[Step 4] XGBoost tuning (RandomizedSearchCV)")
    print("=" * 60)
    best_model, best_params, best_cv_score = tune_model(
        model_type="xgboost",
        X_train=X_train,
        y_train_enc=y_train_enc,
        param_grid=XGB_PARAM_GRID,
        cv=args.cv,
        n_iter=args.n_iter,
    )

    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test_enc, y_pred, average="macro")

    print("\n" + "=" * 60)
    print("[Result] Tuning summary")
    print("=" * 60)
    print(f"  Best CV score: {best_cv_score:.4f}")
    print(f"  Best params:   {best_params}")
    print(f"  Test F1-macro: {f1:.4f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = MODEL_DIR / "experiments" / f"xgb_tuned_{stamp}.joblib"
    save_model_with_meta(
        model=best_model,
        path=str(save_path),
        label_encoder=le,
        f1_score_val=f1,
        model_name=f"XGBoost (tuned, cv={args.cv}, n_iter={args.n_iter})",
        all_results={
            "best_params": best_params,
            "best_cv_score": best_cv_score,
        },
        train_size=X_train.shape[0],
        test_size=X_test.shape[0],
        feature_count=X_train.shape[1],
    )

    print("\nTuning pipeline complete.")


if __name__ == "__main__":
    main()
