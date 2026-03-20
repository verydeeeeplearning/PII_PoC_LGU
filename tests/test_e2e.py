"""End-to-End 파이프라인 테스트

단일 명령어로 전체 파이프라인을 검증합니다:
  더미 데이터 생성 → 3-Layer Filter → Feature Engineering → 모델 학습 → 평가

실행:
    pytest tests/test_e2e.py -v
    pytest tests/ -v                    # 전체 (단위 + E2E)
    pytest tests/ -m e2e -v             # E2E만
    pytest tests/ -m "not e2e" -v       # E2E 제외 (단위만)
"""
import sys
import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import set_seed
from src.utils.constants import (
    RANDOM_SEED, TEXT_COLUMN, LABEL_COLUMN, FILE_PATH_COLUMN,
    LABEL_NAMES, LABEL_TP, TFIDF_MAX_FEATURES, TEST_SIZE,
)
from scripts.generate_dummy_data import generate_dummy_data


# ── Fixtures ──

@pytest.fixture(scope="module")
def e2e_tmp_dir():
    """E2E 테스트용 임시 디렉토리 (모듈 단위 공유)"""
    tmp = tempfile.mkdtemp(prefix="pii_e2e_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def dummy_df():
    """더미 데이터 생성 (모듈 단위 캐시)"""
    set_seed(RANDOM_SEED)
    df = generate_dummy_data(samples_per_class=50)
    return df


# ── 1. 데이터 생성 검증 ──

@pytest.mark.e2e
class TestDataGeneration:
    """Phase 1: 더미 데이터 생성 및 기본 검증"""

    def test_dummy_data_shape(self, dummy_df):
        """8 클래스 x 50건 = 400건 생성"""
        assert len(dummy_df) == 8 * 50
        assert TEXT_COLUMN in dummy_df.columns
        assert LABEL_COLUMN in dummy_df.columns
        assert FILE_PATH_COLUMN in dummy_df.columns

    def test_all_classes_present(self, dummy_df):
        """8개 클래스 모두 존재"""
        unique_labels = set(dummy_df[LABEL_COLUMN].unique())
        for label in LABEL_NAMES:
            assert label in unique_labels, f"Missing label: {label}"

    def test_class_balance(self, dummy_df):
        """각 클래스 50건씩 균등 분포"""
        counts = dummy_df[LABEL_COLUMN].value_counts()
        for label in LABEL_NAMES:
            assert counts[label] == 50

    def test_no_null_text(self, dummy_df):
        """텍스트 컬럼에 null 없음"""
        assert dummy_df[TEXT_COLUMN].isna().sum() == 0

    def test_csv_round_trip(self, dummy_df, e2e_tmp_dir):
        """CSV 저장/로드 라운드트립"""
        csv_path = e2e_tmp_dir / "test_data.csv"
        dummy_df.to_csv(csv_path, index=False, encoding="utf-8")
        df_loaded = pd.read_csv(csv_path)
        assert len(df_loaded) == len(dummy_df)
        assert list(df_loaded.columns) == list(dummy_df.columns)


# ── 2. 3-Layer Filter 검증 ──

@pytest.mark.e2e
class TestFilterPipeline:
    """Phase 2: 3-Layer Filter 파이프라인 통합 검증"""

    def test_filter_pipeline_runs(self, dummy_df):
        """FilterPipeline이 에러 없이 실행"""
        from src.filters import FilterPipeline

        pipeline = FilterPipeline()
        result = pipeline.apply(
            df=dummy_df,
            text_column=TEXT_COLUMN,
            file_path_column=FILE_PATH_COLUMN,
        )

        assert result.total_input == len(dummy_df)
        assert result.total_filtered + result.total_ml_input == result.total_input

    def test_filter_preserves_columns(self, dummy_df):
        """필터 통과 후 컬럼 보존"""
        from src.filters import FilterPipeline

        pipeline = FilterPipeline()
        result = pipeline.apply(
            df=dummy_df,
            text_column=TEXT_COLUMN,
            file_path_column=FILE_PATH_COLUMN,
        )

        # ML 입력 데이터에 필수 컬럼 존재
        if len(result.ml_input_df) > 0:
            assert TEXT_COLUMN in result.ml_input_df.columns
            assert LABEL_COLUMN in result.ml_input_df.columns

    def test_filter_no_data_loss(self, dummy_df):
        """필터 적용 후 데이터 유실 없음"""
        from src.filters import FilterPipeline

        pipeline = FilterPipeline()
        result = pipeline.apply(
            df=dummy_df,
            text_column=TEXT_COLUMN,
            file_path_column=FILE_PATH_COLUMN,
        )

        total_accounted = len(result.filtered_df) + len(result.ml_input_df)
        assert total_accounted == len(dummy_df)

    def test_filter_single_text(self):
        """단일 텍스트 테스트 (디버깅 인터페이스)"""
        from src.filters import FilterPipeline

        pipeline = FilterPipeline()

        # 내부 도메인 → Layer 1에서 필터링
        result = pipeline.test_single("user@lguplus.co.kr")
        assert result["goes_to_ml"] is False

        # 일반 텍스트 → ML로 전달
        result = pipeline.test_single("고객 홍길동의 주문이 접수되었습니다")
        assert result["goes_to_ml"] is True


# ── 3. Feature Engineering 검증 ──

@pytest.mark.e2e
class TestFeatureEngineering:
    """Phase 3: Feature Engineering 파이프라인 검증"""

    @pytest.fixture(scope="class")
    def ml_input_df(self, dummy_df):
        """필터 통과 후 ML 입력 데이터"""
        from src.filters import FilterPipeline

        pipeline = FilterPipeline()
        result = pipeline.apply(
            df=dummy_df,
            text_column=TEXT_COLUMN,
            file_path_column=FILE_PATH_COLUMN,
        )
        return result.ml_input_df

    def test_tfidf_features(self, ml_input_df):
        """TF-IDF Feature 생성"""
        from src.features.text_features import create_tfidf_features
        from sklearn.model_selection import train_test_split

        texts = ml_input_df[TEXT_COLUMN].fillna("")
        train_texts, test_texts = train_test_split(
            texts, test_size=0.2, random_state=RANDOM_SEED
        )

        tfidf_train, tfidf_test, vectorizer = create_tfidf_features(
            train_texts, test_texts, max_features=500  # E2E용 축소
        )

        assert tfidf_train.shape[0] == len(train_texts)
        assert tfidf_test.shape[0] == len(test_texts)
        assert tfidf_train.shape[1] == tfidf_test.shape[1]
        assert tfidf_train.shape[1] <= 500

    def test_keyword_features(self, ml_input_df):
        """키워드 Feature 생성"""
        from src.features.text_features import create_keyword_features

        kw_df = create_keyword_features(ml_input_df[TEXT_COLUMN])
        assert len(kw_df) == len(ml_input_df)
        assert kw_df.shape[1] > 0

    def test_text_stat_features(self, ml_input_df):
        """텍스트 통계 Feature 생성"""
        from src.features.text_features import create_text_stat_features

        stat_df = create_text_stat_features(ml_input_df[TEXT_COLUMN])
        assert len(stat_df) == len(ml_input_df)
        assert "text_length" in stat_df.columns

    def test_path_features(self, ml_input_df):
        """파일 경로 Feature 생성"""
        from src.features.tabular_features import create_file_path_features

        fp_df = create_file_path_features(ml_input_df, path_column=FILE_PATH_COLUMN)
        assert len(fp_df) == len(ml_input_df)

    def test_build_features_pipeline(self, ml_input_df):
        """통합 Feature 파이프라인 (build_features)"""
        from src.features.pipeline import build_features

        result = build_features(
            ml_input_df,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            test_size=0.2,
            tfidf_max_features=500,
        )

        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result
        assert result["X_train"].shape[0] > 0
        assert result["X_test"].shape[0] > 0
        assert result["X_train"].shape[1] == result["X_test"].shape[1]


# ── 4. 모델 학습 검증 ──

@pytest.mark.e2e
class TestModelTraining:
    """Phase 4: 모델 학습 및 예측 검증"""

    @pytest.fixture(scope="class")
    def feature_result(self, dummy_df):
        """필터 + Feature 파이프라인 결과"""
        set_seed(RANDOM_SEED)

        from src.filters import FilterPipeline
        from src.features.pipeline import build_features

        pipeline = FilterPipeline()
        filter_result = pipeline.apply(
            df=dummy_df,
            text_column=TEXT_COLUMN,
            file_path_column=FILE_PATH_COLUMN,
        )

        ml_df = filter_result.ml_input_df
        result = build_features(
            ml_df,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            test_size=0.2,
            tfidf_max_features=500,
        )
        return result

    def test_label_encoding(self, feature_result):
        """레이블 인코딩"""
        from src.models.trainer import encode_labels

        y_train_enc, y_test_enc, le = encode_labels(
            feature_result["y_train"], feature_result["y_test"]
        )

        assert len(le.classes_) > 1
        assert y_train_enc.max() < len(le.classes_)
        assert y_test_enc.max() < len(le.classes_)

    def test_baseline_training(self, feature_result):
        """Baseline 모델 (DummyClassifier) 학습"""
        from src.models.trainer import encode_labels, train_baseline

        y_train_enc, y_test_enc, le = encode_labels(
            feature_result["y_train"], feature_result["y_test"]
        )

        model, f1 = train_baseline(
            feature_result["X_train"], y_train_enc,
            feature_result["X_test"], y_test_enc,
        )

        assert f1 >= 0.0
        assert hasattr(model, "predict")

    def test_xgboost_training(self, feature_result):
        """XGBoost 모델 학습"""
        from src.models.trainer import encode_labels, train_xgboost

        y_train_enc, y_test_enc, le = encode_labels(
            feature_result["y_train"], feature_result["y_test"]
        )

        model, f1, report = train_xgboost(
            feature_result["X_train"], y_train_enc,
            feature_result["X_test"], y_test_enc,
            label_encoder=le,
            params={
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_estimators": 50,  # E2E 속도 최적화
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "verbosity": 0,
            },
        )

        assert f1 > 0.0
        assert hasattr(model, "predict")
        y_pred = model.predict(feature_result["X_test"])
        assert len(y_pred) == feature_result["X_test"].shape[0]

    def test_lightgbm_training(self, feature_result):
        """LightGBM 모델 학습"""
        from src.models.trainer import encode_labels, train_lightgbm

        y_train_enc, y_test_enc, le = encode_labels(
            feature_result["y_train"], feature_result["y_test"]
        )

        model, f1, report = train_lightgbm(
            feature_result["X_train"], y_train_enc,
            feature_result["X_test"], y_test_enc,
            label_encoder=le,
            params={
                "objective": "multiclass",
                "metric": "multi_logloss",
                "n_estimators": 50,  # E2E 속도 최적화
                "num_leaves": 16,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "verbose": -1,
            },
        )

        assert f1 > 0.0
        assert hasattr(model, "predict")


# ── 5. 평가 검증 ──

@pytest.mark.e2e
class TestEvaluation:
    """Phase 5: 모델 평가 파이프라인 검증"""

    @pytest.fixture(scope="class")
    def trained_result(self, dummy_df, e2e_tmp_dir):
        """전체 파이프라인 결과 (필터 → Feature → 학습 → 예측)"""
        set_seed(RANDOM_SEED)

        from src.filters import FilterPipeline
        from src.features.pipeline import build_features
        from src.models.trainer import encode_labels, train_xgboost

        # 필터
        pipeline = FilterPipeline()
        filter_result = pipeline.apply(
            df=dummy_df,
            text_column=TEXT_COLUMN,
            file_path_column=FILE_PATH_COLUMN,
        )

        # Feature
        ml_df = filter_result.ml_input_df
        feat = build_features(
            ml_df,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            test_size=0.2,
            tfidf_max_features=500,
        )

        # 학습
        y_train_enc, y_test_enc, le = encode_labels(
            feat["y_train"], feat["y_test"]
        )

        model, f1, report = train_xgboost(
            feat["X_train"], y_train_enc,
            feat["X_test"], y_test_enc,
            label_encoder=le,
            params={
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "verbosity": 0,
            },
        )

        y_pred = model.predict(feat["X_test"])

        return {
            "model": model,
            "le": le,
            "y_test_enc": y_test_enc,
            "y_pred": y_pred,
            "f1": f1,
            "feature_names": feat["feature_names"],
            "df_test": feat["df_test"],
            "X_test": feat["X_test"],
            "filter_result": filter_result,
        }

    def test_full_evaluation(self, trained_result, e2e_tmp_dir):
        """전체 평가 리포트 생성"""
        from src.evaluation.evaluator import full_evaluation

        report_dir = str(e2e_tmp_dir / "reports")

        results = full_evaluation(
            trained_result["y_test_enc"],
            trained_result["y_pred"],
            list(trained_result["le"].classes_),
            save_dir=report_dir,
        )

        assert "f1_macro" in results
        assert "confusion_matrix" in results
        assert "classification_report" in results
        assert results["f1_macro"] >= 0.0

        # 리포트 파일 생성 확인
        assert (Path(report_dir) / "classification_report.txt").exists()

    def test_poc_criteria_check(self, trained_result):
        """PoC 성공 기준 판정 실행"""
        from src.evaluation.evaluator import check_poc_criteria

        criteria = check_poc_criteria(
            trained_result["y_test_enc"],
            trained_result["y_pred"],
            list(trained_result["le"].classes_),
        )

        # 결과 구조 검증 (PASS/FAIL은 데이터 의존이므로 구조만 확인)
        assert isinstance(criteria, dict)
        for key, val in criteria.items():
            assert "value" in val
            assert "pass" in val
            assert isinstance(val["value"], (float, np.floating))
            assert val["pass"] in (True, False)

    def test_error_analysis(self, trained_result, e2e_tmp_dir):
        """오분류 분석"""
        from src.evaluation.evaluator import analyze_errors

        error_path = str(e2e_tmp_dir / "error_analysis.csv")
        error_df = analyze_errors(
            trained_result["y_test_enc"],
            trained_result["y_pred"],
            trained_result["df_test"],
            list(trained_result["le"].classes_),
            text_column=TEXT_COLUMN,
            save_path=error_path,
        )

        assert isinstance(error_df, pd.DataFrame)

    def test_feature_importance(self, trained_result, e2e_tmp_dir):
        """Feature Importance 분석"""
        from src.evaluation.evaluator import feature_importance_analysis

        fig_path = str(e2e_tmp_dir / "feature_importance.png")
        report_path = str(e2e_tmp_dir / "feature_importance.csv")

        fi_df = feature_importance_analysis(
            trained_result["model"],
            trained_result["feature_names"],
            top_n=10,
            save_path=fig_path,
            report_path=report_path,
        )

        assert isinstance(fi_df, pd.DataFrame)
        assert len(fi_df) > 0
        assert Path(fig_path).exists()
        assert Path(report_path).exists()

    def test_model_save_load(self, trained_result, e2e_tmp_dir):
        """모델 저장/로드 라운드트립"""
        from src.models.trainer import save_model_with_meta, load_model_with_meta

        model_path = str(e2e_tmp_dir / "test_model.joblib")

        save_model_with_meta(
            model=trained_result["model"],
            path=model_path,
            label_encoder=trained_result["le"],
            f1_score_val=trained_result["f1"],
            model_name="e2e_test_xgboost",
        )

        assert Path(model_path).exists()

        artifact = load_model_with_meta(model_path)
        assert "model" in artifact
        assert "label_encoder" in artifact

        # 로드된 모델로 예측 가능한지 확인
        y_pred_loaded = artifact["model"].predict(trained_result["X_test"])
        np.testing.assert_array_equal(y_pred_loaded, trained_result["y_pred"])


# ── 6. 전체 파이프라인 통합 (단일 테스트) ──

@pytest.mark.e2e
class TestFullPipeline:
    """Phase 6: 전체 E2E 파이프라인 - 데이터 생성부터 평가까지 단일 흐름"""

    def test_end_to_end_pipeline(self, e2e_tmp_dir):
        """
        전체 파이프라인을 하나의 테스트로 검증:
        데이터 생성 → 필터 → Feature → 학습 → 예측 → 평가 → 모델 저장
        """
        set_seed(RANDOM_SEED)

        from src.filters import FilterPipeline
        from src.features.pipeline import build_features, save_feature_artifacts
        from src.features.text_features import create_all_text_features
        from src.features.tabular_features import create_all_path_features
        from src.models.trainer import (
            encode_labels, train_xgboost, save_model_with_meta,
        )
        from src.evaluation.evaluator import full_evaluation

        # --- 1. 데이터 생성 ---
        df = generate_dummy_data(samples_per_class=30)
        assert len(df) == 8 * 30

        # --- 2. 3-Layer Filter ---
        pipeline = FilterPipeline()
        filter_result = pipeline.apply(
            df=df,
            text_column=TEXT_COLUMN,
            file_path_column=FILE_PATH_COLUMN,
        )
        ml_df = filter_result.ml_input_df
        assert len(ml_df) > 0, "ML 입력 데이터가 비어있음"

        # 필터된 데이터 + ML 데이터 = 전체
        assert filter_result.total_filtered + filter_result.total_ml_input == len(df)

        # --- 3. 확장 Feature 추가 ---
        texts = ml_df[TEXT_COLUMN].fillna("")
        df_text_feat = create_all_text_features(texts)
        df_path_feat = create_all_path_features(ml_df, FILE_PATH_COLUMN)

        for col in df_text_feat.columns:
            if col not in ml_df.columns:
                ml_df[col] = df_text_feat[col].values
        for col in df_path_feat.columns:
            if col not in ml_df.columns:
                ml_df[col] = df_path_feat[col].values

        # --- 4. Feature Pipeline ---
        feat = build_features(
            ml_df,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            test_size=0.2,
            tfidf_max_features=300,
        )

        # Feature 아티팩트 저장
        feat_dir = str(e2e_tmp_dir / "features")
        model_dir = str(e2e_tmp_dir / "models")
        save_feature_artifacts(feat, feat_dir, model_dir)

        # --- 5. 레이블 인코딩 ---
        y_train_enc, y_test_enc, le = encode_labels(
            feat["y_train"], feat["y_test"]
        )
        assert len(le.classes_) > 1

        # --- 6. 모델 학습 (XGBoost - 경량 파라미터) ---
        model, f1, report = train_xgboost(
            feat["X_train"], y_train_enc,
            feat["X_test"], y_test_enc,
            label_encoder=le,
            params={
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_estimators": 30,
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_jobs": -1,
                "verbosity": 0,
            },
        )
        assert f1 > 0.0, f"F1-macro가 0: 모델이 학습되지 않았을 수 있음"

        # --- 7. 평가 ---
        y_pred = model.predict(feat["X_test"])
        report_dir = str(e2e_tmp_dir / "reports")

        eval_results = full_evaluation(
            y_test_enc, y_pred, list(le.classes_), save_dir=report_dir,
        )

        assert eval_results["f1_macro"] > 0.0
        assert (Path(report_dir) / "classification_report.txt").exists()

        # --- 8. 모델 저장 ---
        final_model_path = str(e2e_tmp_dir / "models" / "best_model.joblib")
        save_model_with_meta(
            model=model,
            path=final_model_path,
            label_encoder=le,
            f1_score_val=f1,
            model_name="e2e_xgboost",
            train_size=feat["X_train"].shape[0],
            test_size=feat["X_test"].shape[0],
            feature_count=feat["X_train"].shape[1],
        )
        assert Path(final_model_path).exists()

        # --- 최종 요약 출력 ---
        print("\n" + "=" * 60)
        print("[E2E 파이프라인 완료]")
        print("=" * 60)
        print(f"  입력 데이터:       {len(df):,}건")
        print(f"  필터 분류:         {filter_result.total_filtered:,}건")
        print(f"  ML 학습 대상:      {filter_result.total_ml_input:,}건")
        print(f"  학습셋:            {feat['X_train'].shape[0]:,}건")
        print(f"  테스트셋:          {feat['X_test'].shape[0]:,}건")
        print(f"  Feature 수:        {feat['X_train'].shape[1]:,}")
        print(f"  클래스 수:         {len(le.classes_)}")
        print(f"  F1-macro:          {eval_results['f1_macro']:.4f}")
        print(f"  모델 저장:         {final_model_path}")
        print("=" * 60)
