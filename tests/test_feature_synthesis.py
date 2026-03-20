"""합성 변수 확장 Feature 테스트"""
import pytest
import pandas as pd

from scripts.generate_dummy_data import generate_dummy_data
from src.features.pipeline import build_features, create_synthetic_interaction_features
from src.utils.common import set_seed
from src.utils.constants import RANDOM_SEED, TEXT_COLUMN, LABEL_COLUMN


class TestSyntheticInteractionFeatures:
    """텍스트/경로 상호작용 합성변수 검증"""

    def test_interaction_feature_values(self):
        text_df = pd.DataFrame(
            {
                "has_internal_domain": [1, 0],
                "has_os_copyright_domain": [1, 0],
                "has_dummy_domain": [0, 1],
                "has_timestamp_pattern": [1, 0],
                "has_bytes_pattern": [1, 0],
                "has_version_pattern": [1, 1],
                "masking_ratio": [0.5, 0.1],
                "digit_ratio": [0.3, 0.2],
                "special_char_ratio": [0.4, 0.2],
            }
        )
        path_df = pd.DataFrame(
            {
                "file_is_log": [1, 0],
                "path_is_system": [1, 0],
                "path_is_temp": [0, 1],
                "path_has_test": [0, 1],
                "path_has_dev": [1, 0],
                "path_has_hadoop": [1, 0],
                "path_has_legacy_date": [1, 0],
                "path_depth": [5, 2],
            }
        )

        syn = create_synthetic_interaction_features(text_df, path_df)

        assert syn["syn_internal_domain_x_log_path"].tolist() == [1, 0]
        assert syn["syn_timestamp_x_legacy_path"].tolist() == [1, 0]
        assert syn["syn_dummy_domain_x_test_path"].tolist() == [0, 1]
        assert syn["syn_masking_ratio_x_path_depth"].tolist() == pytest.approx([2.5, 0.2])


class TestBuildFeaturesSyntheticExpansion:
    """build_features의 합성 변수 확장 동작 검증"""

    def test_expansion_adds_more_features(self):
        set_seed(RANDOM_SEED)
        df = generate_dummy_data(samples_per_class=12)

        baseline = build_features(
            df=df,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            test_size=0.2,
            tfidf_max_features=200,
            use_synthetic_expansion=False,
        )
        expanded = build_features(
            df=df,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            test_size=0.2,
            tfidf_max_features=200,
            use_synthetic_expansion=True,
        )

        assert expanded["X_train"].shape[1] > baseline["X_train"].shape[1]
        assert any(name.startswith("syn_") for name in expanded["feature_names"])
