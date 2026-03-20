"""
src/features - Feature Engineering Modules (Architecture.md §6)

Stage S2 신규 모듈:
    text_prep.py        make_raw_text / make_shape_text / make_path_text
    keyword_flags.py    compute_keyword_flags (25+개 has_* 플래그)
    path_features.py    extract_path_features (경로 구조 피처 8개)
    synthetic.py        build_synthetic_features (Tier 0/1/2)
    file_agg.py         compute_file_aggregates / merge_file_aggregates
    schema_validator.py save_feature_schema / validate_feature_schema

기존 모듈 (하위 호환):
    text_features.py    TF-IDF / keyword / domain / masking features
    tabular_features.py 경로 피처 + extract_tabular_features (신규 추가)
    pipeline.py         build_features 파이프라인
"""
