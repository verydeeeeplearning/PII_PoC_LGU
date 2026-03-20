"""src/models - ML 모델 모듈 (Architecture v1.2)

기존 모듈 (하위 호환):
    trainer.py          : encode_labels, train_xgboost, train_lightgbm,
                          save_model_with_meta, load_model_with_meta

S3b 신규 모듈:
    feature_builder.py  : MLFeatureBuilder (TF-IDF 4채널 + 수동 피처)
    calibrator.py       : ClasswiseCalibrator (Platt/Isotonic 차등 보정)
    ood_detector.py     : OODDetector (Mahalanobis Distance OOD 탐지)
    evidence_generator.py: generate_lightweight_evidence / generate_shap_evidence

S4+S5 신규 모듈:
    decision_combiner.py : combine_decisions (Case 0~3)
    output_writer.py     : write_predictions_main / write_prediction_evidence
    auto_adjudicator.py  : AutoAdjudicator (Step E Tier 1)
"""
