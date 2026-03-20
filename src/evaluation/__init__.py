"""src/evaluation - 평가 모듈 (Architecture v1.2)

기존 모듈 (하위 호환):
    evaluator.py        : full_evaluation, check_poc_criteria, analyze_errors
    eda.py              : EDA 분석

S6 신규 모듈:
    split_strategies.py : group_time_split, server_group_split, event_random_split
    kpi_monitor.py      : compute_monthly_kpis, check_alarms, save_monthly_metrics
    calibration_eval.py : compute_ece, compute_mce, plot_reliability_diagram
    confident_learning.py: ConfidentLearningAuditor (Step E Tier 1)
"""
