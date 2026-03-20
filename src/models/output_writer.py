"""S5 Output Writer - Architecture v1.2 §8.12

판정 결과를 최종 출력 형식으로 변환한다.

출력:
    predictions_main  : 이벤트별 최종 판정 (Silver 메타 포함)
    prediction_evidence: long-format evidence (RULE + ML 결합)
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def build_predictions_main(
    df_decisions: pd.DataFrame,
    df_silver: pd.DataFrame,
    run_id: str,
) -> pd.DataFrame:
    """판정 결과 + Silver 메타데이터 결합 -> predictions_main DataFrame.

    Args:
        df_decisions: combine_decisions 결과 DataFrame
                      (pk_event, pk_file, primary_class, reason_code,
                       confidence, decision_source, risk_flag)
        df_silver   : Silver layer DataFrame (메타데이터 포함)
        run_id      : 실행 식별자 (e.g. "run001")

    Returns:
        predictions_main DataFrame
    """
    _SILVER_META_COLS = [
        "pk_event", "pk_file", "server_name", "agent_ip",
        "file_path", "file_name", "pii_type_inferred", "detection_time",
    ]
    meta_cols = [c for c in _SILVER_META_COLS if c in df_silver.columns]
    df_meta = df_silver[meta_cols]

    # pk_event 기준 left join
    if "pk_event" in df_decisions.columns and "pk_event" in df_meta.columns:
        df = df_decisions.merge(df_meta, on="pk_event", how="left", suffixes=("", "_silver"))
    else:
        df = df_decisions.copy()

    df["run_id"]   = run_id
    df["run_date"] = date.today().isoformat()

    return df.reset_index(drop=True)


def build_prediction_evidence(
    rule_evidence: pd.DataFrame,
    ml_evidence: pd.DataFrame,
) -> pd.DataFrame:
    """RULE evidence + ML evidence -> long-format prediction_evidence DataFrame.

    Args:
        rule_evidence: RuleLabeler rule_evidence_df (long-format)
        ml_evidence  : generate_lightweight_evidence / SHAP evidence df

    Returns:
        pd.DataFrame with pk_event column and combined evidence rows
    """
    frames = []
    if rule_evidence is not None and len(rule_evidence) > 0:
        frames.append(rule_evidence)
    if ml_evidence is not None and len(ml_evidence) > 0:
        frames.append(ml_evidence)

    if not frames:
        return pd.DataFrame(columns=["pk_event"])

    return pd.concat(frames, ignore_index=True)
