"""월별 추론 실행 스크립트 (Inference-only) - Architecture v1.2 §20

학습된 모델로 신규 데이터를 추론하고 predictions_main.parquet을 생성한다.

사용법:
    python scripts/run_inference.py
    python scripts/run_inference.py --silver-path data/processed/silver_detections.parquet
    python scripts/run_inference.py --run-id 2025-02

입력:
    models/final/best_model_v1.joblib
    data/processed/silver_detections.parquet  (또는 --silver-path)

출력:
    outputs/predictions/predictions_main_{run_id}.parquet
    outputs/predictions/prediction_evidence_{run_id}.parquet
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import FINAL_MODEL_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="PII FPR 월별 추론 실행")
    parser.add_argument(
        "--silver-path",
        default=str(PROCESSED_DATA_DIR / "silver_detections.parquet"),
        help="Silver Parquet 경로",
    )
    parser.add_argument(
        "--model-path",
        default=str(FINAL_MODEL_DIR / "best_model_v1.joblib"),
        help="학습된 모델 경로",
    )
    parser.add_argument(
        "--rules-yaml",
        default=str(PROJECT_ROOT / "config" / "rules.yaml"),
        help="rules.yaml 경로",
    )
    parser.add_argument(
        "--rule-stats-json",
        default=str(PROJECT_ROOT / "config" / "rule_stats.json"),
        help="rule_stats.json 경로",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PREDICTIONS_DIR),
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y-%m"),
        help="실행 ID (기본: YYYY-MM)",
    )
    parser.add_argument(
        "--stage",
        choices=["s3a", "s3b", "s4s5", "all"],
        default="all",
        help="실행 단계 (기본: all - 전체 추론 파이프라인)",
    )
    return parser.parse_args()


def run_inference(args):
    """S3a -> S3b -> S4 -> S5 추론 파이프라인."""
    import pandas as pd
    from src.filters.rule_labeler import RuleLabeler
    from src.models.output_writer import build_predictions_main, build_prediction_evidence

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id
    print(f"\n{'=' * 60}")
    print(f"[PII FPR Inference] run_id={run_id}")
    print(f"{'=' * 60}")

    # ── Silver 데이터 로드 ────────────────────────────────────────────────────
    silver_path = Path(args.silver_path)
    if not silver_path.exists():
        print(f"[ERROR] Silver 파일 없음: {silver_path}")
        print("  먼저 run_data_pipeline.py를 실행하세요.")
        return

    print(f"\n[S1] Silver 로드: {silver_path}")
    df_silver = pd.read_parquet(silver_path)
    print(f"  -> {len(df_silver):,}건")

    # ── S3a: RULE Labeling ───────────────────────────────────────────────────
    rule_labels_df = None
    rule_evidence_df = None

    if args.stage in ("s3a", "all"):
        print("\n[S3a] RULE Labeling...")
        rules_yaml = Path(args.rules_yaml)
        rule_stats_json = Path(args.rule_stats_json)

        if rules_yaml.exists():
            labeler = RuleLabeler.from_config_files(
                str(rules_yaml), str(rule_stats_json)
            )
            rule_labels_df, rule_evidence_df = labeler.label_batch(df_silver)
            print(f"  -> rule_labels: {len(rule_labels_df):,}건")
            matched = rule_labels_df["rule_matched"].sum()
            print(f"  -> 매칭됨: {matched:,}건 ({matched/len(rule_labels_df)*100:.1f}%)")
        else:
            print(f"  [SKIP] rules.yaml 없음: {rules_yaml}")

    # ── S3b: ML Inference ────────────────────────────────────────────────────
    ml_predictions_df = None

    if args.stage in ("s3b", "all"):
        print("\n[S3b] ML Inference...")
        model_path = Path(args.model_path)

        if model_path.exists():
            from src.models.trainer import load_model_with_meta, predict_with_uncertainty
            from src.models.feature_builder import MLFeatureBuilder
            import joblib

            meta = load_model_with_meta(str(model_path))
            model = meta["model"]
            print(f"  -> 모델 로드: {model_path.name}")

            # 피처 빌더 로드 (모델과 동일 디렉토리에서)
            fb_path = model_path.parent / "feature_builder.joblib"
            if fb_path.exists():
                builder = MLFeatureBuilder.load(str(fb_path))
                X = builder.transform(df_silver)
                pk_events = df_silver["pk_event"].tolist() if "pk_event" in df_silver.columns else None
                ml_predictions_df = predict_with_uncertainty(model, X, pk_events=pk_events)
                print(f"  -> ml_predictions: {len(ml_predictions_df):,}건")
            else:
                print(f"  [SKIP] feature_builder.joblib 없음: {fb_path}")
        else:
            print(f"  [SKIP] 모델 없음: {model_path}")

    # ── S4+S5: Decision Combiner + Output Writer ──────────────────────────────
    if args.stage in ("s4s5", "all"):
        print("\n[S4+S5] Decision Combiner + Output Writer...")

        if rule_labels_df is None or ml_predictions_df is None:
            print("  [SKIP] S3a 또는 S3b 결과 없음")
        else:
            from src.models.decision_combiner import combine_decisions
            import pandas as pd

            decisions = []
            for i, rule_row in rule_labels_df.iterrows():
                ml_row = (
                    ml_predictions_df[ml_predictions_df["pk_event"] == rule_row["pk_event"]]
                    .iloc[0].to_dict()
                    if "pk_event" in ml_predictions_df.columns
                    else ml_predictions_df.iloc[i].to_dict()
                )
                dec = combine_decisions(rule_row.to_dict(), ml_row)
                dec["pk_event"] = rule_row.get("pk_event", f"evt_{i}")
                dec["pk_file"] = df_silver.iloc[i].get("pk_file", "") if i < len(df_silver) else ""
                decisions.append(dec)

            df_decisions = pd.DataFrame(decisions)
            df_main = build_predictions_main(df_decisions, df_silver, run_id=run_id)

            # Evidence 결합
            if rule_evidence_df is not None:
                from src.models.evidence_generator import generate_lightweight_evidence
                ml_evidence_rows = []
                for _, row in df_silver.iterrows():
                    ev_list = generate_lightweight_evidence(row.to_dict())
                    for ev in ev_list:
                        ev["pk_event"] = row.get("pk_event", "")
                    ml_evidence_rows.extend(ev_list)
                ml_ev_df = pd.DataFrame(ml_evidence_rows) if ml_evidence_rows else pd.DataFrame()
                df_evidence = build_prediction_evidence(rule_evidence_df, ml_ev_df)
            else:
                df_evidence = pd.DataFrame(columns=["pk_event"])

            # 저장
            main_path = output_dir / f"predictions_main_{run_id}.parquet"
            ev_path = output_dir / f"prediction_evidence_{run_id}.parquet"

            df_main.to_parquet(main_path, index=False)
            df_evidence.to_parquet(ev_path, index=False)

            print(f"  -> predictions_main: {len(df_main):,}건 -> {main_path}")
            print(f"  -> prediction_evidence: {len(df_evidence):,}건 -> {ev_path}")

            # 빠른 요약
            if "primary_class" in df_main.columns:
                dist = df_main["primary_class"].value_counts()
                print("\n[판정 분포]")
                for cls, cnt in dist.items():
                    print(f"  {cls}: {cnt:,}건 ({cnt/len(df_main)*100:.1f}%)")

    print(f"\n{'=' * 60}")
    print("[완료] 추론 파이프라인 종료")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
