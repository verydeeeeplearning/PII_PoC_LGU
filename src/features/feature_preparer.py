"""공통 피처 준비 함수 — training/inference 동형성 보장.

run_training.py Steps 2-4의 공통 로직을 단일 함수로 추출하여,
학습(run_training.py)과 추론(feature_builder_snapshot.py transform)이
동일한 피처 생성 경로를 사용하도록 한다.

내부 처리:
    1. build_meta_features() — fname/detection/server 피처
    2. extract_path_features() — 경로 기반 15개 바이너리 피처
    3. RuleLabeler — rule_matched, rule_id, rule_primary_class, rule_confidence_lb
    4. 숫자 변환 — rule_matched→int, rule_confidence_lb→float
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def prepare_phase1_features(
    df: pd.DataFrame,
    rules_yaml_path: Optional[Path] = None,
    rule_stats_path: Optional[Path] = None,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """메타/경로/룰 피처를 생성하여 DataFrame에 추가.

    Parameters
    ----------
    df : S1 파싱 완료된 DataFrame (silver_label 또는 silver_detections)
    rules_yaml_path : rules.yaml 경로 (None이면 RuleLabeler 건너뜀)
    rule_stats_path : rule_stats.json 경로
    chunk_size : 메모리 절약용 청크 크기

    Returns
    -------
    피처가 추가된 DataFrame (원본 복사본)
    """
    from src.features.meta_features import build_meta_features
    from src.features.path_features import extract_path_features

    # ── 1. Meta + Path 피처 (청크 처리) ────────────────────────────────
    print(f"\n[피처 준비] 메타/경로 피처 추출 (청크: {chunk_size:,})")
    meta_chunks = []
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = build_meta_features(df.iloc[start:end].copy())

        if "file_path" in chunk.columns:
            path_feats = chunk["file_path"].apply(extract_path_features)
            path_df = pd.DataFrame(list(path_feats), index=chunk.index)
            for col in path_df.columns:
                if col not in chunk.columns:
                    chunk[col] = path_df[col]

        meta_chunks.append(chunk)
        print(f"  [{start:,}~{end:,}] 완료", flush=True)

    df = pd.concat(meta_chunks, ignore_index=True)
    del meta_chunks
    print(f"  메타/경로 피처 완료. 컬럼: {df.shape[1]}")

    # ── 2. RuleLabeler (청크 처리) ──────────────────────────────────────
    _rules_path = Path(rules_yaml_path) if rules_yaml_path else None
    _stats_path = Path(rule_stats_path) if rule_stats_path else None

    if _rules_path and _rules_path.exists():
        _stats_exists = _stats_path and _stats_path.exists()
        _stats_str = str(_stats_path) if _stats_exists else None

        from src.filters.rule_labeler import RuleLabeler

        print(f"\n[피처 준비] RuleLabeler 적용 (청크: {chunk_size:,})")
        labeler = RuleLabeler.from_config_files(str(_rules_path), _stats_str)
        label_chunks = []
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            chunk_labels, _ = labeler.label_batch(df.iloc[start:end])
            rule_cols = ["rule_matched", "rule_primary_class", "rule_id", "rule_confidence_lb"]
            rule_cols_present = [c for c in rule_cols if c in chunk_labels.columns]
            label_chunks.append(chunk_labels[rule_cols_present])
            print(f"  [{start:,}~{end:,}] 완료", flush=True)

        rule_df = pd.concat(label_chunks, ignore_index=True)
        del label_chunks

        df = df.reset_index(drop=True)
        for col in ["rule_matched", "rule_primary_class", "rule_id"]:
            if col in rule_df.columns:
                df[col] = rule_df[col].values
        if "rule_confidence_lb" in rule_df.columns:
            df["rule_confidence_lb"] = pd.to_numeric(
                rule_df["rule_confidence_lb"], errors="coerce"
            ).fillna(0.0).values
        del rule_df
        print(f"  RuleLabeler 완료. rule_matched + rule_confidence_lb 추가")
    else:
        print(f"\n[피처 준비] RuleLabeler 건너뜀 (rules.yaml 없음)")
        # 추론 시 RuleLabeler 없이도 동작하도록 기본값 설정
        if "rule_matched" not in df.columns:
            df["rule_matched"] = 0
        if "rule_confidence_lb" not in df.columns:
            df["rule_confidence_lb"] = 0.0
        if "rule_primary_class" not in df.columns:
            df["rule_primary_class"] = ""
        if "rule_id" not in df.columns:
            df["rule_id"] = ""

    # ── 3. 숫자 변환 ───────────────────────────────────────────────────
    if "rule_matched" in df.columns:
        df["rule_matched"] = df["rule_matched"].astype(int)
    if "rule_confidence_lb" in df.columns:
        df["rule_confidence_lb"] = pd.to_numeric(
            df["rule_confidence_lb"], errors="coerce"
        ).fillna(0.0)

    return df
