"""모델 성능 신뢰성 진단 스크립트 - Phase 1: 데이터 편향 진단

폐쇄망에서 실행 가능. pandas/numpy/sklearn만 사용.
silver_label.parquet 필수.

사용법:
    python scripts/diagnose_data_bias.py
    python scripts/diagnose_data_bias.py --parquet data/processed/silver_joined.parquet
    python scripts/diagnose_data_bias.py --skip-single-feature  # 단일 피처 F1 생략 (빠른 실행)

산출물:
    outputs/diagnosis/
        column_bias_report.txt       - 컬럼별 Cramer's V, MI, 교차표
        row_quality_report.txt       - 중복/충돌/분포 통계
        single_feature_f1.csv        - 단일 피처 F1 랭킹
        server_label_crosstab.csv    - server_name × label_raw 교차표
        month_label_crosstab.csv     - label_work_month × label_raw 교차표
        bias_summary.md              - 종합 Go/No-Go 판정
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COL = "label_raw"
from src.utils.constants import DIAGNOSIS_DIR

OUTPUT_DIR = DIAGNOSIS_DIR

# 범주형 진단 대상 컬럼
CATEGORICAL_COLS = [
    "server_name",
    "organization",
    "service",
    "label_work_month",
    "_source_file",
]

# 수치형 진단 대상 컬럼
NUMERIC_COLS = [
    "pattern_count",
    "ssn_count",
    "phone_count",
    "email_count",
]

# 경로 피처 (extract_path_features 결과)
PATH_FEATURE_COLS = [
    "is_log_file", "is_docker_overlay", "has_license_path",
    "is_temp_or_dev", "is_system_device", "is_package_path",
    "has_cron_path", "has_date_in_path", "has_business_token",
    "has_system_token",
]

# 메타 피처 (build_meta_features 결과)
META_FEATURE_COLS = [
    "fname_has_date", "fname_has_hash", "fname_has_rotation_num",
    "pattern_count_log1p", "pattern_count_bin",
    "is_mass_detection", "is_extreme_detection", "pii_type_ratio",
    "created_hour", "created_weekday", "is_weekend", "created_month",
]

# Cramer's V 임계값
V_DANGER = 0.5
V_WARNING = 0.3

# 단일 피처 F1 임계값
F1_DANGER = 0.85
F1_WARNING = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V: 범주형 × 범주형 연관성 (0~1)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    if k == 0 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def single_class_ratio(col: pd.Series, label: pd.Series) -> dict:
    """범주형 컬럼에서 단일 클래스만 가진 값의 비율."""
    ct = pd.crosstab(col, label, normalize="index")
    total_groups = len(ct)
    if total_groups == 0:
        return {"total_groups": 0, "pure_tp": 0, "pure_fp": 0, "pure_ratio": 0.0}

    pure_tp = int((ct.max(axis=1) == 1.0).sum()) if "TP" in ct.columns else 0
    pure_fp = int((ct.max(axis=1) == 1.0).sum()) if "FP" in ct.columns else 0
    # 실제로는 max==1이면 어느 한쪽이 100%
    pure_count = int((ct.max(axis=1) >= 0.999).sum())
    return {
        "total_groups": total_groups,
        "pure_groups": pure_count,
        "pure_ratio": pure_count / total_groups,
    }


def write_line(f, text: str = ""):
    """파일과 콘솔 동시 출력."""
    # Windows cp949 콘솔에서 인코딩 불가 문자 대체
    safe_text = text.encode("cp949", errors="replace").decode("cp949", errors="replace")
    print(safe_text)
    f.write(text + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1-A: Column-wise 진단
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_categorical(df: pd.DataFrame, f) -> list:
    """범주형 컬럼 × label_raw 진단. returns list of (col, V, pure_ratio)."""
    write_line(f, "=" * 70)
    write_line(f, "A-1. 범주형 컬럼 × label_raw (Cramer's V + 단일클래스 비율)")
    write_line(f, "=" * 70)

    results = []
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            write_line(f, f"\n  [{col}] - 컬럼 없음 (skip)")
            continue

        valid = df[[col, LABEL_COL]].dropna()
        if len(valid) < 10:
            write_line(f, f"\n  [{col}] - 유효 행 {len(valid)}건 (skip)")
            continue

        v = cramers_v(valid[col], valid[LABEL_COL])
        sc = single_class_ratio(valid[col], valid[LABEL_COL])

        flag = ""
        if v > V_DANGER:
            flag = " *** DANGER: label proxy 가능성 ***"
        elif v > V_WARNING:
            flag = " ** WARNING: 유의미한 편향 **"

        write_line(f, f"\n  [{col}]{flag}")
        write_line(f, f"    Cramer's V:     {v:.4f}")
        write_line(f, f"    고유값 수:      {sc['total_groups']:,}")
        write_line(f, f"    단일클래스 그룹: {sc['pure_groups']:,} ({sc['pure_ratio']:.1%})")

        # 교차표 상위 10개
        ct = pd.crosstab(valid[col], valid[LABEL_COL], normalize="index")
        ct["count"] = pd.crosstab(valid[col], valid[LABEL_COL]).sum(axis=1)
        ct = ct.sort_values("count", ascending=False).head(10)
        write_line(f, f"    상위 10개 값:")
        for idx, row in ct.iterrows():
            vals = "  ".join(f"{c}={row[c]:.2f}" for c in ct.columns if c != "count")
            write_line(f, f"      {str(idx):40s}  n={int(row['count']):>6,}  {vals}")

        results.append((col, v, sc["pure_ratio"]))

        # server_name 교차표 전체 저장
        if col == "server_name":
            ct_full = pd.crosstab(valid[col], valid[LABEL_COL], margins=True)
            ct_full.to_csv(OUTPUT_DIR / "server_label_crosstab.csv", encoding="utf-8-sig")
            write_line(f, f"    [저장] server_label_crosstab.csv ({len(ct_full)-1}개 서버)")

        if col == "label_work_month":
            ct_full = pd.crosstab(valid[col], valid[LABEL_COL], margins=True)
            ct_full.to_csv(OUTPUT_DIR / "month_label_crosstab.csv", encoding="utf-8-sig")
            write_line(f, f"    [저장] month_label_crosstab.csv")

    return results


def diagnose_numeric(df: pd.DataFrame, f):
    """수치형 컬럼 × label_raw 분포 비교."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "A-2. 수치형 컬럼 × label_raw (분포 비교)")
    write_line(f, "=" * 70)

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        valid = df[[col, LABEL_COL]].dropna()
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
        valid = valid.dropna()

        tp_vals = valid.loc[valid[LABEL_COL] == "TP", col]
        fp_vals = valid.loc[valid[LABEL_COL] == "FP", col]

        if len(tp_vals) < 5 or len(fp_vals) < 5:
            write_line(f, f"\n  [{col}] - TP={len(tp_vals)}, FP={len(fp_vals)} (부족, skip)")
            continue

        # Mann-Whitney U test
        try:
            stat, pval = mannwhitneyu(tp_vals, fp_vals, alternative="two-sided")
        except Exception:
            stat, pval = 0, 1.0

        write_line(f, f"\n  [{col}]")
        write_line(f, f"    TP: mean={tp_vals.mean():.2f}  median={tp_vals.median():.2f}  std={tp_vals.std():.2f}  n={len(tp_vals):,}")
        write_line(f, f"    FP: mean={fp_vals.mean():.2f}  median={fp_vals.median():.2f}  std={fp_vals.std():.2f}  n={len(fp_vals):,}")
        write_line(f, f"    Mann-Whitney U p-value: {pval:.6f}  {'(유의)' if pval < 0.05 else '(비유의)'}")


def diagnose_path_features(df: pd.DataFrame, f) -> list:
    """경로 피처 × label_raw 연관성."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "A-3. 경로/메타 이진 피처 × label_raw")
    write_line(f, "=" * 70)

    from src.features.path_features import extract_path_features
    from src.features.meta_features import build_meta_features

    # 피처 추출 (아직 없으면)
    need_compute = not all(c in df.columns for c in PATH_FEATURE_COLS[:3])
    if need_compute:
        write_line(f, "  [피처 계산 중...] build_meta_features + extract_path_features")
        df = build_meta_features(df)
        if "file_path" in df.columns:
            path_feats = df["file_path"].apply(extract_path_features)
            path_df = pd.DataFrame(list(path_feats), index=df.index)
            for col in path_df.columns:
                if col not in df.columns:
                    df[col] = path_df[col]

    all_binary_cols = PATH_FEATURE_COLS + META_FEATURE_COLS
    results = []

    for col in all_binary_cols:
        if col not in df.columns:
            continue

        valid = df[[col, LABEL_COL]].dropna()
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
        valid = valid.dropna()

        ct = pd.crosstab(valid[col], valid[LABEL_COL])
        if ct.shape[0] < 2:
            continue

        v = cramers_v(valid[col].astype(str), valid[LABEL_COL])

        # flag=1일 때 TP/FP 비율
        flag_1 = valid[valid[col] >= 1]
        if len(flag_1) > 0:
            ct_norm = pd.crosstab(valid[col], valid[LABEL_COL], normalize="index")
            tp_when_1 = ct_norm.loc[ct_norm.index >= 1, "TP"].values[0] if "TP" in ct_norm.columns and (ct_norm.index >= 1).any() else 0
            fp_when_1 = ct_norm.loc[ct_norm.index >= 1, "FP"].values[0] if "FP" in ct_norm.columns and (ct_norm.index >= 1).any() else 0
        else:
            tp_when_1, fp_when_1 = 0, 0

        flag = " !!!" if v > V_WARNING else ""
        write_line(f, f"  {col:35s}  V={v:.3f}  flag=1→TP:{tp_when_1:.2f} FP:{fp_when_1:.2f}  (n={len(flag_1):,}){flag}")
        results.append((col, v))

    return results


def diagnose_mutual_information(df: pd.DataFrame, f):
    """상호정보량 종합 랭킹."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "A-4. 상호정보량 (Mutual Information) 종합 랭킹")
    write_line(f, "=" * 70)

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    # 수치형 + 이진 피처만 수집
    feature_cols = []
    for col in PATH_FEATURE_COLS + META_FEATURE_COLS + NUMERIC_COLS:
        if col in df.columns:
            feature_cols.append(col)

    # extension 추가
    if "file_name" in df.columns:
        df = df.copy()
        df["_ext"] = df["file_name"].str.rsplit(".", n=1).str[-1].str.lower().fillna("unknown")
        le_ext = LabelEncoder()
        df["_ext_enc"] = le_ext.fit_transform(df["_ext"])
        feature_cols.append("_ext_enc")

    # server_name frequency encoding (학습 시와 동일 방식)
    if "server_name" in df.columns:
        df = df.copy() if "_ext" not in df.columns else df
        _sn_freq = df["server_name"].value_counts(normalize=True)
        df["_server_freq"] = df["server_name"].map(_sn_freq).fillna(0)
        feature_cols.append("_server_freq")

    valid = df[feature_cols + [LABEL_COL]].dropna()
    if len(valid) < 20:
        write_line(f, "  유효 행 부족 (skip)")
        return

    X = valid[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = LabelEncoder().fit_transform(valid[LABEL_COL])

    mi = mutual_info_classif(X, y, discrete_features="auto", random_state=42)
    mi_df = pd.DataFrame({"feature": feature_cols, "MI": mi}).sort_values("MI", ascending=False)

    write_line(f, f"\n  {'Rank':>4s}  {'Feature':40s}  {'MI':>8s}")
    write_line(f, f"  {'-'*4}  {'-'*40}  {'-'*8}")
    for rank, (_, row) in enumerate(mi_df.iterrows(), 1):
        flag = " ← HIGH" if row["MI"] > 0.3 else ""
        write_line(f, f"  {rank:4d}  {row['feature']:40s}  {row['MI']:8.4f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1-B: Row-wise 진단
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_rows(df: pd.DataFrame, f):
    """행 단위 품질 진단."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "B. Row-wise 진단")
    write_line(f, "=" * 70)

    n_total = len(df)
    write_line(f, f"\n  전체 행 수: {n_total:,}")

    # B-1: pk_file 분포
    write_line(f, f"\n  [B-1] pk_file 분포")
    if "pk_file" in df.columns:
        pk_counts = df.groupby("pk_file").size()
        write_line(f, f"    고유 pk_file 수: {len(pk_counts):,}")
        write_line(f, f"    pk_file당 행 수: mean={pk_counts.mean():.1f}  median={pk_counts.median():.0f}  "
                      f"max={pk_counts.max():,}  min={pk_counts.min():,}")

        # pk_file 레이블 충돌 검사
        pk_labels = df.groupby("pk_file")[LABEL_COL].nunique()
        conflict_pks = pk_labels[pk_labels > 1]
        write_line(f, f"    레이블 충돌 pk_file: {len(conflict_pks):,}건 "
                      f"({len(conflict_pks)/len(pk_counts):.2%})")
        if len(conflict_pks) > 0:
            write_line(f, f"    *** 동일 파일이 TP와 FP 모두에 존재 - 라벨링 일관성 문제 ***")
            # 상위 5개 충돌 pk_file 상세
            for pk in conflict_pks.index[:5]:
                subset = df[df["pk_file"] == pk]
                write_line(f, f"      pk={pk[:16]}...  TP={sum(subset[LABEL_COL]=='TP')}  "
                              f"FP={sum(subset[LABEL_COL]=='FP')}")

        # 클래스별 pk_file 수
        write_line(f, f"\n    클래스별 통계:")
        for label in df[LABEL_COL].unique():
            subset = df[df[LABEL_COL] == label]
            n_pk = subset["pk_file"].nunique()
            write_line(f, f"      {label}: {len(subset):>8,}행  {n_pk:>6,} pk_file  "
                          f"(pk_file당 평균 {len(subset)/max(n_pk,1):.1f}행)")

    # B-2: 완전 중복 행
    write_line(f, f"\n  [B-2] 중복 행 검사")
    dup_cols = [c for c in df.columns if c not in ["pk_event", "pk_file", "_source_file"]]
    n_dup = df.duplicated(subset=dup_cols, keep=False).sum()
    write_line(f, f"    완전 중복 행 (PK 제외): {n_dup:,}건 ({n_dup/n_total:.2%})")

    if "pk_event" in df.columns:
        n_pk_dup = df.duplicated(subset=["pk_event"], keep=False).sum()
        write_line(f, f"    pk_event 중복:          {n_pk_dup:,}건 ({n_pk_dup/n_total:.2%})")

    # B-3: 클래스 불균형
    write_line(f, f"\n  [B-3] 클래스 불균형")
    vc = df[LABEL_COL].value_counts()
    for label, count in vc.items():
        write_line(f, f"    {label}: {count:>8,} ({count/n_total:.1%})")
    imbalance_ratio = vc.max() / max(vc.min(), 1)
    write_line(f, f"    불균형 비율: {imbalance_ratio:.1f}:1")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1-C: 단일 피처 예측력
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_single_feature_f1(df: pd.DataFrame, f) -> pd.DataFrame:
    """각 피처 하나만으로 label 예측 시 F1-macro."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "C. 단일 피처 예측력 (DecisionTree max_depth=1)")
    write_line(f, "=" * 70)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder

    feature_cols = []
    for col in PATH_FEATURE_COLS + META_FEATURE_COLS + NUMERIC_COLS:
        if col in df.columns:
            feature_cols.append(col)

    # extension & server_freq
    df_work = df.copy()
    if "file_name" in df_work.columns:
        df_work["_ext_enc"] = LabelEncoder().fit_transform(
            df_work["file_name"].str.rsplit(".", n=1).str[-1].str.lower().fillna("unknown")
        )
        feature_cols.append("_ext_enc")
    if "server_name" in df_work.columns:
        _sn = df_work["server_name"].value_counts(normalize=True)
        df_work["_server_freq"] = df_work["server_name"].map(_sn).fillna(0)
        feature_cols.append("_server_freq")

    y_enc = LabelEncoder().fit_transform(df_work[LABEL_COL])

    # Split (GroupShuffleSplit if possible)
    if "pk_file" in df_work.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df_work, y_enc, groups=df_work["pk_file"]))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(df_work, y_enc))

    results = []
    for col in feature_cols:
        vals = pd.to_numeric(df_work[col], errors="coerce").fillna(0).values.reshape(-1, 1)
        X_tr, X_te = vals[train_idx], vals[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        clf = DecisionTreeClassifier(max_depth=1, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        results.append({"feature": col, "f1_macro": f1})

    result_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)

    write_line(f, f"\n  {'Rank':>4s}  {'Feature':40s}  {'F1-macro':>8s}  {'판정':10s}")
    write_line(f, f"  {'-'*4}  {'-'*40}  {'-'*8}  {'-'*10}")
    for rank, (_, row) in enumerate(result_df.iterrows(), 1):
        if row["f1_macro"] > F1_DANGER:
            flag = "*** DANGER"
        elif row["f1_macro"] > F1_WARNING:
            flag = "** WARNING"
        else:
            flag = "OK"
        write_line(f, f"  {rank:4d}  {row['feature']:40s}  {row['f1_macro']:8.4f}  {flag}")

    result_df.to_csv(OUTPUT_DIR / "single_feature_f1.csv", index=False, encoding="utf-8-sig")
    write_line(f, f"\n  [저장] single_feature_f1.csv")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# 종합 판정
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(cat_results, path_results, sf_df, f):
    """Go/No-Go 종합 판정."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "종합 판정 (Go / No-Go)")
    write_line(f, "=" * 70)

    findings = []

    # Cramer's V - "왜 성능이 높은지" 해석용
    for col, v, pure_ratio in cat_results:
        if v > V_DANGER:
            findings.append(f"HIGH_CORR: {col} Cramer's V={v:.3f} - label과 강한 연관")
        elif v > V_WARNING:
            findings.append(f"MOD_CORR:  {col} Cramer's V={v:.3f} - label과 중간 연관")

    # 단일 피처 F1 - 피처 의존 구조 파악용
    if sf_df is not None and len(sf_df) > 0:
        strong = sf_df[sf_df["f1_macro"] > F1_DANGER]
        for _, row in strong.iterrows():
            findings.append(f"STRONG:    {row['feature']} 단일피처 F1={row['f1_macro']:.3f} - 강한 판별력")

        moderate = sf_df[(sf_df["f1_macro"] > F1_WARNING) & (sf_df["f1_macro"] <= F1_DANGER)]
        for _, row in moderate.iterrows():
            findings.append(f"MODERATE:  {row['feature']} 단일피처 F1={row['f1_macro']:.3f}")

    write_line(f, f"\n  [데이터 특성 요약]")
    write_line(f, f"  label과 강한 연관: {sum(1 for i in findings if 'HIGH_CORR' in i)}건")
    write_line(f, f"  강한 단일 판별력:  {sum(1 for i in findings if 'STRONG' in i)}건")

    if findings:
        write_line(f, f"\n  상세:")
        for finding in findings:
            write_line(f, f"    - {finding}")

    write_line(f, f"\n  [핵심 액션]")
    write_line(f, f"  이 결과만으로는 모델의 옳고 그름을 판단할 수 없음.")
    write_line(f, f"  -> Temporal Holdout (10~12월)으로 '미래 예측 성능'을 반드시 확인할 것.")
    write_line(f, f"  -> Feature Ablation으로 '어떤 피처가 빠지면 취약한지'를 파악할 것.")

    # 별도 summary 파일
    with open(OUTPUT_DIR / "bias_summary.md", "w", encoding="utf-8") as sf:
        sf.write(f"# 데이터 특성 및 일반화 성능 진단\n\n")
        sf.write(f"**진단 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        sf.write(f"## 데이터 특성 요약\n\n")
        sf.write(f"데이터 편향 분석은 '왜 성능이 높은지'를 해석하는 도구이다.\n")
        sf.write(f"'편향이 있으니 모델이 틀렸다'는 결론과는 다르다.\n\n")
        for finding in findings:
            sf.write(f"- {finding}\n")
        sf.write(f"\n## 핵심 질문\n\n")
        sf.write(f"**3~9월로 학습한 모델이 10~12월을 잘 예측하는가?**\n\n")
        sf.write(f"이 답은 `split_robustness_report.csv`의 temporal_3m 결과에 있다.\n\n")
        sf.write(f"## 해석 기준\n\n")
        sf.write(f"| Temporal F1 vs Random F1 | 의미 |\n")
        sf.write(f"|--------------------------|------|\n")
        sf.write(f"| 하락 < 5pp | 성능 유지. 모델 신뢰 가능 |\n")
        sf.write(f"| 하락 5~15pp | 보완 필요. 시간에 따른 분포 변화 존재 |\n")
        sf.write(f"| 하락 > 15pp | 모델이 과거 패턴에 과적합. 피처 재설계 필요 |\n")

    write_line(f, f"\n  [저장] bias_summary.md")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Phase D: Split Robustness (다른 Agent Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_matrix_for_split(df_train, df_test, exclude_blocks=None):
    """진단용 간이 feature matrix 생성. exclude_blocks로 블록 제거."""
    from src.features.meta_features import build_meta_features
    from src.features.path_features import extract_path_features
    from src.features.tabular_features import create_file_path_features
    from sklearn.preprocessing import LabelEncoder

    exclude_blocks = set(exclude_blocks or [])

    # 메타/경로 피처 추가 (이미 있으면 skip)
    for part in [df_train, df_test]:
        if "fname_has_date" not in part.columns:
            tmp = build_meta_features(part)
            for c in tmp.columns:
                if c not in part.columns:
                    part[c] = tmp[c].values
        if "is_log_file" not in part.columns and "file_path" in part.columns:
            pf = part["file_path"].apply(extract_path_features)
            pdf = pd.DataFrame(list(pf), index=part.index)
            for c in pdf.columns:
                if c not in part.columns:
                    part[c] = pdf[c].values

    # 피처 블록 정의
    server_cols = []
    if "server_name" in df_train.columns and "server" not in exclude_blocks:
        _sn = df_train["server_name"].value_counts(normalize=True)
        df_train["_server_freq"] = df_train["server_name"].map(_sn).fillna(0)
        df_test["_server_freq"] = df_test["server_name"].map(_sn).fillna(0)
        server_cols = ["_server_freq"]

    time_cols = [c for c in ["created_hour", "created_weekday", "is_weekend", "created_month"]
                 if c in df_train.columns and "time" not in exclude_blocks]

    count_cols = [c for c in ["pattern_count_log1p", "pattern_count_bin",
                              "is_mass_detection", "is_extreme_detection", "pii_type_ratio"]
                  if c in df_train.columns and "count" not in exclude_blocks]

    fname_cols = [c for c in ["fname_has_date", "fname_has_hash", "fname_has_rotation_num"]
                  if c in df_train.columns and "fname" not in exclude_blocks]

    path_cols = [c for c in ["path_depth", "is_log_file", "is_docker_overlay",
                             "has_license_path", "is_temp_or_dev", "is_system_device",
                             "is_package_path", "has_cron_path", "has_date_in_path",
                             "has_business_token", "has_system_token"]
                 if c in df_train.columns and "path" not in exclude_blocks]

    # tabular path features
    tab_path_cols = []
    if "path" not in exclude_blocks and "file_path" in df_train.columns:
        pt_tr = create_file_path_features(df_train, path_column="file_path")
        pt_te = create_file_path_features(df_test, path_column="file_path")
        for c in pt_tr.select_dtypes(include=[np.number]).columns:
            if c not in df_train.columns:
                df_train[c] = pt_tr[c].values
                df_test[c] = pt_te[c].values
                tab_path_cols.append(c)

    all_cols = server_cols + time_cols + count_cols + fname_cols + path_cols + tab_path_cols
    if not all_cols:
        return None, None, []

    X_tr = df_train[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_te = df_test[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    return X_tr, X_te, all_cols


def diagnose_split_robustness(df: pd.DataFrame, f):
    """다양한 split 전략별 성능 비교.

    핵심 질문: 3~9월로 학습한 모델이 10~12월을 잘 예측하는가?
    Temporal holdout이 1순위, 나머지는 보조 해석용.
    """
    write_line(f, "\n" + "=" * 70)
    write_line(f, "D. 일반화 성능 검증 (Temporal Holdout 중심)")
    write_line(f, "=" * 70)
    write_line(f, "")
    write_line(f, "  핵심 질문: 과거 데이터로 학습 -> 미래 데이터를 잘 예측하는가?")
    write_line(f, "  Temporal holdout (마지막 2~3개월)이 가장 현실적인 검증.")
    write_line(f, "")

    # ------------------------------------------------------------------
    # 체크포인트 모델 로드 (run_training.py 학습 결과 재사용)
    # ------------------------------------------------------------------
    from pathlib import Path
    _project_root = Path(__file__).resolve().parents[1]
    _ckpt_dir = _project_root / "models" / "checkpoints"
    _ckpt5 = _ckpt_dir / "step5_features_silver_label.pkl"
    _ckpt6 = _ckpt_dir / "step6_model_silver_label.pkl"

    _use_checkpoint = _ckpt5.exists() and _ckpt6.exists()

    if _use_checkpoint:
        import joblib
        from sklearn.metrics import f1_score as _f1_score_fn
        write_line(f, "  [체크포인트 모델 사용] run_training.py 학습 결과와 동일한 모델/피처로 평가")
        _d5 = joblib.load(_ckpt5)
        _d6 = joblib.load(_ckpt6)
        _result = _d5["result"]
        _model = _d6["model"]
        _le = _d5["le"]
        _X_test = _result["X_test"]
        _y_test_enc = _d5["y_test_enc"]
        _y_pred = _model.predict(_X_test)

        _f1 = _f1_score_fn(_y_test_enc, _y_pred, average="macro", zero_division=0)

        # Get split info from result
        _split_meta = _result.get("split_meta", {})
        _split_strategy = _split_meta.get("split_strategy", "unknown")

        write_line(f, f"  [checkpoint] F1={_f1:.4f}  "
                   f"train={_result['X_train'].shape[0]:,}  test={_X_test.shape[0]:,}  "
                   f"(split: {_split_strategy})")
        write_line(f, "")
        write_line(f, "  [참고] 아래 독립 split 실험은 진단 참조용입니다.")
        write_line(f, "         실제 모델 성능은 위 체크포인트 결과입니다.")
        write_line(f, "")

    if not _use_checkpoint:
        write_line(f, "  [SKIP] 체크포인트 없음 — run_training.py 실행 후 다시 시도")
        write_line(f, "  체크포인트 경로: %s", str(_ckpt5))
        return

    # 체크포인트 결과를 split_robustness_report.csv에도 저장
    import pandas as _pd_diag
    _pd_diag.DataFrame([{
        "split": _split_strategy,
        "f1_macro": _f1,
        "train_n": _result["X_train"].shape[0],
        "test_n": _X_test.shape[0],
        "desc": "checkpoint model (run_training.py 학습 결과)",
    }]).to_csv(OUTPUT_DIR / "split_robustness_report.csv",
               index=False, encoding="utf-8-sig")
    write_line(f, f"\n  [저장] split_robustness_report.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Phase E: Feature Block Ablation (다른 Agent Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_ablation(df: pd.DataFrame, f):
    """피처 블록 제거 실험 - 모델의 피처 의존 구조 파악."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "E. Feature Block Ablation (피처 의존 구조 파악)")
    write_line(f, "=" * 70)
    write_line(f, "")
    write_line(f, "  목적: 어떤 피처 블록이 빠지면 성능이 떨어지는지 파악")
    write_line(f, "  '이 블록만으로 성능이 나옴' = 모델이 그 신호에 의존 (정당할 수 있음)")
    write_line(f, "  '이 블록 빠지면 급락' = 해당 블록 없으면 취약 (새 환경에서 위험)")
    write_line(f, "")

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score

    le = LabelEncoder()
    y_all = le.fit_transform(df[LABEL_COL])

    # 고정 split
    if "pk_file" in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(df, y_all, groups=df["pk_file"]))
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(sss.split(df, y_all))

    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    # 블록 정의
    ablation_configs = [
        ("full",              set()),
        ("full - server",     {"server"}),
        ("full - path",       {"path"}),
        ("full - time",       {"time"}),
        ("full - count",      {"count"}),
        ("full - server&path", {"server", "path"}),
        ("server only",       {"time", "count", "fname", "path"}),
        ("path only",         {"server", "time", "count", "fname"}),
        ("count only",        {"server", "time", "path", "fname"}),
    ]

    results = []
    for name, exclude in ablation_configs:
        df_tr = df.iloc[tr_idx].copy().reset_index(drop=True)
        df_te = df.iloc[te_idx].copy().reset_index(drop=True)

        X_tr, X_te, feat_names = _build_feature_matrix_for_split(df_tr, df_te, exclude_blocks=exclude)
        if X_tr is None or X_tr.shape[1] == 0:
            write_line(f, f"  [{name:25s}] SKIP (피처 없음)")
            continue

        try:
            model = LGBMClassifier(
                n_estimators=100, max_depth=6, num_leaves=31,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        except Exception as e:
            f1 = -1.0
            write_line(f, f"  [{name:25s}] 학습 실패: {e}")
            continue

        results.append({"config": name, "f1_macro": f1, "n_features": X_tr.shape[1]})
        write_line(f, f"  [{name:25s}] F1={f1:.4f}  n_features={X_tr.shape[1]}")

    if results:
        full_f1 = results[0]["f1_macro"] if results[0]["config"] == "full" else None
        if full_f1 is not None:
            write_line(f, f"\n  [해석] full F1={full_f1:.4f}")
            write_line(f, f"  --- 블록 제거 시 ---")
            for r in results[1:]:
                diff = r["f1_macro"] - full_f1
                if "only" in r["config"]:
                    continue  # only는 아래에서 별도 출력
                note = ""
                if diff < -0.15:
                    note = " -> 이 블록 없으면 취약"
                elif diff < -0.05:
                    note = " -> 의미있는 기여"
                else:
                    note = " -> 다른 피처로 보완 가능"
                write_line(f, f"  {r['config']:25s}  F1 변화: {diff:+.4f}{note}")

            write_line(f, f"\n  --- 단일 블록만 사용 시 ---")
            for r in results[1:]:
                if "only" not in r["config"]:
                    continue
                ratio = r["f1_macro"] / max(full_f1, 0.001)
                note = ""
                if ratio > 0.9:
                    note = " -> 이 블록만으로 거의 충분 (강한 신호)"
                elif ratio > 0.7:
                    note = " -> 상당한 판별력"
                else:
                    note = " -> 단독으로는 부족"
                write_line(f, f"  {r['config']:25s}  full 대비: {ratio:.1%}{note}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "ablation_report.csv",
                                  index=False, encoding="utf-8-sig")
    write_line(f, f"\n  [저장] ablation_report.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Phase F: Column Risk Registry (다른 Agent Phase 1 보강)
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_column_registry(df: pd.DataFrame, f):
    """모든 컬럼의 리스크 등급을 CSV로 출력."""
    write_line(f, "\n" + "=" * 70)
    write_line(f, "F. Column Risk Registry")
    write_line(f, "=" * 70)

    records = []
    for col in df.columns:
        n_total = len(df)
        n_null = int(df[col].isna().sum())
        null_ratio = n_null / max(n_total, 1)
        cardinality = df[col].nunique()

        # label purity: 컬럼 값별 label이 얼마나 단일 클래스인지
        try:
            ct = pd.crosstab(df[col].fillna("__NULL__"), df[LABEL_COL], normalize="index")
            purity = float(ct.max(axis=1).mean())
        except Exception:
            purity = 0.0

        # 리스크 등급 결정
        is_target = col in [LABEL_COL, "label_binary"]
        is_post_label = col in ["_source_file", "label_work_month", "label_review",
                                "fp_description", "exception_requested"]
        is_pk = col in ["pk_file", "pk_event"]
        is_model_input = col in (
            ["server_name", "file_path", "file_name", "pattern_count",
             "ssn_count", "phone_count", "email_count", "file_created_at"] +
            PATH_FEATURE_COLS + META_FEATURE_COLS
        )

        if is_target:
            risk = "TARGET"
        elif is_pk:
            risk = "PK (safe)"
        elif is_post_label:
            risk = "Post-label artifact"
        elif purity > 0.95 and is_model_input:
            risk = "Target proxy"
        elif purity > 0.85 and is_model_input:
            risk = "Possible leakage"
        elif is_model_input:
            risk = "Needs review"
        else:
            risk = "Safe (unused)"

        records.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_ratio": round(null_ratio, 3),
            "cardinality": cardinality,
            "label_purity": round(purity, 3),
            "model_input": "Y" if is_model_input else "N",
            "risk_level": risk,
        })

        flag = f" <<<" if "proxy" in risk.lower() or "leakage" in risk.lower() else ""
        write_line(f, f"  {col:30s}  null={null_ratio:.1%}  card={cardinality:>6,}  "
                      f"purity={purity:.2f}  [{risk}]{flag}")

    reg_df = pd.DataFrame(records)
    reg_df.to_csv(OUTPUT_DIR / "column_risk_registry.csv", index=False, encoding="utf-8-sig")
    write_line(f, f"\n  [저장] column_risk_registry.csv")

    # 요약
    for level in ["Target proxy", "Possible leakage", "Post-label artifact", "Needs review", "Safe (unused)"]:
        n = sum(1 for r in records if r["risk_level"] == level)
        if n > 0:
            write_line(f, f"  {level}: {n}개")


def parse_args():
    parser = argparse.ArgumentParser(description="모델 성능 신뢰성 진단 (Phase 1: 데이터 편향)")
    parser.add_argument(
        "--parquet",
        default=str(PROJECT_ROOT / "data" / "processed" / "silver_label.parquet"),
        help="진단 대상 Parquet 파일 경로 (기본: silver_label.parquet)",
    )
    parser.add_argument(
        "--skip-single-feature",
        action="store_true",
        help="단일 피처 F1 진단 생략 (빠른 실행)",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Ablation + Split Robustness 진단 생략 (빠른 실행)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"[오류] 파일 없음: {parquet_path}")
        print("  실 데이터 silver_label.parquet 필요")
        print("  먼저 python scripts/run_data_pipeline.py --source label 실행")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[진단 시작] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  대상: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"  로드: {df.shape[0]:,}행 × {df.shape[1]}열")

    if LABEL_COL not in df.columns:
        print(f"[오류] {LABEL_COL} 컬럼 없음")
        sys.exit(1)

    report_path = OUTPUT_DIR / "column_bias_report.txt"
    row_report_path = OUTPUT_DIR / "row_quality_report.txt"

    # ── Column-wise 진단 ──
    with open(report_path, "w", encoding="utf-8") as f:
        write_line(f, f"모델 성능 신뢰성 진단 - Column-wise")
        write_line(f, f"진단 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_line(f, f"대상: {parquet_path.name} ({df.shape[0]:,}행 x {df.shape[1]}열)")
        write_line(f, "")

        cat_results = diagnose_categorical(df, f)
        diagnose_numeric(df, f)
        path_results = diagnose_path_features(df, f)
        diagnose_mutual_information(df, f)

        sf_df = None
        if not args.skip_single_feature:
            sf_df = diagnose_single_feature_f1(df, f)

        # Column Risk Registry
        diagnose_column_registry(df, f)

        write_summary(cat_results, path_results, sf_df, f)

    print(f"\n[저장] {report_path}")

    # ── Row-wise 진단 ──
    with open(row_report_path, "w", encoding="utf-8") as f:
        write_line(f, f"모델 성능 신뢰성 진단 - Row-wise")
        write_line(f, f"진단 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_line(f, f"대상: {parquet_path.name} ({df.shape[0]:,}행 x {df.shape[1]}열)")
        write_line(f, "")
        diagnose_rows(df, f)

    print(f"[저장] {row_report_path}")

    # ── Split Robustness + Ablation 진단 ──
    if not args.skip_ablation:
        robust_path = OUTPUT_DIR / "split_ablation_report.txt"
        with open(robust_path, "w", encoding="utf-8") as f:
            write_line(f, f"모델 성능 신뢰성 진단 - Split Robustness & Ablation")
            write_line(f, f"진단 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            write_line(f, f"대상: {parquet_path.name} ({df.shape[0]:,}행 x {df.shape[1]}열)")
            write_line(f, "")
            diagnose_split_robustness(df, f)
            diagnose_ablation(df, f)
        print(f"[저장] {robust_path}")
    else:
        print("[SKIP] Ablation + Split Robustness (--skip-ablation)")

    print(f"\n[진단 완료] 결과: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
