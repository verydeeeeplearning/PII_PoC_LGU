"""Feature Engineering 통합 파이프라인

Architecture.md v1.2 반영:
- GroupShuffleSplit(groups=pk_file) - File-level Leakage 방지 (§14)
- Multi-view TF-IDF 3개 뷰: raw_text / shape_text / path_text (§12)
- 산출물: df_train/df_test Parquet + CSV, split_meta.json
"""
import json
import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import joblib
from pathlib import Path
from typing import Optional

from src.features.text_features import (
    create_keyword_features,
    create_text_stat_features,
    create_all_text_features,
)
from src.features.tabular_features import (
    create_file_path_features,
    create_all_path_features,
    encode_categorical,
)
from src.utils.constants import RANDOM_SEED, TEST_SIZE, TFIDF_MAX_FEATURES, FILE_PATH_COLUMN


# ── 텍스트 뷰 변환 헬퍼 ─────────────────────────────────────────────────────

def _to_shape_text(text: str) -> str:
    """문자를 유형 기호로 추상화하여 구조적 패턴 신호 생성

    변환 규칙:
        A-Z  -> U (uppercase)
        a-z  -> l (lowercase)
        0-9  -> D (digit)
        @ . - _  -> 그대로 유지 (이메일/경로 구분자 보존)
        공백  -> 공백
        기타  -> S (special)
    """
    buf = []
    for c in text:
        if c.isupper():
            buf.append("U")
        elif c.islower():
            buf.append("l")
        elif c.isdigit():
            buf.append("D")
        elif c in " \t\n\r":
            buf.append(" ")
        elif c in "@.-_":
            buf.append(c)
        else:
            buf.append("S")
    return "".join(buf)


def _to_path_text(text: str) -> str:
    """파일 경로 구분자 기준으로 토큰화하여 경로 컨텍스트 신호 생성

    / \\ . _ - 기준으로 분리 후 소문자 변환
    """
    tokens = re.split(r"[/\\._\-\s]+", text or "")
    return " ".join(t.lower() for t in tokens if t)


def _build_tfidf_view(
    train_texts: pd.Series,
    test_texts: pd.Series,
    params: dict,
    view_name: str,
) -> tuple:
    """단일 TF-IDF 뷰 학습/변환

    Returns:
        (tfidf_train, tfidf_test, vectorizer)
    """
    vec = TfidfVectorizer(**params, sublinear_tf=True)
    tr = vec.fit_transform(train_texts.fillna(""))
    te = vec.transform(test_texts.fillna(""))
    print(f"  [{view_name}] vocab={len(vec.vocabulary_):,}  train={tr.shape}  test={te.shape}")
    return tr, te, vec


# ── 합성 상호작용 변수 ────────────────────────────────────────────────────────

def create_synthetic_interaction_features(
    text_features: pd.DataFrame,
    path_features: pd.DataFrame,
) -> pd.DataFrame:
    """텍스트/경로 신호를 조합한 합성(상호작용) 변수를 생성합니다."""
    if text_features.empty and path_features.empty:
        return pd.DataFrame()

    index = text_features.index if not text_features.empty else path_features.index

    def get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=index, dtype=float)

    has_internal_domain      = get_series(text_features, "has_internal_domain")
    has_os_copyright_domain  = get_series(text_features, "has_os_copyright_domain")
    has_dummy_domain         = get_series(text_features, "has_dummy_domain")
    has_timestamp_pattern    = get_series(text_features, "has_timestamp_pattern")
    has_bytes_pattern        = get_series(text_features, "has_bytes_pattern")
    has_version_pattern      = get_series(text_features, "has_version_pattern")
    masking_ratio            = get_series(text_features, "masking_ratio")
    digit_ratio              = get_series(text_features, "digit_ratio")
    special_char_ratio       = get_series(text_features, "special_char_ratio")

    file_is_log      = get_series(path_features, "file_is_log")
    path_is_system   = get_series(path_features, "path_is_system")
    path_is_temp     = get_series(path_features, "path_is_temp")
    path_has_test    = get_series(path_features, "path_has_test")
    path_has_dev     = get_series(path_features, "path_has_dev")
    path_has_hadoop  = get_series(path_features, "path_has_hadoop")
    path_has_legacy_date = get_series(path_features, "path_has_legacy_date")
    path_depth       = get_series(path_features, "path_depth")

    synthetic = pd.DataFrame(
        {
            "syn_internal_domain_x_log_path":      has_internal_domain * file_is_log,
            "syn_internal_domain_x_system_path":   has_internal_domain * path_is_system,
            "syn_dummy_domain_x_test_path":        has_dummy_domain * path_has_test,
            "syn_os_copyright_x_hadoop_path":      has_os_copyright_domain * path_has_hadoop,
            "syn_timestamp_x_legacy_path":         has_timestamp_pattern * path_has_legacy_date,
            "syn_bytes_x_system_path":             has_bytes_pattern * path_is_system,
            "syn_version_x_dev_path":              has_version_pattern * path_has_dev,
            "syn_masking_ratio_x_path_depth":      masking_ratio * path_depth,
            "syn_digit_ratio_x_log_path":          digit_ratio * file_is_log,
            "syn_special_char_ratio_x_temp_path":  special_char_ratio * path_is_temp,
        },
        index=index,
    )

    print(f"[합성 상호작용 Feature] {synthetic.shape[1]}개 생성")
    return synthetic


# ── 메인 Feature Engineering 파이프라인 ──────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    text_column: str = "full_context_raw",
    label_column: str = "label",
    test_size: float = TEST_SIZE,
    tfidf_max_features: int = TFIDF_MAX_FEATURES,
    use_synthetic_expansion: bool = True,
    use_group_split: bool = True,
    use_multiview_tfidf: bool = True,
    use_phase1_tfidf: bool = False,
    use_variance_threshold: bool = False,
    n_splits: int = 1,
    config: Optional[dict] = None,
    split_strategy: str = "group",
    test_months: int = 3,
) -> dict:
    """전체 Feature Engineering 파이프라인

    처리 순서:
        1. Train/Test 분할
           - pk_file 컬럼 있으면: GroupShuffleSplit (Architecture §14, Leakage 방지)
           - pk_file 없으면: StratifiedSplit (fallback, 경고 출력)
        2. Multi-view TF-IDF Feature 생성 (raw / shape / path)
        3. Dense Feature 생성
           - 확장 모드: 텍스트/경로 확장 Feature + 합성 상호작용
           - 기본 모드: 키워드/통계/기본 경로 Feature
        4. sparse(TF-IDF 3개) + dense Feature 결합

    Args:
        df: 전처리된 DataFrame
        text_column: 텍스트 컬럼명 (full_context_raw)
        label_column: 레이블 컬럼명
        test_size: 테스트셋 비율
        tfidf_max_features: Multi-view OFF 시 단일 TF-IDF 최대 Feature 수
        use_synthetic_expansion: 합성 변수 확장 사용 여부
        use_group_split: pk_file GroupShuffleSplit 사용 여부 (기본 True)
        use_multiview_tfidf: Multi-view TF-IDF 사용 여부 (기본 True)
        config: feature_config.yaml 내용 (None이면 자동 로드)
        split_strategy: 분할 전략 선택
            - "group": GroupShuffleSplit(pk_file) (기본값, 기존 동작)
            - "temporal": label_work_month 기준 시간 분할
            - "server": server_name 기준 서버 그룹 분할
        test_months: temporal split 시 테스트 월 수 (기본 3 -> 마지막 3개월)

    Returns:
        dict:
            X_train, X_test: scipy.sparse matrix
            y_train, y_test: pd.Series
            feature_names: list[str]
            tfidf_vectorizers: dict {raw/shape/path} or {single}
            df_train, df_test: pd.DataFrame
            split_meta: dict (split 정보)
    """
    if config is None:
        try:
            from src.data.loader import load_config
            config = load_config()
        except Exception:
            config = {}

    _eval_cfg = config.get("evaluation", {}).get("split_strategy", {})
    _features_cfg = config.get("features", {}).get("tfidf", {})

    # ── 1. Train/Test 분할 ────────────────────────────────────────────────────
    y = df[label_column]
    split_strategy_used = "stratified_random"

    has_pk_file = "pk_file" in df.columns and use_group_split

    if split_strategy == "temporal" and "label_work_month" in df.columns:
        # Temporal split: 마지막 N개월을 test로 분리
        from src.evaluation.split_strategies import work_month_time_split
        train_idx_list, test_idx_list = work_month_time_split(
            df, test_months=test_months,
        )
        train_idx = df.index[train_idx_list]
        test_idx = df.index[test_idx_list]
        split_strategy_used = f"temporal_work_month(test_months={test_months})"

    elif split_strategy == "server" and "server_name" in df.columns:
        # Server-level split: 서버 단위로 분리
        from src.evaluation.split_strategies import server_group_split
        train_idx_list, test_idx_list = server_group_split(
            df, test_ratio=test_size,
        )
        train_idx = df.index[train_idx_list]
        test_idx = df.index[test_idx_list]
        split_strategy_used = f"server_group_split(test_ratio={test_size})"

    elif has_pk_file:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
        train_idx_arr, test_idx_arr = next(gss.split(df, y, groups=df["pk_file"]))
        train_idx = df.index[train_idx_arr]
        test_idx  = df.index[test_idx_arr]
        split_strategy_used = "pk_file_group_split"
    else:
        if use_group_split:
            print("  [WARN] pk_file 컬럼 없음 -> Stratified random split으로 fallback")
            print("         run_data_pipeline.py 실행 후 pk_file 생성 권장")
        train_idx, test_idx = train_test_split(
            df.index, test_size=test_size, stratify=y, random_state=RANDOM_SEED
        )

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)
    y_train  = df_train[label_column]
    y_test   = df_test[label_column]

    print(f"[데이터 분할] 전략={split_strategy_used}")
    print(f"  학습셋:   {len(df_train):,}건")
    print(f"  테스트셋: {len(df_test):,}건")

    # temporal/server split 상세 출력
    if split_strategy == "temporal" and "label_work_month" in df.columns:
        tr_months = sorted(df_train["label_work_month"].unique()) if "label_work_month" in df_train.columns else []
        te_months = sorted(df_test["label_work_month"].unique()) if "label_work_month" in df_test.columns else []
        print(f"  train 월: {tr_months}")
        print(f"  test 월:  {te_months}")
    if split_strategy == "server" and "server_name" in df.columns:
        n_tr_srv = df_train["server_name"].nunique() if "server_name" in df_train.columns else 0
        n_te_srv = df_test["server_name"].nunique() if "server_name" in df_test.columns else 0
        print(f"  train 서버: {n_tr_srv}개  test 서버: {n_te_srv}개")

    if has_pk_file:
        train_pkfiles = set(df_train["pk_file"])
        test_pkfiles  = set(df_test["pk_file"])
        overlap = train_pkfiles & test_pkfiles
        print(f"  train pk_file: {len(train_pkfiles):,}  test pk_file: {len(test_pkfiles):,}")
        print(f"  pk_file Leakage: {len(overlap)}건  "
              f"({'OK - Leakage 없음' if len(overlap) == 0 else 'WARN - Leakage 존재'})")

    # 클래스 비율 검증 - 소수 클래스 누락 또는 심각한 분포 편향 경보
    _y_train_vc = y_train.value_counts(normalize=True)
    _y_test_vc  = y_test.value_counts(normalize=True)
    for _cls in _y_train_vc.index:
        if _cls not in _y_test_vc.index:
            print(f"  [WARN] 테스트셋에 클래스 '{_cls}' 없음 - 평가 편향 가능")
        else:
            _tr = _y_train_vc[_cls]
            _te = _y_test_vc[_cls]
            if abs(_tr - _te) > 0.15:
                print(f"  [WARN] 클래스 '{_cls}' train/test 비율 편차 큼: "
                      f"train={_tr:.2f} test={_te:.2f}")

    # §2.2 Repeated split - 분산 추정 (n_splits > 1)
    if n_splits > 1 and has_pk_file:
        print(f"\n[Repeated GroupShuffleSplit] n_splits={n_splits} - 분산 추정")
        _gss_rep = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=RANDOM_SEED)
        _rep_test_ratios: dict = {}
        for _s_idx, (_tr, _te) in enumerate(_gss_rep.split(df, y, groups=df["pk_file"])):
            _y_te = y.iloc[_te]
            for _c, _cnt in _y_te.value_counts(normalize=True).items():
                _rep_test_ratios.setdefault(_c, []).append(_cnt)
        print("  클래스별 test 비율 mean ± std:")
        for _c, _vals in sorted(_rep_test_ratios.items()):
            print(f"    {_c}: {np.mean(_vals):.3f} ± {np.std(_vals):.3f}")

    # ── 2. TF-IDF Feature 생성 ────────────────────────────────────────────────
    tfidf_vectorizers = {}
    tfidf_parts_train = []
    tfidf_parts_test  = []
    tfidf_names_all   = []

    has_text_column = text_column in df_train.columns

    if has_text_column:
        train_raw = df_train[text_column].fillna("")
        test_raw  = df_test[text_column].fillna("")
    else:
        train_raw = pd.Series([""] * len(df_train), index=df_train.index)
        test_raw  = pd.Series([""] * len(df_test),  index=df_test.index)

    print("\n[Multi-view TF-IDF]" if (use_multiview_tfidf and has_text_column) else "\n[Dense Only - TF-IDF 비활성화]")

    if use_multiview_tfidf and has_text_column:
        # View 1: raw_text (원문)
        raw_params = {
            "max_features": _features_cfg.get("raw_text", {}).get("max_features", 5000),
            "ngram_range":  tuple(_features_cfg.get("raw_text", {}).get("ngram_range", [1, 2])),
            "min_df":       _features_cfg.get("raw_text", {}).get("min_df", 3),
            "max_df":       _features_cfg.get("raw_text", {}).get("max_df", 0.95),
        }
        tr_raw, te_raw, vec_raw = _build_tfidf_view(train_raw, test_raw, raw_params, "raw_text")
        tfidf_vectorizers["raw"] = vec_raw
        tfidf_parts_train.append(tr_raw)
        tfidf_parts_test.append(te_raw)
        tfidf_names_all += [f"tfidf_raw_{n}" for n in vec_raw.get_feature_names_out()]

        # View 2: shape_text (문자 유형 추상화)
        train_shape = train_raw.apply(_to_shape_text)
        test_shape  = test_raw.apply(_to_shape_text)
        shape_params = {
            "max_features": _features_cfg.get("shape_text", {}).get("max_features", 2000),
            "ngram_range":  tuple(_features_cfg.get("shape_text", {}).get("ngram_range", [1, 3])),
            "min_df":       _features_cfg.get("shape_text", {}).get("min_df", 2),
            "max_df":       _features_cfg.get("shape_text", {}).get("max_df", 0.98),
        }
        tr_sh, te_sh, vec_sh = _build_tfidf_view(train_shape, test_shape, shape_params, "shape_text")
        tfidf_vectorizers["shape"] = vec_sh
        tfidf_parts_train.append(tr_sh)
        tfidf_parts_test.append(te_sh)
        tfidf_names_all += [f"tfidf_shape_{n}" for n in vec_sh.get_feature_names_out()]

        # View 3: path_text (파일 경로 토큰)
        path_col = FILE_PATH_COLUMN
        if path_col in df_train.columns:
            train_path_txt = df_train[path_col].fillna("").apply(_to_path_text)
            test_path_txt  = df_test[path_col].fillna("").apply(_to_path_text)
        else:
            train_path_txt = train_raw.apply(_to_path_text)
            test_path_txt  = test_raw.apply(_to_path_text)

        path_params = {
            "max_features": _features_cfg.get("path_text", {}).get("max_features", 1000),
            "ngram_range":  tuple(_features_cfg.get("path_text", {}).get("ngram_range", [1, 2])),
            "min_df":       _features_cfg.get("path_text", {}).get("min_df", 2),
            "max_df":       _features_cfg.get("path_text", {}).get("max_df", 0.95),
        }
        tr_pt, te_pt, vec_pt = _build_tfidf_view(train_path_txt, test_path_txt, path_params, "path_text")
        tfidf_vectorizers["path"] = vec_pt
        tfidf_parts_train.append(tr_pt)
        tfidf_parts_test.append(te_pt)
        tfidf_names_all += [f"tfidf_path_{n}" for n in vec_pt.get_feature_names_out()]

        tfidf_feature_count = sum(p.shape[1] for p in tfidf_parts_train)
        print(f"  총 TF-IDF Feature: {tfidf_feature_count:,}")

    elif not use_multiview_tfidf:
        # Phase 1 TF-IDF: file_name char + shape + file_path path_text (label-only 모드 보강)
        if use_phase1_tfidf:
            print("[Phase 1 TF-IDF] file_name char + shape + file_path path_text 추가")
            fname_col = "file_name"
            if fname_col in df_train.columns:
                train_fname = df_train[fname_col].fillna("")
                test_fname  = df_test[fname_col].fillna("")

                # View 1: file_name raw char n-gram
                fname_params = {
                    "analyzer": "char_wb",
                    "ngram_range": (2, 5),
                    "max_features": 200,
                    "min_df": 2,
                    "max_df": 0.98,
                }
                try:
                    tr_fn, te_fn, vec_fn = _build_tfidf_view(
                        train_fname, test_fname, fname_params, "phase1_fname_char"
                    )
                    tfidf_vectorizers["phase1_fname"] = vec_fn
                    tfidf_parts_train.append(tr_fn)
                    tfidf_parts_test.append(te_fn)
                    tfidf_names_all += [f"tfidf_fname_{n}" for n in vec_fn.get_feature_names_out()]
                except ValueError as _e:
                    print(f"  [WARN] file_name TF-IDF 건너뜀 (어휘 없음): {_e}")

                # [Tier 2 B3] View 2: file_name shape (숫자→D, 문자→L 변환)
                # 숫자 n-gram 과적합 감소: "02506" → "DDDDD" 패턴 학습
                train_fname_shape = train_fname.apply(_to_shape_text)
                test_fname_shape  = test_fname.apply(_to_shape_text)
                fname_shape_params = {
                    "analyzer": "char_wb",
                    "ngram_range": (2, 5),
                    "max_features": 100,
                    "min_df": 2,
                    "max_df": 0.98,
                }
                try:
                    tr_fns, te_fns, vec_fns = _build_tfidf_view(
                        train_fname_shape, test_fname_shape, fname_shape_params, "phase1_fname_shape"
                    )
                    tfidf_vectorizers["phase1_fname_shape"] = vec_fns
                    tfidf_parts_train.append(tr_fns)
                    tfidf_parts_test.append(te_fns)
                    tfidf_names_all += [f"tfidf_fshape_{n}" for n in vec_fns.get_feature_names_out()]
                except ValueError as _e:
                    print(f"  [WARN] file_name shape TF-IDF 건너뜀: {_e}")

            # View 3: file_path path_text (토큰화)
            path_col_p1 = FILE_PATH_COLUMN
            if path_col_p1 in df_train.columns:
                train_path_p1 = df_train[path_col_p1].fillna("").apply(_to_path_text)
                test_path_p1  = df_test[path_col_p1].fillna("").apply(_to_path_text)
                path1_params = {
                    "max_features": 200,
                    "ngram_range": (1, 2),
                    "min_df": 2,
                    "max_df": 0.95,
                }
                try:
                    tr_p1, te_p1, vec_p1 = _build_tfidf_view(
                        train_path_p1, test_path_p1, path1_params, "phase1_path_text"
                    )
                    tfidf_vectorizers["phase1_path"] = vec_p1
                    tfidf_parts_train.append(tr_p1)
                    tfidf_parts_test.append(te_p1)
                    tfidf_names_all += [
                        f"tfidf_phase1path_{n}" for n in vec_p1.get_feature_names_out()
                    ]
                except ValueError as _e:
                    print(f"  [WARN] file_path TF-IDF 건너뜀 (어휘 없음): {_e}")
            phase1_tfidf_count = sum(p.shape[1] for p in tfidf_parts_train)
            if phase1_tfidf_count > 0:
                print(f"  Phase 1 TF-IDF 총 피처: {phase1_tfidf_count:,}")

    else:
        # use_multiview_tfidf=True but text_column missing - warn and skip
        print(f"\n[경고] text_column='{text_column}' 컬럼 없음 -> TF-IDF 건너뜀")

    # ── 3. Dense Feature 생성 ─────────────────────────────────────────────────
    if use_synthetic_expansion and has_text_column:
        text_train = create_all_text_features(df_train[text_column])
        text_test  = create_all_text_features(df_test[text_column])
        path_train = create_all_path_features(df_train, path_column=FILE_PATH_COLUMN)
        path_test  = create_all_path_features(df_test,  path_column=FILE_PATH_COLUMN)
    elif not use_synthetic_expansion and has_text_column:
        kw_train   = create_keyword_features(train_raw)
        kw_test    = create_keyword_features(test_raw)
        stat_train = create_text_stat_features(train_raw)
        stat_test  = create_text_stat_features(test_raw)
        text_train = pd.concat([kw_train, stat_train], axis=1)
        text_test  = pd.concat([kw_test, stat_test], axis=1)
        path_train = create_file_path_features(df_train, path_column=FILE_PATH_COLUMN)
        path_test  = create_file_path_features(df_test,  path_column=FILE_PATH_COLUMN)
    else:
        # dense only - no text features (label-only mode)
        text_train = pd.DataFrame(index=df_train.index)
        text_test  = pd.DataFrame(index=df_test.index)
        path_train = create_file_path_features(df_train, path_column=FILE_PATH_COLUMN)
        path_test  = create_file_path_features(df_test,  path_column=FILE_PATH_COLUMN)
        # server_freq 제거됨 — train 통계 누수 위험 (model_performance_report.md §2.4 참조)
        # precomputed meta/path 컬럼 포함 (build_meta_features + extract_path_features 결과)
        # run_training.py Step 2에서 df에 이미 추가된 컬럼을 feature matrix에 반영한다.
        _PRECOMPUTED_DENSE_COLS = [
            # build_meta_features 결과 (시간 피처 4개 제거: created_hour/weekday/weekend/month)
            "fname_has_date", "fname_has_hash", "fname_has_rotation_num",
            "pattern_count_log1p", "pattern_count_bin",
            "is_mass_detection", "is_extreme_detection", "pii_type_ratio",
            # extract_path_features 결과 (10개)
            "is_log_file", "is_docker_overlay", "has_license_path",
            "is_temp_or_dev", "is_system_device", "is_package_path",
            "has_cron_path", "has_date_in_path", "has_business_token", "has_system_token",
            # Rule Labeler 결과
            "rule_matched",
            # exception_requested — Sumologic에 없음, 추론 불가 → 제거
            # [Tier 2 B7] 서버 의미 토큰 (server_freq 대체 일반화 신호)
            "server_is_prod",
            # [Tier 2 B8] RULE 세부 신호 (rule_matched binary → 12개 룰 도메인 지식)
            "rule_confidence_lb",
            # [Tier 2 B9] file-level aggregation (compute_file_aggregates_label)
            "file_event_count", "file_pii_diversity",
            # 파일 크기
            "file_size_log1p",
        ]

        # [Tier 2 B1+B7+B8] 범주형 피처 — Label Encoding으로 정수 변환
        _CATEGORICAL_COLS = [
            "service", "ops_dept", "organization", "retention_period",  # B1
            "server_env", "server_stack",       # B7
            "rule_id", "rule_primary_class",    # B8
        ]
        _cat_cols_present = [c for c in _CATEGORICAL_COLS
                            if c in df_train.columns and c in df_test.columns]
        _categorical_encoders = {}
        if _cat_cols_present:
            from sklearn.preprocessing import LabelEncoder as _LE
            for _cc in _cat_cols_present:
                _le_cat = _LE()
                # train+test 합쳐서 fit (unseen 방지)
                _combined = pd.concat([
                    df_train[_cc].fillna("__MISSING__").astype(str),
                    df_test[_cc].fillna("__MISSING__").astype(str),
                ])
                _le_cat.fit(_combined)
                df_train[_cc + "_enc"] = _le_cat.transform(
                    df_train[_cc].fillna("__MISSING__").astype(str)
                )
                df_test[_cc + "_enc"] = _le_cat.transform(
                    df_test[_cc].fillna("__MISSING__").astype(str)
                )
                _PRECOMPUTED_DENSE_COLS.append(_cc + "_enc")
                _categorical_encoders[_cc] = _le_cat
            print(f"  [Tier 2 B1/B7/B8] 범주형 Label Encoding: {_cat_cols_present}")

        _pc_train = df_train.reindex(
            columns=[c for c in _PRECOMPUTED_DENSE_COLS if c in df_train.columns]
        )
        _pc_test = df_test.reindex(
            columns=[c for c in _PRECOMPUTED_DENSE_COLS if c in df_test.columns]
        )
        _pc_train_num = _pc_train.select_dtypes(include=[np.number])
        _pc_test_num  = _pc_test.select_dtypes(include=[np.number])
        if len(_pc_train_num.columns) > 0:
            path_train = pd.concat([path_train, _pc_train_num], axis=1)
            path_test  = pd.concat([path_test,  _pc_test_num],  axis=1)
            # 중복 컬럼 제거 (path_depth 등이 양쪽에 있을 경우 첫 번째 유지)
            path_train = path_train.loc[:, ~path_train.columns.duplicated()]
            path_test  = path_test.loc[:,  ~path_test.columns.duplicated()]

    if "file_extension" in path_train.columns:
        path_train, path_test, _ = encode_categorical(
            path_train, ["file_extension"], path_test
        )

    if use_synthetic_expansion:
        syn_train = create_synthetic_interaction_features(text_train, path_train)
        syn_test  = create_synthetic_interaction_features(text_test, path_test)
    else:
        syn_train = pd.DataFrame(index=df_train.index)
        syn_test  = pd.DataFrame(index=df_test.index)

    dense_train_parts = [text_train]
    dense_test_parts  = [text_test]

    if len(path_train.columns) > 0:
        dense_train_parts.append(path_train.select_dtypes(include=[np.number]))
        dense_test_parts.append(path_test.select_dtypes(include=[np.number]))

    if len(syn_train.columns) > 0:
        dense_train_parts.append(syn_train)
        dense_test_parts.append(syn_test)

    # §7.2 수정: fillna(0) 제거 - LightGBM/XGBoost는 NaN을 missing으로 native 처리
    # "값이 0"과 "값이 없음(NaN)"을 구분하여 더 정확한 split 학습 가능
    dense_train = pd.concat(dense_train_parts, axis=1)
    dense_test  = pd.concat(dense_test_parts, axis=1)
    for _col in dense_train.columns:
        dense_train[_col] = pd.to_numeric(dense_train[_col], errors="coerce")
        dense_test[_col]  = pd.to_numeric(dense_test[_col],  errors="coerce")
    dense_feature_names = list(dense_train.columns)

    # ── 4. sparse + dense 결합 ────────────────────────────────────────────────
    # LightGBM은 float32/float64만 허용하므로 항상 float64로 변환
    dense_train_arr = dense_train.values.astype(np.float64)
    dense_test_arr  = dense_test.values.astype(np.float64)
    if tfidf_parts_train:
        X_train = hstack(tfidf_parts_train + [csr_matrix(dense_train_arr)])
        X_test  = hstack(tfidf_parts_test  + [csr_matrix(dense_test_arr)])
    else:
        X_train = csr_matrix(dense_train_arr)
        X_test  = csr_matrix(dense_test_arr)
    all_feature_names = tfidf_names_all + dense_feature_names

    # §4.3 Feature Selection: Near-Zero Variance 제거
    if use_variance_threshold and X_train.shape[1] > 0:
        from sklearn.feature_selection import VarianceThreshold
        n_before = X_train.shape[1]
        print(f"\n[Feature Selection] VarianceThreshold (threshold=1e-5)  before={n_before:,}")
        selector = VarianceThreshold(threshold=1e-5)
        X_train = selector.fit_transform(X_train)
        X_test  = selector.transform(X_test)
        selected_mask = selector.get_support()
        if len(all_feature_names) == len(selected_mask):
            all_feature_names = [n for n, m in zip(all_feature_names, selected_mask) if m]
        print(f"  피처: {n_before:,} -> {X_train.shape[1]:,}  ({n_before - X_train.shape[1]:,}개 제거)")

    print("\n" + "=" * 60)
    print("Feature Matrix 생성 완료")
    print("=" * 60)
    print(f"  최종 Feature 수: {X_train.shape[1]:,}")
    if tfidf_parts_train and use_multiview_tfidf:
        print(f"    - TF-IDF raw_text:   {tfidf_parts_train[0].shape[1]:,}")
        print(f"    - TF-IDF shape_text: {tfidf_parts_train[1].shape[1]:,}")
        print(f"    - TF-IDF path_text:  {tfidf_parts_train[2].shape[1]:,}")
    elif tfidf_parts_train:
        print(f"    - TF-IDF: {tfidf_parts_train[0].shape[1]:,}")
    else:
        print(f"    - TF-IDF: 비활성화 (dense only)")
    print(f"    - Dense:  {dense_train.shape[1]}")
    if use_synthetic_expansion:
        print(f"    - 합성 상호작용: {syn_train.shape[1]}")
    print(f"  학습 행렬: {X_train.shape}")
    print(f"  테스트 행렬: {X_test.shape}")
    print(f"  Split 전략: {split_strategy_used}")

    # split_meta 구성
    split_meta = {
        "split_strategy": split_strategy_used,
        "test_size_target": test_size,
        "train_size": len(df_train),
        "test_size_actual": len(df_test),
        "train_n_pkfiles": int(df_train["pk_file"].nunique()) if "pk_file" in df_train.columns else None,
        "test_n_pkfiles":  int(df_test["pk_file"].nunique())  if "pk_file" in df_test.columns  else None,
        "tfidf_mode": "multiview" if (use_multiview_tfidf and has_text_column) else ("none" if not use_multiview_tfidf else "skipped_no_text_column"),
        "tfidf_views": list(tfidf_vectorizers.keys()),
        "random_seed": RANDOM_SEED,
        "use_synthetic_expansion": use_synthetic_expansion,
        "total_features": int(X_train.shape[1]),
    }

    # categorical_encoders가 정의되지 않은 경우 (non-dense 경로)
    try:
        _categorical_encoders
    except NameError:
        _categorical_encoders = {}

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_names": all_feature_names,
        "tfidf_vectorizers": tfidf_vectorizers,
        # backward compat key: 첫 번째 vectorizer 반환 (dense-only이면 None)
        "tfidf_vectorizer": next(iter(tfidf_vectorizers.values())) if tfidf_vectorizers else None,
        "df_train": df_train,
        "df_test":  df_test,
        "split_meta": split_meta,
        "categorical_encoders": _categorical_encoders,
    }


# ── 저장 / 로드 ───────────────────────────────────────────────────────────────

def save_feature_artifacts(result: dict, feature_dir: str, model_dir: str) -> None:
    """Feature 결과물을 디스크에 저장합니다.

    저장 파일:
        feature_dir/
            X_train.npz, X_test.npz
            y_train.csv, y_test.csv
            feature_names.joblib
            df_train.parquet, df_test.parquet   <- 기본 (Parquet)
            df_train.csv,    df_test.csv        <- 사람이 읽을 수 있는 백업 (UTF-8 BOM)
            split_meta.json
        model_dir/
            tfidf_raw_vectorizer.joblib
            tfidf_shape_vectorizer.joblib
            tfidf_path_vectorizer.joblib
            tfidf_vectorizer.joblib             <- backward compat (raw 별칭)
    """
    fp = Path(feature_dir)
    mp = Path(model_dir)
    fp.mkdir(parents=True, exist_ok=True)
    mp.mkdir(parents=True, exist_ok=True)

    # Feature Matrix
    sparse.save_npz(fp / "X_train.npz", result["X_train"].tocsr())
    sparse.save_npz(fp / "X_test.npz",  result["X_test"].tocsr())

    # 레이블
    result["y_train"].to_csv(fp / "y_train.csv", index=False, encoding="utf-8")
    result["y_test"].to_csv(fp  / "y_test.csv",  index=False, encoding="utf-8")

    # Feature 이름
    joblib.dump(result["feature_names"], fp / "feature_names.joblib")

    # TF-IDF Vectorizers
    vecs = result.get("tfidf_vectorizers", {})
    for view_name, vec in vecs.items():
        joblib.dump(vec, mp / f"tfidf_{view_name}_vectorizer.joblib")
    # backward compat
    if "raw" in vecs:
        joblib.dump(vecs["raw"], mp / "tfidf_vectorizer.joblib")
    elif "single" in vecs:
        joblib.dump(vecs["single"], mp / "tfidf_vectorizer.joblib")
    elif vecs:
        joblib.dump(next(iter(vecs.values())), mp / "tfidf_vectorizer.joblib")

    # 원본 DataFrame - Parquet (primary) + CSV (human-readable)
    # pyarrow/fastparquet 미설치 환경(폐쇄망)에서는 CSV만 저장
    _parquet_ok = False
    try:
        result["df_train"].to_parquet(fp / "df_train.parquet", index=False)
        result["df_test"].to_parquet(fp  / "df_test.parquet",  index=False)
        _parquet_ok = True
    except ImportError:
        print("  [경고] pyarrow/fastparquet 미설치 - Parquet 저장 건너뜀, CSV로만 저장합니다.")
    # utf-8-sig: Windows Excel에서 한글 깨짐 방지
    result["df_train"].to_csv(fp / "df_train.csv", index=False, encoding="utf-8-sig")
    result["df_test"].to_csv(fp  / "df_test.csv",  index=False, encoding="utf-8-sig")

    # split_meta
    split_meta = result.get("split_meta", {})
    with open(fp / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)

    print("[저장 완료]")
    print(f"  X_train/X_test:      {fp}/X_train.npz, X_test.npz")
    print(f"  y_train/y_test:      {fp}/y_train.csv, y_test.csv")
    print(f"  feature_names:       {fp}/feature_names.joblib")
    _df_label = f"{fp}/df_train.parquet + .csv" if _parquet_ok else f"{fp}/df_train.csv (Parquet 생략)"
    print(f"  df_train/df_test:    {_df_label}")
    print(f"  split_meta:          {fp}/split_meta.json")
    vec_names = ", ".join(f"tfidf_{k}_vectorizer.joblib" for k in vecs)
    print(f"  vectorizers:         {mp}/{vec_names}")


def load_feature_artifacts(feature_dir: str) -> dict:
    """저장된 Feature 결과물을 로드합니다.

    df_train/df_test: Parquet 우선, 없으면 CSV fallback
    tfidf_vectorizers: 다중 뷰 우선, 없으면 단일 vectorizer
    """
    fp = Path(feature_dir)

    X_train = sparse.load_npz(fp / "X_train.npz")
    X_test  = sparse.load_npz(fp / "X_test.npz")
    y_train = pd.read_csv(fp / "y_train.csv").iloc[:, 0]
    y_test  = pd.read_csv(fp / "y_test.csv").iloc[:, 0]
    feature_names = joblib.load(fp / "feature_names.joblib")

    # DataFrame - Parquet 우선, ImportError(폐쇄망 미설치) 또는 파일 없으면 CSV fallback
    _pq_loaded = False
    if (fp / "df_train.parquet").exists():
        try:
            df_train = pd.read_parquet(fp / "df_train.parquet")
            df_test  = pd.read_parquet(fp / "df_test.parquet")
            _pq_loaded = True
        except ImportError:
            pass
    if not _pq_loaded:
        df_train = pd.read_csv(fp / "df_train.csv", encoding="utf-8-sig")
        df_test  = pd.read_csv(fp / "df_test.csv",  encoding="utf-8-sig")

    # split_meta
    split_meta = {}
    if (fp / "split_meta.json").exists():
        with open(fp / "split_meta.json", "r", encoding="utf-8") as f:
            split_meta = json.load(f)

    print("[Feature 로드 완료]")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}  y_test: {y_test.shape}")
    print(f"  Feature 수: {len(feature_names):,}")
    if split_meta:
        print(f"  Split 전략: {split_meta.get('split_strategy', 'unknown')}")

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_names": feature_names,
        "df_train": df_train,
        "df_test":  df_test,
        "split_meta": split_meta,
    }
