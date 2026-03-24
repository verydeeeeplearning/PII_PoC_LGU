"""프로젝트 전역 상수"""
import os
from pathlib import Path
from typing import Any, Dict

# ── 재현성 ──
RANDOM_SEED: int = 42

# ── 경로 ──
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR: Path = PROJECT_ROOT / "data" / "features"
MODEL_DIR: Path = PROJECT_ROOT / "models"
FINAL_MODEL_DIR: Path = MODEL_DIR / "final"
CHECKPOINT_DIR: Path = MODEL_DIR / "checkpoints"
FIGURE_DIR: Path = PROJECT_ROOT / "outputs"          # legacy alias (= REPORT_DIR)
REPORT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = PROJECT_ROOT / "outputs" / "figures"
DIAGNOSIS_DIR: Path = PROJECT_ROOT / "outputs" / "diagnosis"
PREDICTIONS_DIR: Path = PROJECT_ROOT / "outputs" / "predictions"
EXPORTS_DIR: Path = PROJECT_ROOT / "outputs" / "exports"

# ── 컬럼명 ──
TEXT_COLUMN: str = "detected_text_with_context"
LABEL_COLUMN: str = "label"
FILE_PATH_COLUMN: str = "file_path"
PK_COLUMNS: list = ["detection_id"]

# ── 파일명 ──
DETECTION_FILE: str = "detection_results.csv"
LABEL_FILE: str = "labels.csv"
MERGED_CLEANED_FILE: str = "merged_cleaned.csv"
ENCODING: str = "utf-8"

# ── 레이블 체계 (회의록 2026-01 확정) ──
# 총 8개 클래스: 정탐 1종 + 오탐 7종
LABEL_TP: str = "TP-실제개인정보"                    # 0: 정탐 - 실제 개인정보 -> 파기 대상
LABEL_FP_NUMERIC_CODE: str = "FP-숫자나열/코드"      # 1: 단순 숫자/문자 나열, 버전번호, 일련번호
LABEL_FP_DUMMY_DATA: str = "FP-더미데이터"           # 2: 개발용 테스트/더미 데이터
LABEL_FP_TIMESTAMP: str = "FP-타임스탬프"            # 3: 시간/날짜/타임스탬프 값
LABEL_FP_INTERNAL_DOMAIN: str = "FP-내부도메인"      # 4: LG U+ 내부 이메일/도메인
LABEL_FP_BYTES: str = "FP-bytes크기"                 # 5: 파일 크기(바이트) 값
LABEL_FP_OS_COPYRIGHT: str = "FP-OS저작권"           # 6: OS/오픈소스 저작권 표기 내 이메일
LABEL_FP_CONTEXT: str = "FP-패턴맥락"                # 7: 앞뒤 컨텍스트로 판단 가능한 기타 오탐

# 레이블 리스트 (인덱스 순서 = 클래스 번호)
LABEL_NAMES: list = [
    LABEL_TP,               # 0
    LABEL_FP_NUMERIC_CODE,  # 1
    LABEL_FP_DUMMY_DATA,    # 2
    LABEL_FP_TIMESTAMP,     # 3
    LABEL_FP_INTERNAL_DOMAIN,  # 4
    LABEL_FP_BYTES,         # 5
    LABEL_FP_OS_COPYRIGHT,  # 6
    LABEL_FP_CONTEXT,       # 7
]

# 클래스별 상세 설명 (리포트/문서용)
CLASS_DESCRIPTIONS: dict = {
    LABEL_TP: "실존 인물과 연결 가능한 실제 개인정보",
    LABEL_FP_NUMERIC_CODE: "JGNORE 코드값, SW 버전번호(1.3.3.32-2087-1512), 일련번호 등",
    LABEL_FP_DUMMY_DATA: "개발용 테스트 이메일(@entry.sc, @cherry.email), 더미 데이터",
    LABEL_FP_TIMESTAMP: "Unix timestamp, 날짜형식(xpiryDate=170603), x-timestamp 등",
    LABEL_FP_INTERNAL_DOMAIN: "@lguplus.co.kr, @bdp.lguplus.co.kr, Kerberos 토큰 내 도메인",
    LABEL_FP_BYTES: "파일 크기값(45 bytes 141022), 바이트 정보",
    LABEL_FP_OS_COPYRIGHT: "@redhat.com, @fedora-project, @apache.org 등 오픈소스 저작권",
    LABEL_FP_CONTEXT: "파일 경로·로그 구조상 명백한 비개인정보 (기타)",
}

# 정탐/오탐 구분
TP_LABELS: list = [LABEL_TP]
FP_LABELS: list = [
    LABEL_FP_NUMERIC_CODE,
    LABEL_FP_DUMMY_DATA,
    LABEL_FP_TIMESTAMP,
    LABEL_FP_INTERNAL_DOMAIN,
    LABEL_FP_BYTES,
    LABEL_FP_OS_COPYRIGHT,
    LABEL_FP_CONTEXT,
]

# ── TF-IDF 파라미터 ──
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: tuple = (1, 2)
TFIDF_MIN_DF: int = 5
TFIDF_MAX_DF: float = 0.95

# ── 키워드 그룹 ──
KEYWORD_GROUPS: dict = {
    "has_test_keyword": ["test", "테스트", "tset", "testing"],
    "has_sample_keyword": ["sample", "샘플", "example", "예시", "예제"],
    "has_dev_keyword": ["dev", "debug", "개발", "mock", "dummy", "stub"],
    "has_temp_keyword": ["temp", "tmp", "임시", "temporary"],
    "has_doc_keyword": ["문서", "매뉴얼", "manual", "guide", "가이드"],
    "has_system_keyword": ["system", "시스템", "auto", "generated", "자동"],
    "has_sequential_digits": ["000", "111", "123", "1234", "0000"],
}

# ── 모델 기본 파라미터 ──
XGB_DEFAULT_PARAMS: dict = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "n_jobs": -1,
    "verbosity": 0,
}

LGB_DEFAULT_PARAMS: dict = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "n_estimators": 500,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 200,       # [Tier 2 B6] 일반화 강제
    "reg_alpha": 0.5,               # [Tier 2 B6] L1 정규화
    "max_depth": 10,                # [Tier 2 B6] 트리 깊이 제한
    "n_jobs": -1,
    "verbose": -1,
}

# ── 하이퍼파라미터 튜닝 ──
XGB_PARAM_GRID: dict = {
    "n_estimators": [100, 300, 500, 700],
    "max_depth": [3, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
}

TUNING_N_ITER: int = 30
TUNING_CV_FOLDS: int = 3

# ── 학습/평가 ──
TEST_SIZE: float = 0.2
TOP_N_FEATURES: int = 30
EARLY_STOPPING_ROUNDS: int = 30

# ── PoC 성공 기준 ──
POC_F1_MACRO_THRESHOLD: float = 0.70
POC_TP_RECALL_THRESHOLD: float = 0.75
POC_FP_PRECISION_THRESHOLD: float = 0.80

# ── 시각화 ──
FONT_FAMILY: str = "DejaVu Sans"
FIGURE_DPI: int = 150


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow+deep merge for nested dicts (in-place on dst)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        # Keep imports resilient in air-gapped environments; fall back to defaults.
        return {}


def load_yaml(path: Path) -> Dict[str, Any]:
    """외부에서 사용 가능한 YAML 로더 (예외 안전, yaml 미설치 시 {} 반환)."""
    return _load_yaml(path)


def _apply_feature_config(cfg: Dict[str, Any]) -> None:
    global RANDOM_SEED, TEST_SIZE
    global TEXT_COLUMN, LABEL_COLUMN, FILE_PATH_COLUMN, PK_COLUMNS, ENCODING, DETECTION_FILE, LABEL_FILE
    global TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF
    global KEYWORD_GROUPS
    global POC_F1_MACRO_THRESHOLD, POC_TP_RECALL_THRESHOLD, POC_FP_PRECISION_THRESHOLD
    global FONT_FAMILY, FIGURE_DPI

    data = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    split = cfg.get("split", {}) if isinstance(cfg.get("split", {}), dict) else {}
    tfidf = cfg.get("tfidf", {}) if isinstance(cfg.get("tfidf", {}), dict) else {}
    keywords = cfg.get("keywords", {}) if isinstance(cfg.get("keywords", {}), dict) else {}
    evaluation = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    viz = cfg.get("viz", {}) if isinstance(cfg.get("viz", {}), dict) else {}

    # Data schema
    TEXT_COLUMN = data.get("text_column", TEXT_COLUMN)
    LABEL_COLUMN = data.get("label_column", LABEL_COLUMN)
    PK_COLUMNS = data.get("pk_columns", PK_COLUMNS)
    FILE_PATH_COLUMN = data.get("file_path_column", FILE_PATH_COLUMN)
    ENCODING = data.get("encoding", ENCODING)
    DETECTION_FILE = data.get("detection_file", DETECTION_FILE)
    LABEL_FILE = data.get("label_file", LABEL_FILE)

    # Train/test split
    RANDOM_SEED = int(split.get("random_seed", RANDOM_SEED))
    TEST_SIZE = float(split.get("test_size", TEST_SIZE))

    # TF-IDF
    TFIDF_MAX_FEATURES = int(tfidf.get("max_features", TFIDF_MAX_FEATURES))
    ngram = tfidf.get("ngram_range", TFIDF_NGRAM_RANGE)
    if isinstance(ngram, (list, tuple)) and len(ngram) == 2:
        TFIDF_NGRAM_RANGE = (int(ngram[0]), int(ngram[1]))
    TFIDF_MIN_DF = int(tfidf.get("min_df", TFIDF_MIN_DF))
    TFIDF_MAX_DF = float(tfidf.get("max_df", TFIDF_MAX_DF))

    # Keywords
    groups = keywords.get("groups", None)
    if isinstance(groups, dict) and groups:
        KEYWORD_GROUPS = groups

    # PoC thresholds
    POC_F1_MACRO_THRESHOLD = float(evaluation.get("poc_f1_macro_threshold", POC_F1_MACRO_THRESHOLD))
    POC_TP_RECALL_THRESHOLD = float(evaluation.get("poc_tp_recall_threshold", POC_TP_RECALL_THRESHOLD))
    POC_FP_PRECISION_THRESHOLD = float(evaluation.get("poc_fp_precision_threshold", POC_FP_PRECISION_THRESHOLD))

    # Viz
    FONT_FAMILY = viz.get("font_family", FONT_FAMILY)
    FIGURE_DPI = int(viz.get("figure_dpi", FIGURE_DPI))


def _apply_model_config(cfg: Dict[str, Any]) -> None:
    global XGB_DEFAULT_PARAMS, LGB_DEFAULT_PARAMS, XGB_PARAM_GRID, TUNING_N_ITER, TUNING_CV_FOLDS, TOP_N_FEATURES, EARLY_STOPPING_ROUNDS

    xgb = cfg.get("xgboost", {}) if isinstance(cfg.get("xgboost", {}), dict) else {}
    lgb = cfg.get("lightgbm", {}) if isinstance(cfg.get("lightgbm", {}), dict) else {}
    tuning = cfg.get("tuning", {}) if isinstance(cfg.get("tuning", {}), dict) else {}
    training = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}

    xgb_params = xgb.get("default_params", None)
    if isinstance(xgb_params, dict):
        _deep_update(XGB_DEFAULT_PARAMS, xgb_params)

    lgb_params = lgb.get("default_params", None)
    if isinstance(lgb_params, dict):
        _deep_update(LGB_DEFAULT_PARAMS, lgb_params)

    grid = tuning.get("xgb_param_grid", None)
    if isinstance(grid, dict):
        XGB_PARAM_GRID = grid

    if "n_iter" in tuning:
        TUNING_N_ITER = int(tuning["n_iter"])
    if "cv_folds" in tuning:
        TUNING_CV_FOLDS = int(tuning["cv_folds"])

    if "top_n_features" in training:
        TOP_N_FEATURES = int(training["top_n_features"])
    if "early_stopping_rounds" in training:
        EARLY_STOPPING_ROUNDS = int(training["early_stopping_rounds"])


def _load_project_overrides() -> None:
    # Allow operators to point to an external config directory without editing code.
    # Example: export PII_FPR_CONFIG_DIR=/path/to/config
    cfg_dir = Path(os.environ.get("PII_FPR_CONFIG_DIR", str(PROJECT_ROOT / "config")))
    feature_cfg = _load_yaml(cfg_dir / "feature_config.yaml")
    model_cfg = _load_yaml(cfg_dir / "model_config.yaml")

    if isinstance(feature_cfg, dict) and feature_cfg:
        _apply_feature_config(feature_cfg)
    if isinstance(model_cfg, dict) and model_cfg:
        _apply_model_config(model_cfg)


# Apply config overrides at import time so scripts/modules pick them up consistently.
_load_project_overrides()
