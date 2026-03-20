"""데이터 품질 검증 모듈

회의록 2026-01 반영:
- 마스킹 데이터 검증 (~10자 컨텍스트)
- 패턴 종류 필드 신뢰도 검증
- PII 노출 검사
"""
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# PII 패턴 정규식
PATTERNS = {
    "phone": re.compile(
        r"01[016789]-?\d{3,4}-?\d{4}"  # 휴대폰 (마스킹 안 된 경우)
    ),
    "email": re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # 이메일 (마스킹 안 된 경우)
    ),
    "jumin": re.compile(
        r"\d{6}-?[1-4]\d{6}"  # 주민번호 (마스킹 안 된 경우)
    ),
    "masked": re.compile(
        r"\*{3,}"  # 마스킹 패턴 (*** 이상)
    ),
}

# 패턴 종류 판별용 정규식
PATTERN_TYPE_DETECTORS = {
    "이메일": re.compile(r"@[a-zA-Z0-9.-]+", re.IGNORECASE),
    "휴대폰번호": re.compile(r"01[016789][\*\d]{5,}"),
    "주민등록번호": re.compile(r"\d{6}[\*-]?[1-4\*][\d\*]{6}"),
}


@dataclass
class MaskingValidationResult:
    """마스킹 검증 결과"""
    total_rows: int = 0
    masked_rows: int = 0
    unmasked_rows: int = 0
    masking_rate: float = 0.0

    # 컨텍스트 길이 통계
    context_length_stats: Dict[str, float] = field(default_factory=dict)

    # PII 노출 검사 결과
    exposed_phone: int = 0
    exposed_email: int = 0
    exposed_jumin: int = 0

    # 노출된 샘플 (디버깅용)
    exposed_samples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_exposure(self) -> bool:
        """PII 노출이 있는지 여부"""
        return (self.exposed_phone + self.exposed_email + self.exposed_jumin) > 0

    def summary(self) -> str:
        """결과 요약"""
        lines = [
            "=" * 60,
            "[마스킹 검증 결과]",
            "=" * 60,
            f"총 데이터:      {self.total_rows:,}건",
            f"마스킹 적용:    {self.masked_rows:,}건 ({self.masking_rate * 100:.1f}%)",
            f"마스킹 미적용:  {self.unmasked_rows:,}건",
            "",
            "[컨텍스트 길이 통계]",
            f"  평균: {self.context_length_stats.get('mean', 0):.1f}자",
            f"  최소: {self.context_length_stats.get('min', 0):.0f}자",
            f"  최대: {self.context_length_stats.get('max', 0):.0f}자",
            f"  중앙값: {self.context_length_stats.get('median', 0):.1f}자",
            "",
            "[PII 노출 검사]",
            f"  휴대폰 노출: {self.exposed_phone:,}건",
            f"  이메일 노출: {self.exposed_email:,}건",
            f"  주민번호 노출: {self.exposed_jumin:,}건",
        ]

        if self.has_exposure:
            lines.append("")
            lines.append("⚠️ 경고: 마스킹되지 않은 PII가 발견되었습니다!")
        else:
            lines.append("")
            lines.append("✅ PII 노출 없음")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class PatternTypeValidationResult:
    """패턴 종류 검증 결과"""
    total_rows: int = 0
    matched_rows: int = 0
    mismatched_rows: int = 0
    mismatch_rate: float = 0.0

    # 불일치 유형별 카운트
    mismatch_details: Dict[str, int] = field(default_factory=dict)

    # 불일치 샘플 (디버깅용)
    mismatch_samples: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """결과 요약"""
        lines = [
            "=" * 60,
            "[패턴 종류 신뢰도 검증 결과]",
            "=" * 60,
            f"총 데이터:     {self.total_rows:,}건",
            f"일치:          {self.matched_rows:,}건",
            f"불일치:        {self.mismatched_rows:,}건 ({self.mismatch_rate * 100:.1f}%)",
        ]

        if self.mismatch_details:
            lines.append("")
            lines.append("[불일치 유형별 분포]")
            for mismatch_type, count in sorted(
                self.mismatch_details.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {mismatch_type}: {count:,}건")

        if self.mismatch_rate > 0.1:  # 10% 이상 불일치
            lines.append("")
            lines.append("⚠️ 경고: 패턴 종류 필드 신뢰도가 낮습니다 (불일치 10% 이상)")
            lines.append("   -> 검출 내역 기반 패턴 종류 재분류를 권장합니다.")

        lines.append("=" * 60)
        return "\n".join(lines)


def validate_masking(
    df: pd.DataFrame,
    text_column: str,
    max_samples: int = 10,
) -> MaskingValidationResult:
    """
    마스킹 데이터를 검증합니다.

    검증 항목:
    1. 마스킹 패턴 (*****) 존재 여부
    2. 컨텍스트 길이 통계 (~10자 예상)
    3. 마스킹되지 않은 PII 노출 여부

    Args:
        df: 검증할 DataFrame
        text_column: 텍스트 컬럼명 (검출 내역)
        max_samples: 저장할 최대 노출 샘플 수

    Returns:
        MaskingValidationResult
    """
    result = MaskingValidationResult(total_rows=len(df))

    if text_column not in df.columns:
        print(f"[경고] 텍스트 컬럼 '{text_column}'이 없습니다.")
        return result

    texts = df[text_column].fillna("")

    # 1. 마스킹 패턴 검사
    masked_mask = texts.str.contains(r"\*{3,}", regex=True, na=False)
    result.masked_rows = masked_mask.sum()
    result.unmasked_rows = result.total_rows - result.masked_rows
    result.masking_rate = result.masked_rows / max(result.total_rows, 1)

    # 2. 컨텍스트 길이 통계
    lengths = texts.str.len()
    result.context_length_stats = {
        "mean": lengths.mean(),
        "min": lengths.min(),
        "max": lengths.max(),
        "median": lengths.median(),
        "std": lengths.std(),
    }

    # 3. PII 노출 검사
    exposed_samples = []

    for idx, text in enumerate(texts):
        if pd.isna(text) or not text:
            continue

        exposures = []

        # 휴대폰 노출 검사
        phone_matches = PATTERNS["phone"].findall(text)
        if phone_matches:
            result.exposed_phone += len(phone_matches)
            exposures.append(("phone", phone_matches))

        # 이메일 노출 검사 (마스킹 패턴 제외)
        email_matches = PATTERNS["email"].findall(text)
        # 내부 도메인이나 시스템 이메일은 제외 (이미 마스킹된 것)
        real_emails = [e for e in email_matches if "****" not in e and "@" in e]
        if real_emails:
            result.exposed_email += len(real_emails)
            exposures.append(("email", real_emails))

        # 주민번호 노출 검사
        jumin_matches = PATTERNS["jumin"].findall(text)
        if jumin_matches:
            result.exposed_jumin += len(jumin_matches)
            exposures.append(("jumin", jumin_matches))

        # 노출 샘플 저장
        if exposures and len(exposed_samples) < max_samples:
            exposed_samples.append({
                "index": idx,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "exposures": exposures,
            })

    result.exposed_samples = exposed_samples

    print(result.summary())
    return result


def validate_pattern_type(
    df: pd.DataFrame,
    content_column: str,
    pattern_type_column: str,
    max_samples: int = 20,
) -> PatternTypeValidationResult:
    """
    패턴 종류 필드의 신뢰도를 검증합니다.

    회의록 발견사항:
    - 이메일 패턴인데 "주민번호"로 분류된 케이스 다수
    - Server-i가 숫자 패턴을 주민번호로 오인

    Args:
        df: 검증할 DataFrame
        content_column: 검출 내역 컬럼명
        pattern_type_column: 패턴 종류 컬럼명
        max_samples: 저장할 최대 불일치 샘플 수

    Returns:
        PatternTypeValidationResult
    """
    result = PatternTypeValidationResult(total_rows=len(df))

    if content_column not in df.columns:
        print(f"[경고] 검출 내역 컬럼 '{content_column}'이 없습니다.")
        return result

    if pattern_type_column not in df.columns:
        print(f"[경고] 패턴 종류 컬럼 '{pattern_type_column}'이 없습니다.")
        return result

    mismatch_samples = []
    mismatch_details = {}

    for idx, row in df.iterrows():
        content = str(row[content_column]) if pd.notna(row[content_column]) else ""
        declared_type = str(row[pattern_type_column]) if pd.notna(row[pattern_type_column]) else ""

        # 검출 내역에서 실제 패턴 유형 추정
        detected_type = _detect_pattern_type(content)

        if not detected_type:
            continue

        # 패턴 종류 정규화
        declared_normalized = _normalize_pattern_type(declared_type)

        # 불일치 검사
        if detected_type != declared_normalized:
            result.mismatched_rows += 1

            mismatch_key = f"{declared_normalized} -> 실제:{detected_type}"
            mismatch_details[mismatch_key] = mismatch_details.get(mismatch_key, 0) + 1

            if len(mismatch_samples) < max_samples:
                mismatch_samples.append({
                    "index": idx,
                    "content": content[:80] + "..." if len(content) > 80 else content,
                    "declared_type": declared_type,
                    "detected_type": detected_type,
                })
        else:
            result.matched_rows += 1

    result.mismatch_rate = result.mismatched_rows / max(result.total_rows, 1)
    result.mismatch_details = mismatch_details
    result.mismatch_samples = mismatch_samples

    print(result.summary())
    return result


def _detect_pattern_type(text: str) -> Optional[str]:
    """검출 내역에서 실제 패턴 유형을 추정합니다."""
    if not text:
        return None

    # 이메일 패턴 (@ 포함)
    if PATTERN_TYPE_DETECTORS["이메일"].search(text):
        return "이메일"

    # 휴대폰 패턴 (01x로 시작)
    if PATTERN_TYPE_DETECTORS["휴대폰번호"].search(text):
        return "휴대폰번호"

    # 주민번호 패턴 (6자리-7자리)
    if PATTERN_TYPE_DETECTORS["주민등록번호"].search(text):
        return "주민등록번호"

    return None


def _normalize_pattern_type(pattern_type: str) -> str:
    """패턴 종류 문자열을 정규화합니다."""
    pt = pattern_type.lower().strip()

    if "이메일" in pt or "email" in pt or "e-mail" in pt:
        return "이메일"
    if "휴대" in pt or "핸드폰" in pt or "phone" in pt or "mobile" in pt:
        return "휴대폰번호"
    if "주민" in pt or "jumin" in pt or "ssn" in pt:
        return "주민등록번호"

    return pattern_type


def auto_correct_pattern_type(
    df: pd.DataFrame,
    content_column: str,
    pattern_type_column: str,
    output_column: str = "pattern_type_corrected",
) -> pd.DataFrame:
    """
    검출 내역 기반으로 패턴 종류를 자동 보정합니다.

    Args:
        df: DataFrame
        content_column: 검출 내역 컬럼명
        pattern_type_column: 패턴 종류 컬럼명
        output_column: 보정된 패턴 종류를 저장할 컬럼명

    Returns:
        보정된 DataFrame
    """
    df = df.copy()

    def correct_type(row):
        content = str(row[content_column]) if pd.notna(row[content_column]) else ""
        original = str(row[pattern_type_column]) if pd.notna(row[pattern_type_column]) else ""

        detected = _detect_pattern_type(content)
        if detected:
            return detected
        return _normalize_pattern_type(original)

    df[output_column] = df.apply(correct_type, axis=1)

    # 변경 통계
    changed = (df[output_column] != df[pattern_type_column].apply(_normalize_pattern_type)).sum()
    print(f"[패턴 종류 보정] {changed:,}건 변경됨")

    return df


def validate_data(
    df: pd.DataFrame,
    label_column: str = "label",
) -> dict:
    """
    데이터 품질을 검증하고 리포트를 출력합니다.

    검증 항목:
        1. 기본 정보 (행/열 수)
        2. 결측치 (컬럼별 결측 수/비율)
        3. 중복 행 수
        4. 레이블 분포
        5. 데이터 타입 요약

    Args:
        df: 검증할 DataFrame
        label_column: 레이블 컬럼명

    Returns:
        dict: {
            "n_rows": int,
            "n_cols": int,
            "missing_columns": dict,
            "n_duplicates": int,
            "label_distribution": dict
        }
    """
    report = {}

    print("=" * 60)
    print("데이터 품질 검증 리포트")
    print("=" * 60)

    # 1. 기본 정보
    print(f"\n[1] 기본 정보")
    print(f"  총 행 수: {len(df):,}")
    print(f"  총 열 수: {len(df.columns)}")
    report["n_rows"] = len(df)
    report["n_cols"] = len(df.columns)

    # 2. 결측치
    print(f"\n[2] 결측치")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "결측 수": missing,
        "결측 비율(%)": missing_pct
    })
    missing_cols = missing_df[missing_df["결측 수"] > 0]
    if len(missing_cols) > 0:
        print(missing_cols.to_string())
    else:
        print("  결측치 없음")
    report["missing_columns"] = missing_cols.to_dict() if len(missing_cols) > 0 else {}

    # 3. 중복 행
    n_dup = df.duplicated().sum()
    print(f"\n[3] 중복 행: {n_dup:,}건")
    report["n_duplicates"] = n_dup

    # 4. 레이블 분포
    if label_column in df.columns:
        print(f"\n[4] 레이블 분포")
        label_dist = df[label_column].value_counts()
        label_pct = (df[label_column].value_counts(normalize=True) * 100).round(2)
        for label in label_dist.index:
            print(f"  {label}: {label_dist[label]:,}건 ({label_pct[label]}%)")
        report["label_distribution"] = label_dist.to_dict()

    # 5. 데이터 타입 요약
    print(f"\n[5] 데이터 타입 요약")
    print(f"  {df.dtypes.value_counts().to_string()}")

    print("=" * 60)
    return report


def full_validation(
    df: pd.DataFrame,
    text_column: str,
    label_column: str = "label",
    pattern_type_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    전체 데이터 품질 검증을 수행합니다.

    Args:
        df: 검증할 DataFrame
        text_column: 텍스트 컬럼명
        label_column: 레이블 컬럼명
        pattern_type_column: 패턴 종류 컬럼명 (선택)

    Returns:
        검증 결과 딕셔너리
    """
    results = {}

    # 1. 기본 데이터 검증
    print("\n" + "=" * 60)
    print("[1/3] 기본 데이터 품질 검증")
    print("=" * 60)
    results["basic"] = validate_data(df, label_column)

    # 2. 마스킹 검증
    print("\n" + "=" * 60)
    print("[2/3] 마스킹 데이터 검증")
    print("=" * 60)
    results["masking"] = validate_masking(df, text_column)

    # 3. 패턴 종류 검증 (컬럼이 있는 경우)
    if pattern_type_column and pattern_type_column in df.columns:
        print("\n" + "=" * 60)
        print("[3/3] 패턴 종류 신뢰도 검증")
        print("=" * 60)
        results["pattern_type"] = validate_pattern_type(
            df, text_column, pattern_type_column
        )

    return results
