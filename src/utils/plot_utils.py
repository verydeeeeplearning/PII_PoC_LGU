"""공통 시각화 설정 유틸리티"""
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.constants import FONT_FAMILY

_FONT_NOTICE_SHOWN = False

# 환경별 한글 폰트 후보 목록 (우선순위 순)
_KOREAN_FONT_CANDIDATES = [
    "Malgun Gothic",     # Windows 기본 한글 폰트
    "NanumGothic",       # Linux/서버 Nanum 폰트
    "NanumBarunGothic",  # Nanum 계열
    "AppleGothic",       # macOS 한글 폰트
    "Gulim",             # Windows 구버전 한글 폰트
    "Dotum",             # Windows 구버전 한글 폰트
]


def _find_available_korean_font() -> str | None:
    """설치된 폰트 목록에서 첫 번째로 사용 가능한 한글 폰트를 반환한다."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in _KOREAN_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return None


def setup_plot() -> None:
    """시각화 공통 설정 (한글 폰트 자동 설정 + rcParams 설정)"""
    global _FONT_NOTICE_SHOWN

    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font", category=UserWarning)

    font_to_use = FONT_FAMILY

    # 설정된 폰트가 DejaVu Sans(기본값)이면 한글 폰트를 자동 탐색
    if font_to_use == "DejaVu Sans":
        detected = _find_available_korean_font()
        if detected:
            font_to_use = detected
            if not _FONT_NOTICE_SHOWN:
                print(f"[시각화] 한글 폰트 자동 감지: {detected}")
                _FONT_NOTICE_SHOWN = True
        else:
            if not _FONT_NOTICE_SHOWN:
                print(
                    "[참고] 한글 폰트를 찾을 수 없습니다. "
                    "서버에 한글 폰트를 설치한 뒤 config/feature_config.yaml의 "
                    "viz.font_family를 NanumGothic 등으로 변경하세요."
                )
                _FONT_NOTICE_SHOWN = True
    else:
        # 설정에 명시된 폰트가 실제로 설치되어 있는지 확인
        available = {f.name for f in fm.fontManager.ttflist}
        if font_to_use not in available:
            if not _FONT_NOTICE_SHOWN:
                print(
                    f"[경고] config에 설정된 폰트 '{font_to_use}'를 찾을 수 없습니다. "
                    "DejaVu Sans로 대체합니다."
                )
                _FONT_NOTICE_SHOWN = True
            font_to_use = "DejaVu Sans"

    # sns.set_style 이 font 관련 rcParams를 덮어쓰므로 먼저 호출
    sns.set_style("whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = font_to_use
    plt.rcParams["font.sans-serif"] = [font_to_use, "DejaVu Sans"]
