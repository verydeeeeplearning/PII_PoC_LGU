"""
Stage S2-3: Path Features - Architecture.md §6.3

extract_path_features: 파일 경로에서 구조적 피처 11+개 추출.

파일 경로는 텍스트 컨텍스트(약 10자)가 제한된 환경에서
'생성 맥락'을 대리하는 가장 강력한 판별 피처 중 하나다.

[변경 이력 - docs/rule_labeler_analysis.md]
2026-03-15:
  - is_temp_or_dev: /dev/ 제거 (Linux 디바이스 경로 오매핑 방지)
  - is_system_device: 신규 - /dev/ /proc/ /sys/ OS 가상 파일시스템
  - is_package_path: 신규 - pip/npm/conda/RPM 패키지 설치 경로
  - has_cron_path: 신규 - cron/배치 로그 경로
  - is_log_file: .gz/.bz2 확장자 + 로테이션 패턴 추가
    (meta_features.py의 is_log_file 중복 제거에 따른 보강)
"""

from __future__ import annotations

import re
from typing import Optional


# 업무 관련 토큰 -> TP 방향 신호
_BUSINESS_TOKENS = [
    "customer", "crm", "hr", "billing", "export", "report",
    "contact", "personal", "member", "회원", "고객",
]

# 시스템/인프라 토큰 -> FP 방향 신호
_SYSTEM_TOKENS = [
    "var/log", "proc", "kernel", "auth", "krb", "kerberos", "hadoop",
    "hdfs", "datanode", "namenode", "syslog", "journald",
]

# 패키지/라이브러리 설치 경로 토큰 -> FP-라이브러리 방향 신호
_PACKAGE_PATH_TOKENS = [
    "dist-packages", "site-packages", "node_modules",
    "/var/cache/yum", "/var/cache/dnf", "/opt/conda/pkgs",
    "/usr/local/lib/python", "/usr/share/perl5",
    "/usr/lib/jvm", "/opt/java",
]

# 크론/배치 로그 경로 토큰
_CRON_PATH_TOKENS = [
    "/var/log/cron", "/etc/cron", "/cron.d/",
    "/var/spool/cron", "/var/log/batch",
    "/var/log/scheduler", "/system/generated",
    "/var/log/anacron",
]

# 로그 파일 확장자
_LOG_EXTENSIONS_PATH = {".log", ".gz", ".bz2"}

# 로테이션 패턴 (파일명 끝 - meta_features.py와 동일)
_RE_LOG_ROTATION = re.compile(r"\.log\.\d+$|\.\d+$|-\d{3,}$|-\d{4}$", re.IGNORECASE)


def extract_path_features(file_path: Optional[str]) -> dict:
    """
    파일 경로에서 구조적 피처를 추출 (Architecture.md §6.3).

    Parameters
    ----------
    file_path : 파일 전체 경로 문자열

    Returns
    -------
    피처 dict (이진 플래그 + 수치 피처):
        path_depth          int  : 경로 깊이 (/ 개수)
        extension           str  : 파일 확장자 (없으면 'unknown')
        is_log_file         0/1  : .log/.gz/.bz2 파일이거나 /log/ 경로 또는 로테이션 패턴
        is_docker_overlay   0/1  : Docker overlay 경로
        has_license_path    0/1  : 라이선스/저작권 경로
        is_temp_or_dev      0/1  : 임시/개발/테스트 경로 (/dev/ 제외)
        is_system_device    0/1  : OS 시스템 디바이스 경로 (/dev/, /proc/, /sys/)
        is_package_path     0/1  : 패키지/라이브러리 설치 경로
        has_cron_path       0/1  : 크론/배치 로그 경로
        has_date_in_path    0/1  : /YYYYMMDD/ 날짜 패턴
        has_business_token  0/1  : 업무 관련 토큰 (TP 방향)
        has_system_token    0/1  : 시스템/인프라 토큰 (FP 방향)
    """
    fp = (file_path or "").lower()

    extension = "unknown"
    if "." in fp:
        ext_candidate = fp.rsplit(".", 1)[-1]
        # 확장자로 보이는 부분만 (최대 6자, 알파숫자)
        if re.match(r"^[a-z0-9]{1,6}$", ext_candidate):
            extension = ext_candidate

    # 파일명 부분만 분리 (로테이션 패턴 검사용)
    fname = fp.rsplit("/", 1)[-1]

    # 확장자 기반 로그 파일 여부
    _ext_is_log = any(fname.endswith(ext) for ext in _LOG_EXTENSIONS_PATH)
    # 로테이션 패턴 기반 로그 파일 여부 (access.log.1, app-001 등)
    _rotation_is_log = bool(_RE_LOG_ROTATION.search(fname))

    features: dict = {
        "path_depth": fp.count("/"),
        "extension": extension,

        # is_log_file: meta_features.py 중복 제거에 따라 단일 소스로 통합
        # .gz/.bz2 압축 로그, 로테이션 패턴, /log/ 경로 모두 포함
        "is_log_file": int(
            _ext_is_log
            or _rotation_is_log
            or "/log/" in fp
            or "/logs/" in fp
        ),

        "is_docker_overlay": int(
            "overlay" in fp or "docker" in fp or "/var/lib/docker/" in fp
        ),

        "has_license_path": int(
            any(t in fp for t in [
                "/usr/share/doc/", "/license", "/copyright", "changelog",
                "/licenses/",
            ])
        ),

        # ⚠️ /dev/ 제거: Linux 디바이스 경로가 FP-더미테스트로 오분류되던 문제 수정
        # /dev/ 는 is_system_device로 분리
        "is_temp_or_dev": int(
            any(t in fp for t in [
                "/tmp/", "/test/", "/tests/", "/sample/",
                "/mock/", "/sandbox/", "/debug/", "/fixture/",
            ])
        ),

        # 신규: OS 시스템 디바이스/가상 파일시스템 경로
        # ⚠️ 경로 시작 기준으로 매칭 - /home/dev/ 같은 사용자 디렉토리 오매핑 방지
        "is_system_device": int(
            fp.startswith("/dev/")
            or fp.startswith("/proc/")
            or fp.startswith("/sys/")
        ),

        # 신규: pip/npm/conda/RPM 패키지 설치 경로
        "is_package_path": int(
            any(t in fp for t in _PACKAGE_PATH_TOKENS)
        ),

        # 신규: 크론/배치 로그 경로
        "has_cron_path": int(
            any(t in fp for t in _CRON_PATH_TOKENS)
        ),

        "has_date_in_path": int(bool(re.search(r"/\d{8}/", fp))),

        "has_business_token": int(
            any(t in fp for t in _BUSINESS_TOKENS)
        ),

        "has_system_token": int(
            any(t in fp for t in _SYSTEM_TOKENS)
        ),
    }

    return features
