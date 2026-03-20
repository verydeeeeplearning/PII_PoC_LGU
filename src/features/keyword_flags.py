"""
Stage S2-4: Keyword Flags - Architecture.md §6 (원칙 E-5)

25+개 has_* 바이너리 플래그 생성.
설정 파일(config/feature_config.yaml) 기반 확장 가능.

각 플래그는 텍스트에 특정 키워드/패턴이 존재하면 1, 없으면 0.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 키워드 그룹 정의 (코드 기본값 - config/feature_config.yaml로 오버라이드 가능)
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_GROUPS: dict[str, list[str]] = {
    # ── FP 방향 신호 ───────────────────────────────────────────────────────
    "has_timestamp_kw": [
        "timestamp", "xpirydate", "expiry", "expir", "expire",
        "duration", "created_at", "updated_at", "last_modified",
        "createddate", "modifieddate", "accessdate",
    ],
    "has_byte_kw": [
        "bytes", "byte", "바이트", " kb", " mb", " gb", "kbytes",
        "mbytes", "filesize", "file size",
    ],
    "has_code_kw": [
        "version", "ver.", "v1.", "v2.", "serial", "id=", "code=",
        "revision", "build", "release", "patch",
    ],
    "has_domain_kw": [
        "lguplus", "lgup", "bdp.", "devnet", "lgcns", "lge.lgt",
        "lgelectronics",
    ],
    "has_os_copyright_kw": [
        "redhat", "fedora", "gnu", "apache", "debian", "ubuntu",
        "centos", "oracle", "microsoft", "cisco", "ibm", "paloalto",
        "copyright", "open source",
    ],
    "has_dev_kw": [
        "test", "sample", "dummy", "example", "mock", "stub", "fixture",
        "테스트", "더미", "예시", "sandbox", "demo", "fake",
    ],
    "has_json_structure_kw": [
        '{"', '"email":', '"username":', '"password":', '"user":', '"name":',
        "json", "xml", "yaml",
    ],
    "has_license_kw": [
        "license", "copyright", "redistribution", "warranty",
        "the software is provided", "all rights reserved",
    ],
    "has_kerberos_kw": [
        "kerberos", "krb", "kdc", "principal", "kinit", "kerberoast",
        "ticket", "realm",
    ],
    "has_hadoop_kw": [
        "hadoop", "hdfs", "mapreduce", "hive", "spark", "yarn",
        "datanode", "namenode", "hbase", "zookeeper",
    ],
    "has_docker_kw": [
        "docker", "container", "overlay", "layer", "image", "dockerfile",
        "k8s", "kubernetes", "pod",
    ],
    "has_config_kw": [
        "config", "conf", "cfg", "setting", "properties", "ini",
        "environment", "env", "profile",
    ],
    "has_path_traversal_kw": [
        "../", "..", "%2e%2e", "etc/passwd", "etc/shadow",
    ],
    "has_encryption_kw": [
        "aes", "rsa", "sha", "md5", "encrypt", "decrypt", "cipher",
        "ssl", "tls", "certificate",
    ],
    "has_db_kw": [
        "select ", "insert ", "update ", "delete ", "from ", "where ",
        "mysql", "postgresql", "oracle", "jdbc",
    ],
    "has_internal_domain_exact": [
        "@lguplus.co.kr", "@bdp.lguplus.co.kr", "@lgcns.com",
        "@lge.lgt.co.kr", "@lgupluscns.com",
    ],
    "has_os_domain_exact": [
        "@redhat.com", "@fedoraproject.org", "@apache.org", "@gnu.org",
        "@ubuntu.com", "@debian.org", "@centos.org", "@kernel.org",
    ],
    "has_dummy_domain_exact": [
        "@example.com", "@example.org", "@test.com", "@localhost",
    ],
    "has_ip_pattern_kw": [
        "10.0.", "192.168.", "172.16.", "127.0.0.1",
    ],
    "has_log_entry_kw": [
        " info ", " warn ", " error ", " debug ", " trace ",
        "exception", "stacktrace", "at com.", "at org.",
    ],
    "has_backup_kw": [
        "backup", "bak", ".bak", "archive", "old", "restore",
    ],
    "has_report_kw": [
        "report", "export", "download", "csv", "xlsx", "xls",
    ],
    "has_personal_kw": [
        "customer", "고객", "회원", "사용자", "user", "member",
        "crm", "contact", "personal",
    ],
    "has_auth_kw": [
        "password", "passwd", "pwd", "credential", "token", "api_key",
        "apikey", "secret", "auth",
    ],
    "has_system_proc_kw": [
        "proc", "kernel", "daemon", "service", "systemd", "cron",
        "init", "pid", "cpu", "memory",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────────────────────

def compute_keyword_flags(
    texts: pd.Series,
    keyword_groups: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    텍스트 Series에서 키워드 플래그 DataFrame 생성.

    Parameters
    ----------
    texts:          입력 텍스트 (full_context 또는 raw_text)
    keyword_groups: {플래그명: [키워드, ...]} (기본값: KEYWORD_GROUPS)

    Returns
    -------
    각 플래그가 컬럼인 0/1 DataFrame
    """
    if keyword_groups is None:
        keyword_groups = KEYWORD_GROUPS

    lower_texts = texts.fillna("").str.lower()
    flags: dict[str, pd.Series] = {}

    for flag_name, keywords in keyword_groups.items():
        pattern = "|".join(re.escape(kw.lower()) for kw in keywords)
        flags[flag_name] = lower_texts.str.contains(pattern, regex=True).astype(int)

    return pd.DataFrame(flags, index=texts.index)
