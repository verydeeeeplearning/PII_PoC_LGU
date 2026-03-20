"""레이블 Excel + Sumologic Excel 원본형 mock 데이터 생성.

문서/이미지 기준으로 실제 수집 포맷과 유사한 샘플을 생성한다.
실 데이터는 레이블/Sumologic 모두 Excel(.xlsx) 형식이며, CSV도 로더에서 허용됨.

출력:
  - data/raw/label/25년 정탐 (3월~12월)/*/*.xlsx
  - data/raw/label/25년 오탐 (3월~12월)/*/*.xlsx
  - data/raw/dataset_a/sumologic_mock_202506_202507.xlsx  (기본, Excel)
  - data/raw/dataset_a/sumologic_mock_202506_202507.csv   (--csv 옵션 시)

기본값:
  - label 200행
  - sumologic 200행

사용 예:
  python scripts/generate_mock_raw_data.py
  python scripts/generate_mock_raw_data.py --label-rows 200 --sumologic-rows 200
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RANDOM_SEED = 42

LABEL_ROOT = PROJECT_ROOT / "data" / "raw" / "label"
SUMOLOGIC_OUTPUT_XLSX = PROJECT_ROOT / "data" / "raw" / "dataset_a" / "sumologic_mock_202506_202507.xlsx"
SUMOLOGIC_OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "dataset_a" / "sumologic_mock_202506_202507.csv"

TP_FOLDER = "25년 정탐 (3월~12월)"
FP_FOLDER = "25년 오탐 (3월~12월)"

LABEL_GROUPS = [
    ("TP", "6월", "CTO"),
    ("TP", "6월", "NW부문"),
    ("TP", "7월", "CTO"),
    ("TP", "7월", "NW부문"),
    ("FP", "6월", "CTO"),
    ("FP", "6월", "NW부문"),
    ("FP", "7월", "CTO"),
    ("FP", "7월", "NW부문"),
]

OPS_DEPT_MAP = {
    "CTO": ["AX선행기술팀", "AI플랫폼운영팀", "클라우드보안팀"],
    "NW부문": ["NW운영1팀", "전송망품질팀", "서비스플랫폼운영팀"],
}

SERVICE_MAP = {
    "CTO": ["비디오 포털(신동향)검색", "모바일인증", "AI메시징허브"],
    "NW부문": ["NW관제", "문자중계", "MMS 운영"],
}

ORG_DEPTNAME_MAP = {
    "CTO": "CTO사업부",
    "NW부문": "NW운영부",
}

NAMES = [
    "홍길동", "김민수", "이서연", "박지훈", "최유진", "정현우", "오하늘", "한지민",
    "강민지", "서도윤", "조수빈", "윤태호", "임수아", "배지호", "송다은",
]

EMAIL_IDS = ["kim.ms", "park.js", "lee.sw", "choi.yj", "jung.hw", "han.jm", "oh.hn"]
INTERNAL_DOMAINS = ["lguplus.co.kr", "bdp.lguplus.co.kr", "map.lguplus.co.kr"]
OSS_DOMAINS = ["redhat.com", "apache.org", "python.org", "openssl.org", "haxx.se"]

HEADER_VARIANTS = [
    {
        "서버이름": "서버 이름",
        "에이전트IP": "에이전트 IP",
        "패턴개수": "패턴 개수",
        "파일경로": "파일 경로",
        "파일이름": "파일 이름",
        "주민등록번호개수": "주민 등록 번호 개수",
        "핸드폰번호개수": "핸드폰 번호 개수",
        "이메일주소개수": "E-Mail 주소 개수",
        "파일생성일시": "파일 생성 일시",
        "파일크기": "파일 크기",
    },
    {
        "서버이름": "서버이름",
        "에이전트IP": "에이전트IP",
        "패턴개수": "패턴개수",
        "파일경로": "파일경로",
        "파일이름": "파일이름",
        "주민등록번호개수": "주민등록번호개수",
        "핸드폰번호개수": "핸드폰번호개수",
        "이메일주소개수": "이메일주소개수",
        "파일생성일시": "파일생성일시",
        "파일크기": "파일크기",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원본형 mock 데이터 생성")
    parser.add_argument("--label-rows", type=int, default=200, help="생성할 label 총 행 수")
    parser.add_argument("--sumologic-rows", type=int, default=200, help="생성할 sumologic 총 행 수")
    parser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="Sumologic mock을 CSV로도 출력 (기본: Excel .xlsx만 출력)",
    )
    return parser.parse_args()


def _seed() -> None:
    random.seed(RANDOM_SEED)


def _masked_phone() -> str:
    return f"010-****-{random.randint(1000, 9999)}"


def _phone_plain() -> str:
    return f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"


def _masked_ssn() -> str:
    year = random.randint(70, 99)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year:02d}{month:02d}{day:02d}-{random.randint(1,4)}*****"


def _masked_email(internal: bool = False, oss: bool = False) -> str:
    user = random.choice(EMAIL_IDS)
    if internal:
        domain = random.choice(INTERNAL_DOMAINS)
    elif oss:
        domain = random.choice(OSS_DOMAINS)
    else:
        domain = random.choice(["gmail.com", "naver.com", "partner.co.kr"])
    return f"{user}***@{domain}"


def _format_label_datetime(dt: datetime, variant_idx: int) -> str:
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%y.%m.%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
    ]
    return dt.strftime(formats[variant_idx % len(formats)])


def _canonical_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _work_month_base(month: str) -> datetime:
    if month == "6월":
        return datetime(2025, 6, 15, 9, 0, 0)
    return datetime(2025, 7, 15, 9, 0, 0)


def _make_server_name(org: str, label_raw: str, index: int) -> str:
    env = "prd" if label_raw == "TP" else random.choice(["dev", "stg", "sbx"])
    role = "app" if org == "CTO" else "mms"
    return f"lgup-{env}-{role}{(index % 7) + 1:02d}"


def _make_agent_ip(org: str, label_raw: str, index: int) -> str:
    if org == "CTO":
        base = "10.100.1" if label_raw == "TP" else "10.200.2"
    else:
        base = "10.50.3" if label_raw == "TP" else "172.21.56"
    return f"{base}.{10 + (index % 40)}"


def _tp_payload(index: int, org: str) -> dict[str, object]:
    person = random.choice(NAMES)
    scenario = index % 4

    if scenario == 0:
        file_path = random.choice(["/crm/customer/export", "/data/customer/master", "/shared/sales/export"])
        file_name = f"customer_export_{20250601 + index:08d}.xlsx"
        ssn_count = random.randint(15, 120)
        phone_count = random.randint(10, 90)
        email_count = random.randint(3, 25)
        masked = _masked_ssn()
        context = f"{person} {masked} {_masked_phone()} 서울시 강남구 고객정보"
        pattern_name = "주민등록 번호"
        fp_desc = "삭제"
    elif scenario == 1:
        file_path = random.choice(["/hr/data/export", "/payroll/archive", "/partner/share/employee"])
        file_name = f"employee_roster_{20250601 + index:08d}.csv"
        ssn_count = random.randint(5, 60)
        phone_count = random.randint(5, 40)
        email_count = random.randint(15, 80)
        masked = _masked_email()
        context = f"{person} 부장 {masked} {_masked_phone()} 인사팀"
        pattern_name = "E-Mail"
        fp_desc = "실제 임직원 개인정보 포함으로 삭제 조치"
    elif scenario == 2:
        file_path = random.choice(["/insurance/claim/intake", "/billing/customer/input", "/erp/member/upload"])
        file_name = f"claim_intake_{20250601 + index:08d}.txt"
        ssn_count = random.randint(30, 180)
        phone_count = random.randint(5, 25)
        email_count = random.randint(0, 10)
        masked = _masked_ssn()
        context = f"보험청구 대상자 {person} 주민번호 {masked} 확인 요청"
        pattern_name = "주민등록 번호"
        fp_desc = "실제 고객 정보 포함"
    else:
        file_path = random.choice(["/mail/customer/queue", "/cs/voice-mail/archive", "/partner/contact/export"])
        file_name = f"contact_list_{20250601 + index:08d}.mbox"
        ssn_count = random.randint(0, 5)
        phone_count = random.randint(25, 120)
        email_count = random.randint(25, 120)
        masked = _masked_phone()
        context = f"담당자 {person} 연락처 {masked} 이메일 {_masked_email()} 발송 대상"
        pattern_name = "핸드폰 번호"
        fp_desc = "삭제"

    pattern_count = ssn_count + phone_count + email_count
    return {
        "file_path": file_path,
        "file_name": file_name,
        "pattern_count": pattern_count,
        "ssn_count": ssn_count,
        "phone_count": phone_count,
        "email_count": email_count,
        "masked_content": masked,
        "context": context,
        "pattern_name": pattern_name,
        "fp_description": fp_desc,
        "exception_requested": "N",
        "retention_period": random.choice(["즉시 삭제", "3개월", "6개월"]),
        "file_size": random.randint(40_000, 4_000_000),
        "deptname": ORG_DEPTNAME_MAP[org],
    }


def _fp_payload(index: int, org: str) -> dict[str, object]:
    scenario = index % 7

    if scenario == 0:
        file_path = "/var/log/apache2"
        file_name = f"access.log.{(index % 9) + 1}"
        ssn_count, phone_count, email_count = random.randint(500, 5000), 0, 0
        masked = _masked_ssn()
        context = f"uid=33(www-data) {masked} gid=33 www apache access log"
        pattern_name = "주민등록 번호"
        fp_desc = "로그 파일 경로 및 시스템 UID 문맥으로 오탐"
    elif scenario == 1:
        file_path = random.choice([
            "/var/lib/docker/overlay2/a3f1b2c4d5/diff/var/log",
            "/docker/overlay2/merged/logs",
        ])
        file_name = random.choice(["app.log", "container.log", "nginx.log"])
        ssn_count, phone_count, email_count = random.randint(1000, 8000), 0, 0
        masked = _masked_ssn()
        context = f"container_id=a3f1b restart=3 {masked} overlay docker"
        pattern_name = "주민등록 번호"
        fp_desc = "docker overlay 로그 맥락으로 오탐"
    elif scenario == 2:
        file_path = random.choice(["/usr/share/doc/libssl1.1", "/usr/share/licenses/python3", "/usr/share/doc/curl"])
        file_name = random.choice(["copyright", "LICENSE", "changelog.gz"])
        ssn_count, phone_count, email_count = 0, 0, random.randint(1, 8)
        masked = _masked_email(oss=True)
        context = f"Maintainer: {masked} open source license contact"
        pattern_name = "E-Mail"
        fp_desc = "오픈소스 저작권/라이선스 문구"
    elif scenario == 3:
        file_path = random.choice(["/home/dev/workspace/test", "/tmp/test", "/tests/fixtures"])
        file_name = random.choice(["dummy_data.json", "sample_customer.csv", "test_case.sql"])
        ssn_count, phone_count, email_count = random.randint(1, 30), random.randint(1, 15), random.randint(0, 5)
        masked = random.choice([_masked_phone(), _masked_ssn()])
        context = f"test fixture mock data {masked} validation only"
        pattern_name = "핸드폰 번호"
        fp_desc = "테스트/더미 데이터"
    elif scenario == 4:
        file_path = random.choice(["/home/mail/inbox", "/var/app/mail/archive", "/auth/internal"])
        file_name = random.choice(["mailbox.db", "sent_archive.mbox", "auth_audit.txt"])
        ssn_count, phone_count, email_count = 0, 0, random.randint(10, 90)
        masked = _masked_email(internal=True)
        context = f"From: {masked} Team meeting internal notification"
        pattern_name = "E-Mail"
        fp_desc = "내부 도메인 메일 주소"
    elif scenario == 5:
        file_path = random.choice(["/var/log/cron", "/var/log/scheduler", "/system/generated"])
        file_name = random.choice(["cron_daily.log", "scheduler.log.1", "batch_202507.log"])
        ssn_count, phone_count, email_count = random.randint(80, 500), 0, 0
        masked = _masked_ssn()
        context = f"[2025-06-06 14:30:22] {masked} task_id={random.randint(100,9999)} generated timestamp"
        pattern_name = "주민등록 번호"
        fp_desc = "타임스탬프/배치 로그 문맥"
    else:
        file_path = random.choice(["/var/log/disk_monitor", "/storage/system/report", "/logs/iostat"])
        file_name = random.choice(["disk_usage.log", "storage_size.txt", "iostat.log"])
        ssn_count, phone_count, email_count = random.randint(20, 300), 0, 0
        masked = _masked_ssn()
        context = f"/dev/sda1 size=52428800000 used={masked.replace('-', '')} avail bytes"
        pattern_name = "주민등록 번호"
        fp_desc = "bytes/용량 수치가 주민번호 패턴으로 오검출"

    pattern_count = ssn_count + phone_count + email_count
    return {
        "file_path": file_path,
        "file_name": file_name,
        "pattern_count": pattern_count,
        "ssn_count": ssn_count,
        "phone_count": phone_count,
        "email_count": email_count,
        "masked_content": masked,
        "context": context,
        "pattern_name": pattern_name,
        "fp_description": fp_desc,
        "exception_requested": random.choice(["Y", "N"]),
        "retention_period": random.choice(["1개월", "3개월", "예외승인시 보관"]),
        "file_size": random.randint(10_000, 1_500_000),
        "deptname": ORG_DEPTNAME_MAP[org],
    }


def _build_events(total_rows: int) -> list[dict[str, object]]:
    counts = {group: total_rows // len(LABEL_GROUPS) for group in LABEL_GROUPS}
    for group in LABEL_GROUPS[: total_rows % len(LABEL_GROUPS)]:
        counts[group] += 1

    events: list[dict[str, object]] = []
    global_index = 0

    for group_idx, group in enumerate(LABEL_GROUPS):
        label_raw, month, org = group
        for local_idx in range(counts[group]):
            global_index += 1
            month_anchor = _work_month_base(month)
            created_at = month_anchor - timedelta(
                days=random.randint(0, 35),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            scan_time = created_at + timedelta(minutes=random.randint(5, 240))

            server_name = _make_server_name(org, label_raw, global_index)
            agent_ip = _make_agent_ip(org, label_raw, global_index)

            payload = _tp_payload(global_index, org) if label_raw == "TP" else _fp_payload(global_index, org)

            label_row = {
                "조직": org,
                "운영/클렌징 부서": random.choice(OPS_DEPT_MAP[org]),
                "서비스": random.choice(SERVICE_MAP[org]),
                "오탐여부(정탐/오탐)": "정탐" if label_raw == "TP" else "오탐",
                "오탐: 오탐 설명(검토 후 예외처리 목적)정탐: 삭제조치 OR 삭제불가 사유": payload["fp_description"],
                "예외요청": payload["exception_requested"],
                "파일 보관기간": payload["retention_period"],
                "서버이름": server_name,
                "에이전트IP": agent_ip,
                "패턴개수": payload["pattern_count"],
                "파일경로": payload["file_path"],
                "파일이름": payload["file_name"],
                "주민등록번호개수": payload["ssn_count"],
                "핸드폰번호개수": payload["phone_count"],
                "이메일주소개수": payload["email_count"],
                "파일생성일시": _format_label_datetime(created_at, global_index + group_idx),
                "파일크기": payload["file_size"],
            }

            sumologic_row = {
                "_time": scan_time.isoformat(timespec="seconds") + "+0900",
                "_messagetime": _canonical_datetime(scan_time),
                "computername": server_name,
                "agentip": agent_ip,
                "dfile_computername": server_name,
                "dfile_agentip": agent_ip,
                "dfile_userid": agent_ip,
                "dfile_username": server_name.replace("lgup-", ""),
                "dfile_patternname": payload["pattern_name"],
                "dfile_patterncnt": payload["pattern_count"],
                "dfile_inspectcount": payload["pattern_count"],
                "dfile_ssnpatterncnt": payload["ssn_count"],
                "dfile_phonepatterncnt": payload["phone_count"],
                "dfile_emailpatterncnt": payload["email_count"],
                "dfile_inspectmaskedcontent": payload["masked_content"],
                "dfile_inspectcontentwithcontext": payload["context"],
                "dfile_firstscannedpath": f"{payload['file_path']}/{payload['file_name']}",
                "dfile_filepath": payload["file_path"],
                "dfile_filedirectedpath": payload["file_path"],
                "dfile_filename": payload["file_name"],
                "dfile_filecreatedtime": _canonical_datetime(created_at),
                "dfile_deptname": payload["deptname"],
                "dfile_filesize": payload["file_size"],
                "dfile_fileextension": Path(str(payload["file_name"])).suffix.lstrip("."),
                "dfile_filemodifiedtime": _canonical_datetime(created_at + timedelta(hours=random.randint(1, 72))),
                "dfile_durationdays": random.randint(0, 30),
                "snapshot_week": ((scan_time.day - 1) // 7) + 1,
                "mock_label_raw": label_raw,
                "mock_work_month": month,
                "mock_org": org,
            }

            events.append(
                {
                    "group": group,
                    "label_row": label_row,
                    "sumologic_row": sumologic_row,
                }
            )

    return events


def _clear_previous_outputs() -> None:
    if LABEL_ROOT.exists():
        shutil.rmtree(LABEL_ROOT)
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)
    SUMOLOGIC_OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)


def _apply_variant(df: pd.DataFrame, variant_idx: int) -> pd.DataFrame:
    rename_map = HEADER_VARIANTS[variant_idx % len(HEADER_VARIANTS)]
    return df.rename(columns=rename_map)


def _write_label_files(events: list[dict[str, object]]) -> dict[str, int]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        grouped[event["group"]].append(event["label_row"])

    file_row_counts: dict[str, int] = {}
    for idx, group in enumerate(LABEL_GROUPS):
        label_raw, month, org = group
        rows = grouped[group]
        if not rows:
            continue

        folder_name = TP_FOLDER if label_raw == "TP" else FP_FOLDER
        prefix = "정탐" if label_raw == "TP" else "오탐"
        out_dir = LABEL_ROOT / folder_name / month
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{prefix} 취합 보고 자료_{month}_{org}.xlsx"

        df = pd.DataFrame(rows)
        df = _apply_variant(df, idx)

        # 실제 레이블 Excel 구조 재현:
        #   Row 1 (Excel): 타이틀 행 (헤더가 아님 - ingestion_config header_row=1로 스킵)
        #   Row 2 (Excel): 컬럼 헤더 (연번 포함)
        #   Row 3+:        데이터 (연번: 1, 2, 3, ...)
        df_with_serial = df.copy()
        df_with_serial.insert(0, "연번", range(1, len(df) + 1))

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            pd.DataFrame(
                {
                    "설명": [
                        "Mock label workbook for loader testing",
                        f"label={label_raw}, month={month}, org={org}",
                        f"rows={len(df)}",
                    ]
                }
            ).to_excel(writer, sheet_name="Summary", index=False)
            # startrow=1: 헤더를 Excel 2행에 배치, 1행은 빈 타이틀 자리
            df_with_serial.to_excel(
                writer, sheet_name=f"{org}_{prefix}", index=False, startrow=1
            )
            # Excel 1행에 타이틀 기입
            ws = writer.sheets[f"{org}_{prefix}"]
            ws.cell(row=1, column=1, value=f"{prefix} 취합 보고 자료_{month}_{org}")

        file_row_counts[str(out_path.relative_to(PROJECT_ROOT))] = len(df)

    return file_row_counts


def _write_sumologic_file(events: list[dict[str, object]], total_rows: int, also_csv: bool = False) -> Path:
    rows = [event["sumologic_row"] for event in events[:total_rows]]
    df = pd.DataFrame(rows)

    # 기본: Excel (.xlsx) - 실 데이터와 동일한 포맷
    with pd.ExcelWriter(SUMOLOGIC_OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)

    # 옵션: CSV도 함께 출력
    if also_csv:
        df.to_csv(SUMOLOGIC_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    return SUMOLOGIC_OUTPUT_XLSX


def main() -> None:
    args = parse_args()
    if args.label_rows <= 0 or args.sumologic_rows <= 0:
        raise ValueError("label/sumologic row 수는 1 이상이어야 합니다.")

    _seed()
    _clear_previous_outputs()

    total_events = max(args.label_rows, args.sumologic_rows)
    events = _build_events(total_events)

    label_events = events[: args.label_rows]
    sumologic_events = events[: args.sumologic_rows]

    label_files = _write_label_files(label_events)
    sumologic_path = _write_sumologic_file(sumologic_events, args.sumologic_rows, also_csv=args.csv)

    print("[mock data 생성 완료]")
    print(f"  label rows:     {args.label_rows}")
    print(f"  sumologic rows: {args.sumologic_rows}")
    print("  label files:")
    for path, n_rows in label_files.items():
        print(f"    - {path}: {n_rows} rows")
    print(f"  sumologic file: {sumologic_path.relative_to(PROJECT_ROOT)}  (Excel)")
    if args.csv:
        print(f"  sumologic csv:  {SUMOLOGIC_OUTPUT_CSV.relative_to(PROJECT_ROOT)}  (CSV)")


if __name__ == "__main__":
    main()
