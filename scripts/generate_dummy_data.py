"""더미 데이터 생성 스크립트

전체 ML 파이프라인의 End-to-End 동작 검증을 위한 더미 데이터를 생성합니다.

사용법:
    python scripts/generate_dummy_data.py
    python scripts/generate_dummy_data.py --samples-per-class 500

출력:
    data/processed/merged_cleaned.csv
"""
import sys
from pathlib import Path

# Make imports work even if the script is executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import random
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from src.utils.constants import (
    RANDOM_SEED, PROCESSED_DATA_DIR, MERGED_CLEANED_FILE,
    TEXT_COLUMN, LABEL_COLUMN,
    LABEL_TP,
    LABEL_FP_NUMERIC_CODE,
    LABEL_FP_DUMMY_DATA,
    LABEL_FP_TIMESTAMP,
    LABEL_FP_INTERNAL_DOMAIN,
    LABEL_FP_BYTES,
    LABEL_FP_OS_COPYRIGHT,
    LABEL_FP_CONTEXT,
)

# ── 이름/숫자 풀 ──
NAMES_KR = ["홍길동", "김철수", "이영희", "박민수", "최지영", "정수현", "강다은", "윤서준",
             "조미래", "한소희", "임도윤", "오지호", "서하늘", "배수지", "송민호"]
NAMES_EN = ["John", "Alice", "Bob", "Emma", "David", "Sarah", "Michael", "Lisa"]
DOMAINS = ["company.com", "corp.co.kr", "firm.net", "org.kr", "biz.co.kr"]
EXTENSIONS_CODE = [".py", ".js", ".java", ".cpp", ".ts", ".go", ".cs"]
EXTENSIONS_DOC = [".md", ".docx", ".pdf", ".txt", ".hwp"]


def _rand_phone() -> str:
    return f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"


def _rand_ssn() -> str:
    y = random.randint(70, 99)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y:02d}{m:02d}{d:02d}-{random.randint(1,4)}{random.randint(100000,999999)}"


def _rand_email(name: str = None) -> str:
    if name is None:
        name = random.choice(NAMES_EN).lower()
    return f"{name.lower()}{random.randint(1,99)}@{random.choice(DOMAINS)}"


def _rand_biz_no() -> str:
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10000,99999)}"


def _rand_corp_no() -> str:
    return f"{random.randint(100000,999999)}-{random.randint(1000000,9999999)}"


def _rand_date_str() -> str:
    y = random.randint(2020, 2025)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}{m:02d}{d:02d}"


# ── 클래스별 텍스트 템플릿 ──

def _gen_tp(n: int) -> List[Tuple[str, str]]:
    """TP-실제개인정보: 실제 PII 패턴을 포함하는 텍스트"""
    templates = [
        lambda: f"고객 {random.choice(NAMES_KR)}의 연락처는 {_rand_phone()} 입니다",
        lambda: f"주민등록번호: {_rand_ssn()} 확인 바랍니다",
        lambda: f"이메일 주소: {_rand_email()} 으로 발송해주세요",
        lambda: f"{random.choice(NAMES_KR)} 님의 휴대폰 번호 {_rand_phone()} 로 연락 부탁드립니다",
        lambda: f"성명: {random.choice(NAMES_KR)} 주민번호: {_rand_ssn()}",
        lambda: f"담당자 {random.choice(NAMES_KR)} ({_rand_email()}) 에게 문의하세요",
        lambda: f"환자 {random.choice(NAMES_KR)} 생년월일 {random.randint(1960,2000)}.{random.randint(1,12):02d}.{random.randint(1,28):02d}",
        lambda: f"계약자 정보 - 이름: {random.choice(NAMES_KR)}, 전화: {_rand_phone()}, SSN: {_rand_ssn()}",
        lambda: f"배송지 정보: {random.choice(NAMES_KR)} {_rand_phone()} 서울시 강남구",
        lambda: f"직원 {random.choice(NAMES_KR)} 사번 {random.randint(10000,99999)} 연락처 {_rand_phone()}",
        lambda: f"수신인: {random.choice(NAMES_KR)} 핸드폰: {_rand_phone()}",
        lambda: f"고객 ID {random.randint(1000,9999)} {random.choice(NAMES_KR)} 주민번호 {_rand_ssn()} 등록",
        lambda: f"회원가입 정보 이름={random.choice(NAMES_KR)} email={_rand_email()} phone={_rand_phone()}",
        lambda: f"보험 청구서 피보험자 {random.choice(NAMES_KR)} 주민등록번호 {_rand_ssn()}",
        lambda: f"{random.choice(NAMES_KR)} 고객님 계좌 이체 완료 ({_rand_phone()} 인증)",
    ]
    paths = [
        "/data/customers/records.csv",
        "/home/user/documents/contacts.xlsx",
        "/shared/hr/employee_data.csv",
        "/data/crm/client_info.csv",
        "/backup/db/user_table.csv",
        "/data/insurance/claims.xlsx",
        "/shared/sales/customer_list.csv",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


def _gen_fp_test(n: int) -> List[Tuple[str, str]]:
    """FP-테스트데이터"""
    templates = [
        lambda: f"test case: {_rand_phone()} 검증용 데이터",
        lambda: f"테스트 계정 test_user@test.com 입니다",
        lambda: f"testing phone number 000-0000-0000 for validation",
        lambda: f"assert validate_ssn('{_rand_ssn()}') == True  # test",
        lambda: f"테스트 환경 더미 전화번호: 010-0000-{random.randint(1000,9999)}",
        lambda: f"unittest: expected email = tester{random.randint(1,99)}@test.org",
        lambda: f"QA test data - name: 테스트유저 phone: {_rand_phone()}",
        lambda: f"integration test: POST /api/user body={{phone: '{_rand_phone()}'}}",
        lambda: f"test fixture: ssn={_rand_ssn()} (not real)",
        lambda: f"테스트 데이터 생성: 이름=테스트{random.randint(1,99)} 연락처=010-1111-{random.randint(1000,9999)}",
        lambda: f"load test mock data: email=loadtest@testing.com phone=000-0000-0000",
        lambda: f"test_validate_phone: input={_rand_phone()} expected=True",
        lambda: f"regression test 전화번호 형식 체크: {_rand_phone()}",
        lambda: f"테스트 시나리오 #{random.randint(1,100)}: 사용자 정보 입력 검증 010-0000-0000",
    ]
    paths = [
        "/test/unit/test_validator.py",
        "/tests/test_data.csv",
        "/src/test/resources/mock_users.json",
        "/test/integration/test_api.py",
        "/tests/fixtures/dummy_data.csv",
        f"/test/test_case_{random.randint(1,50)}.py",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


def _gen_fp_sample(n: int) -> List[Tuple[str, str]]:
    """FP-샘플데이터"""
    templates = [
        lambda: f"sample data: {random.choice(NAMES_KR)} {_rand_phone()}",
        lambda: f"예시 이메일 example@sample.org 을 입력하세요",
        lambda: f"예제 데이터셋에 포함된 번호 {_rand_phone()}",
        lambda: f"SAMPLE_USER = {{'name': '{random.choice(NAMES_KR)}', 'phone': '{_rand_phone()}'}}",
        lambda: f"샘플 주민번호 000000-0000000 (실제 아님)",
        lambda: f"example input: email=user@example.com phone={_rand_phone()}",
        lambda: f"데모 데이터: 홍길동 010-1234-5678 (예시)",
        lambda: f"예시 파일 - sample customer: name=샘플유저 ssn=000000-0000000",
        lambda: f"sample record #{random.randint(1,100)}: {random.choice(NAMES_KR)} {_rand_email()}",
        lambda: f"교육용 예제 데이터 이름={random.choice(NAMES_KR)} 전화={_rand_phone()}",
        lambda: f"샘플 CSV 행: id,name,phone\\n1,예시사용자,010-0000-0000",
        lambda: f"example API response: {{name: '{random.choice(NAMES_EN)}', email: 'sample@example.com'}}",
    ]
    paths = [
        "/sample/data/users.csv",
        "/examples/demo_data.csv",
        "/demo/sample_input.json",
        "/data/sample/example_records.xlsx",
        "/resources/examples/mock_data.csv",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


def _gen_fp_code(n: int) -> List[Tuple[str, str]]:
    """FP-개발코드"""
    templates = [
        lambda: f"def validate_phone(num='010-0000-0000'):  # dev mock data",
        lambda: f"const DUMMY_SSN = '000000-0000000';  // placeholder",
        lambda: f"debug: stub email = dummy@dev.local for development",
        lambda: f"mock_user = User(name='dev_user', phone='{_rand_phone()}')",
        lambda: f"# TODO: replace dummy data 010-0000-0000 with real input",
        lambda: f"String REGEX_PHONE = \"010-\\\\d{{4}}-\\\\d{{4}}\";  // 전화번호 정규식",
        lambda: f"logger.debug(f'parsed phone: {{phone}}')  # dev only",
        lambda: f"PHONE_PATTERN = re.compile(r'\\d{{3}}-\\d{{4}}-\\d{{4}}')",
        lambda: f"stub_data = {{'ssn': '000000-0000000', 'name': 'stub_user'}}",
        lambda: f"// development 환경 설정: mock email = dev{random.randint(1,9)}@localhost",
        lambda: f"func mockSSN() string {{ return \"{_rand_ssn()}\" }} // dev dummy",
        lambda: f"private static final String DEV_PHONE = \"010-0000-0000\";",
        lambda: f"var debugEmail = \"debug@dev.internal\";  // 개발용 더미",
    ]
    paths = [
        f"/src/main/validators/phone{random.choice(EXTENSIONS_CODE)}",
        f"/src/utils/regex{random.choice(EXTENSIONS_CODE)}",
        f"/lib/helpers/mock{random.choice(EXTENSIONS_CODE)}",
        f"/dev/scripts/setup{random.choice(EXTENSIONS_CODE)}",
        f"/src/services/user_service{random.choice(EXTENSIONS_CODE)}",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


def _gen_fp_doc(n: int) -> List[Tuple[str, str]]:
    """FP-문서예시"""
    templates = [
        lambda: f"매뉴얼: 전화번호 형식은 010-XXXX-XXXX 입니다",
        lambda: f"guide: 이메일 입력 예시 user@example.com",
        lambda: f"사용자 가이드 문서 - 주민번호 입력 방법: YYMMDD-NNNNNNN",
        lambda: f"입력 가이드: 연락처 필드에 010-0000-0000 형태로 입력",
        lambda: f"[매뉴얼] 개인정보 마스킹 처리 예시: 홍** 010-****-5678",
        lambda: f"API 문서: phone 필드는 010-XXXX-XXXX 형식 (string)",
        lambda: f"설치 가이드: 관리자 이메일을 admin@your-domain.com 으로 설정",
        lambda: f"FAQ: Q. 주민번호 형식이 어떻게 되나요? A. YYMMDD-NNNNNNN",
        lambda: f"문서 템플릿 - 고객명: [이름], 연락처: [전화번호]",
        lambda: f"운영 매뉴얼 제3장 - 개인정보 탐지 패턴: 전화번호, 이메일, 주민번호",
        lambda: f"가이드라인: 이메일 형식 검증 규칙 - user@domain.com 형태",
        lambda: f"사용 설명서 p.{random.randint(1,100)}: 전화번호 입력란에 010-XXXX-XXXX 입력",
    ]
    paths = [
        f"/docs/user_guide{random.choice(EXTENSIONS_DOC)}",
        f"/manual/admin_guide{random.choice(EXTENSIONS_DOC)}",
        f"/docs/api/reference{random.choice(EXTENSIONS_DOC)}",
        f"/docs/faq{random.choice(EXTENSIONS_DOC)}",
        f"/manual/operations{random.choice(EXTENSIONS_DOC)}",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


def _gen_fp_system(n: int) -> List[Tuple[str, str]]:
    """FP-시스템생성"""
    templates = [
        lambda: f"auto-generated ID: SYS-{_rand_date_str()}-{random.randint(100000,999999)}",
        lambda: f"시스템 자동 생성 로그: session={random.randint(1000000000,9999999999)}",
        lambda: f"generated timestamp: {_rand_date_str()}{random.randint(100000,235959)}",
        lambda: f"[AUTO] batch job #{random.randint(1,999)} completed, process_id={random.randint(10000,99999)}",
        lambda: f"system log: auto cleanup task ref={random.randint(100000000,999999999)}",
        lambda: f"자동 생성 보고서 ID: RPT-{_rand_date_str()}-{random.randint(1000,9999)}",
        lambda: f"cron job output: processed {random.randint(100,9999)} records, txn={random.randint(10000000,99999999)}",
        lambda: f"[SYSTEM] auto-backup completed, archive_id={random.randint(100000,999999)}",
        lambda: f"generated config: node_id={random.randint(1000,9999)} cluster={random.randint(1,10)}",
        lambda: f"시스템 알림: 자동 스캔 완료 파일수={random.randint(100,10000)} 시간={_rand_date_str()}",
        lambda: f"auto notification: event_id=EVT{random.randint(100000,999999)} generated by system",
    ]
    paths = [
        "/var/log/system/audit.log",
        "/var/log/batch/daily.log",
        "/system/generated/reports/auto_report.csv",
        "/logs/cron/weekly.log",
        "/system/auto/backup.log",
        "/var/log/app/auto_scan.log",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


def _gen_fp_corp(n: int) -> List[Tuple[str, str]]:
    """FP-법인/사업자"""
    corp_names = ["(주)한국상사", "(주)테크솔루션", "삼성전자(주)", "(주)미래에너지",
                  "한국통신(주)", "(주)글로벌트레이드", "대한건설(주)", "(주)서울식품"]
    templates = [
        lambda: f"사업자등록번호: {_rand_biz_no()} {random.choice(corp_names)}",
        lambda: f"법인등록번호 {_rand_corp_no()} 확인",
        lambda: f"거래처: {random.choice(corp_names)} 사업자번호 {_rand_biz_no()}",
        lambda: f"세금계산서 발행 - 공급자: {random.choice(corp_names)} ({_rand_biz_no()})",
        lambda: f"법인번호 {_rand_corp_no()} 법인명: {random.choice(corp_names)}",
        lambda: f"계약서 상대방: {random.choice(corp_names)} 사업자 {_rand_biz_no()} 대표 {random.choice(NAMES_KR)}",
        lambda: f"매출처 {random.choice(corp_names)} 사업자등록번호 {_rand_biz_no()} 매출액 {random.randint(1,999)}억",
        lambda: f"공급받는자: {random.choice(corp_names)} ({_rand_biz_no()}) 품목: 소프트웨어",
        lambda: f"법인카드 사용내역 - {random.choice(corp_names)} 카드번호 {random.randint(1000,9999)}-****-****-{random.randint(1000,9999)}",
        lambda: f"사업자 정보: 상호={random.choice(corp_names)} 번호={_rand_biz_no()} 업태=서비스",
    ]
    paths = [
        "/accounting/invoices/2024.csv",
        "/finance/tax/business_registry.xlsx",
        "/biz/contracts/vendor_list.csv",
        "/accounting/reports/quarterly.csv",
        "/finance/ap/supplier_info.xlsx",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


def _gen_fp_other(n: int) -> List[Tuple[str, str]]:
    """FP-기타"""
    templates = [
        lambda: f"주문번호 {_rand_date_str()[:4]}-{random.randint(1000,9999)}-{random.randint(1000,9999)} 배송완료",
        lambda: f"제품코드 AB-{random.randint(1000000,9999999)}-CD 재고 확인",
        lambda: f"IP 주소 192.168.{random.randint(0,255)}.{random.randint(1,254)} 에서 접속",
        lambda: f"송장번호 {random.randint(100000000000,999999999999)} CJ대한통운",
        lambda: f"바코드 {random.randint(1000000000000,9999999999999)} 스캔 완료",
        lambda: f"내선번호 {random.randint(100,999)}-{random.randint(1000,9999)} 연결",
        lambda: f"차량번호 {random.randint(10,99)}가{random.randint(1000,9999)} 주차등록",
        lambda: f"시리얼넘버 SN-{random.randint(100000,999999)}-{random.choice(['A','B','C','D'])}{random.randint(10,99)}",
        lambda: f"우편번호 {random.randint(10000,99999)} 서울시 중구",
        lambda: f"결제번호 PAY-{_rand_date_str()}-{random.randint(10000,99999)} 승인완료",
        lambda: f"MAC 주소 {':'.join(f'{random.randint(0,255):02X}' for _ in range(6))} 등록",
        lambda: f"회의실 예약 #{random.randint(1000,9999)} 참석자 {random.randint(3,15)}명",
    ]
    paths = [
        "/misc/orders/shipping_log.csv",
        "/archive/inventory/stock_2024.xlsx",
        "/data/network/access_log.csv",
        "/shared/logistics/tracking.csv",
        "/misc/facilities/parking.csv",
        "/archive/payments/transactions.csv",
    ]
    results = []
    for _ in range(n):
        text = random.choice(templates)()
        fp = random.choice(paths)
        results.append((text, fp))
    return results


# ── 클래스 ↔ 생성 함수 매핑 ──
CLASS_GENERATORS: Dict[str, callable] = {
    LABEL_TP:                _gen_tp,
    LABEL_FP_NUMERIC_CODE:   _gen_fp_test,
    LABEL_FP_DUMMY_DATA:     _gen_fp_sample,
    LABEL_FP_TIMESTAMP:      _gen_fp_code,
    LABEL_FP_INTERNAL_DOMAIN: _gen_fp_doc,
    LABEL_FP_BYTES:          _gen_fp_system,
    LABEL_FP_OS_COPYRIGHT:   _gen_fp_corp,
    LABEL_FP_CONTEXT:        _gen_fp_other,
}


def generate_dummy_data(samples_per_class: int = 200) -> pd.DataFrame:
    """더미 데이터를 생성합니다.

    Args:
        samples_per_class: 클래스당 생성 건수

    Returns:
        생성된 DataFrame
    """
    rows = []
    idx = 0

    for label, gen_func in CLASS_GENERATORS.items():
        data_pairs = gen_func(samples_per_class)
        for text, file_path in data_pairs:
            rows.append({
                "detection_id": f"DET-{idx:06d}",
                TEXT_COLUMN: text,
                LABEL_COLUMN: label,
                "file_path": file_path,
            })
            idx += 1

    df = pd.DataFrame(rows)
    # 셔플
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="더미 데이터 생성")
    parser.add_argument(
        "--samples-per-class", type=int, default=200,
        help="클래스당 생성 건수 (기본: 200)",
    )
    args = parser.parse_args()

    # 시드 고정
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("[Dummy Data Generator]")
    print("=" * 60)
    print(f"  클래스 수: {len(CLASS_GENERATORS)}")
    print(f"  클래스당 건수: {args.samples_per_class}")
    print(f"  총 건수: {len(CLASS_GENERATORS) * args.samples_per_class}")

    # 생성
    df = generate_dummy_data(args.samples_per_class)

    # 출력 디렉토리 생성
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 저장
    output_path = PROCESSED_DATA_DIR / MERGED_CLEANED_FILE
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n[저장 완료] {output_path}")
    print(f"  행: {len(df):,}  |  열: {len(df.columns)}")
    print(f"\n[레이블 분포]")
    print(df[LABEL_COLUMN].value_counts().to_string())
    print(f"\n[컬럼] {list(df.columns)}")
    print(f"\n[샘플 데이터 (상위 3건)]")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
