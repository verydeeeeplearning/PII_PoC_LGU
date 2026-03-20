"""테스트 실행 스크립트

사용법:
    python scripts/run_tests.py              # 단위 테스트만 (E2E 제외)
    python scripts/run_tests.py --e2e        # E2E 테스트만
    python scripts/run_tests.py --all        # 전체 (단위 + E2E)
    python scripts/run_tests.py --filter     # 필터 테스트만
    python scripts/run_tests.py --validator  # 검증 테스트만
    python scripts/run_tests.py --merger     # 병합 테스트만
    python scripts/run_tests.py --coverage   # 커버리지 포함
"""
import sys
import argparse
from pathlib import Path

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="테스트 실행")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--filter",
        action="store_true",
        help="필터 테스트만 실행",
    )
    group.add_argument(
        "--validator",
        action="store_true",
        help="검증 테스트만 실행",
    )
    group.add_argument(
        "--merger",
        action="store_true",
        help="병합 테스트만 실행",
    )
    group.add_argument(
        "--e2e",
        action="store_true",
        help="E2E 파이프라인 테스트만 실행",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="전체 테스트 실행 (단위 + E2E)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="커버리지 리포트 생성",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 출력",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # pytest가 설치되어 있는지 확인
    try:
        import pytest
    except ImportError:
        print("[오류] pytest가 설치되어 있지 않습니다.")
        print("  pip install pytest")
        return 1

    # 테스트 경로 설정
    test_dir = PROJECT_ROOT / "tests"

    if args.filter:
        test_path = str(test_dir / "test_filters.py")
        marker_expr = None
    elif args.validator:
        test_path = str(test_dir / "test_validator.py")
        marker_expr = None
    elif args.merger:
        test_path = str(test_dir / "test_merger.py")
        marker_expr = None
    elif args.e2e:
        test_path = str(test_dir)
        marker_expr = "e2e"
    elif args.all:
        test_path = str(test_dir)
        marker_expr = None
    else:
        # 기본: 단위 테스트만 (E2E 제외)
        test_path = str(test_dir)
        marker_expr = "not e2e"

    # pytest 인자 구성
    pytest_args = [test_path, "-v", "--tb=short"]

    # 마커 필터
    if marker_expr:
        pytest_args.extend(["-m", marker_expr])

    # 커버리지 옵션
    if args.coverage:
        try:
            import pytest_cov  # noqa: F401
            pytest_args.extend([
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:outputs/coverage",
            ])
        except ImportError:
            print("[경고] pytest-cov가 설치되어 있지 않습니다.")
            print("  pip install pytest-cov")

    # 실행 모드 표시
    if args.e2e:
        mode = "E2E 파이프라인 테스트"
    elif args.all:
        mode = "전체 테스트 (단위 + E2E)"
    elif args.filter:
        mode = "필터 단위 테스트"
    elif args.validator:
        mode = "검증 단위 테스트"
    elif args.merger:
        mode = "병합 단위 테스트"
    else:
        mode = "단위 테스트 (E2E 제외)"

    print("=" * 60)
    print(f"테스트 실행: {mode}")
    print("=" * 60)
    print(f"  테스트 경로: {test_path}")
    print(f"  pytest 인자: {pytest_args}")
    print()

    # 테스트 실행
    result = pytest.main(pytest_args)

    return result


if __name__ == "__main__":
    sys.exit(main())
