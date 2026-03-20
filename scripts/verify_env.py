"""환경 검증 스크립트

사용법:
    python scripts/verify_env.py
"""
import sys
import warnings
from pathlib import Path

# Make imports work even if the script is executed from a different cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("PII False Positive Reduction - 환경 검증")
    print("=" * 60)

    print(f"\nPython: {sys.version}")
    print(f"경로:   {sys.executable}\n")

    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("imblearn", "imbalanced-learn"),
        ("joblib", "joblib"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
    ]

    ok_count = 0
    fail_count = 0
    results = []

    # Some binary wheels (notably XGBoost manylinux2014) may emit a FutureWarning about glibc.
    # It's not an import failure; suppress it here to avoid confusing non-ML operators.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"^xgboost(\.|$)")

    for import_name, display_name in packages:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "OK")
            results.append((display_name, ver, True))
            ok_count += 1
        except ImportError as e:
            results.append((display_name, str(e), False))
            fail_count += 1

    # 결과 출력
    print(f"{'패키지':<22s} {'버전':<18s} {'상태'}")
    print("-" * 50)
    for name, ver, ok in results:
        status = "[OK]" if ok else "[X] "
        print(f"  {status} {name:<20s} {ver}")

    print()
    print("=" * 60)
    if fail_count == 0:
        print(f"  전체 {ok_count}개 패키지 정상 확인")
        print("  환경이 정상적으로 구성되었습니다.")
    else:
        print(f"  성공: {ok_count}개 / 실패: {fail_count}개")
        print("  실패한 패키지를 설치하세요.")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
