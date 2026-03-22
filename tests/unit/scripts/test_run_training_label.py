"""Unit tests for run_training.py --source label path (Phase 3).

Uses mocks to avoid requiring actual data files.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def _make_silver_label_df(n: int = 80) -> pd.DataFrame:
    """최소 silver_label DataFrame 생성 (label_raw 포함)"""
    return pd.DataFrame({
        "pk_event": [f"evt{i:03d}" for i in range(n)],
        "pk_file": [f"pkf{i // 4}" for i in range(n)],
        "file_path": [f"/var/log/app{i % 5}/service.log" for i in range(n)],
        "file_name": [f"service_{i % 10}.log" for i in range(n)],
        "server_name": [f"lgup-{'prd' if i % 2 == 0 else 'dev'}-app{i % 3:02d}" for i in range(n)],
        "label_raw": (["TP", "FP"] * (n // 2))[:n],
        "pattern_count": np.float64([10] * n),
        "service": [f"svc{i % 3}" for i in range(n)],
        "ops_dept": [f"dept{i % 2}" for i in range(n)],
        "organization": [f"org{i % 2}" for i in range(n)],
        "retention_period": [f"{(i % 3) + 1}년" for i in range(n)],
    })


class TestRunTrainingLabelArgparse:
    """--source label argparse 테스트"""

    def test_source_detection_is_default(self):
        """기본값은 detection"""
        import scripts.run_training as rt
        with patch("sys.argv", ["run_training.py"]):
            args = rt.parse_args()
        assert args.source == "detection"

    def test_source_label_parsed(self):
        """--source label 파싱"""
        import scripts.run_training as rt
        with patch("sys.argv", ["run_training.py", "--source", "label"]):
            args = rt.parse_args()
        assert args.source == "label"

    def test_dry_run_flag_parsed(self):
        """--dry-run 플래그 파싱"""
        import scripts.run_training as rt
        with patch("sys.argv", ["run_training.py", "--source", "label", "--dry-run"]):
            args = rt.parse_args()
        assert args.dry_run is True

    def test_invalid_source_raises(self):
        """잘못된 source 값 → argparse 오류"""
        import scripts.run_training as rt
        with patch("sys.argv", ["run_training.py", "--source", "invalid"]):
            with pytest.raises(SystemExit):
                rt.parse_args()


class TestRunLabelModeFileNotFound:
    """silver_label.parquet 파일 없을 때 조기 종료"""

    def test_label_mode_exits_when_no_parquet(self, tmp_path, capsys):
        """파일 없으면 함수 조기 반환 (예외 없이)"""
        import scripts.run_training as rt
        args = MagicMock()
        args.dry_run = False
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # PROCESSED_DATA_DIR를 tmp_path로 패치 (파일 없는 디렉토리)
        with (
            patch("scripts.run_training.PROCESSED_DATA_DIR", tmp_path),
            patch("scripts.run_training.CHECKPOINT_DIR", ckpt_dir),
        ):
            rt._run_label_mode(args)  # 예외 없이 반환

        captured = capsys.readouterr()
        assert "오류" in captured.out or "없음" in captured.out


class TestRunLabelModeDryRun:
    """--dry-run 모드: 피처까지만 실행"""

    def test_dry_run_skips_model_training(self, tmp_path, capsys):
        """--dry-run 시 모델 학습 없이 반환"""
        import scripts.run_training as rt

        df = _make_silver_label_df(n=80)
        # silver_label.parquet 존재하는 것처럼 더미 파일 생성
        parquet_path = tmp_path / "silver_label.parquet"
        parquet_path.touch()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        args = MagicMock()
        args.dry_run = True

        with (
            patch("scripts.run_training.PROCESSED_DATA_DIR", tmp_path),
            patch("scripts.run_training.MODEL_DIR", tmp_path),
            patch("scripts.run_training.CHECKPOINT_DIR", ckpt_dir),
            patch("scripts.run_training.PROJECT_ROOT", PROJECT_ROOT),
            patch("pandas.read_parquet", return_value=df),
        ):
            rt._run_label_mode(args)

        captured = capsys.readouterr()
        assert "dry-run" in captured.out or "건너뜀" in captured.out
        # 모델 파일이 생성되지 않아야 함
        assert not (tmp_path / "phase1_lgb.joblib").exists()


class TestRunLabelModeIntegration:
    """_run_label_mode() 전체 경로 검증 (모델 학습 포함)"""

    def test_label_mode_creates_model_file(self, tmp_path):
        """정상 실행 시 phase1_lgb.joblib 생성"""
        import scripts.run_training as rt

        df = _make_silver_label_df(n=80)
        # silver_label.parquet 존재하는 것처럼 더미 파일 생성
        parquet_path = tmp_path / "silver_label.parquet"
        parquet_path.touch()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_dir = tmp_path / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        args = MagicMock()
        args.dry_run = False
        args.tp_weight = 1.0

        with (
            patch("scripts.run_training.PROCESSED_DATA_DIR", tmp_path),
            patch("scripts.run_training.MODEL_DIR", tmp_path),
            patch("scripts.run_training.CHECKPOINT_DIR", ckpt_dir),
            patch("scripts.run_training.FINAL_MODEL_DIR", final_dir),
            patch("scripts.run_training.PROJECT_ROOT", PROJECT_ROOT),
            patch("pandas.read_parquet", return_value=df),
        ):
            rt._run_label_mode(args)

        assert (final_dir / "phase1_lgb.joblib").exists()
