"""공통 유틸리티"""
import random
import numpy as np
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """재현성 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(*dirs: Path) -> None:
    """디렉토리 존재 보장"""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
