from pathlib import Path

import numpy as np
import pytest

from baulens.data.concrete import load_concrete

FIXTURE = Path(__file__).parent / "fixtures" / "concrete_sample.csv"


def test_load_concrete_rejects_non_finite_values(tmp_path: Path) -> None:
    frame = load_concrete(FIXTURE)
    frame.loc[0, "cement"] = np.inf
    broken = tmp_path / "broken.csv"
    frame.to_csv(broken, index=False)
    with pytest.raises(ValueError, match=r"Non-finite values .*cement"):
        load_concrete(broken)
