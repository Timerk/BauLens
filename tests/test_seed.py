import random

import numpy as np
import torch

from baulens.seed import seed_everything


def test_seed_everything_reproduces_random_values() -> None:
    seed_everything(123)
    python_sample = random.random()
    numpy_sample = np.random.random(3)
    torch_sample = torch.rand(3, device="cpu")

    seed_everything(123)
    assert random.random() == python_sample
    np.testing.assert_array_equal(np.random.random(3), numpy_sample)
    assert torch.equal(torch.rand(3, device="cpu"), torch_sample)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
