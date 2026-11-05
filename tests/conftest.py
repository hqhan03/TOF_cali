import numpy as np
import pytest

from hhg_tof.config import load_config


@pytest.fixture
def sim_cfg():
    def make(name="sphere", **overrides):
        return load_config(f"config/sim_{name}.yaml", overrides or None)
    return make


@pytest.fixture
def rng():
    return np.random.default_rng(0)
