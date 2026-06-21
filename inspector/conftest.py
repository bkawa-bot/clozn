"""Make `clozn` importable without an editable install (Windows-friendly), and load the
real RWKV-4 source once for the gated `model` tests."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def rwkv():
    """A real recurrent model as a StateSource (cached). Skips cleanly if deps/checkpoint absent."""
    try:
        from clozn.sources.hf_rwkv import RwkvStateSource
        return RwkvStateSource()
    except Exception as e:                      # missing torch/transformers or offline cache
        pytest.skip(f"RWKV-4 source unavailable: {e}")
