"""NumPy tensor codec for the native engine's float32 wire format.

NumPy is an optional dependency so gateway-only and score-only users retain the
standard-library-only install. Array methods fail with a focused message when the
``arrays`` extra is not installed.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from ._transport import CloznProtocolError


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra
        raise RuntimeError(
            "activation tensor methods require NumPy; install 'clozn-client[arrays]'"
        ) from exc
    return np


def decode_float32_tensor(value: object, *, label: str = "tensor"):
    """Decode ``{dtype, shape, data}`` into a read-only little-endian float32 ndarray."""
    if not isinstance(value, Mapping):
        raise CloznProtocolError(f"{label} must be a JSON object")
    if value.get("dtype") != "float32":
        raise CloznProtocolError(f"{label}.dtype must be 'float32'")
    shape_value = value.get("shape")
    if isinstance(shape_value, (str, bytes, bytearray)) or not isinstance(shape_value, Sequence):
        raise CloznProtocolError(f"{label}.shape must be an integer array")
    shape: list[int] = []
    for dim in shape_value:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise CloznProtocolError(f"{label}.shape must contain non-negative integers")
        shape.append(dim)
    data = value.get("data")
    if not isinstance(data, str):
        raise CloznProtocolError(f"{label}.data must be base64 text")
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise CloznProtocolError(f"{label}.data is not valid base64") from exc
    np = require_numpy()
    array = np.frombuffer(raw, dtype="<f4")
    expected = 1
    for dim in shape:
        expected *= dim
    if array.size != expected:
        raise CloznProtocolError(
            f"{label} byte count contains {array.size} float32 values; shape requires {expected}"
        )
    result = array.reshape(tuple(shape))
    result.setflags(write=False)
    return result


def flatten_float32(values: Any, *, label: str = "values") -> tuple[list[float], tuple[int, ...]]:
    """Normalize array-like values to contiguous little-endian float32, row-major."""
    np = require_numpy()
    try:
        array = np.asarray(values, dtype="<f4")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric array-like") from exc
    if array.ndim not in (1, 2):
        raise ValueError(f"{label} must be a vector or matrix")
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    contiguous = np.ascontiguousarray(array, dtype="<f4")
    return contiguous.reshape(-1).tolist(), tuple(int(dim) for dim in contiguous.shape)
