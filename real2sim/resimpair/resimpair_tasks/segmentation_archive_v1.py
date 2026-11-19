"""Explicit lossless NPZ writer with bounded DEFLATE effort.

Array bytes, dtype, shape, member names and order are preserved. This utility
is opt-in: it does not modify NumPy or any existing renderer implementation.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

VERSION = 'lossless_numpy_zip_deflate1_v1'


def write_npz(path, **arrays):
    """Write a new archive only; failed partial output remains for inspection."""
    path = Path(path)
    if not arrays:
        raise ValueError('nonempty_explicit_arrays_required')
    values = {}
    for name, value in arrays.items():
        if (not isinstance(name, str) or not name or name in {'.', '..'}
                or '/' in name or '\\' in name or '\x00' in name):
            raise ValueError('plain_numpy_member_names_required')
        array = np.asanyarray(value)
        if array.dtype.hasobject:
            raise ValueError('object_arrays_and_pickle_forbidden')
        values[name] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        with ZipFile(stream, mode='w', compression=ZIP_DEFLATED,
                     compresslevel=1, allowZip64=True) as archive:
            for name, value in values.items():
                with archive.open(name + '.npy', mode='w', force_zip64=True) as member:
                    np.lib.format.write_array(member, value, allow_pickle=False)
    return path
