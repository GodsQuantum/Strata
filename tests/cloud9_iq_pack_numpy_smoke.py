#!/usr/bin/env python3
"""Small real NumPy/memmap packing A/B, no GPU, no large model weights.

Intended for isolated CPU Docker with numpy installed; not needed in basic CI.
"""
import ast
import hashlib
import io
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "tools/iq_pack.py").read_text(encoding="utf-8"))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "write_expert_views"]
assert len(functions) == 1
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "iq_pack.py", "exec"), namespace)
writer = namespace["write_expert_views"]

with tempfile.TemporaryDirectory() as folder:
    num_experts = 12
    arrays = []
    for i, width in enumerate((32, 64, 48)):
        path = Path(folder) / f"experts_{i}.dat"
        mapped = np.memmap(path, dtype=np.uint8, mode="w+", shape=(num_experts, width))
        mapped[:] = np.arange(num_experts * width, dtype=np.uint32).reshape(num_experts, width) % 251
        mapped.flush()
        arrays.append(np.memmap(path, dtype=np.uint8, mode="r", shape=(num_experts, width)))
    expected = np.concatenate(arrays, axis=1).tobytes()
    out = io.BytesIO()
    writer(out, arrays, num_experts)
    actual = out.getvalue()
    assert actual == expected, (len(actual), len(expected))
    assert len(actual) == num_experts * sum(arr.shape[1] for arr in arrays)
    print("CLOUD9_NUMPY_MEMMAP_EXPERT_PACKING_PASS", len(actual), hashlib.sha256(actual).hexdigest(), flush=True)
