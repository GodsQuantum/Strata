#!/usr/bin/env python3
"""CPU-only expert packing byte-layout contract, no GPU or NumPy dependency.

Execute the actual streaming writer AST from tools/iq_pack.py in isolation,
rather than testing a duplicate algorithm or importing large ML dependencies.
"""
import ast
import io
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def writer_from_source():
    source = (ROOT / "tools/iq_pack.py").read_text(encoding="utf-8")
    syntax = ast.parse(source)
    functions = [node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "write_expert_views"]
    if len(functions) != 1:
        raise AssertionError("Expected exactly one production streaming writer")
    standalone = ast.Module(body=functions, type_ignores=[])
    scope = {}
    exec(compile(ast.fix_missing_locations(standalone), "tools/iq_pack.py", "exec"), scope)
    return scope["write_expert_views"]


class Row:
    def __init__(self, data: bytes):
        self.data = data

    def tobytes(self) -> bytes:
        return self.data


class View:
    def __init__(self, rows):
        self.rows = [Row(item) for item in rows]

    def __getitem__(self, index):
        return self.rows[index]


class CountingWriter(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.max_chunk = 0

    def write(self, buffer):
        self.calls += 1
        self.max_chunk = max(self.max_chunk, len(buffer))
        return super().write(buffer)


class ExpertStreamingLayoutTests(unittest.TestCase):
    def test_exact_expert_major_gate_up_down_byte_layout(self):
        writer = writer_from_source()
        gate = [b"\x00\x01", b"\x10\x11"]
        up = [b"A", b"B"]
        down = [b"\xff\xfe\xfd", b"\x03\x02\x01"]
        views = [View(gate), View(up), View(down)]
        out = CountingWriter()
        writer(out, views, 2)
        reference = b"".join(b"".join(arr[e] for arr in (gate, up, down)) for e in range(2))
        self.assertEqual(out.getvalue(), reference)
        self.assertEqual(out.calls, 6)
        self.assertEqual(out.max_chunk, 3)

    def test_variable_sizes_five_experts_and_no_concatenation(self):
        writer = writer_from_source()
        arrays = [
            [bytes([100 + i]) * 2 for i in range(5)],
            [bytes([120 + i]) * 7 for i in range(5)],
            [bytes([200 + i]) for i in range(5)],
        ]
        out = CountingWriter()
        writer(out, [View(a) for a in arrays], 5)
        self.assertEqual(out.getvalue(), b"".join(b"".join(a[i] for a in arrays) for i in range(5)))
        self.assertEqual(out.calls, 15)
        self.assertLess(out.max_chunk, out.tell())

    def test_real_packer_uses_tested_helper(self):
        tree = ast.parse((ROOT / "tools/iq_pack.py").read_text(encoding="utf-8"))
        calls = [
            call for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "write_expert_views"
        ]
        self.assertEqual(len(calls), 1, "the real experts.bin write must use the validated streaming writer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
