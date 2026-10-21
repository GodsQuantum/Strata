#!/usr/bin/env python3
"""No-GPU, no-NumPy test: low-memory expert packing with Strata v0.1.44.1 NVFP4 tails."""
import ast
import io
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def real_writer():
    source = (ROOT / "tools/iq_pack.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)
                 and item.name == "write_expert_views"]
    assert len(functions) == 1
    namespace = {}
    unit = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(unit, "iq_pack.py", "exec"), namespace)
    return namespace["write_expert_views"]


class Row:
    def __init__(self, value):
        self.value = value

    def tobytes(self):
        return self.value


class View:
    def __init__(self, rows):
        self.rows = [Row(x) for x in rows]

    def __getitem__(self, index):
        return self.rows[index]


class TrackingWriter(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.highest_chunk = 0
        self.write_count = 0

    def write(self, value):
        self.write_count += 1
        self.highest_chunk = max(self.highest_chunk, len(value))
        return super().write(value)


class TestNVFP4Layout(unittest.TestCase):
    def test_nvfp4_tail_after_every_expert(self):
        writer = real_writer()
        a = [[b"gate0", b"gate1"], [b"up0", b"up1"], [b"down0", b"down1"]]
        tails = [struct.pack("<ffff", 1., 2., 3., 0.),
                 struct.pack("<ffff", 1.25, 0.5, 4., 0.)]
        output = TrackingWriter()
        writer(output, [View(x) for x in a], 2, View(tails))
        expected = b"".join(b"".join(col[e] for col in a) + tails[e] for e in range(2))
        self.assertEqual(output.getvalue(), expected)
        self.assertEqual(output.write_count, 8)
        self.assertEqual(output.highest_chunk, 16)

    def test_mixed_projection_tail_one_for_non_nvfp4(self):
        writer = real_writer()
        data = [[b"g"] * 3, [b"u"] * 3, [b"d"] * 3]
        scales = [struct.pack("<ffff", 2., 4., 1., 0.) for _ in range(3)]
        output = TrackingWriter()
        writer(output, [View(x) for x in data], 3, View(scales))
        self.assertEqual(output.getvalue(), (b"gud" + scales[0]) * 3)
        self.assertLess(output.highest_chunk, output.tell())

    def test_standard_weights_no_tails_preserved(self):
        writer = real_writer()
        out = TrackingWriter()
        writer(out, [View([b"AA", b"BB"]), View([b"C", b"D"])], 2)
        self.assertEqual(out.getvalue(), b"AACBBD")
        self.assertEqual(out.write_count, 4)

    def test_real_main_streams_not_full_layer_concat(self):
        source = (ROOT / "tools/iq_pack.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [item for item in ast.walk(tree) if isinstance(item, ast.Call)
                 and isinstance(item.func, ast.Name)
                 and item.func.id == "write_expert_views"]
        self.assertEqual(len(calls), 1)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertFalse(any(isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute) and n.func.attr == "concatenate"
                             for n in ast.walk(main)))
        code = ast.get_source_segment(source, main)
        self.assertIn("nvfp4_tails", code)
        self.assertIn("not np.isfinite(tail).all()", code)
        self.assertIn("tail[:, :3] <= 0", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
