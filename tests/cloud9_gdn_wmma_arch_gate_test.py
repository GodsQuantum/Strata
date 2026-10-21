#!/usr/bin/env python3
"""CPU-only regression: Radeon 780M must never launch the unsafe optional
GDN chunked-WMMA kernel until gfx1103 parity/stability has been validated.
Does not open the GPU or require a ROCm SDK.
"""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

class GdnWmmaGate(unittest.TestCase):
    def test_gfx1103_fail_closed_without_disabling_other_wmma(self):
        source = r"""
#include <strata/kernels/gfx_arch.hpp>
#include <cassert>
using namespace strata::kernels;
int main() {
    assert(gfx_arch_is_gfx11_wmma("gfx1103"));
    assert(!gfx_arch_has_safe_gdn_chunked_wmma("gfx1103"));
    assert(!gfx_arch_has_safe_gdn_chunked_wmma("gfx1103:sramecc-:xnack-"));
    assert(gfx_arch_has_safe_gdn_chunked_wmma("gfx1100"));
    assert(gfx_arch_has_safe_gdn_chunked_wmma("gfx1101"));
    assert(gfx_arch_has_safe_gdn_chunked_wmma("gfx1151"));
    assert(!gfx_arch_has_safe_gdn_chunked_wmma("gfx1201"));
    assert(!gfx_arch_has_safe_gdn_chunked_wmma(nullptr));
}
"""
        with tempfile.TemporaryDirectory(prefix="strata-gdn-gfx1103-") as tmp:
            binary=pathlib.Path(tmp)/"gate"
            subprocess.run(["c++","-std=c++17","-Wall","-Wextra","-Werror",
                            "-I"+str(ROOT/"include"),"-x","c++","-","-o",str(binary)],
                            input=source,text=True,check=True,capture_output=True,timeout=20)
            subprocess.run([str(binary)],check=True,timeout=8)

    def test_runtime_dispatch_uses_gdn_specific_gate(self):
        code=(ROOT/"src/prefill/kernels.cu").read_text()
        fn=code.split("bool gdn_rec_wmma(",1)[1].split("#endif   // __HIPCC__",1)[0]
        self.assertIn("gfx_arch_has_safe_gdn_chunked_wmma(p.gcnArchName)",fn)
        self.assertIn("if (!gfx11) return false;",fn)
        self.assertIn("gdn_wmma_prep_kernel<<<",fn)
        self.assertLess(fn.index("if (!gfx11) return false;"),fn.index("gdn_wmma_prep_kernel<<<"))

if __name__=="__main__":
    unittest.main(verbosity=2)
