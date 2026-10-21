#!/usr/bin/env python3
"""Offline contract tests; no API/model/GPU calls."""
import unittest
from unittest.mock import patch
import io
import json
import socket
import sys

import cloud9_strata_admission as adm


def sample(**kwargs):
    d = {"allow": True, "reasons": [], "mem_available_gib": 62.0,
         "safe_total_gtt_gib": 16.0, "psi_memory_full_avg10": 0.0,
         "swap_used_percent": 12.0}
    d.update(kwargs)
    return d


class TestAdmission(unittest.TestCase):
    def test_safe_guard_and_ample_memory(self):
        self.assertTrue(adm.review(sample())["safe_to_attempt"])

    def test_memory_shortage_blocks_even_guard_allow(self):
        r = adm.review(sample(mem_available_gib=22))
        self.assertFalse(r["safe_to_attempt"])
        self.assertTrue(any("Strata 125B floor" in x for x in r["reasons"]))

    def test_guard_refusal_is_never_overridden(self):
        r = adm.review(sample(allow=False, reasons=["low host memory"]))
        self.assertFalse(r["safe_to_attempt"])
        self.assertIn("low host memory", r["reasons"])

    def test_gtt_request_check(self):
        r = adm.review(sample(safe_total_gtt_gib=5))
        self.assertFalse(r["safe_to_attempt"])

    def test_psi_check(self):
        r = adm.review(sample(psi_memory_full_avg10=0.6))
        self.assertFalse(r["safe_to_attempt"])

    def test_missing_metrics_fail_closed(self):
        for k in ("mem_available_gib", "safe_total_gtt_gib", "psi_memory_full_avg10"):
            broken = sample()
            del broken[k]
            self.assertFalse(adm.review(broken)["safe_to_attempt"])

    def test_nan_and_infinity_fail_closed(self):
        for value in (float("nan"), float("inf"), -1):
            self.assertFalse(adm.review(sample(mem_available_gib=value))["safe_to_attempt"])

    def test_socket_unavailable_fails(self):
        with patch("socket.create_connection", side_effect=OSError("test failure")):
            with self.assertRaises(OSError):
                adm.query_guard("guard.example", 19755, "coexist-light", 8)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            adm.query_guard("guard.example", 19755, "bypass-guard", 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
