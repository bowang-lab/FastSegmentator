"""Vendored subset of TotalSegmentator (https://github.com/wasserth/TotalSegmentator).

Only the modules FastSegmentator's inference pipeline depends on are vendored here,
copied verbatim from upstream commit 44151b4 (v2.13.0-32-g44151b4) so the package is
self-contained and pinned to the exact code the parity results were validated against.

TotalSegmentator is licensed under Apache-2.0 (see ./LICENSE). See ../NOTICE for
attribution and the list of vendored modules.
"""
