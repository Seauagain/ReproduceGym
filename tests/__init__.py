"""Local test package for shared helpers.

This prevents imports like `tests.helpers` from resolving to an unrelated
third-party package named `tests` in some virtual environments.
"""
