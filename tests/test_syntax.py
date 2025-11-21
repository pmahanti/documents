#!/usr/bin/env python3
"""Test script to validate Python syntax of all modules."""

import py_compile
import sys
from pathlib import Path

print("=" * 60)
print("Syntax Validation Tests")
print("=" * 60)

project_root = Path(__file__).parent.parent
modules = [
    'src/crater_analysis/__init__.py',
    'src/crater_analysis/config.py',
    'src/crater_analysis/refinement.py',
    'src/crater_analysis/morphometry.py',
    'src/crater_analysis/cratools.py',
    'main.py',
]

failed = []
passed = []

for module in modules:
    module_path = project_root / module
    print(f"\nTesting: {module}")
    try:
        py_compile.compile(str(module_path), doraise=True)
        print(f"  ✓ Syntax OK")
        passed.append(module)
    except py_compile.PyCompileError as e:
        print(f"  ✗ Syntax Error: {e}")
        failed.append(module)

print("\n" + "=" * 60)
print(f"Results: {len(passed)} passed, {len(failed)} failed")
print("=" * 60)

if failed:
    print("\nFailed modules:")
    for module in failed:
        print(f"  - {module}")
    sys.exit(1)
else:
    print("\n✓ All modules have valid Python syntax!")
    sys.exit(0)
