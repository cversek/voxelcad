#!/usr/bin/env python3
"""Run all VoxelCAD benchmarks and export JSON results.

Usage:
    python benchmarks/run_benchmarks.py                  # small (quick)
    python benchmarks/run_benchmarks.py --size medium    # standard
    python benchmarks/run_benchmarks.py --size large     # full (needs Cython + multi-core)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from voxelcad._kernels import CYTHON_AVAILABLE

# Import all benchmark classes
from benchmarks.benchmark_render import (
    BenchmarkCubeRender, BenchmarkSphereRender,
    BenchmarkCylinderRender, BenchmarkGyroidRender,
)
from benchmarks.benchmark_boolean import (
    BenchmarkSameGridUnion, BenchmarkSameGridIntersection,
    BenchmarkCSGUnionRender, BenchmarkCSGDepth4Render,
)
from benchmarks.benchmark_fallback_breaking import (
    BenchmarkManualComposition, BenchmarkTier2CompatibleGrid,
    BenchmarkTransformedInCSG, BenchmarkTransformOverhead,
)

ALL_BENCHMARKS = [
    # Render
    BenchmarkCubeRender,
    BenchmarkSphereRender,
    BenchmarkCylinderRender,
    BenchmarkGyroidRender,
    # Boolean
    BenchmarkSameGridUnion,
    BenchmarkSameGridIntersection,
    BenchmarkCSGUnionRender,
    BenchmarkCSGDepth4Render,
    # Fallback-breaking
    BenchmarkManualComposition,
    BenchmarkTier2CompatibleGrid,
    BenchmarkTransformedInCSG,
    BenchmarkTransformOverhead,
]


def get_system_info():
    """Capture system spec for cross-machine comparison."""
    import platform
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cython_available": CYTHON_AVAILABLE,
    }
    try:
        from super_utils.system import get_system_spec
        info["system_spec"] = get_system_spec()
    except ImportError:
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description="VoxelCAD benchmark suite")
    parser.add_argument("--size", choices=["small", "medium", "large"],
                        default="small", help="Problem size (default: small)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Timed iterations per benchmark (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: stdout summary)")
    args = parser.parse_args()

    results = {
        "system": get_system_info(),
        "config": {"size": args.size, "iterations": args.iterations},
        "benchmarks": {},
    }

    print(f"VoxelCAD Benchmark Suite — size={args.size}, "
          f"iterations={args.iterations}, cython={CYTHON_AVAILABLE}")
    print("=" * 70)

    for bench_cls in ALL_BENCHMARKS:
        bench = bench_cls(size=args.size)
        name = bench.name
        print(f"\n  {name}: {bench.description}")
        try:
            result = bench.benchmark(
                iterations=args.iterations, warmup=1,
                track_memory=True, sample_interval_ms=10,
            )
            results["benchmarks"][name] = result
            valid = result.get("valid", False)
            mean = result.get("mean_ms", 0)
            std = result.get("std_ms", 0)
            peak = result.get("run_peak_mb", 0)
            status = "PASS" if valid else "FAIL"
            print(f"    [{status}] {mean:.1f} +/- {std:.1f} ms "
                  f"| peak {peak:.1f} MB")
        except Exception as e:
            results["benchmarks"][name] = {"error": str(e)}
            print(f"    [ERROR] {e}")

    print("\n" + "=" * 70)
    passed = sum(1 for r in results["benchmarks"].values()
                 if r.get("valid", False))
    total = len(results["benchmarks"])
    print(f"Results: {passed}/{total} passed")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"JSON exported to: {args.output}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
