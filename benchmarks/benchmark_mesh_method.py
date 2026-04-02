"""Benchmark: render_surface_mesh() CDT vs fast_smooth across resolutions.

Compares the two surface extraction pipelines:
- CDT: signed distance field + Butterworth convolution (precision path)
- fast_smooth: fused scale + convolve, no CDT (fast path)

Measures time, triangle count, open edges, and peak memory.
"""
import time
import numpy as np
from voxelcad import Sphere, GyroidCube
from voxelcad._kernels import CYTHON_AVAILABLE


def _count_open_edges(verts, faces):
    """Count open (non-manifold) edges in a triangle mesh."""
    edge_count = {}
    for f in faces:
        for i in range(3):
            e = tuple(sorted((f[i], f[(i + 1) % 3])))
            edge_count[e] = edge_count.get(e, 0) + 1
    return sum(1 for c in edge_count.values() if c != 2)


def _mesh_stats(mesh_result):
    """Extract stats from render_surface_mesh result (PyVista PolyData)."""
    if mesh_result is None:
        return {"verts": 0, "tris": 0, "open_edges": "N/A"}
    n_points = mesh_result.n_points
    faces_arr = mesh_result.faces.reshape(-1, 4)[:, 1:]
    n_tris = len(faces_arr)
    open_edges = _count_open_edges(
        mesh_result.points, faces_arr
    )
    return {"verts": n_points, "tris": n_tris, "open_edges": open_edges}


def benchmark_method(model, method, n_runs=3, **kwargs):
    """Benchmark a single method, return timing and mesh stats."""
    # Warmup
    model.pv_surf = None
    model.render_surface_mesh(method=method, cache=False, **kwargs)

    times = []
    for _ in range(n_runs):
        model.pv_surf = None
        t0 = time.perf_counter()
        mesh = model.render_surface_mesh(method=method, cache=False, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    stats = _mesh_stats(mesh)
    stats["time_median"] = np.median(times)
    stats["time_min"] = np.min(times)
    stats["method"] = method
    return stats


def run_benchmark_suite():
    """Run CDT vs fast_smooth benchmarks across geometries and resolutions."""
    print(f"Cython available: {CYTHON_AVAILABLE}")
    print("=" * 80)

    configs = [
        ("Sphere", 128, lambda vs: Sphere(r=5, voxel_size=vs)),
        ("Sphere", 256, lambda vs: Sphere(r=5, voxel_size=vs)),
        ("GyroidCube", 128, lambda vs: GyroidCube(size=10, voxel_size=vs, center=True)),
        ("GyroidCube", 256, lambda vs: GyroidCube(size=10, voxel_size=vs, center=True)),
    ]

    # Add 384 if Cython available (too slow without it)
    if CYTHON_AVAILABLE:
        configs.extend([
            ("Sphere", 384, lambda vs: Sphere(r=5, voxel_size=vs)),
            ("GyroidCube", 384, lambda vs: GyroidCube(size=10, voxel_size=vs, center=True)),
        ])

    methods = ["cdt", "fast_smooth"]
    strides = [1, 2]

    results = []

    for geom_name, target_res, factory in configs:
        vs = 10.0 / target_res
        model = factory(vs)
        model.render_volume()
        rx, ry, rz = model.grid.res_vector
        print(f"\n{geom_name} {rx}x{ry}x{rz} (voxel_size={vs:.4f})")
        print("-" * 60)
        print(f"  {'Method':<20} {'Stride':>6} {'Time(s)':>8} {'Tris':>10} {'Open':>6}")

        for method in methods:
            for stride in strides:
                stats = benchmark_method(
                    model, method, n_runs=3, mc_stride=stride
                )
                label = f"{method}"
                print(
                    f"  {label:<20} {stride:>6} "
                    f"{stats['time_median']:>8.3f} "
                    f"{stats['tris']:>10,} "
                    f"{stats['open_edges']:>6}"
                )
                stats["geometry"] = geom_name
                stats["target_res"] = target_res
                stats["stride"] = stride
                results.append(stats)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: fast_smooth speedup over CDT")
    print("=" * 80)
    for geom_name, target_res, _ in configs:
        for stride in strides:
            cdt = [r for r in results
                   if r["geometry"] == geom_name
                   and r["method"] == "cdt"
                   and r["stride"] == stride
                   and r["target_res"] == target_res]
            fs = [r for r in results
                  if r["geometry"] == geom_name
                  and r["method"] == "fast_smooth"
                  and r["stride"] == stride
                  and r["target_res"] == target_res]
            if cdt and fs:
                speedup = cdt[0]["time_median"] / fs[0]["time_median"]
                tri_match = "MATCH" if cdt[0]["tris"] == fs[0]["tris"] else "DIFFER"
                print(
                    f"  {geom_name:<12} ~{target_res}^3 stride={stride}: "
                    f"{speedup:.1f}x faster, tris {tri_match}, "
                    f"open_edges CDT={cdt[0]['open_edges']} "
                    f"fast_smooth={fs[0]['open_edges']}"
                )


if __name__ == "__main__":
    run_benchmark_suite()
