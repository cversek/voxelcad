#!/usr/bin/env python
"""Spectral analysis demo — visualize SDF frequency content.

Generates diagnostic plots for Sphere and GyroidCube at multiple
resolutions, showing radial power spectra, cumulative energy curves,
2D spectrum slices, and Butterworth filter effects.

Usage:
    python examples/spectral_analysis_demo.py
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(name)s: %(message)s')

from voxelcad import ENV
from voxelcad.sphere import Sphere
from voxelcad.gyroid_cube import GyroidCube

OUT_DIR = 'spectral_output'
os.makedirs(OUT_DIR, exist_ok=True)


def analyze_model(model, label, resolution):
    """Run spectral analysis on a model and print summary."""
    print(f"\n{'='*60}")
    print(f"  {label}  (resolution {resolution}^3)")
    print(f"{'='*60}")

    result = model.analyze_spectrum(
        lowpass_cutoff=0.25,
        plot=True,
        save_path=OUT_DIR,
    )

    print(f"  bandwidth_99:       {result['bandwidth_99']:.4f} cyc/vox")
    print(f"  bandwidth_95:       {result['bandwidth_95']:.4f} cyc/vox")
    print(f"  recommended_cutoff: {result['recommended_cutoff']:.4f} cyc/vox")
    print(f"  safe_stride:        {result['safe_stride']}")
    print(f"  stride-2 safe?      "
          f"{'YES' if result['bandwidth_99'] < 0.25 else 'MARGINAL/NO'}")
    return result


def main():
    resolutions = [64, 128, 200]

    for res in resolutions:
        voxel_size = 12.0 / res  # 12mm cube
        ENV.voxel_size = voxel_size

        # Sphere: smooth geometry, no thin features
        sphere = Sphere(5)
        sphere.render_volume()
        analyze_model(sphere, 'Sphere(r=5)', res)
        del sphere

        # GyroidCube: thin periodic walls
        gyroid = GyroidCube(12)
        gyroid.render_volume()
        analyze_model(gyroid, 'GyroidCube(size=12)', res)
        del gyroid

    print(f"\nPlots saved to: {os.path.abspath(OUT_DIR)}/")
    print("Done.")


if __name__ == '__main__':
    main()
