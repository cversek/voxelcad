# VoxelCAD Examples

## Quick Start

| Example | Difficulty | Features | Run |
|---------|-----------|----------|-----|
| [hello_gyroid_sphere.py](hello_gyroid_sphere.py) | Beginner | Sphere, GyroidCube, intersection, STL export | `python hello_gyroid_sphere.py` |
| [transforms_demo.py](transforms_demo.py) | Beginner | Translate, rotate, scale, compose, union | `python transforms_demo.py` |

## Interactive

| Example | Difficulty | Features | Run |
|---------|-----------|----------|-----|
| [ice_cream_cone_demo.ipynb](ice_cream_cone_demo.ipynb) | Intermediate | Full walkthrough: primitives, CSG, transforms, rendering, profiling | `jupyter notebook ice_cream_cone_demo.ipynb` |

## 3D Printing

| Example | Difficulty | Features | Run |
|---------|-----------|----------|-----|
| [gyroid_sponge.py](gyroid_sponge.py) | Intermediate | High-res gyroid slab, smoothing, downsampling | `python -c "from gyroid_sponge import export; export()"` |
| [gyroid_sponge_disks.py](gyroid_sponge_disks.py) | Intermediate | Gyroid-cylinder intersection, parameterized export | `python -c "from gyroid_sponge_disks import export; export()"` |
| [gyroid_mold.py](gyroid_mold.py) | Advanced | Multi-component assembly, CSG difference (negative mold) | `python gyroid_mold.py` |
| [gyroid_electrode_support_plug_conn.py](gyroid_electrode_support_plug_conn.py) | Advanced | Real-world part: tapered cylinders, multi-union, parameterized dims | `python -c "from gyroid_electrode_support_plug_conn import export; export()"` |
| [multi_union_test.py](multi_union_test.py) | Beginner | union_all() with list comprehension | `python multi_union_test.py` |

## Notes

- **Resolution**: Most examples use `ENV.voxel_size` to control resolution globally. Smaller values give finer detail but use more memory. See the [performance guide](../docs/user/performance-guide.md) for memory scaling.
- **STL export** requires PyVista: `pip install voxelcad[viz]`
- **High-res examples** (gyroid_sponge, electrode_support) default to 512-1024 resolution and need several GB of RAM.
