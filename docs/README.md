# VoxelCAD Documentation

## Audience Guide

| You are a... | Start here |
|--------------|------------|
| **User** (3D printing, research, engineering) | [User Guide](user/) |
| **Developer** (extending VoxelCAD, contributing) | [Developer Guide](developer/) |
| **Architect** (optimization internals, GPU design) | [Architecture Guide](architecture/) |

## User Documentation

- [Getting Started](user/getting-started.md) -- Install, first model, STL export
- [Geometry Catalog](user/geometry-catalog.md) -- All primitives with parameters
- [Boolean Operations](user/boolean-operations.md) -- Union, intersection, difference, XOR
- [Transforms](user/transforms.md) -- Rotate, scale, translate, composition
- [Performance Guide](user/performance-guide.md) -- Resolution, memory, Cython acceleration
- [Troubleshooting](user/troubleshooting.md) -- Common errors and fixes

## Developer Documentation

- [Extension Guide](developer/extension-guide.md) -- Adding new geometry primitives
- [Testing Strategy](developer/testing-strategy.md) -- Test structure and correctness criteria
- [Build System](developer/build-system.md) -- Cython, cross-platform OpenMP, pip workflow
- [API Reference](developer/api-reference.md) -- Class hierarchy, method signatures

## Architecture Documentation

- [Optimization System](architecture/optimization-system.md) -- Three-tier dispatch, composition
- [Storage Format](architecture/storage-format.md) -- F-order packed booleans, byte alignment
- [Query Planner](architecture/query-planner.md) -- CSG tree analysis and execution planning
- [Memory Model](architecture/memory-model.md) -- Scaling laws, streaming, peak tracking
- [GPU Design](architecture/gpu-design.md) -- OpenMP target offloading, CuPy/MLX backends
