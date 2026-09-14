"""Memory-bounded PyTorch implementation of GeoUDF's E-MC extractor.

The implementation keeps the original E-MC decisions intact:

* only cubes whose first corner has UDF <= 2 * voxel_size are examined;
* the same 28 pairwise edge tests and 128 occupancy candidates are used;
* ties are resolved in the original candidate order; and
* the original interpolation threshold (5e-4) is preserved.

Unlike the original Python/Numba loop, cubes are evaluated in vectorized
batches.  We only materialize the near-surface cubes and their crossing grid
edges, so a 256^3 grid does not require an 8-corner copy of every voxel.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


Array = Union[np.ndarray, torch.Tensor]

_CUBE_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 0, 1),
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 1, 1),
    (0, 1, 1),
)

_CUBE_EDGES = (
    (0, 1), (1, 2), (3, 2), (0, 3),
    (4, 5), (5, 6), (7, 6), (4, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_ALL_VERTEX_PAIRS = tuple(
    (first, second)
    for first in range(7)
    for second in range(first + 1, 8)
)

_SURFACE_EPSILON = 5e-4


def _resolve_device(arrays: Sequence[Array], device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    for array in arrays:
        if isinstance(array, torch.Tensor) and array.is_cuda:
            return array.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(array: Array, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.detach().to(device=device)
    return torch.as_tensor(np.asarray(array), device=device)


def _cube_vertex_ids(cube_ijk: torch.Tensor, resolution: int, corners: torch.Tensor) -> torch.Tensor:
    base = (
        cube_ijk[:, 0] * resolution * resolution
        + cube_ijk[:, 1] * resolution
        + cube_ijk[:, 2]
    )
    corner_ids = (
        corners[:, 0] * resolution * resolution
        + corners[:, 1] * resolution
        + corners[:, 2]
    )
    return base[:, None] + corner_ids[None, :]


def _detect_surface_pairs(
    cube_udf: torch.Tensor,
    cube_grad: torch.Tensor,
    vertex_pairs: torch.Tensor,
    pair_half_delta: torch.Tensor,
    voxel_size: float,
) -> torch.Tensor:
    """Vectorized equivalent of the original ``edge_detector_all``."""
    first, second = vertex_pairs[:, 0], vertex_pairs[:, 1]
    udf_first, udf_second = cube_udf[:, first], cube_udf[:, second]
    grad_first, grad_second = cube_grad[:, first], cube_grad[:, second]

    close_to_surface = (udf_first < _SURFACE_EPSILON) | (udf_second < _SURFACE_EPSILON)
    gradients_face_each_other = (grad_first * grad_second).sum(dim=-1) < 0
    within_voxel = (udf_first < voxel_size * 1.1) & (udf_second < voxel_size * 1.1)

    delta = pair_half_delta[None, :, :]
    points_towards_midpoint = (
        (delta * grad_first).sum(dim=-1) > 0
    ) & ((-delta * grad_second).sum(dim=-1) > 0)

    return close_to_surface | (gradients_face_each_other & within_voxel & points_towards_midpoint)


def _optimize_occupancy(
    detected_pairs: torch.Tensor,
    occupancy_patterns: torch.Tensor,
    pattern_pair_crossings: torch.Tensor,
) -> torch.Tensor:
    """Solve the original 28-bit Hamming objective for a batch of cubes."""
    target = detected_pairs.to(torch.float32)
    candidates = pattern_pair_crossings.to(torch.float32)
    # Hamming(a, b) = |a| + |b| - 2 * <a, b>.  All values are small
    # integers, so float32 evaluates the objective and its ties exactly.
    losses = (
        target.sum(dim=1, keepdim=True)
        + candidates.sum(dim=1)[None, :]
        - 2 * target @ candidates.T
    )
    # torch.argmin returns the first minimum, matching the strict '<' update
    # in GeoUDF's original loop.
    return occupancy_patterns[losses.argmin(dim=1)]


def _gather_edge_points(
    grids_coords: Array,
    edge_vertices: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather only the grid coordinates touched by output edges."""
    if isinstance(grids_coords, torch.Tensor):
        flat = grids_coords.detach().reshape(-1, 3)
        ids = edge_vertices.to(flat.device)
        points = flat[ids].to(device=device)
    else:
        flat = np.asarray(grids_coords).reshape(-1, 3)
        ids = edge_vertices.detach().cpu().numpy()
        points = torch.as_tensor(flat[ids], device=device)
    return points[:, 0], points[:, 1]


def _interpolate_edges(
    edge_vertices: torch.Tensor,
    grids_coords: Array,
    udf_flat: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    p1, p2 = _gather_edge_points(grids_coords, edge_vertices, device)
    udf1, udf2 = udf_flat[edge_vertices[:, 0]], udf_flat[edge_vertices[:, 1]]
    close1, close2 = udf1 <= _SURFACE_EPSILON, udf2 <= _SURFACE_EPSILON

    vertices = torch.empty_like(p1)
    only1, only2, both = close1 & ~close2, ~close1 & close2, close1 & close2
    neither = ~close1 & ~close2
    vertices[only1] = p1[only1]
    vertices[only2] = p2[only2]
    vertices[both] = (p1[both] + p2[both]) / 2
    vertices[neither] = (
        p1[neither] * udf2[neither, None]
        + p2[neither] * udf1[neither, None]
    ) / (udf1[neither] + udf2[neither])[:, None]
    return vertices


def _deduplicate_and_order_vertices(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match legacy coordinate-based merging and first-use vertex ordering."""
    vertices, coordinate_map = torch.unique(
        vertices, dim=0, sorted=True, return_inverse=True
    )
    faces = coordinate_map[faces]

    # NumPy's return_index gives a stable first-use order and keeps this module
    # compatible with the PyTorch 1.10 environment documented by GeoUDF
    # (``Tensor.scatter_reduce_`` was introduced later).
    flattened = faces.reshape(-1)
    flattened_cpu = flattened.detach().cpu().numpy()
    _, first_use = np.unique(flattened_cpu, return_index=True)
    order = torch.as_tensor(
        flattened_cpu[np.sort(first_use)], dtype=torch.long, device=vertices.device
    )

    remap = torch.full(
        (vertices.shape[0],), -1, dtype=torch.long, device=vertices.device
    )
    remap[order] = torch.arange(order.numel(), device=vertices.device)
    return vertices[order], remap[faces]


@torch.no_grad()
def custom_marching_cube_torch(
    grids_coords: Array,
    grids_udf: Array,
    grids_udf_grad: Array,
    voxel_size: Union[float, torch.Tensor],
    resolution: int,
    triangle_table: Sequence[Sequence[int]],
    *,
    device: Optional[Union[str, torch.device]] = None,
    cube_batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract an E-MC mesh and return tensors on the selected device."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    expected_udf_shape = (resolution, resolution, resolution)
    expected_grad_shape = expected_udf_shape + (3,)
    expected_coord_shape = expected_udf_shape + (3,)
    if tuple(grids_udf.shape) != expected_udf_shape:
        raise ValueError(f"grids_udf must have shape {expected_udf_shape}")
    if tuple(grids_udf_grad.shape) != expected_grad_shape:
        raise ValueError(f"grids_udf_grad must have shape {expected_grad_shape}")
    if tuple(grids_coords.shape) != expected_coord_shape:
        raise ValueError(f"grids_coords must have shape {expected_coord_shape}")

    device = _resolve_device((grids_udf, grids_udf_grad), device)
    voxel_size = float(voxel_size.detach().item() if isinstance(voxel_size, torch.Tensor) else voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")

    if cube_batch_size is None:
        cube_batch_size = int(os.environ.get("GEOUDF_EMC_BATCH_SIZE", 32768))
    if cube_batch_size <= 0:
        raise ValueError("cube_batch_size must be positive")

    udf = _as_tensor(grids_udf, device)
    grad = _as_tensor(grids_udf_grad, device)
    if not (udf.is_floating_point() and grad.is_floating_point()):
        raise TypeError("UDF values and gradients must be floating-point arrays")
    if grad.dtype != udf.dtype:
        grad = grad.to(dtype=udf.dtype)
    udf_flat, grad_flat = udf.reshape(-1), grad.reshape(-1, 3)

    corners = torch.tensor(_CUBE_CORNERS, dtype=torch.long, device=device)
    cube_edges = torch.tensor(_CUBE_EDGES, dtype=torch.long, device=device)
    vertex_pairs = torch.tensor(_ALL_VERTEX_PAIRS, dtype=torch.long, device=device)
    triangle_table = torch.as_tensor(triangle_table, dtype=torch.long, device=device)
    if tuple(triangle_table.shape) != (256, 16):
        raise ValueError("triangle_table must have shape (256, 16)")

    states = torch.arange(128, dtype=torch.long, device=device)[:, None]
    bits = torch.arange(8, dtype=torch.long, device=device)[None, :]
    occupancy_patterns = ((states >> bits) & 1).to(torch.bool)
    pattern_pair_crossings = (
        occupancy_patterns[:, vertex_pairs[:, 0]]
        != occupancy_patterns[:, vertex_pairs[:, 1]]
    )
    pair_half_delta = (
        corners[vertex_pairs[:, 0]] - corners[vertex_pairs[:, 1]]
    ).to(dtype=udf.dtype) * (voxel_size / 2)

    # This is the exact legacy early-out, applied before any 8-corner gather.
    candidate_ijk = torch.nonzero(
        udf[: resolution - 1, : resolution - 1, : resolution - 1]
        <= voxel_size * 2,
        as_tuple=False,
    )
    if candidate_ijk.numel() == 0:
        return (
            torch.empty((0, 3), dtype=udf.dtype, device=device),
            torch.empty((0, 3), dtype=torch.long, device=device),
        )

    valid_cubes, valid_cases = [], []
    case_weights = 1 << torch.arange(8, dtype=torch.long, device=device)
    for start in range(0, candidate_ijk.shape[0], cube_batch_size):
        ijk = candidate_ijk[start : start + cube_batch_size]
        cube_ids = _cube_vertex_ids(ijk, resolution, corners)
        detected = _detect_surface_pairs(
            udf_flat[cube_ids],
            grad_flat[cube_ids],
            vertex_pairs,
            pair_half_delta,
            voxel_size,
        )
        occupancy = _optimize_occupancy(
            detected, occupancy_patterns, pattern_pair_crossings
        )
        cases = (occupancy.to(torch.long) * case_weights).sum(dim=1)
        keep = (cases > 0) & (cases < 255)
        if keep.any():
            valid_cubes.append(cube_ids[keep])
            valid_cases.append(cases[keep])

    if not valid_cubes:
        return (
            torch.empty((0, 3), dtype=udf.dtype, device=device),
            torch.empty((0, 3), dtype=torch.long, device=device),
        )

    cubes = torch.cat(valid_cubes)
    cases = torch.cat(valid_cases)
    occupancy = occupancy_patterns[cases]

    edge_vertices = cubes[:, cube_edges]
    edge_vertices = edge_vertices.sort(dim=-1).values
    crossing = (
        occupancy[:, cube_edges[:, 0]] != occupancy[:, cube_edges[:, 1]]
    )
    flat_edges = edge_vertices.reshape(-1, 2)
    flat_crossing = crossing.reshape(-1)
    crossing_edges = flat_edges[flat_crossing]

    vertex_count = resolution ** 3
    edge_keys = crossing_edges[:, 0] * vertex_count + crossing_edges[:, 1]
    unique_keys, crossing_to_unique = torch.unique(
        edge_keys, sorted=True, return_inverse=True
    )
    unique_edges = torch.stack(
        (unique_keys // vertex_count, unique_keys % vertex_count), dim=1
    )

    cube_edge_map = torch.full(
        (flat_edges.shape[0],), -1, dtype=torch.long, device=device
    )
    cube_edge_map[flat_crossing] = crossing_to_unique
    cube_edge_map = cube_edge_map.reshape(-1, 12)

    table_edges = triangle_table[cases, :15].reshape(-1, 5, 3)
    triangle_exists = table_edges[:, :, 0] >= 0
    mapped = cube_edge_map.gather(
        1, table_edges.clamp_min(0).reshape(-1, 15)
    ).reshape(-1, 5, 3)
    faces = mapped[triangle_exists]
    if faces.numel() == 0:
        return (
            torch.empty((0, 3), dtype=udf.dtype, device=device),
            torch.empty((0, 3), dtype=torch.long, device=device),
        )
    if (faces < 0).any():
        raise RuntimeError("triangle table referenced a non-crossing cube edge")

    vertices = _interpolate_edges(unique_edges, grids_coords, udf_flat, device)
    vertices, faces = _deduplicate_and_order_vertices(vertices, faces)
    return vertices, faces


def custom_marching_cube(
    grids_coords: Array,
    grids_udf: Array,
    grids_udf_grad: Array,
    voxel_size: Union[float, torch.Tensor],
    resolution: int,
    triangle_table: Sequence[Sequence[int]],
    *,
    device: Optional[Union[str, torch.device]] = None,
    cube_batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy-compatible wrapper matching GeoUDF's original public API."""
    vertices, faces = custom_marching_cube_torch(
        grids_coords,
        grids_udf,
        grids_udf_grad,
        voxel_size,
        resolution,
        triangle_table,
        device=device,
        cube_batch_size=cube_batch_size,
    )
    return vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
