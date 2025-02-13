from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

from numba import njit as _njit, prange
import numpy as np

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    MAX_DIMS,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator for JIT compiling functions with NUMBA."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# inv = njit(inv)
# inv_back  = njit(inv_back)

to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # ASSIGN2.2:
        if (
            (len(in_shape) == len(out_shape))  # Same number of dimensions.
            and (in_shape == out_shape).all()  # Shapes match.
            and (in_strides == out_strides).all()  # Strides are aligned.
        ):
            # Directly apply the function to all elements in parallel.
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            # General case: Perform element-wise mapping with indexing.
            for i in prange(len(out)):
                # Temporary index buffers for input and output tensors.
                in_idx = in_shape.copy()
                out_idx = out_shape.copy()
                # Convert flat index `i` to multidimensional index.
                to_index(i, out_shape, out_idx)
                # Map output index to corresponding input index.
                broadcast_index(out_idx, out_shape, in_shape, in_idx)
                # Compute and store the result.
                out[i] = fn(in_storage[index_to_position(in_idx, in_strides)])

    # Compile the function with Numba for parallel execution.
    return njit(_map, parallel=True)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_size = len(out)  # Total number of elements in the output tensor.

        # Check if input and output tensors are aligned.
        if (
            (
                len(a_shape) == len(b_shape)
            )  # Both inputs have the same number of dimensions.
            and (a_shape == b_shape).all()  # Shapes match.
            and (a_strides == b_strides).all()  # Strides are aligned.
            and (b_strides == out_strides).all()  # Output strides also match.
        ):
            # Directly apply the function to all elements in parallel.
            for i in prange(out_size):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # General case: Perform element-wise zipping with indexing.
            for i in prange(out_size):
                out_index = np.empty(len(out_shape), np.int32)
                a_index = np.empty(len(a_shape), np.int32)
                b_index = np.empty(len(b_shape), np.int32)
                # Convert flat index `i` to multidimensional index.
                to_index(i, out_shape, out_index)
                # Map output index to corresponding input indices.
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                # Compute and store the result.
                a_pos = index_to_position(a_index, a_strides)
                b_pos = index_to_position(b_index, b_strides)
                out[i] = fn(a_storage[a_pos], b_storage[b_pos])

    # Compile the function with Numba for parallel execution.
    return njit(_zip, parallel=True)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. Optimized for parallel execution."""

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # reduce_size = a_shape[reduce_dim]  # Size of the reduction dimension.
        # Iterate over all elements in the output tensor in parallel
        for i in prange(len(out)):
            out_index = np.empty(MAX_DIMS, np.int32)  # Thread-local index buffer
            size = a_shape[reduce_dim]  # Size of the reduction dimension

            # Convert flat index `i` to multidimensional index.
            to_index(i, out_shape, out_index)
            # Position in the output storage.
            o = index_to_position(out_index, out_strides)

            # Initialize accumulator and compute the initial position in input storage
            acc = out[o]
            j = index_to_position(out_index, a_strides)
            step = a_strides[reduce_dim]

            # Reduce across dimension, update the index for the reduction dimension.
            for s in range(size):
                acc = fn(acc, a_storage[j])
                j += step

            out[o] = acc  # Write the final accumulated value

    # Compile the function with Numba for parallel execution.
    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None: Fills in `out`.

    """
    # batch_size, out_rows, out_cols = out_shape
    # a_rows, a_cols = a_shape[-2], a_shape[-1]
    # b_rows, b_cols = b_shape[-2], b_shape[-1]

    # # Ensure matrix dimensions match for multiplication
    # assert a_cols == b_rows

    # for n in prange(batch_size):  # Parallelize over batches
    #     for i in range(out_rows):  # Iterate over rows of the output
    #         for j in range(out_cols):  # Iterate over columns of the output
    #             out_index = n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
    #             sum_value = 0.0  # Local variable to accumulate the dot product

    #             for k in range(a_cols):  # Iterate over the inner dimension
    #                 a_index = n * a_strides[0] + i * a_strides[1] + k * a_strides[2]
    #                 b_index = n * b_strides[0] + k * b_strides[1] + j * b_strides[2]

    #                 sum_value += a_storage[a_index] * b_storage[b_index]

    #             out[out_index] = sum_value  # Write the result to the output tensor
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for x in prange(out_shape[0]):
        for y in prange(out_shape[1]):
            for z in prange(out_shape[2]):
                val = 0.0
                posA = x * a_batch_stride + y * a_strides[1]
                posB = x * b_batch_stride + z * b_strides[2]
                for a in range(a_shape[2]):
                    val += a_storage[posA] * b_storage[posB]
                    posA += a_strides[2]
                    posB += b_strides[1]
                outPos = x * out_strides[0] + y * out_strides[1] + z * out_strides[2]
                out[outPos] = val


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
