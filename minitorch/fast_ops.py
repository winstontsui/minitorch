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
    from .tensor_data import Shape, Storage, Strides, Index


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
    func: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function.

    Optimizations:
    * Parallelizes the main loop.
    * Uses NumPy buffers for index calculations.
    * Avoids indexing when input and output are stride-aligned.

    Args:
    ----
        func: A function mapping floats-to-floats to apply to each element.

    Returns:
    -------
        Tensor map function.

    """

    def _apply_map(
        output: Storage,
        output_shape: Shape,
        output_strides: Strides,
        input_storage: Storage,
        input_shape: Shape,
        input_strides: Strides,
    ) -> None:
        # Check for stride alignment
        if (
            len(output_strides) != len(input_strides)
            or (output_strides != input_strides).any()
            or (output_shape != input_shape).any()
        ):
            # Use indexing when stride alignment fails
            for idx in prange(len(output)):
                output_index = np.zeros(MAX_DIMS, np.int16)
                input_index = np.zeros(MAX_DIMS, np.int16)
                to_index(idx, output_shape, output_index)
                broadcast_index(output_index, output_shape, input_shape, input_index)
                output_pos = index_to_position(output_index, output_strides)
                input_pos = index_to_position(input_index, input_strides)
                output[output_pos] = func(input_storage[input_pos])
        else:
            # Direct computation when stride alignment holds
            for idx in prange(len(output)):
                output[idx] = func(input_storage[idx])

    return njit(_apply_map, parallel=True)


def tensor_zip(
    func: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function.

    Optimizations:
    * Parallelizes the main loop.
    * Uses NumPy buffers for index calculations.
    * Avoids indexing when stride alignment holds.

    Args:
    ----
        func: A function combining two floats into one to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _apply_zip(
        output: Storage,
        output_shape: Shape,
        output_strides: Strides,
        storage_a: Storage,
        shape_a: Shape,
        strides_a: Strides,
        storage_b: Storage,
        shape_b: Shape,
        strides_b: Strides,
    ) -> None:
        # Check for stride alignment
        if (
            len(output_strides) != len(strides_a)
            or len(output_strides) != len(strides_b)
            or (output_strides != strides_a).any()
            or (output_strides != strides_b).any()
            or (output_shape != shape_a).any()
            or (output_shape != shape_b).any()
        ):
            # Use indexing when stride alignment fails
            for idx in prange(len(output)):
                output_index = np.zeros(MAX_DIMS, np.int32)
                a_index = np.zeros(MAX_DIMS, np.int32)
                b_index = np.zeros(MAX_DIMS, np.int32)
                to_index(idx, output_shape, output_index)
                output_pos = index_to_position(output_index, output_strides)
                broadcast_index(output_index, output_shape, shape_a, a_index)
                a_pos = index_to_position(a_index, strides_a)
                broadcast_index(output_index, output_shape, shape_b, b_index)
                b_pos = index_to_position(b_index, strides_b)
                output[output_pos] = func(storage_a[a_pos], storage_b[b_pos])
        else:
            # Direct computation when stride alignment holds
            for idx in prange(len(output)):
                output[idx] = func(storage_a[idx], storage_b[idx])

    return njit(_apply_zip, parallel=True)


def tensor_reduce(
    func: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function.

    Optimizations:
    * Parallelizes the main loop.
    * Uses NumPy buffers for index calculations.
    * Inner loop avoids global writes or function calls.

    Args:
    ----
        func: A reduction function combining two floats.

    Returns:
    -------
        Tensor reduce function.

    """

    def _apply_reduce(
        output: Storage,
        output_shape: Shape,
        output_strides: Strides,
        input_storage: Storage,
        input_shape: Shape,
        input_strides: Strides,
        reduce_axis: int,
    ) -> None:
        for idx in prange(len(output)):
            output_index = np.empty(MAX_DIMS, np.int32)
            size = input_shape[reduce_axis]
            to_index(idx, output_shape, output_index)
            output_pos = index_to_position(output_index, output_strides)
            input_pos = index_to_position(output_index, input_strides)
            acc = output[output_pos]
            stride = input_strides[reduce_axis]
            for step in range(size):
                acc = func(acc, input_storage[input_pos])
                input_pos += stride
            output[output_pos] = acc

    return njit(_apply_reduce, parallel=True)


def _tensor_matrix_multiply(
    output: Storage,
    output_shape: Shape,
    output_strides: Strides,
    storage_a: Storage,
    shape_a: Shape,
    strides_a: Strides,
    storage_b: Storage,
    shape_b: Shape,
    strides_b: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Optimizations:
    * Parallelizes the outer loop.
    * Avoids index buffers and global writes in the inner loop.

    Args:
    ----
        output (Storage): Storage for output tensor.
        output_shape (Shape): Shape of output tensor.
        output_strides (Strides): Strides for output tensor.
        storage_a (Storage): Storage for tensor `A`.
        shape_a (Shape): Shape for tensor `A`.
        strides_a (Strides): Strides for tensor `A`.
        storage_b (Storage): Storage for tensor `B`.
        shape_b (Shape): Shape for tensor `B`.
        strides_b (Strides): Strides for tensor `B`.

    Returns:
    -------
        None : Fills in `output`.

    """
    a_batch_stride = strides_a[0] if shape_a[0] > 1 else 0
    b_batch_stride = strides_b[0] if shape_b[0] > 1 else 0

    for batch in prange(output_shape[0]):
        for row in prange(output_shape[1]):
            for col in prange(output_shape[2]):
                a_offset = batch * a_batch_stride + row * strides_a[1]
                b_offset = batch * b_batch_stride + col * strides_b[2]
                accumulator = 0.0
                for step in range(shape_a[2]):
                    accumulator += storage_a[a_offset] * storage_b[b_offset]
                    a_offset += strides_a[2]
                    b_offset += strides_b[1]
                output_position = (
                    batch * output_strides[0]
                    + row * output_strides[1]
                    + col * output_strides[2]
                )
                output[output_position] = accumulator


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
