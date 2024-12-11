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
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            # print("zip as a tensor", fn, a, b, out)
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

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
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
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

        # print("start matrix_multiply", out.shape, a.shape, b.shape, a, b)
        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
        # print("end matrix_multiply", out, a, b, a.shape, b.shape)

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
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
        if np.array_equal(out_shape, in_shape) and np.array_equal(
            out_strides, in_strides
        ):
            for out_pos in prange(len(out)):
                out[out_pos] = fn(in_storage[out_pos])
            return

        for outi in prange(len(out)):
            out_index = np.zeros(
                len(out_shape), dtype=np.int32
            )  # will be automatically hoisted
            in_index = np.zeros(len(in_shape), dtype=np.int32)
            to_index(outi, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            if out_pos != outi:
                print("ERROR!", out_pos, outi)
            # assert out_pos == i
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            out[out_pos] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
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
        a_need_shape_broadcast = not np.array_equal(
            a_shape, out_shape
        ) or not np.array_equal(a_strides, out_strides)
        b_need_shape_broadcast = not np.array_equal(
            b_shape, out_shape
        ) or not np.array_equal(b_strides, out_strides)
        # print("_zip", out_shape, out_strides, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides)
        for outi in prange(len(out)):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)
            b_index = np.zeros(len(b_shape), dtype=np.int32)
            to_index(outi, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            if out_pos != outi:
                print("ERROR!", out_pos, outi)
            a_pos = b_pos = out_pos
            if a_need_shape_broadcast or b_need_shape_broadcast:
                if a_need_shape_broadcast:
                    broadcast_index(out_index, out_shape, a_shape, a_index)
                    a_pos = index_to_position(a_index, a_strides)
                if b_need_shape_broadcast:
                    broadcast_index(out_index, out_shape, b_shape, b_index)
                    b_pos = index_to_position(b_index, b_strides)
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
            # print("zip, out_pos", out_pos, a_pos, b_pos, out[out_pos], a_storage[a_pos], b_storage[b_pos], out_shape, a_shape, b_shape, out_index, a_index, b_index)

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        assert (
            len(out_shape) == len(a_shape)
            and np.array_equal(out_shape[:reduce_dim], a_shape[:reduce_dim])
            and np.array_equal(out_shape[reduce_dim + 1 :], a_shape[reduce_dim + 1 :])
        )
        selected_dim_shape = a_shape[reduce_dim]
        selected_dim_stride = a_strides[reduce_dim]
        for outi in prange(len(out)):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            to_index(outi, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)
            if out_pos != outi:
                print("ERROR!", out_pos, outi)
            a_pos_start = index_to_position(out_index, a_strides)
            v = out[out_pos]
            for a_pos in range(
                a_pos_start,
                a_pos_start + selected_dim_shape * selected_dim_stride,
                selected_dim_stride,
            ):
                # assert a_pos < len(a_storage) # !!!!!!!!!!!!numba do not allow assert or other exit point within prange
                # print("reduce, out_pos", out_pos, a_pos, out_index, out_shape, a_shape)
                v = fn(v, a_storage[a_pos])
            out[out_pos] = v

    return njit(parallel=True)(_reduce)  # type: ignore


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
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns None : Fills in `out`
    """
    assert out_shape[-2] == a_shape[-2] and out_shape[-1] == b_shape[-1]
    # MxN, NxD
    shape_n = a_shape[-1]
    a_need_broadcast = not np.array_equal(
        out_shape[:-2], a_shape[:-2]
    )  # do not need to handle strides since we are copying indexx
    b_need_broadcast = not np.array_equal(
        out_shape[:-2], b_shape[:-2]
    )  # do not need to handle strides since we are copying indexx
    a_key_stride = a_strides[-1]
    b_key_stride = b_strides[-2]
    out_shape_pseudo_a = np.zeros(len(out_shape), dtype=np.int32)
    out_shape_pseudo_a[:-2] = out_shape[:-2]
    out_shape_pseudo_a[-2:] = a_shape[-2:]
    # np.concatenate(out_shape[:-2], a_shape[-2:])
    out_shape_pseudo_b = np.zeros(len(out_shape), dtype=np.int32)
    out_shape_pseudo_b[:-2] = out_shape[:-2]
    out_shape_pseudo_b[-2:] = b_shape[-2:]

    for outi in prange(len(out)):
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(out_shape), dtype=np.int32)
        b_index = np.zeros(len(out_shape), dtype=np.int32)
        to_index(outi, out_shape, out_index)
        out_pos = index_to_position(out_index, out_strides)
        if out_pos != outi:
            print("ERROR!", out_pos, outi)

        # calculate a_start_pos
        out_index_cache = out_index[-1]
        out_index[-1] = 0
        if a_need_broadcast:
            broadcast_index(out_index, out_shape_pseudo_a, a_shape, a_index)
        else:
            a_index[:] = out_index
        a_start_pos = index_to_position(a_index, a_strides)
        out_index[-1] = out_index_cache

        # calculate b_start_pos
        out_index_cache = out_index[-2]
        out_index[-2] = 0
        if b_need_broadcast:
            broadcast_index(out_index, out_shape_pseudo_b, b_shape, b_index)
        else:
            b_index[:] = out_index
        b_start_pos = index_to_position(b_index, b_strides)
        out_index[-2] = out_index_cache

        # calculate line x column
        v = 0.0
        for a_pos, b_pos in zip(
            range(a_start_pos, a_start_pos + shape_n * a_key_stride, a_key_stride),
            range(b_start_pos, b_start_pos + shape_n * b_key_stride, b_key_stride),
        ):
            # print("out_pos", out_pos, a_pos, b_pos, a_start_pos, b_start_pos, out_index, a_index, b_index, a_storage[a_pos] * b_storage[b_pos], a_storage[a_pos], b_storage[b_pos], v,  v + a_storage[a_pos] * b_storage[b_pos])
            v += a_storage[a_pos] * b_storage[b_pos]
        out[out_pos] = v


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)