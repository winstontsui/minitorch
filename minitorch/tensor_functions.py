"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)  # type: ignore
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Neg function.

        Args:
        ----
            ctx : Context
                The context object that can be used to store information for backward computation.
            t1 : Tensor
                The input tensor to be negated.

        Returns:
        -------
            Tensor
                A new tensor with the negated values of the input tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Neg function.

        Args:
        ----
            ctx : Context
            The context object that can be used to store information for backward computation.
            grad_output : Tensor
            The gradient of the loss with respect to the output of the Neg function.

        Returns:
        -------
            Tensor
            The gradient of the loss with respect to the input of the Neg function.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the inv_map function.

        Args:
        ----
            ctx : The context object that can be used to store information for backward computation.
            t1 : The input tensor to be transformed using the inv_map function.

        Returns:
        -------
            Tensor
            A new tensor resulting from applying the inv_map function to the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Inv function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the Inv function.

        Returns:
        -------
            Tensor
                The gradient of the loss with respect to the input of the Inv function.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward pass for the Add function.

        Args:
        ----
            ctx : The context object that can be used to store information for backward computation.
            t1 : The first input tensor to be added.
            t2 : The second input tensor to be added.

        Returns:
        -------
            Tensor
                A new tensor resulting from adding the two input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the Add function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the Add function.

        Returns:
        -------
            Tuple[Tensor, Tensor]
                A tuple containing the gradients of the loss with respect to the inputs of the Add function.

        """
        return grad_output, grad_output


# class All(Function):
#     @staticmethod
#     def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
#         """Return 1 if all are true"""
#         if dim is not None:
#             return a.f.mul_reduce(a, int(dim.item()))
#         else:
#             return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Forward pass for the Mul function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for forward computation.
            a: Tensor
                The first input tensor.
            b: Tensor
                The second input tensor.

        Returns:
        -------
            Tensor
                A new tensor resulting from multiplying the two input tensors.

        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the Mul function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the Mul function.

        Returns:
        -------
            Tuple[Tensor, Tensor]
                A tuple containing the gradients of the loss with respect to the inputs of the Mul function.

        """
        (a, b) = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


# old:
# class Sigmoid(Function):
#     @staticmethod
#     def forward(ctx: Context, a: Tensor) -> Tensor:
#         ctx.save_for_backward(a)
#         return a.f.sigmoid_map(a)


#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tensor:
#         a, = ctx.saved_values
#         return grad_output * a.f.sigmoid_map(a) * (1 - a.f.sigmoid_map(a))
# more efficient
class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Sigmoid function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            t1: Tensor
                The input tensor to the Sigmoid function.

        Returns:
        -------
            Tensor: The output tensor resulting from applying the Sigmoid function to the input tensor.

        """
        sigma = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sigma)
        return sigma

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Sigmoid function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the Sigmoid function.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the Sigmoid function.

        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Forward pass for the ReLU function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            a: Tensor
                The input tensor to the ReLU function.

        Returns:
        -------
            Tensor: The output tensor resulting from applying the ReLU function to the input tensor.

        """
        ctx.save_for_backward(a)
        return a.f.relu_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the ReLU function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the ReLU function.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the ReLU function.

        """
        (a,) = ctx.saved_values
        # return grad_output * a.f.relu_back_zip(a, grad_output)
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Log function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            t1: Tensor
                The input tensor to the Log function.

        Returns:
        -------
            Tensor: The output tensor resulting from applying the Log function to the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Log function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the Log function.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the Log function.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward pass for the Exp function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            t1: Tensor
                The input tensor to the Exp function.

        Returns:
        -------
            Tensor: The output tensor resulting from applying the Exp function to the input tensor.

        """
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward pass for the Exp function.

        Args:
        ----
            ctx: Context
                The context object that can be used to store information for backward computation.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the Exp function.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the Exp function.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.mul_zip(t1, grad_output)


# sum with dim argument TODO: Check if correct old:
# class Sum(Function):
#     @staticmethod
#     def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
#         if dim is None:
#             ctx.save_for_backward(a)
#             return a.f.add_reduce(a, -1)  # Use -1 for full reduction
#         else:
#             dim_val = int(dim.item())
#             ctx.save_for_backward(a, dim)  # Save dim, not dim_val
#             return a.f.add_reduce(a, dim_val)


#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
#         saved_values = ctx.saved_values
#         if len(saved_values) == 1:
#             # Full reduction case
#             a, = saved_values
#             ones = zeros(a.shape) + 1
#             grad_input = grad_output * ones
#             return (grad_input,)  # Note the comma to make it a tuple
#         else:
#             # Reduction along specific dimension
#             a, dim = saved_values
#             ones = zeros(a.shape) + 1
#             grad_input = grad_output * ones
#             return (grad_input, zeros((1,)))
# new:
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the sum of a tensor along a specified dimension.

        Args:
        ----
            ctx: Context
                The context object to save information for backward computation.
            a: Tensor
                The input tensor.
            dim: Tensor
                The dimension along which to sum.

        Returns:
        -------
            Tensor: The sum of the input tensor along the specified dimension.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the loss with respect to the input tensors.

        Args:
        ----
            ctx: Context
                The context object containing saved tensors from the forward pass.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient with respect to the input tensor and the dimension.

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise less than comparison between two tensors.

        Args:
        ----
            ctx: Context
                The context object to save information for backward computation.
            a: Tensor
                The first input tensor.
            b: Tensor
                The second input tensor.

        Returns:
        -------
            Tensor: A tensor of boolean values where True indicates a < b.

        """
        ctx.save_for_backward(a, b)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the loss with respect to the input tensors.

        Args:
        ----
            ctx: Context
                The context object containing saved tensors from the forward pass.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the input tensors.

        """
        a, b = ctx.saved_values
        a_zeros = zeros(a.shape)
        b_zeros = zeros(b.shape)
        return (a_zeros, b_zeros)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise equality comparison between two tensors.

        Args:
        ----
            ctx: Context
                The context object to save information for backward computation.
            a: Tensor
                The first input tensor.
            b: Tensor
                The second input tensor.

        Returns:
        -------
            Tensor: A tensor of boolean values where True indicates a == b.

        """
        ctx.save_for_backward(a, b)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the loss with respect to the input tensors.

        Args:
        ----
            ctx: Context
                The context object containing saved tensors from the forward pass.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradients with respect to the input tensors.

        """
        a, b = ctx.saved_values
        a_zeros = zeros(a.shape)
        b_zeros = zeros(b.shape)
        return (a_zeros, b_zeros)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise closeness comparison between two tensors.

        Args:
        ----
            ctx: Context
                The context object to save information for backward computation.
            a: Tensor
                The first input tensor.
            b: Tensor
                The second input tensor.

        Returns:
        -------
            Tensor: A tensor of boolean values where True indicates a is close to b.

        """
        ctx.save_for_backward(a, b)
        return a.f.is_close_zip(a, b)


# TODO: review this
class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dims: Tensor) -> Tensor:
        """Permute the dimensions of a tensor according to the given dimensions.

        Args:
        ----
            ctx: Context
                The context object to save information for backward computation.
            a: Tensor
                The input tensor to be permuted.
            dims: Tensor
                The tensor specifying the permutation of dimensions.

        Returns:
        -------
            Tensor: The permuted tensor.

        """
        ctx.save_for_backward(dims)
        dims_tuple = tuple(int(dims[i]) for i in range(dims.shape[0]))  # type: ignore
        return minitorch.Tensor.make(
            a._tensor._storage,
            tuple(a.shape[d] for d in dims_tuple),
            tuple(a._tensor.strides[d] for d in dims_tuple),
            backend=a.backend,
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the loss with respect to the input tensors.

        Args:
        ----
            ctx: Context
                The context object containing saved tensors from the forward pass.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple[Tensor, float]: The gradients with respect to the input tensors.

        """
        (dims,) = ctx.saved_values
        # Calculate the inverse permutation
        inv_dims = [0] * dims.shape[0]
        for i in range(dims.shape[0]):
            inv_dims[int(dims[i])] = i
        inv_dims_tuple = tuple(inv_dims)
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage,
                tuple(grad_output.shape[d] for d in inv_dims_tuple),
                tuple(grad_output._tensor.strides[d] for d in inv_dims_tuple),
                backend=grad_output.backend,
            ),
            0.0,
        )


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """View a tensor with a new shape.

        Args:
        ----
            ctx: Context
                The context object to save information for backward computation.
            a: Tensor
                The input tensor to be viewed.
            shape: Tensor
                The tensor specifying the new shape.

        Returns:
        -------
            Tensor: The tensor viewed with the new shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = tuple(int(shape[i]) for i in range(shape.size))  # type: ignore
        assert (
            a._tensor.size == operators.prod(list(shape2))
        ), f"New shape {shape2} must have same number of elements as original shape {a.shape}"
        return minitorch.Tensor.make(a._tensor._storage, shape2, backend=a.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the loss with respect to the input tensors.

        Args:
        ----
            ctx: Context
                The context object containing saved tensors from the forward pass.
            grad_output: Tensor
                The gradient of the loss with respect to the output of the forward pass.

        Returns:
        -------
            Tuple[Tensor, float]: The gradients with respect to the input tensors.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))  # type: ignore
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(list(shape))), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(list(shape))))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the gradient of a function using central difference method.

    This function calculates the gradient of a given function `f` at a specific argument `arg` and index `ind` using the central difference method. It does this by perturbing the value at the specified index by a small amount `epsilon`, computing the difference in the function's output, and then dividing by the perturbation amount.

    Args:
    ----
        f: The function to compute the gradient of.
        *vals: The tensors to pass to the function.
        arg: The index of the tensor to perturb.
        epsilon: The amount to perturb the value by.
        ind: The index within the tensor to perturb.

    Returns:
    -------
        float: The computed gradient value.

    """
    x = vals[arg]  # type: ignore
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)  # type: ignore


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
