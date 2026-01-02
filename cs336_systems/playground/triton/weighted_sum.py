from typing import Any

import torch
import triton
import triton.language as tl
from einops import rearrange

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def weighted_sum_fwd(x_ptr, weight_ptr, output_ptr,
                     x_stride_row, x_stride_dim,
                     weight_stride_dim,
                     output_stride_row,
                     ROWS, D,
                     ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        offsets=(0,),
        shape=(D,),
        strides=(weight_stride_dim,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        shape=(ROWS,),
        strides=(output_stride_row,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE, )
        weight = weight[None, :]  # (1, D_TILE_SIZE)
        output += tl.sum(tile * weight,
                         axis=1)  # (ROWS_TILE_SIZE, D_TILE_SIZE) *  (1, D_TILE_SIZE) -> (ROWS_TILE_SIZE, D_TILE_SIZE) -> (ROWS_TILE_SIZE, )

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
        x_ptr, weight_ptr,
        grad_output_ptr,
        grad_x_ptr, partial_grad_weight_ptr,
        stride_x_row, stride_x_dim,
        stride_weight,
        stride_gr,
        stride_gxr, stride_gxd,
        stride_gwb, stride_gwd,
        ROWS, D,
        ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr):
    row_tile_idx = tl.program_id(0)
    assert tl.num_programs(0) == tl.cdiv(ROWS, ROWS_TILE_SIZE)

    # inputs
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(stride_x_row, stride_x_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        offsets=(0,),
        strides=(stride_weight,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    # outputs
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(tl.cdiv(ROWS, ROWS_TILE_SIZE), D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0)
    )

    for idx in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option='zero')  # (ROWS_TILE_SIZE, )
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option='zero')  # (D_TILE_SIZE, )
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')  # (ROWS_TILE_SIZE, D_TILE_SIZE)

        grad_x = grad_output[:, None] * weight[None, :]  # outer product
        tl.store(grad_x_block_ptr, grad_x, boundary_check=(0, 1))

        grad_partial_weight = tl.sum(grad_output[:, None] * x, axis=0, keep_dims=True)  # (1, D_TILE_SIZE)
        tl.store(partial_grad_weight_block_ptr, grad_partial_weight, boundary_check=(0, 1))

        # advance input ptrs
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))

        # advance output ptrs
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        D, output_dims = x.shape[-1], x.shape[:-1]
        x_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)
        ctx.D_TILE_SIZE = triton.next_power_of_2(D)
        ctx.ROW_TILE_SIZE = 16
        ctx.input_shape = x_shape

        y = torch.empty(output_dims, device=x.device, dtype=x.dtype)
        n_rows = y.numel()
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROW_TILE_SIZE),)](
            x, weight, y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.view(-1).stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROW_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE
        )

        return y

    @staticmethod
    def backward(ctx, grad_output) -> Any:
        x, weight = ctx.saved_tensors
        ROW_TILE_SIZE, D_TILE_SIZE = ctx.ROW_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        grad_output = rearrange(grad_output, "... -> (...)").contiguous()

        grad_partial_weight = torch.empty(size=(cdiv(n_rows, ROW_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(cdiv(n_rows, ROW_TILE_SIZE),)](
            x, weight,
            grad_output,
            grad_x, grad_partial_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_output.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            grad_partial_weight.stride(0), grad_partial_weight.stride(1),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROW_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE
        )

        grad_weight = grad_partial_weight.sum(dim=0)

        return grad_x.view(ctx.input_shape), grad_weight


def test_customized_inputs():
    torch.manual_seed(0)
    x = torch.rand((3, 2, 172, 35), device=DEVICE)
    weight = torch.rand((35,), device=DEVICE)

    x_tri = x.detach().clone().requires_grad_(True)
    weight_tri = weight.detach().clone().requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    output_torch = torch.sum(x_ref * weight_ref[None, :], dim=-1)
    output_triton = WeightedSumFunc.apply(x_tri, weight_tri)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_triton))}')

    loss = output_triton.mean()
    loss.backward()

    torch_loss = output_torch.mean()
    torch_loss.backward()

    # Compare Gradients
    grad_x_diff = torch.max(torch.abs(x_tri.grad - x_ref.grad))
    grad_w_diff = torch.max(torch.abs(weight_tri.grad - weight_ref.grad))

    print(f"Grad X Max Diff:      {grad_x_diff:.6f}")
    print(f"Grad Weight Max Diff: {grad_w_diff:.6f}")


def test_autograd():
    x = torch.rand((6, 172, 35), device=DEVICE, requires_grad=True)
    weight = torch.rand((35,), device=DEVICE, requires_grad=True)

    from torch.autograd import gradcheck
    test_passed = gradcheck(WeightedSumFunc.apply, (x, weight), eps=1e-2, atol=1e-2, rtol=1e-2)
    assert test_passed


if __name__ == '__main__':
    test_autograd()
