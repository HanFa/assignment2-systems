import math
from typing import Any

import torch
import triton
import triton.language as tl
from einops import rearrange, einsum
from triton.language import make_block_ptr


@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,  # input pointers
        O_ptr, L_ptr,  # output pointers
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # inputs
    q_block_ptr = make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    k_block_ptr = make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    v_block_ptr = make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    # outputs
    o_block_ptr = make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    l_block_ptr = make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # buffer
    o_i = tl.zeros(shape=(Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full(shape=(Q_TILE_SIZE,), value=-float('inf'), dtype=tl.float32)

    q_i = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')

    k_base_offset = tl.arange(0, K_TILE_SIZE)
    t_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(t_k):
        k_j = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')

        s_ij = tl.dot(q_i, tl.trans(k_j), allow_tf32=False) * scale
        # mask s_ij
        mask = (j * K_TILE_SIZE + k_base_offset) < N_KEYS  # (K_TILE_SIZE,)
        s_ij = tl.where(mask[None, :], s_ij, -float('inf'))

        new_m_i = tl.maximum(m_i, tl.max(s_ij, axis=-1, keep_dims=False))

        p_ij = tl.exp(s_ij - new_m_i[:, None])
        l_i = tl.exp(m_i - new_m_i) * l_i + tl.sum(p_ij, axis=-1, keep_dims=False)
        o_i = tl.exp(m_i - new_m_i)[:, None] * o_i + tl.dot(p_ij, v_j, allow_tf32=False)

        k_block_ptr = k_block_ptr.advance((K_TILE_SIZE, 0))
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))
        m_i = new_m_i

    tl.store(o_block_ptr, o_i / l_i[:, None], boundary_check=(0, 1))
    tl.store(l_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))


class TritonFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> Any:
        leading_shape, seq_len, d_model = q.shape[:-2], q.shape[-2], q.shape[-1]
        assert leading_shape == k.shape[:-2] and leading_shape == v.shape[:-2], "q, k, v must have same leading shape"
        assert d_model == k.shape[-1] and d_model == v.shape[-1]

        q = rearrange(q, "... seq_len d_model -> (...) seq_len d_model")
        k = rearrange(k, "... seq_len d_model -> (...) seq_len d_model")
        v = rearrange(v, "... seq_len d_model -> (...) seq_len d_model")

        collapsed_leading_shape = q.shape[0]
        q_seq_len, k_seq_len = q.shape[-2], k.shape[-2]

        b_q, b_k = 16, 16  # tile size
        grid_calc = lambda meta: (triton.cdiv(meta['N_QUERIES'], meta['Q_TILE_SIZE']), collapsed_leading_shape)

        o = torch.empty_like(q)
        l = torch.empty(size=q.shape[:-1], device=q.device, dtype=q.dtype)
        scale = math.sqrt(d_model) ** -1
        flash_fwd_kernel[grid_calc](
            q, k, v,  # input pointers
            o, l,  # output pointers
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            q_seq_len, k_seq_len,
            scale,
            D=d_model,
            Q_TILE_SIZE=b_q,
            K_TILE_SIZE=b_k
        )

        ctx.save_for_backward(o, l, q, k, v)
        return o.view(*leading_shape, q_seq_len, d_model)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError("not implemented")


class PytorchFlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> Any:
        leading_shape, seq_len, d_model = q.shape[:-2], q.shape[-2], q.shape[-1]

        assert leading_shape == k.shape[:-2] and leading_shape == v.shape[:-2], "q, k, v must have same leading shape"
        assert d_model == k.shape[-1] and d_model == v.shape[-1]

        q = rearrange(q, "... seq_len d_model -> (...) seq_len d_model")
        k = rearrange(k, "... seq_len d_model -> (...) seq_len d_model")
        v = rearrange(v, "... seq_len d_model -> (...) seq_len d_model")

        collapsed_leading_shape = q.shape[0]

        q_seq_len, k_seq_len = q.shape[-2], k.shape[-2]

        b_q, b_k = 16, 16  # tile size
        t_q, t_k = math.ceil(q_seq_len / b_q), math.ceil(k_seq_len / b_k)  # number of query, key tiles

        # output tensors
        o = torch.empty_like(q)
        l = torch.empty(size=q.shape[:-1], device=q.device, dtype=q.dtype)

        for i in range(0, t_q):

            start_b_q, end_b_q = i * b_q, min((i + 1) * b_q, q_seq_len)

            q_i = q[..., start_b_q: end_b_q, :]  # b_q, d_model

            o_i = torch.zeros(size=(collapsed_leading_shape, end_b_q - start_b_q, d_model),
                              device=q.device)  # output tile
            l_i = torch.zeros(size=(collapsed_leading_shape, end_b_q - start_b_q,),
                              device=q.device)  # unnormalized softmax values
            m_i = torch.full(size=(collapsed_leading_shape, end_b_q - start_b_q,),
                             fill_value=-float('inf'), device=q.device)  # running maximum of each query

            for j in range(0, t_k):
                start_b_k, end_b_k = j * b_k, min(k_seq_len, (j + 1) * b_k)

                k_j = k[..., start_b_k: end_b_k, :]  # (..., b_k, d_model)
                v_j = v[..., start_b_k: end_b_k, :]  # (..., b_k, d_model)

                s_ij = einsum(q_i, k_j, "... b_q d_model, ... b_k d_model -> ... b_q b_k") / math.sqrt(d_model)

                new_m_i = torch.maximum(m_i, torch.max(s_ij, dim=-1, keepdim=False).values)  # (..., b_q, )

                p_ij = torch.exp(s_ij - new_m_i.unsqueeze(dim=-1))  # (..., b_q, b_k)
                new_l_i = torch.exp(m_i - new_m_i) * l_i + torch.sum(p_ij, dim=-1, keepdim=False)  # (..., b_q)
                new_o_i = torch.exp(m_i - new_m_i).unsqueeze(dim=-1) * o_i + einsum(p_ij, v_j,
                                                                                    "... b_q b_k, ... b_k d_model -> ... b_q d_model")  # (..., b_q, 1) * (..., b_q, d_model) + (... b_q, d_model) -> (..., b_q, d_model)

                m_i = new_m_i
                l_i = new_l_i
                o_i = new_o_i

            o_i /= l_i.unsqueeze(dim=-1) + 1e-8  # (..., b_q, d_model) / (..., b_q, ) -> (..., b_q, d_model)
            o[..., start_b_q: end_b_q, :] = o_i
            l[..., start_b_q: end_b_q] = m_i + torch.log(l_i)

        ctx.save_for_backward(o, l, q, k, v)

        return o.view(*leading_shape, q_seq_len, d_model)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError("not implemented")


def native_self_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d_model = q.shape[-1]
    s = einsum(q, k, "... q_seq_len d_model, ... k_seq_len d_model -> ... q_seq_len k_seq_len") / math.sqrt(d_model)
    p = torch.softmax(s, dim=-1)
    o = einsum(p, v, "... q_seq_len k_seq_len, ... k_seq_len d_model -> ... q_seq_len d_model")
    return o


if __name__ == '__main__':
    torch.manual_seed(0)

    q = torch.rand(size=(64, 3, 17, 256), device='cuda')
    k = torch.rand(size=(64, 3, 23, 256), device='cuda')
    v = torch.rand(size=(64, 3, 23, 256), device='cuda')

    o_native_attn = native_self_attn(q, k, v)
    o_flash_attn = PytorchFlashAttnFunc.apply(q, k, v)
    o_triton_attn = TritonFlashAttnFunc.apply(q, k, v)

    print(f'The maximum difference between native and pytorch flash attention is '
          f'{torch.max(torch.abs(o_native_attn - o_flash_attn))}')

    print(f'The maximum difference between native and triton flash attention is '
          f'{torch.max(torch.abs(o_native_attn - o_triton_attn))}')
