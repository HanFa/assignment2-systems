import math
from typing import Any

import torch
from einops import rearrange, einsum


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

            o_i = torch.zeros(size=(collapsed_leading_shape, end_b_q - start_b_q, d_model))  # output tile
            l_i = torch.zeros(size=(collapsed_leading_shape, end_b_q - start_b_q,))  # unnormalized softmax values
            m_i = torch.full(size=(collapsed_leading_shape, end_b_q - start_b_q,),
                             fill_value=-float('inf'))  # running maximum of each query

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

    q = torch.rand(size=(64, 3, 17, 256))
    k = torch.rand(size=(64, 3, 23, 256))
    v = torch.rand(size=(64, 3, 23, 256))

    o_native_attn = native_self_attn(q, k, v)
    o_flash_attn = PytorchFlashAttnFunc.apply(q, k, v)

    print(f'The maximum difference between native and flash attention is '
          f'{torch.max(torch.abs(o_native_attn - o_flash_attn))}')
