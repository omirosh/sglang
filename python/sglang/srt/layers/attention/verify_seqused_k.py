"""One seq_lens+add buffer per forward for the vattn_asm plan cache.

The plan cache keys on data_ptr. `seq_lens + max_q_len` allocates a new tensor
per layer, so the plan kernel relaunches on every layer.
"""

import torch


def fill_stable_seqused_k(buf, seq_lens: torch.Tensor, add: int):
    n = seq_lens.shape[0]
    if (
        buf is None
        or buf.device != seq_lens.device
        or buf.dtype != seq_lens.dtype
        or buf.shape[0] < n
    ):
        buf = torch.empty(n, dtype=seq_lens.dtype, device=seq_lens.device)
    out = buf[:n]
    out.copy_(seq_lens)
    out.add_(add)
    return buf, out


if __name__ == "__main__":
    seq = torch.tensor([3, 8, 1], dtype=torch.int32)
    buf, first = fill_stable_seqused_k(None, seq, 4)
    buf, second = fill_stable_seqused_k(buf, seq, 4)
    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(second, torch.tensor([7, 12, 5], dtype=torch.int32))
    seq2 = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    buf, third = fill_stable_seqused_k(buf, seq2, 4)
    assert third.shape[0] == 4
    assert torch.equal(third, torch.tensor([5, 5, 5, 5], dtype=torch.int32))
    print("ok")
