"""One seq_lens+add buffer per forward for the vattn_asm plan cache.

The plan cache keys on data_ptr. `seq_lens + max_q_len` allocates a new tensor
per layer, so the plan kernel relaunches on every layer.
"""

import torch


class StableSequsedK:
    def __init__(self, max_bs: int):
        self.max_bs = max_bs
        # Never reallocated: captured graphs keep writing to these addresses.
        self.bufs = {}
        self.out = None
        self.filled_in_capture = None

    def reset(self):
        self.filled_in_capture = None

    def get(self, seq_lens: torch.Tensor, add: int, capturing: bool) -> torch.Tensor:
        # Graph capture runs eager warmups first. A fill from a warmup is not
        # recorded, so the captured forward has to fill again.
        if self.filled_in_capture == capturing:
            return self.out
        buf = self.bufs.get(seq_lens.dtype)
        if buf is None:
            buf = torch.empty(self.max_bs, dtype=seq_lens.dtype, device=seq_lens.device)
            self.bufs[seq_lens.dtype] = buf
        self.out = torch.add(seq_lens, add, out=buf[: seq_lens.shape[0]])
        self.filled_in_capture = capturing
        return self.out


if __name__ == "__main__":
    s = StableSequsedK(8)
    seq = torch.tensor([3, 8, 1], dtype=torch.int32)
    first = s.get(seq, 4, capturing=False)
    second = s.get(seq, 4, capturing=False)
    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(second, torch.tensor([7, 12, 5], dtype=torch.int32))

    # Warmup then capture after one reset: the capture pass must refill.
    s.reset()
    s.get(seq, 4, capturing=False)
    seq.fill_(10)
    captured = s.get(seq, 4, capturing=True)
    assert torch.equal(captured, torch.tensor([14, 14, 14], dtype=torch.int32))
    assert s.get(seq, 4, capturing=True) is captured

    # A different batch size reuses the same storage.
    s.reset()
    third = s.get(torch.ones(5, dtype=torch.int32), 4, capturing=False)
    assert third.data_ptr() == first.data_ptr()
    assert torch.equal(third, torch.full((5,), 5, dtype=torch.int32))
    print("ok")
