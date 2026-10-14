# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 Engram tables stored as NGQ2 2-bit trellis records (qtip-row format).

Record per row: DIM/4 bytes of 4-bit trellis symbols (one QTIP bitshift-trellis sequence per row, L=16, k=4,
tail-biting, 2 values per step) + bf16 gain = DIM/4 + 2 bytes (66 B for DIM=256 vs 256 B fp8 + 8 B scales).
Lookup decodes rows in the gather kernel: row = bf16(lut[state_t] * gain), state_t = nib[t-3..t] (16 bits).
Checkpoint: `layers.{i}.engram.embed.weight` uint8 [rows, DIM/4+2] (same head/row order as the original),
`layers.{i}.engram.embed.tlut` fp32 [512, 2]; selected by hf config `engram_table_format == "ngq2"`.
"""
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.nvidia.engram import ParallelEngramEmbedding
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)
TLUT_BITS = 9


def hyb_lut(tlut: torch.Tensor) -> torch.Tensor:
    """QTIP `quantlut_sym` expansion: [512, 2] -> [65536, 2] fp32."""
    tlut = tlut.float()
    s = torch.arange(1 << 16, device=tlut.device, dtype=torch.int64)
    u = (s + 1) * s
    sflp = 1 - ((u >> 15) & 1) * 2
    idx = (u >> (16 - TLUT_BITS - 1)) & ((1 << TLUT_BITS) - 1)
    lut = tlut[idx].clone()
    lut[:, 0] = lut[:, 0] * sflp
    return lut.contiguous()


@triton.jit
def _nib(rec_ptr, base, t, NSTEP: tl.constexpr):
    s = (t + NSTEP) % NSTEP
    b = tl.load(rec_ptr + base + (s >> 1)).to(tl.int32)
    return tl.where((s & 1) == 0, b >> 4, b & 15)


@triton.jit(do_not_specialize=["vocab_start", "vocab_end", "num_rows", "ids_stride_t", "ids_stride_h", "GRID"])
def _engram_ngq2_lookup_kernel(
    rec_ptr, lut_ptr, ids, out, vocab_start, vocab_end, num_rows, ids_stride_t, ids_stride_h,
    HEAD_START: tl.constexpr, LOCAL_HEADS: tl.constexpr, TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr, REC: tl.constexpr, GRID,
):
    """One program per (token, local head) row, persistent grid: gather the row's record, decode DIM values, write bf16.
    Rows of heads this rank does not own (or padded heads) are written as zeros, like the fp8 kernel."""
    NSTEP: tl.constexpr = DIM // 2
    steps = tl.arange(0, NSTEP)
    for row in tl.range(tl.program_id(0), num_rows, GRID):
        head = HEAD_START + row % LOCAL_HEADS
        token = (row // LOCAL_HEADS).to(tl.int64)
        okh = head < TOTAL_HEADS
        index = tl.load(ids + token * ids_stride_t + head * ids_stride_h, mask=okh, other=-1).to(tl.int64)
        owned = okh & (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        base = local * REC
        n0 = _nib(rec_ptr, base, steps - 3, NSTEP)
        n1 = _nib(rec_ptr, base, steps - 2, NSTEP)
        n2 = _nib(rec_ptr, base, steps - 1, NSTEP)
        n3 = _nib(rec_ptr, base, steps, NSTEP)
        st = ((n0 << 12) | (n1 << 8) | (n2 << 4) | n3).to(tl.int64)
        v0 = tl.load(lut_ptr + st * 2)
        v1 = tl.load(lut_ptr + st * 2 + 1)
        g_lo = tl.load(rec_ptr + base + (REC - 2)).to(tl.int32)
        g_hi = tl.load(rec_ptr + base + (REC - 1)).to(tl.int32)
        gain = ((g_hi << 24) | (g_lo << 16)).to(tl.float32, bitcast=True)
        gain = tl.where(owned, gain, 0.0)
        o = out + row.to(tl.int64) * DIM
        tl.store(o + steps * 2, (v0 * gain).to(tl.bfloat16))
        tl.store(o + steps * 2 + 1, (v1 * gain).to(tl.bfloat16))


class Ngq2ParallelEngramEmbedding(ParallelEngramEmbedding):
    """Head-sharded Engram table whose rows are 2-bit trellis records decoded at lookup."""

    def __init__(self, num_embeddings: int, dim: int, head_sizes: tuple[int, ...], block_size: int = 32,
                 cpu_offload: bool = False, dp_shared_memory: bool = False, use_thp: bool = False) -> None:
        if dp_shared_memory or use_thp:
            raise NotImplementedError("ngq2 Engram tables: dp_shared_memory / use_thp not supported")
        self.rec_bytes = dim // 4 + 2
        super().__init__(num_embeddings, dim, head_sizes, block_size, cpu_offload=cpu_offload)
        # the fp8 scale parameter is meaningless here; replace it by the codebook base table
        del self.weight_scale_inv
        self.tlut = nn.Parameter(torch.zeros(1 << TLUT_BITS, 2, dtype=torch.float32), requires_grad=False)
        set_weight_attrs(self.tlut, {"weight_loader": self._tlut_loader})
        self._lut: torch.Tensor | None = None
        self._rec_view: torch.Tensor | None = None
        logger.info("Engram table as ngq2 records: %d rows x %d B (%.2f GiB) %s", self.part_num_embeddings,
                    self.rec_bytes, self.part_num_embeddings * self.rec_bytes / 1024**3,
                    "pinned host" if cpu_offload else "on device")

    @staticmethod
    def _tlut_loader(param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded.float())

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        kw = {"device": "cpu", "pin_memory": True} if self.cpu_offload else {}
        rec = torch.empty(self.part_num_embeddings, self.rec_bytes, dtype=torch.uint8, **kw)
        dummy = torch.empty(1, dtype=torch.uint8)
        return rec, dummy

    def _records(self) -> torch.Tensor:
        if not self.cpu_offload:
            return self.weight.data
        if self._rec_view is None or self._rec_view.data_ptr() == 0:
            self._rec_view = get_accelerator_view_from_cpu_tensor(self.weight.data)
        return self._rec_view

    def lookup(self, indices: torch.Tensor, out: torch.Tensor, background: bool = False) -> None:
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if self._lut is None:
            assert bool(torch.any(self.tlut != 0)), "ngq2 Engram checkpoint is missing its tlut"
            self._lut = hyb_lut(self.tlut.data.to(indices.device))
        grid = min(rows, self._num_sms * 8 if not background else self._num_sms * 4)
        _engram_ngq2_lookup_kernel[(grid,)](
            self._records(), self._lut, indices, out, self.vocab_start_idx, self.vocab_end_idx, rows,
            indices.stride(0), indices.stride(1), HEAD_START=self.head_start, LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols, DIM=self.dim, REC=self.rec_bytes, GRID=grid, num_warps=4,
        )
