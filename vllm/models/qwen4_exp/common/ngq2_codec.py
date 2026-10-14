# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NGQ2 (row-granular 2-bit trellis) record decoder for Qwen4Exp PLE n-gram tables.

Imported by common/ngram_embedding.py (Qwen4ExpPLENgq2EmbeddingMethod): device tables use the fused
gather+decode kernel (``gather_decode``), pinned-host tables decode gathered records (``decode_records``).
Kernels are bit-identical to the torch reference decoder in the engram-quantization research repo
(vllm-port/scripts/ngq2_codec.py).
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

REC_BYTES = 42
T_STEPS = 80
DIM = 160


@triton.jit
def _nib(rec_ptr, base, i, mask):
    """4-bit symbol i (0..79) of the record at byte offset `base`: high nibble of even i, low nibble of odd i."""
    b = tl.load(rec_ptr + base + (i >> 1), mask=mask, other=0).to(tl.uint32)
    return tl.where((i & 1) == 0, b >> 4, b & 0xF)


@triton.jit
def _decode_rows(rec_ptr, rec_base, out_ptr, out_base, tlut_ptr, rmask, lmask,
                 T: tl.constexpr, TP: tl.constexpr, REC: tl.constexpr):
    # rec_base / out_base: [ROWS] int64 byte / element offsets of each row; rmask: [ROWS] rows to write;
    # lmask: [ROWS] rows whose record is read (rows with rmask & ~lmask are written as zeros)
    t = tl.arange(0, TP)
    ld = rmask & lmask
    m2 = ld[:, None] & (t < T)[None, :]
    st = rmask[:, None] & (t < T)[None, :]
    base = rec_base[:, None]
    tt = t[None, :]
    n0 = _nib(rec_ptr, base, (tt + T - 3) % T, m2)
    n1 = _nib(rec_ptr, base, (tt + T - 2) % T, m2)
    n2 = _nib(rec_ptr, base, (tt + T - 1) % T, m2)
    n3 = _nib(rec_ptr, base, tt, m2)
    s = (n0 << 12) | (n1 << 8) | (n2 << 4) | n3                      # uint32 state
    u = (s + 1) * s                                                   # wraps mod 2^32 like QTIP's uint32 kernel
    sg = tl.where(((u >> 15) & 1) == 1, -1.0, 1.0)
    idx = ((u >> 6) & 511).to(tl.int32)
    g_lo = tl.load(rec_ptr + rec_base + 40, mask=ld, other=0).to(tl.uint32)
    g_hi = tl.load(rec_ptr + rec_base + 41, mask=ld, other=0).to(tl.uint32)
    g = ((g_hi << 24) | (g_lo << 16)).to(tl.float32, bitcast=True)   # bf16 -> f32 by placing in the high half
    v0 = tl.load(tlut_ptr + idx * 2, mask=m2, other=0.0) * sg * g[:, None]
    v1 = tl.load(tlut_ptr + idx * 2 + 1, mask=m2, other=0.0) * g[:, None]
    v0 = tl.where(m2, v0, 0.0)
    v1 = tl.where(m2, v1, 0.0)
    o = out_base[:, None] + 2 * tt
    tl.store(out_ptr + o, v0.to(tl.bfloat16), mask=st)
    tl.store(out_ptr + o + 1, v1.to(tl.bfloat16), mask=st)


@triton.jit
def _decode_kernel(rec_ptr, out_ptr, tlut_ptr, n_rows,
                   T: tl.constexpr, TP: tl.constexpr, REC: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = r < n_rows
    _decode_rows(rec_ptr, r.to(tl.int64) * REC, out_ptr, r.to(tl.int64) * D, tlut_ptr, rmask, rmask, T, TP, REC)


@triton.jit
def _gather_decode_kernel(ids_ptr, table_ptr, out_ptr, tlut_ptr, n_ids, vocab_start, vocab_end,
                          T: tl.constexpr, TP: tl.constexpr, REC: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr):
    # One launch per lookup: row id -> 42 B record of the local shard -> bf16 row. Ids outside
    # [vocab_start, vocab_end) belong to another TP/ETP shard and produce an all-zero row.
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = r < n_ids
    gid = tl.load(ids_ptr + r, mask=rmask, other=0).to(tl.int64)
    owned = (gid >= vocab_start) & (gid < vocab_end)
    local = tl.where(owned, gid - vocab_start, 0)
    _decode_rows(table_ptr, local * REC, out_ptr, r.to(tl.int64) * D, tlut_ptr, rmask, owned, T, TP, REC)


def decode_records(records: torch.Tensor, tlut: torch.Tensor, rows_per_prog: int = 4) -> torch.Tensor:
    """records uint8 [N, 42] (CUDA or UVA view) -> bf16 [N, 160]."""
    assert records.dtype == torch.uint8 and records.shape[-1] == REC_BYTES
    rec = records.reshape(-1, REC_BYTES)
    if not rec.is_contiguous():
        rec = rec.contiguous()
    n = rec.shape[0]
    out = torch.empty(n, DIM, dtype=torch.bfloat16, device=tlut.device)
    if n:
        _decode_kernel[(triton.cdiv(n, rows_per_prog),)](rec, out, tlut, n, T=T_STEPS, TP=128, REC=REC_BYTES,
                                                          D=DIM, ROWS=rows_per_prog, num_warps=4)
    return out.view(*records.shape[:-1], DIM)


def gather_decode(table: torch.Tensor, ids: torch.Tensor, tlut: torch.Tensor, vocab_start: int = 0,
                  vocab_end: int | None = None, rows_per_prog: int = 4) -> torch.Tensor:
    """Fused lookup: table uint8 [R, 42] (local shard; CUDA or UVA view), ids int[...] global row ids
    -> bf16 [..., 160]. Row ``vocab_start + i`` lives at table row ``i``; ids outside
    ``[vocab_start, vocab_end)`` (default end: vocab_start + R) decode to zeros."""
    assert table.dtype == torch.uint8 and table.dim() == 2 and table.shape[1] == REC_BYTES
    assert table.is_contiguous()
    if vocab_end is None:
        vocab_end = vocab_start + table.shape[0]
    flat = ids.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    n = flat.numel()
    out = torch.empty(n, DIM, dtype=torch.bfloat16, device=flat.device)
    if n:
        _gather_decode_kernel[(triton.cdiv(n, rows_per_prog),)](flat, table, out, tlut, n, vocab_start, vocab_end,
                                                                 T=T_STEPS, TP=128, REC=REC_BYTES, D=DIM,
                                                                 ROWS=rows_per_prog, num_warps=4)
    return out.view(*ids.shape, DIM)
