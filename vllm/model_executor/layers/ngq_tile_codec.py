# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-granular Triton decoder for NGQ2 `qtip-tile16` n-gram tables (LongCat-Flash-Lite release).

Table layout: 16-row blocks; block = 16 tiles (16 dims each) x 64 B; a tile is one QTIP bitshift-trellis
sequence (L=16, k=4, tail-biting) of 128 steps x 2 values laid out row-major inside the 16x16 tile.
Row r of a block uses steps 8r..8r+7 of every tile, which depend on nibbles 8r-3..8r+7 (cyclic), so a row
needs 16 x 6 bytes of codes (+ its bf16 norm) rather than the whole 1 KB block.
Decode (authors' lc_decode_tiles): q = bf16(lut[state] * W) as fp32; row = bf16(q / |q| * norm).
"""
import torch

from vllm.triton_utils import tl, triton

TLUT_BITS = 9
L = 16


def hyb_lut(tlut: torch.Tensor) -> torch.Tensor:
    """QTIP `quantlut_sym` (HYB) expansion: base table [512, 2] -> [2^16, 2] fp32."""
    tlut = tlut.float()
    s = torch.arange(1 << L, device=tlut.device, dtype=torch.int64)
    u = (s + 1) * s
    sflp = 1 - ((u >> 15) & 1) * 2
    idx = (u >> (16 - TLUT_BITS - 1)) & ((1 << TLUT_BITS) - 1)
    lut = tlut[idx].clone()
    lut[:, 0] = lut[:, 0] * sflp
    return lut.contiguous()


@triton.jit
def _nib_at(sym_ptr, blk_base, tile, step):
    s = (step + 128) % 128
    byte = tl.load(sym_ptr + blk_base + tile * 64 + (s >> 1)).to(tl.int32)
    return tl.where((s & 1) == 0, byte >> 4, byte & 15)


@triton.jit
def _tile_row_kernel(sym_ptr, gain_ptr, lut_ptr, blk_ptr, r_ptr, gidx_ptr, w_ptr, out_ptr, n_rows,
                     NT: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    if pid < n_rows:
        blk = tl.load(blk_ptr + pid).to(tl.int64)
        r = tl.load(r_ptr + pid).to(tl.int32)
        gidx = tl.load(gidx_ptr + pid).to(tl.int64)
        W = tl.load(w_ptr + pid).to(tl.float32)
        tiles = tl.arange(0, NT)[:, None]
        steps = r * 8 + tl.arange(0, NS)[None, :]
        base = blk * (NT * 64)
        n0 = _nib_at(sym_ptr, base, tiles, steps - 3)
        n1 = _nib_at(sym_ptr, base, tiles, steps - 2)
        n2 = _nib_at(sym_ptr, base, tiles, steps - 1)
        n3 = _nib_at(sym_ptr, base, tiles, steps)
        st = (n0 << 12) | (n1 << 8) | (n2 << 4) | n3
        v0 = tl.load(lut_ptr + st.to(tl.int64) * 2)
        v1 = tl.load(lut_ptr + st.to(tl.int64) * 2 + 1)
        q0 = (v0 * W).to(tl.bfloat16).to(tl.float32)
        q1 = (v1 * W).to(tl.bfloat16).to(tl.float32)
        ss = tl.sum(tl.sum(q0 * q0 + q1 * q1, axis=1), axis=0)
        nrm = tl.maximum(tl.sqrt(ss), 1e-12)
        g = tl.load(gain_ptr + gidx).to(tl.float32)
        col = tiles * 16 + tl.arange(0, NS)[None, :] * 2
        o = out_ptr + pid.to(tl.int64) * (NT * 16)
        tl.store(o + col, (q0 / nrm * g).to(tl.bfloat16))
        tl.store(o + col + 1, (q1 / nrm * g).to(tl.bfloat16))


def decode_tile_rows(sym_flat: torch.Tensor, gain: torch.Tensor, lut: torch.Tensor, blk: torch.Tensor,
                     r: torch.Tensor, gidx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """-> bf16 [N, 256]. `sym_flat` (uint8, blocks x 1024 B) and `gain` (bf16) may be UVA views of pinned host memory."""
    n = blk.numel()
    out = torch.empty(n, 256, dtype=torch.bfloat16, device=blk.device)
    if n:
        _tile_row_kernel[(n,)](sym_flat, gain, lut, blk, r, gidx, w, out, n, NT=16, NS=8, num_warps=1)
    return out
