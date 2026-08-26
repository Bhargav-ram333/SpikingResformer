"""
models/decoder_readout.py -- Learned Temporal Decoder readout arm.

Implements the PRD's own mandated Phase-0 no-go pivot (PRD sec 2, "Phase 0 exit
criteria": "No-go: no clear separation -> pivot to a learned temporal decoder
(small GRU/attention head over the raw per-timestep spike sequence) as the
readout mechanism, re-run the same three-way probe comparison with the decoder
added as a fourth arm, and re-gate before proceeding.")

This was triggered: gate_result.json shows pre-reset V_mem lost to spike-rate
significantly (Cohen's d = -0.49, paired-t p = 1.09e-6), not just "no clear
separation" -- a real reversal. Per the PRD's own key-risk register, that is
the exact condition that mandates this module before any further investment
in Phase 1.

Unlike the other 3 readouts (spike-rate mean, pre-reset V_mem, post-reset
V_mem), which are all *fixed* extraction rules with zero learned parameters,
this readout is a trainable module: a single-layer GRU consumes the spatially
-pooled, per-timestep binary spike sequence and learns how to weight/aggregate
timesteps, instead of naive averaging (which is exactly what "spike-rate"
already does and already beat V_mem).
"""
import torch
import torch.nn as nn


class TemporalDecoderReadout(nn.Module):
    """
    Input:  spike_seq [T, B, C, H, W]  (binary spike train from the hooked LIF,
                                          same tensor the spike-rate arm reads)
    Output: feature    [B, C]           (drop-in replacement for the other 3
                                          readouts -- same dim as backbone_dim,
                                          so ConceptBottleneckLayer is unchanged)
    """

    def __init__(self, channels: int, hidden: int = 256):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.gru = nn.GRU(input_size=channels, hidden_size=hidden,
                           num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden, channels)

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """spike_seq: [T, B, C, H, W] (raw, un-pooled). Used during real CBM
        training/inference, where only one batch is in memory at a time."""
        assert spike_seq.dim() == 5, f"expected [T,B,C,H,W], got {tuple(spike_seq.shape)}"
        T, B, C, H, W = spike_seq.shape
        assert C == self.channels, f"channel mismatch: module built for {self.channels}, got {C}"

        per_t = spike_seq.mean(dim=(-2, -1))       # [T, B, C]  spatial GAP per timestep
        per_t = per_t.permute(1, 0, 2)              # [B, T, C]  batch_first for GRU
        return self.forward_pooled(per_t)

    def forward_pooled(self, per_t: torch.Tensor) -> torch.Tensor:
        """per_t: [B, T, C], ALREADY spatially pooled -- the GRU/proj step only.

        Exists so a caller that needs to cache many batches (e.g. the go/no-go
        gate script, which reuses the same extracted data across 3 seeds) can
        pool spatially once at extraction time and cache the ~196x-smaller
        [T,B,C] tensor instead of the full [T,B,C,H,W] spike map -- without
        duplicating the pooling math or risking it drifting out of sync with
        forward() above.
        """
        assert per_t.dim() == 3, f"expected [B,T,C], got {tuple(per_t.shape)}"
        assert per_t.shape[-1] == self.channels, \
            f"channel mismatch: module built for {self.channels}, got {per_t.shape[-1]}"
        _, h_n = self.gru(per_t)                    # h_n: [1, B, hidden]
        feat = self.proj(h_n.squeeze(0))              # [B, C]
        return feat
