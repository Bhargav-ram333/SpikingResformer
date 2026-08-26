"""
models/cbm.py  --  Concept Bottleneck Model on top of SpikingResformer

Architecture:
    SpikingResformerBackbone (frozen)
        |
        v
    LIF Hook (layers.2.6.down.0)  -- captures V_mem or spike features
        |
        v
    ConceptBottleneckLayer         -- Linear(1536 -> n_concepts) + Sigmoid
        |
        v
    ClassificationHead             -- Linear(n_concepts -> n_classes)

readout_type:
    "pre_reset_vmem"   - continuous membrane potential before fire+reset
                         (FAILED the Week-4 go/no-go gate: lost to spike_rate,
                         Cohen's d=-0.49, paired-t p=1.09e-6 -- see gate_result.json)
    "post_reset_vmem"  - membrane potential after hard reset
    "spike_rate"       - discrete firing rate, T-mean (current fallback baseline)
    "learned_decoder"  - GRU over the raw per-timestep spike sequence (PRD's own
                         mandated Phase-0 no-go pivot -- see decoder_readout.py).
                         Unlike the other three, this one has trainable parameters
                         and must be included in trainable_parameters()/optimizer.
"""

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from models.decoder_readout import TemporalDecoderReadout


# ---- LIF hook ----------------------------------------------------------------

def install_vmem_hook(lif_mod):
    """
    Monkey-patch a LIF node's multi_step_forward to store:
      lif_mod._pre_reset_v_seq   [T, B, C, H, W]   pre-reset membrane potential
      lif_mod._post_reset_v_seq  [T, B, C, H, W]   post-reset membrane potential
      lif_mod._spike_seq         [T, B, C, H, W]   spike train
    The patched forward is identical to the original charge/fire/reset equations.
    """
    lif_mod._pre_reset_v_seq  = None
    lif_mod._post_reset_v_seq = None
    lif_mod._spike_seq        = None

    def _patched_msf(self, x_seq: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x_seq[0], self.v)

        tau   = float(self.tau)
        v_thr = float(self.v_threshold)
        v_rst = float(self.v_reset) if self.v_reset is not None else None

        pre_list, post_list, spk_list = [], [], []
        for t in range(x_seq.shape[0]):
            xt = x_seq[t]
            # Charge
            H = (self.v + (xt - (self.v - v_rst)) / tau) if v_rst is not None \
                else (self.v + (xt - self.v) / tau)
            pre_list.append(H.detach().clone())

            # Fire
            spike = (H >= v_thr).to(x_seq.dtype)
            spk_list.append(spike)

            # Reset
            self.v = (v_rst * spike + (1. - spike) * H) if v_rst is not None \
                     else (H - spike * v_thr)
            post_list.append(self.v.detach().clone())

        self._pre_reset_v_seq  = torch.stack(pre_list,  dim=0)
        self._post_reset_v_seq = torch.stack(post_list, dim=0)
        self._spike_seq        = torch.stack(spk_list,  dim=0)
        return self._spike_seq

    lif_mod.multi_step_forward = types.MethodType(_patched_msf, lif_mod)


def _pool_temporal_mean(t5d: torch.Tensor) -> torch.Tensor:
    """[T, B, C, H, W] -> [B, C]  via T-mean then spatial GAP."""
    return t5d.mean(dim=0).mean(dim=(-2, -1))


# ---- CBL Module --------------------------------------------------------------

class ConceptBottleneckLayer(nn.Module):
    """
    Linear layer projecting backbone features to n_concepts concept scores.
    Output is passed through Sigmoid to produce [0,1] per-concept probabilities.
    """
    def __init__(self, in_features: int, n_concepts: int):
        super().__init__()
        self.linear = nn.Linear(in_features, n_concepts, bias=True)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="sigmoid")
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_features] -> concept_scores: [B, n_concepts]"""
        return torch.sigmoid(self.linear(x))


# ---- Classification Head -----------------------------------------------------

class ClassificationHead(nn.Module):
    """
    Linear layer mapping concept scores to class logits.
    Deliberately kept simple so the concept scores are the sole
    information bottleneck (strict CBM interpretation).
    """
    def __init__(self, n_concepts: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_concepts, n_classes, bias=True)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, concept_scores: torch.Tensor) -> torch.Tensor:
        """concept_scores: [B, n_concepts] -> logits: [B, n_classes]"""
        return self.linear(concept_scores)


# ---- Full CBM ----------------------------------------------------------------

class SpikingResformerCBM(nn.Module):
    """
    Frozen SpikingResformer backbone + Concept Bottleneck Layer + Classification Head.

    Args:
        backbone         : Pre-loaded, already-eval SpikingResformer model (will be frozen).
        n_concepts       : Number of concept attributes (112 for CUB-200-2011).
        n_classes        : Number of output classes (200 for CUB species).
        readout_type     : One of "pre_reset_vmem" | "post_reset_vmem" | "spike_rate".
        backbone_dim     : Channel dimension of the hooked LIF layer (default 1536 for Ti).
        target_layer_key : Named-module key of the LIF node to hook.
        lambda_concept   : Weight for concept BCE loss.
        lambda_task      : Weight for classification CE loss.
    """

    READOUT_TYPES = ("pre_reset_vmem", "post_reset_vmem", "spike_rate", "learned_decoder")

    def __init__(
        self,
        backbone,
        n_concepts: int       = 112,
        n_classes: int        = 200,
        readout_type: str     = "pre_reset_vmem",
        backbone_dim: int     = 1536,
        target_layer_key: str = "layers.2.6.down.0",
        lambda_concept: float = 1.0,
        lambda_task: float    = 1.0,
    ):
        super().__init__()

        assert readout_type in self.READOUT_TYPES, \
            f"readout_type must be one of {self.READOUT_TYPES}, got '{readout_type}'"

        self.readout_type   = readout_type
        self.lambda_concept = lambda_concept
        self.lambda_task    = lambda_task

        # ---- Backbone (frozen) -----------------------------------------------
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        # Switch to torch backend to avoid CuPy dependency
        for m in self.backbone.modules():
            if hasattr(m, "backend"):
                m.backend = "torch"

        # ---- Hook target LIF layer -------------------------------------------
        named = dict(self.backbone.named_modules())
        assert target_layer_key in named, \
            f"Layer '{target_layer_key}' not found in backbone. " \
            f"Available keys (partial): {list(named.keys())[:10]}"
        self._hooked_lif = named[target_layer_key]
        install_vmem_hook(self._hooked_lif)

        # ---- Trainable modules -----------------------------------------------
        # The decoder readout is itself a trainable module (GRU + proj) unlike
        # the other three fixed-rule readouts; only instantiate it when selected,
        # so the other readout_types don't pay for unused parameters.
        self.decoder = TemporalDecoderReadout(channels=backbone_dim) \
            if readout_type == "learned_decoder" else None

        self.cbl  = ConceptBottleneckLayer(backbone_dim, n_concepts)
        self.head = ClassificationHead(n_concepts, n_classes)

    # ---- Internal: pull features from hook -----------------------------------
    def _get_features(self) -> torch.Tensor:
        """Returns [B, backbone_dim] based on chosen readout_type."""
        lif = self._hooked_lif
        if self.readout_type == "pre_reset_vmem":
            return _pool_temporal_mean(lif._pre_reset_v_seq)
        elif self.readout_type == "post_reset_vmem":
            return _pool_temporal_mean(lif._post_reset_v_seq)
        elif self.readout_type == "learned_decoder":
            return self.decoder(lif._spike_seq)
        else:  # spike_rate
            return _pool_temporal_mean(lif._spike_seq)

    # ---- Forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [B, C, H, W] image batch (NOT time-expanded — backbone handles that)
        Returns:
            concept_scores : [B, n_concepts]  (Sigmoid probabilities)
            class_logits   : [B, n_classes]
        """
        # Backbone forward (fills hook buffers, backbone itself is frozen)
        functional.reset_net(self.backbone)
        with torch.no_grad():
            self.backbone(x)

        feats          = self._get_features()      # [B, backbone_dim]
        concept_scores = self.cbl(feats)           # [B, n_concepts]
        class_logits   = self.head(concept_scores) # [B, n_classes]
        return concept_scores, class_logits

    # ---- Loss ----------------------------------------------------------------
    def compute_loss(
        self,
        concept_scores: torch.Tensor,    # [B, n_concepts]
        class_logits:   torch.Tensor,    # [B, n_classes]
        concept_targets: torch.Tensor,   # [B, n_concepts]  binary {0, 1}
        class_targets:   torch.Tensor,   # [B]              integer class IDs
    ):
        """
        Joint loss = lambda_concept * L_concept + lambda_task * L_task

        L_concept: mean binary cross-entropy over all concepts
        L_task   : cross-entropy for class prediction
        """
        L_concept = F.binary_cross_entropy(
            concept_scores, concept_targets.float(), reduction="mean"
        )
        L_task = F.cross_entropy(class_logits, class_targets.long(), reduction="mean")

        L_total = self.lambda_concept * L_concept + self.lambda_task * L_task
        return L_total, L_concept, L_task

    # ---- Convenience ---------------------------------------------------------
    def trainable_parameters(self):
        """CBL + head parameters, plus the decoder's GRU/proj when it's the active readout."""
        params = list(self.cbl.parameters()) + list(self.head.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        return params

    def summary(self):
        cbl_params     = sum(p.numel() for p in self.cbl.parameters())
        head_params    = sum(p.numel() for p in self.head.parameters())
        bb_params      = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters()) if self.decoder is not None else 0
        print("=" * 60)
        print("SpikingResformerCBM Summary")
        print("=" * 60)
        print(f"  Readout type        : {self.readout_type}")
        print(f"  Lambda concept      : {self.lambda_concept}")
        print(f"  Lambda task         : {self.lambda_task}")
        print(f"  Backbone params     : {bb_params:,}  [FROZEN]")
        if self.decoder is not None:
            print(f"  Decoder params      : {decoder_params:,}  [trainable]")
        print(f"  CBL params          : {cbl_params:,}  [trainable]")
        print(f"  Head params         : {head_params:,}  [trainable]")
        print(f"  Total trainable     : {cbl_params + head_params + decoder_params:,}")
        print("=" * 60)
