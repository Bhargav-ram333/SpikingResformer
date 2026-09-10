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
import json
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
    def forward(
        self,
        x: torch.Tensor,
        concept_targets: torch.Tensor = None,
        concept_dropout_prob: float = 0.0,
    ):
        """
        Args:
            x : [B, C, H, W] image batch (NOT time-expanded — backbone handles that)
            concept_targets : optional [B, n_concepts] binary ground-truth concept
                labels. Only ever read when concept_dropout_prob > 0 AND the module
                is in training mode (self.training) -- ignored in every other case.
                Every existing eval script (anec5_gap_test.py, calibration_*.py,
                intervention_consistency.py, energy_accounting.py) calls
                model(imgs) with no extra arguments and model.eval() is always set
                first, so this change is a strict no-op for all of them.
            concept_dropout_prob : per-(example, concept) probability of replacing
                the model's own predicted concept score with the ground-truth
                value before it reaches the classification head only -- i.e.
                "concept dropout" / simulated intervention during training. This
                is the documented fix for the ICRC train/inference distribution
                mismatch (Koh et al. 2020 follow-up literature): the head is
                trained on a mix of its own noisy predictions and clean ground
                truth, instead of only ever its own predictions, so a full
                ground-truth substitution at test time is no longer a total
                surprise to it.
        Returns:
            concept_scores : [B, n_concepts]  (Sigmoid probabilities -- always the
                model's own prediction; concept dropout never touches this value,
                only what the classification head receives, below)
            class_logits   : [B, n_classes]
        """
        # Backbone forward (fills hook buffers, backbone itself is frozen)
        functional.reset_net(self.backbone)
        with torch.no_grad():
            self.backbone(x)

        feats          = self._get_features()      # [B, backbone_dim]
        concept_scores = self.cbl(feats)           # [B, n_concepts]

        head_input = concept_scores
        if self.training and concept_dropout_prob > 0.0 and concept_targets is not None:
            mask = torch.rand_like(concept_scores) < concept_dropout_prob
            head_input = torch.where(mask, concept_targets.float(), concept_scores)

        class_logits = self.head(head_input)        # [B, n_classes]
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

    # ---- Display-only calibrated concept output ------------------------------
    # These methods provide Platt-calibrated concept probabilities as an
    # ADDITIONAL, opt-in output.  They do NOT touch forward(), the
    # classification head, or any training path.  Callers that want
    # calibrated concepts call get_calibrated_concepts() on the raw
    # concept_scores that forward() already returns.
    #
    # WHY display-only:  the classification head was trained on raw,
    # uncalibrated sigmoid scores.  Inserting calibration before the head
    # would change the input distribution it was trained on, risking
    # silent accuracy degradation.  This was never tested and must not
    # happen -- see the user-request notes in the commit that added this.

    def load_calibration(self, readout_or_path: str = None) -> None:
        """Load per-concept Platt (a, b) arrays from JSON and register as buffers.

        Args:
            readout_or_path: Path to calibration_params_<readout>.json, or a
                readout name (e.g. 'pre_reset_vmem', 'learned_decoder'), or None
                (defaults to self.readout_type).

        After calling this, get_calibrated_concepts() becomes usable.
        The buffers travel with .to(device), appear in state_dict(), and
        are never included in any gradient computation.
        """
        import os
        if readout_or_path is None:
            readout_or_path = self.readout_type

        # Resolve path
        if os.path.isfile(readout_or_path):
            json_path = readout_or_path
        else:
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "evaluation_results", f"calibration_params_{readout_or_path}.json"),
                os.path.join("evaluation_results", f"calibration_params_{readout_or_path}.json"),
                os.path.join("cbm_checkpoints", f"calibration_params_{readout_or_path}.json"),
            ]
            json_path = None
            for c in candidates:
                if os.path.isfile(c):
                    json_path = c
                    break
            if json_path is None:
                raise FileNotFoundError(
                    f"Could not find calibration params for '{readout_or_path}'. "
                    f"Checked: {candidates}"
                )

        with open(json_path, encoding="utf-8") as f:
            params = json.load(f)

        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        a = torch.tensor(params["per_concept"]["a"], dtype=torch.float32, device=device)
        b = torch.tensor(params["per_concept"]["b"], dtype=torch.float32, device=device)

        # register_buffer makes these part of the module's state (move with
        # .to(device), saved/loaded with state_dict) but NOT parameters
        # (never receive gradients, never returned by .parameters()).
        self.register_buffer("_platt_a", a)
        self.register_buffer("_platt_b", b)

    @property
    def has_calibration(self) -> bool:
        """True if load_calibration() has been called successfully."""
        return hasattr(self, "_platt_a") and self._platt_a is not None

    def get_calibrated_concepts(self, concept_scores: torch.Tensor) -> torch.Tensor:
        """Apply per-concept Platt scaling to raw concept scores (display-only).

        Args:
            concept_scores: [B, n_concepts] raw sigmoid probabilities from
                forward()'s first return value.

        Returns:
            calibrated: [B, n_concepts] Platt-calibrated probabilities.
                Computed as sigmoid(a * logit(concept_scores) + b) where
                (a, b) are the per-concept Platt parameters loaded by
                load_calibration().

        Raises:
            RuntimeError: if load_calibration() has not been called.
        """
        if not self.has_calibration:
            raise RuntimeError(
                "Calibration parameters not loaded. "
                "Call model.load_calibration(json_path) first."
            )

        # Recover pre-sigmoid logits from the sigmoid concept scores.
        # torch.logit is the inverse of torch.sigmoid: logit(p) = log(p/(1-p))
        # Clamp to avoid log(0) at extremes.
        raw_logits = torch.logit(concept_scores.clamp(1e-6, 1 - 1e-6))

        # Per-concept Platt transform: sigmoid(a_c * logit_c + b_c)
        # Ensure buffers match input device
        a = self._platt_a.to(raw_logits.device)
        b = self._platt_b.to(raw_logits.device)
        calibrated = torch.sigmoid(a * raw_logits + b)
        return calibrated
