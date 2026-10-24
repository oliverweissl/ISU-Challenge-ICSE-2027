"""Training-free hint-space fusion on top of Z-Image ControlNet-Union 2.1.

The control branch is run once per modality; the per-layer hints (2 noise-refiner + 15 main-layer hints) are merged
before they are added to the base transformer. Implemented by patching the instance methods that produce hints.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_zimage_cn import ZImageControlBaseline  # noqa: E402
from videox_fun.utils import get_image_latent  # noqa: E402

from .config import FusionConfig  # noqa: E402
from .merge import merge_weighted, project_conflicts  # noqa: E402

DATA = Path(__file__).resolve().parents[3] / "data"
TOKEN = 16  # pixels per transformer token (vae 8 x patch 2)


def load_control(path, seg_path=None) -> Image.Image:
    """Load a control PNG as RGB. 16-bit depth (dataset: bright = far, exterior = noise) is converted to an 8-bit
    near=bright map: exterior pixels (from instance_seg) are set to far, the rest is percentile-stretched.
    PIL's convert('RGB') on the raw 16-bit file saturates to white."""
    im = Image.open(path)
    if not im.mode.startswith("I"):
        return im.convert("RGB")
    d = np.asarray(im, dtype=np.float32) / 65535.0
    ext = np.zeros(d.shape, bool)
    if seg_path is not None:
        seg = np.asarray(Image.open(seg_path).convert("RGB"), dtype=np.int32)
        ext = (np.abs(seg - np.array([8, 63, 122])).sum(-1) == 0)
    v = d[~ext] if (~ext).any() else d.ravel()
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    n[ext] = 1.0
    return Image.fromarray(((1 - n) * 255).astype(np.uint8)).convert("RGB")


class _State:
    t = 0.0
    ctx_extra: list = []  # control contexts of modalities 1.. (modality 0 arrives through the normal path)
    cc: list = []  # per-modality refiner outputs fed to the layer-hint stage
    gate: torch.Tensor | None = None  # (1, Lx, M) per-token modality weights for region_gated
    conflict_frac: list = []
    cos_mean: list = []


class ZImageControlHintFusion(ZImageControlBaseline):
    def __init__(self, cfg: FusionConfig | None = None, **kw):
        super().__init__(**kw)
        self.cfg = cfg or FusionConfig()
        assert self.cfg.mode in ("sum", "mean", "region_gated", "phase_gated", "conflict_aware")
        if self.cfg.mode == "conflict_aware":
            assert len(self.cfg.modalities) == 2
        self.st = _State()
        self._patch()
        self._load_gate_table()

    # ---- gates -------------------------------------------------------------------------------------------
    def _load_gate_table(self):
        c = self.cfg
        cmap_path = Path(c.seg_class_map) if c.seg_class_map else DATA / "instance_seg" / "instance_seg_class_map.json"
        classes = json.loads(Path(cmap_path).read_text())
        table = json.loads(Path(c.gate_table).read_text())
        default = table.get("_default", {m: 1.0 for m in c.modalities})
        self._colors = np.array([k["color_rgb"] for k in classes], dtype=np.int32)
        w = np.array([[table.get(k["name"], default)[m] for m in c.modalities] for k in classes], dtype=np.float32)
        self._class_w = w / w.sum(1, keepdims=True) if c.gate_normalize else w
        self._default_w = np.array([default[m] for m in c.modalities], dtype=np.float32)
        if c.gate_normalize:
            self._default_w /= self._default_w.sum()

    def _token_gate(self, seg: Image.Image, sample_hw):
        H, W = sample_hw
        seg = np.asarray(seg.convert("RGB").resize((W, H), Image.NEAREST), dtype=np.int32)
        dist = np.abs(seg[:, :, None, :] - self._colors[None, None]).sum(-1)  # H W C
        idx, ok = dist.argmin(-1), dist.min(-1) == 0
        pix = np.where(ok[..., None], self._class_w[idx], self._default_w)  # H W M
        pix = torch.from_numpy(pix).permute(2, 0, 1)[None]
        tok = F.avg_pool2d(pix, TOKEN).flatten(2).transpose(1, 2)  # 1 Ltok M
        pad = (-tok.shape[1]) % 32
        if pad:
            tok = torch.cat([tok, torch.from_numpy(self._default_w)[None, None].expand(1, pad, -1)], 1)
        return tok.to(self.device)

    # ---- merging -----------------------------------------------------------------------------------------
    def _weights(self):
        c, M = self.cfg, len(self.cfg.modalities)
        if c.mode == "sum":
            return [c.weights[m] for m in c.modalities]
        if c.mode in ("mean", "conflict_aware"):
            if c.mode == "conflict_aware" and c.conflict_base == "sum":
                return [c.weights[m] for m in c.modalities]
            w = [c.weights[m] for m in c.modalities]
            return [x / sum(w) for x in w]
        if c.mode == "phase_gated":
            t = self.st.t
            if c.phase_shape == "step":
                w = [c.phase_start[m] if t < c.phase_switch else c.phase_end[m] for m in c.modalities]
            else:
                w = [c.phase_start[m] + (c.phase_end[m] - c.phase_start[m]) * t for m in c.modalities]
            if c.phase_normalize:
                s = sum(w) or 1.0
                w = [x / s for x in w]
            return w
        raise ValueError(c.mode)

    def _merge(self, hints_per_mod):
        """hints_per_mod: list over modalities of tuples of (B, L, D) hints. Returns tuple of fused hints."""
        c = self.cfg
        if len(hints_per_mod) == 1:  # plain baseline call through the patched model
            return hints_per_mod[0]
        out = []
        for layer_hints in zip(*hints_per_mod):
            hs = list(layer_hints)
            if c.mode == "region_gated":
                L = hs[0].shape[1]
                g = self.st.gate
                if g is None:
                    g = torch.from_numpy(self._default_w).to(hs[0].device)[None, None].expand(1, 1, -1)
                if g.shape[1] < L:  # caption tokens appended to the unified sequence: use default weights
                    extra = torch.from_numpy(self._default_w).to(g.device)[None, None].expand(1, L - g.shape[1], -1)
                    g = torch.cat([g, extra], 1)
                g = g.to(hs[0].dtype)
                w = [g[..., i:i + 1] for i in range(len(hs))]
            else:
                if c.mode == "conflict_aware":
                    hs, frac, cm = project_conflicts(hs, c.conflict_thresh, c.conflict_strength, c.conflict_center)
                    self.st.conflict_frac.append(frac)
                    self.st.cos_mean.append(cm)
                w = self._weights()
            out.append(merge_weighted(hs, w))
        return tuple(out)

    # ---- patching ----------------------------------------------------------------------------------------
    def _patch(self):
        tr, st = self.pipe.transformer, self.st
        orig_fwd, orig_ref, orig_lay = tr.forward, tr.forward_control_2_0_refiner, tr.forward_control_2_0_layers

        def fwd(x, t, cap_feats, *a, control_context=None, **kw):
            st.t = float(t[0])
            return orig_fwd(x, t, cap_feats, *a, control_context=control_context, **kw)

        def ref(self_, x, cap_feats, control_context, kwargs, **kw):
            ctxs = [control_context] + [e[: len(control_context)] for e in st.ctx_extra]
            hints, st.cc, seqlens = [], [], None
            for ctx in ctxs:
                h, cc, seqlens = orig_ref(x, cap_feats, ctx, kwargs, **kw)
                hints.append(h)
                st.cc.append(cc)
            return self._merge(hints), st.cc[0], seqlens

        def lay(self_, x, cap_feats, control_context, seqlens, kwargs):
            hints = [orig_lay(x, cap_feats, cc, seqlens, kwargs) for cc in st.cc]
            return self._merge(hints)

        tr.forward = fwd
        tr.forward_control_2_0_refiner = types.MethodType(ref, tr)
        tr.forward_control_2_0_layers = types.MethodType(lay, tr)

    # ---- inference ---------------------------------------------------------------------------------------
    def _encode_control(self, img_tensor, sample):
        p = self.pipe
        x = p.image_processor.preprocess(img_tensor, height=sample[0], width=sample[1])
        x = x.to(dtype=p.text_encoder.dtype, device=self.device)
        z = p.vae.encode(x)[0].mode()
        return ((z - p.vae.config.shift_factor) * p.vae.config.scaling_factor).unsqueeze(2)

    @torch.no_grad()
    def translate(self, control, prompt, size, scale=0.8, steps=8, seed=0, seg=None):
        """control: dict modality -> PIL image (all of cfg.modalities). seg: instance_seg PIL (region_gated)."""
        w, h = size
        sample = [round(h / 16) * 16, round(w / 16) * 16]
        mods = self.cfg.modalities
        lat = {m: get_image_latent(control[m], sample_size=sample)[:, :, 0] for m in mods}
        st = self.st
        st.conflict_frac, st.cos_mean = [], []
        st.gate = self._token_gate(seg, sample) if self.cfg.mode == "region_gated" else None
        # modality 0 goes through the pipeline; the others are encoded here and swapped into the 16 control channels
        orig_call = self.pipe.transformer.forward
        extra_lat = [self._encode_control(lat[m], sample) for m in mods[1:]]
        blank = torch.zeros([1, 3, *sample])
        mask = torch.ones([1, 1, *sample]) * 255
        # the pipeline concatenates [control_latent, mask, inpaint_latent]; capture it to build the other contexts
        cap = {}

        def grab(x, t, cap_feats, *a, control_context=None, **kw):
            if "ctx" not in cap:
                cap["ctx"] = control_context
                st.ctx_extra = []
                for el in extra_lat:
                    c2 = control_context.clone()
                    c2[:, : el.shape[1]] = el.to(c2.dtype)
                    st.ctx_extra.append(c2)
            return orig_call(x, t, cap_feats, *a, control_context=control_context, **kw)

        self.pipe.transformer.forward = grab
        try:
            out = self.pipe(
                prompt=prompt, negative_prompt=" ", height=sample[0], width=sample[1],
                generator=torch.Generator(device=self.device).manual_seed(seed),
                guidance_scale=0.0, image=blank, mask_image=mask, control_image=lat[mods[0]],
                num_inference_steps=steps, control_context_scale=scale,
            ).images[0]
        finally:
            self.pipe.transformer.forward = orig_call
            st.ctx_extra = []
        return out.resize((w, h), Image.LANCZOS)
