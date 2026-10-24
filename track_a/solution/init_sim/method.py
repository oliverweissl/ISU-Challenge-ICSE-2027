"""SDEdit-style initialisation from the simulator image on top of the (hint-fused) Z-Image ControlNet.

The sim image is VAE-encoded, noised to the sigma of the first used Turbo step (flow matching:
x_t = (1 - sigma) * x0 + sigma * eps) and only the remaining Turbo steps are run, with the ControlNet hints active.
"""
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hint_fusion import FusionConfig, ZImageControlHintFusion  # noqa: E402


class _PipeProxy:
    """Forwards everything to the pipeline; injects the init latents / truncated schedule into __call__."""

    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, k):
        return getattr(self._owner._real_pipe, k)

    def __setattr__(self, k, v):
        setattr(self._owner._real_pipe, k, v)

    def __call__(self, *a, **kw):
        o = self._owner
        pipe = o._real_pipe
        if o._init is None:
            return pipe(*a, **kw)
        sch, skip = pipe.scheduler, o._init["skip"]
        orig = sch.set_timesteps
        prev_term = sch.config.shift_terminal
        sch.config.shift_terminal = o._init["terminal"]

        def trunc(*aa, **kk):
            orig(*aa, **kk)
            sch.timesteps = sch.timesteps[skip:]
            sch.sigmas = sch.sigmas[skip:]
            sch.num_inference_steps = len(sch.timesteps)

        sch.set_timesteps = trunc
        try:
            return pipe(*a, latents=o._init["latents"], **kw)
        finally:
            del sch.set_timesteps
            sch.config.shift_terminal = prev_term


class ZImageInitSim(ZImageControlHintFusion):
    def __init__(self, cfg: FusionConfig | None = None, **kw):
        super().__init__(cfg, **kw)
        self._real_pipe = self.pipe
        self._init = None
        self.pipe = _PipeProxy(self)

    def _sigmas(self, sample, steps, terminal):
        """Full Turbo sigma schedule (same recipe as the pipeline) for this resolution."""
        from videox_fun.pipeline.pipeline_z_image_control import calculate_shift, retrieve_timesteps
        p = self._real_pipe
        cfg = p.scheduler.config
        n = (sample[0] // 16) * (sample[1] // 16)
        mu = calculate_shift(n, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        p.scheduler.sigma_min = 0.0
        prev = cfg.shift_terminal
        cfg.shift_terminal = terminal
        try:
            retrieve_timesteps(p.scheduler, steps, self.device, mu=mu)
        finally:
            cfg.shift_terminal = prev
        return p.scheduler.sigmas.float().cpu()

    @torch.no_grad()
    def translate(self, control, prompt, size, scale=0.8, steps=8, seed=0, seg=None, sim=None, strength=None,
                  schedule="clean"):
        """sim: PIL sim image; strength s in (0, 1): start at the Turbo step whose sigma is nearest to s (None = baseline).
        schedule: 'clean' = scheduler default (shift_terminal None); 'inherit' = whatever the pipeline mutated the
        scheduler config to (the 2-step warm-up leaves shift_terminal=2/3, which squeezes 8 steps into sigma 1..0.67)."""
        self._init, self.last_sigma = None, None
        if sim is not None and strength is not None and strength < 1.0:
            w, h = size
            sample = [round(h / 16) * 16, round(w / 16) * 16]
            terminal = None if schedule == "clean" else self._real_pipe.scheduler.config.shift_terminal
            sig = self._sigmas(sample, steps, terminal)[:-1]
            skip = int((sig[:-1] - strength).abs().argmin())  # nearest Turbo sigma (last step 0.33/0 excluded)
            s0 = float(sig[skip])
            p = self._real_pipe
            x = p.image_processor.preprocess(sim.convert("RGB"), height=sample[0], width=sample[1])
            x = x.to(dtype=p.text_encoder.dtype, device=self.device)
            z = p.vae.encode(x)[0].mode()
            z = ((z - p.vae.config.shift_factor) * p.vae.config.scaling_factor).float()
            eps = torch.randn(z.shape, generator=torch.Generator(device=self.device).manual_seed(seed + 10_000),
                              device=self.device, dtype=torch.float32)
            self._init = {"skip": skip, "latents": (1 - s0) * z + s0 * eps,
                          "terminal": terminal}
            self.last_sigma = s0
        try:
            return super().translate(control, prompt, size, scale=scale, steps=steps, seed=seed, seg=seg)
        finally:
            self._init = None
