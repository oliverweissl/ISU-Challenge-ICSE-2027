# init_sim: SDEdit-style initialisation from the sim image

Idea: instead of starting the Turbo sampler from pure noise, VAE-encode the sim image, noise it to the sigma of the first
used step (flow matching `x_t = (1-sigma) x0 + sigma eps`) and run only the remaining Turbo sigmas, with the
Union-ControlNet hints (reused `hint_fusion` wrapper) active throughout. Latent initialisation is independent of the
33-channel inpaint path (which stays blank).

Config (`ZImageInitSim.translate`): `sim` (PIL), `strength` s (None/1 = baseline behaviour; otherwise the Turbo step
whose sigma is nearest to s is the first one used), `schedule` ("clean" = scheduler default, shift_terminal None;
"inherit" = leave the pipeline-mutated scheduler config). Turbo sigmas (8 steps): 1, .947, .882, .8, .692, .545, .333, 0
so s=0.85 -> 0.882 (5 steps run), 0.7 -> 0.692 (4), 0.5 -> 0.545 (3).

Run: `python run_experiment.py --name is_s0.7_60 --method init_sim --init-strength 0.7 --fusion-mode sum --prompt label --n 60 --eval`
(single canny: add `--fusion-config '{"modalities":["canny"],"mode":"sum"}'`). Smoke: `python init_sim/smoke.py`.

Note: the 2-step warm-up in run_experiment mutates `scheduler.config.shift_terminal` to 2/3, so all baseline/hint_fusion
8-step runs actually use sigmas 1..0.667 then 0; init_sim overrides this to the clean schedule while it runs.
