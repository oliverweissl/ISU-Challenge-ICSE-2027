# Related work and novelty map (Track A)

Date: 2026-10-17. Owner: literature-scout.
Verification level: every paper below was opened via WebFetch of its arXiv abs page (or ICLR page / HF model card).
Almost all are **abstract-level reads only**; mechanism details marked "not in abstract" were NOT verified.
Update 2026-10-17 (second pass): see 'Full-text verification' below for what is now verified-full-text vs still-abstract. ~34 + 32 searches/fetches used.

## Verified papers (opened)

### Spatial control on DiT (token / adapter based)
1. **OminiControl** (Tan et al., 2411.15098) https://arxiv.org/abs/2411.15098
   Reuses DiT's VAE+blocks, ~0.1% extra params, condition tokens concatenated with image tokens, dynamic position encoding.
   Multi-condition, conflict handling, strength control: not in abstract (unverified).
   Overlap: T1 partial (token conditioning on single-stream-like sequence, but one condition at a time as far as abstract says).
2. **EasyControl** (Zhang et al., 2503.07027) https://arxiv.org/abs/2503.07027
   Condition-injection LoRA, position-aware training, causal attention + KV cache; trained on single conditions,
   zero-shot multi-condition composition. Base frozen. Conflict handling / strength knob: not in abstract.
   Overlap: T1 partial (frozen backbone, trained adapter, multi-cond by composition); F1-F4 none (no hint merging).
3. **UniCombine** (Wang et al., 2503.09277) https://arxiv.org/abs/2503.09277
   Conditional MMDiT Attention, training-free and LoRA-trained variants, SubjectSpatial200K; any combination of text/spatial/subject.
   Conflict handling not in abstract. Overlap: T1 partial (multi-cond in one DiT), F1 partial (training-free multi-cond combo, but via attention not hint-space).
4. **Gated Condition Injection without Multimodal Attention** (Liu et al., 2603.27666) https://arxiv.org/abs/2603.27666
   Unified gated conditioning module, dual-path, integrates aligned and non-aligned cues; backbone SANA (linear attention), not a ControlNet-style or S3-DiT model.
   Training, conflict handling not in abstract. Overlap: F2/F3 partial (gating, details unverified), T1 partial.
5. **UNITY** (Das et al., 2606.20971) https://arxiv.org/abs/2606.20971
   Universal-to-specialized adapter, Morphable Attention Flow; stage 1 trains on all modalities jointly (half the steps), stage 2 specializes. Single and composite conditioning.
   Conflict handling and backbone not in abstract. Overlap: T1 partial (shared universal stage across modalities = similar to shared trunk).
6. **PKA / Rethinking Multi-Condition DiTs** (Zhou et al., 2602.06850) https://arxiv.org/abs/2602.06850
   Position-aligned and keyword-scoped attention to remove redundant cross-condition attention (10x speedup); CSAS reweights training toward critical denoising phases.
   No conflict handling stated. Overlap: T1 none-partial (efficiency, not fusion), F3 partial (timestep-sensitivity reweighting in training only).
7. **ReaDiT Guidance** (Mahajan et al., 2609.04649) https://arxiv.org/abs/2609.04649
   Test-time guidance using features of a single DiT block toward depth/pose/edge targets. Multi-cond/conflict not in abstract.
   Overlap: none with our fusion ideas (guidance, not injection). Low threat.
8. **VolDiT** (Seyfarth et al., 2603.25181) https://arxiv.org/abs/2603.25181
   Frozen-backbone DiT, adapter turning masks into control tokens, "timestep-gated". Single condition (segmentation), medical. Details unverified.
   Overlap: F3 partial (timestep gating of control on DiT exists), T1 partial.

### Multi-ControlNet fusion and conflicts
9. **Cross-ControlNet** (Liu et al., ICLR 2026) https://iclr.cc/virtual/2026/poster/10011222
   Training-free fusion of several ControlNet branches: PixFusion (per-pixel, std-map guided), ChannelFusion (consistency-ratio gate), KV-Injection (fg/bg masks from text). Reported to transfer to FLUX.
   No cosine/projection stated in abstract. Overlap: **F1 full-ish** (training-free per-modality hint merge with gates), **F2 partial** (region masks via text attention), F4 partial (conflict-aware merge but gating/selection, not projection; unverified at equation level), F3 none.
10. **MaxFusion** (Nair et al., ECCV 2024, 2404.09977) https://arxiv.org/abs/2404.09977
   Fuses features of separate single-task ControlNet/T2I-Adapter models; variance maps indicate conditioning intensity; average where correlated above threshold else keep higher-variance (details from search snippet, abstract confirms variance-map idea). SD1.5.
   Overlap: **F1 and F4 partial-to-high** (feature-correlation-based conflict-aware merge of per-modality hints is the same family; correlation threshold = a cosine-style test).
11. **Minimal Impact ControlNet** (Sun et al., ICLR 2025, 2506.01672) https://arxiv.org/abs/2506.01672
   Diagnoses "silent control signals" (edge-type controls suppress texture where no edges); balanced dataset, balanced feature injection, Jacobian-asymmetry fix. Trained.
   Overlap: F2 partial (motivates region-aware use of canny), F4 partial (conflict analysis), T1 partial (training-side fix).
12. **Ctrl-Adapter** (Lin et al., 2404.09967) https://arxiv.org/abs/2404.09967
   Adapts existing ControlNets to new backbones incl. DiT (PixArt-alpha); MoE router gives patch-level multi-condition weighting; <10 GPU h.
   Overlap: F1/F2 partial (learned patch-level weights over per-condition ControlNet outputs), T1 partial.
13. **AtteConDA** (Noguchi, 2605.09425) https://arxiv.org/abs/2605.09425 (body read via HTML, first 100k chars)
   Driving-scene generation with edge+depth+seg. Patch-wise Adaptation Module: per-position gate conditioned on timestep embedding and pooled text, straight-through hard selection of one condition per position. SD1.5, Uni-ControlNet init, only local control branch + PAM trained. Modality dropout: not found. Distillation: not found. Strength knob: fixed 1.0. Evaluates image properties, not downstream training. No DiT.
   Overlap: **F2 high** (region-wise per-position condition selection among canny/depth/seg), F3 partial (timestep enters gate input), F4 partial (explicit conflict suppression), D1 partial (synthetic data purpose, but conditions from real driving images), T1 partial (trained multi-cond branch, no dropout/knob/distillation).
14. **UniControl** (Qin et al., NeurIPS 2023, 2305.11147) https://arxiv.org/abs/2305.11147
   One model over 9 condition tasks with task-aware HyperNet. Multi-cond combination and conflicts not in abstract.
   Overlap: T1 partial (shared model across modalities; modality-aware modulation).

### Timestep / gating
15. **RealityBridge** (Wu et al., 2606.16278) https://arxiv.org/abs/2606.16278
   Sim-to-real for 3DGS driving video; lightweight GateNet controls injection of complementary modality signals across backbone blocks; video foundation model, multi-stage training. Details unverified.
   Overlap: F2/F3 partial (learned gating of injection), T1 partial, sim-to-real application relevant to our task.
16. **TC-LoRA** (Cho et al., 2510.09561) https://arxiv.org/abs/2510.09561
   Hypernetwork generates timestep- and condition-specific LoRA on a frozen backbone; single condition.
   Overlap: F3 partial (time-varying control strength, learned).

### Backbone / baseline / data
17. **Z-Image** (Z-Image Team, 2511.22699) https://arxiv.org/abs/2511.22699 - 6B S3-DiT, Turbo distilled. Abstract does not give layer count or step count (unverified here). No control module described.
18. **Z-Image-Turbo-Fun-Controlnet-Union-2.1** (alibaba-pai model card) https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
   ControlNet on 15 layer blocks + 2 refiner blocks; trained from scratch 70k steps on 1M images (2.0) +11k (2.1); 0.10 text dropout (no modality dropout mentioned); one control image at a time; control_context_scale 0.65-1.0 as the only strength knob; 8-step variant distilled from 2.1; acknowledged loss of acceleration/artifacts at high scale.
   Note: the search snippet said "15 blocks"; our brief said "every 2nd layer" - model card says 15 layers + 2 refiner, matches ~every 2nd of 30. No multi-condition mode documented.
19. **Exploring Generative AI for Sim2Real in Driving Data Synthesis** (Zhao et al., 2404.09111) https://arxiv.org/abs/2404.09111
   Compares GAN methods and ControlNet for turning simulator label maps into realistic driving images; ControlNet had fewer artifacts. Overlap: P1/D1 context only.

Search-snippet only (NOT opened, unverified): OminiControl2 (2503.08280, token compression + feature reuse for efficient conditioning), Laytrol (AAAI, preserves pretrained knowledge in MM-DiT layout control), SimGen (driving sim layouts, cascade diffusion), Uni-ControlNet.

## Overlap matrix (none / partial / full; "?" = abstract silent)

| Paper | F1 | F2 | F3 | F4 | T1 | D1 | P1 |
|---|---|---|---|---|---|---|---|
| OminiControl | none | none | none | none | partial | none | none |
| EasyControl | none | none | none | none | partial | none | none |
| UniCombine | partial | none | none | ? | partial | none | none |
| Gated injection (SANA) | none | partial? | partial? | ? | partial | none | none |
| UNITY | none | none | none | ? | partial | none | none |
| PKA | none | none | partial | none | none | none | none |
| Cross-ControlNet | full-ish | partial | none | partial | none | none | none |
| MaxFusion | partial-high | none | none | partial-high | none | none | none |
| Minimal Impact CN | none | partial | none | partial | partial | none | none |
| Ctrl-Adapter | partial | partial | none | none | partial | none | none |
| AtteConDA | partial | high | partial | partial | partial | partial | none |
| UniControl | none | none | none | none | partial | none | none |
| RealityBridge | none | partial | partial | ? | partial | none | none |
| TC-LoRA | none | none | partial | none | none | none | none |
| VolDiT | none | none | partial | none | partial | none | none |
| Z-Image Fun Union 2.1 | none | none | none | none | partial | none | none |

## Novelty table

| id | Closest prior work | Verdict | Our specific difference |
|---|---|---|---|
| F1 hint-space sum/mean fusion | Cross-ControlNet, MaxFusion, Ctrl-Adapter (MoE router), Multi-ControlNet baseline | **done before** (as a method). Novel only as a *port* | Applied to an S3-DiT additive-hint ControlNet-Union (Z-Image) with 8-step Turbo; first-use-of-Union-net-per-modality is a engineering point, not a contribution. Use as baseline, do not claim. |
| F2 region gates from instance_seg | AtteConDA (per-position selection among edge/depth/seg), Cross-ControlNet KV-Injection (text masks), Minimal Impact CN | **incremental / done in spirit** | Gate derived from the *label* instance masks themselves (available by construction in the sim data) rather than learned or text-derived; training-free, DiT, task-specific. Defensible only as a practical trick on thin objects, not core novelty. |
| F3 timestep gates | TC-LoRA, VolDiT, AtteConDA (timestep input), GateNet/RealityBridge, TASR (U-Net, snippet only) | **incremental** | Training-free, hand-set structure-early/release-late schedule on hint magnitudes for few-step (8) Turbo; with only 8 steps the schedule is coarse, worth an ablation. No prior found on few-step DiT. |
| F4 PCGrad-style projection of conflicting hint components | MaxFusion (verified full text: per-location cosine, delta=0.7, average if above else keep highest-std feature), AtteConDA (verified body: hard per-position selection), Cross-ControlNet (full text NOT reachable; abstract/secondary: std-map and consistency-ratio gates). PCGrad only in gradient space (Yu et al. 2020). | **novel in the exact form as far as verified; empirically null for us** | MaxFusion and AtteConDA verified to contain no projection / negative-cosine removal (both select, not subtract). Cross-ControlNet still abstract-level: no evidence of projection, not excluded. Our difference (remove only the negative-cosine component, keep the rest) is real but our own measurement shows hints are aligned (mean cos ~0.8, ~0.7% tokens conflict), so there is almost nothing to remove. Publishable only as a negative result / diagnostic: 'hint conflicts in Union-ControlNet on Z-Image are rare, so conflict-aware merging has no effect'. Do not claim a performance contribution. |
| T1 shared trunk, modality tokens, fidelity knob, modality dropout, control-off distillation | UNITY (universal stage across modalities), UniControl (task-aware HyperNet), EasyControl/OminiControl (frozen backbone + condition tokens/LoRA), AtteConDA (multi-cond branch), Laytrol (knowledge preservation, unopened) | **partially novel; components individually done** | Individually: shared multi-modality training (UNITY/UniControl), frozen backbone (many), zero-init (ControlNet). Not found in any opened abstract: (a) learned scalar fidelity-knob embedding trained jointly with modality dropout, (b) control-off distillation (matching the frozen backbone's output when control is dropped) for knowledge retention on a few-step distilled model, (c) all on Z-Image S3-DiT. Novelty rests on the *combination* and on the few-step/Turbo setting; claim carefully. Full texts of UNITY, Gated injection, Laytrol must be checked before claiming. |
| D1 self-generated pairs | ControlNet-training guides (snippet only), AtteConDA (real-image-derived conditions), Subjects200K/SubjectSpatial200K (DiT-synthesized pairs for OminiControl/UniCombine) | **done before (generic technique)** | OminiControl/Subjects200K already synthesize training data with the DiT itself. Ours: label-prompt-conditioned in-car scenes with extracted canny/depth/seg and mix with 60 real photos for a sim-to-real target. Treat as data recipe, not a contribution. |
| P1 label-aware prompting + diversity sampling | Sim2Real driving synthesis (2404.09111), AtteConDA evaluation protocol (diversity metrics) | **likely novel for this task, but low scientific weight** | Per-feature (clothing, light, skin) diversity sampling that provably keeps labels valid; no prior found, but search for in-cabin work found only a job posting (no paper). |
| I1 init-from-sim (SDEdit/img2img from sim render) + control hints on Z-Image-Turbo, for sim-to-real | (1) phi-PD / NeuralRemaster (2512.05106; structured-noise init keeping source Fourier phase, radius r knob, FLUX/Wan 4-step, CARLA->planner transfer), (2) SimGen (2406.09386; SDEdit from sim depth/seg *condition* latents, t_s in [0.4,0.65], ControlNet stage from noise), (3) Bosch Shapiro et al. (2508.13592; CARLA -> ControlNet/ALDM, then per-class pixel blending with sim image). Also: Z-Image diffusers ZImageImg2ImgPipeline (strength 0.6, 8 steps) ships plain init-from-source on Turbo. | **done before as a technique; incremental as a combination** | Init-from-source with a flow DiT and few steps is an off-the-shelf pipeline; sim-to-real with structure-preserving init is done (phi-PD, SimGen, Bosch blend). What none of the opened papers has: Union-ControlNet hint + SDEdit-init on a distilled 8-step S3-DiT, on in-cabin scenes, and an explicit analysis of the evaluator exploit (near-copy of sim wins FID/SAM). The defensible contribution is the strength curve plus human-realism check showing where metrics reward near-copies, not the method. |

## Threat assessment
- **Biggest threat for the fusion claim: AtteConDA (2605.09425)** (driving, canny+depth+seg, conflict suppression, timestep-conditioned per-position gating, synthetic-data motive) plus **MaxFusion / Cross-ControlNet** for training-free merging. All are U-Net/SD1.5 era or ControlNet-on-SD; none targets a single-stream few-step DiT, uses modality dropout, a fidelity knob, or control-off distillation (per what I could read).
- **Biggest threat for T1: UNITY** (shared universal stage over all modalities) and **UniControl**. Need full text.

## Suggested ideas (not added to backlog; critic merges)
- **S1. Conflict-aware merge as a learned-free *directional* projection with magnitude preserved** (extends F4): project per-token, per-layer hint of modality B off the conflicting direction of A only where cos<0, then renormalize to keep control strength. Head-to-head vs MaxFusion variance rule and Cross-ControlNet gates. Cheap, clean ablation, plausible novelty.
- **S2. Few-step-aware control schedule** (extends F3 + T1): train fidelity knob embedding to also modulate by denoising-step index for Turbo's 8 steps; ask whether distilled few-step models need different control schedules than 50-step ones (no prior found; TC-LoRA/VolDiT are many-step).
- **S3. Control-off distillation as measurable knowledge retention**: report diversity/realism of control-dropped outputs vs frozen Z-Image-Turbo (e.g., LPIPS/CLIP diversity); position as a retention benchmark, since Z-Image-Fun reports artifacts and acceleration loss after control training.
- **S4. Silent-region diagnostics for canny** (from Minimal Impact CN): check whether canny hints suppress texture in edge-free regions of cabin scenes, motivating F2 with depth/seg filling those regions.

## Full-text verification (2026-10-17)

Legend: verified-full-text = method section read via arXiv HTML/PDF text; still-abstract = only abstract/snippet/secondary summary.

### F4 question: does any of the three remove the conflicting component (projection / gradient surgery)?
- **MaxFusion** (Nair et al., ECCV 2024, arXiv 2404.09977), https://arxiv.org/html/2404.09977 : **verified-full-text**. Per spatial location cosine rho between the two features; threshold delta=0.7 (ablation: >0.5 realistic, 0.75 good); if rho>=delta features are averaged (Eq. 2); else keep the feature with highest normalized std (Eq. 5), and rescale the other model's feature by sigma_i/sigma_max times f_max (Eq. 6, "unmerging"). No projection, no subtraction, no gradient surgery. It does use a cosine test, so it is the closest in spirit; the difference is selection/averaging vs directional removal. (Fetch tool summarised the page; equations come from that summary, not from my own reading of the raw page.)
- **AtteConDA** (Noguchi, arXiv 2605.09425), https://arxiv.org/html/2605.09425 : **verified-full-text for the method section and appendix tail** (first 100k plus the remaining chars checked for the question). Tri-attention is a score generator; softmax then one-hot argmax among edge/depth/seg per position, straight-through weight (Eq. 14-16), tau=1. Cosine appears only for CLIP weather labelling in the evaluation appendix. No projection or orthogonalization. Its own limitation text: hard selection can restrict combination where several conditions are needed; hybrid soft/hard is future work. No modality dropout, no SDEdit, no sim-to-real (SDEdit and CARLA only in references).
- **Cross-ControlNet** (Liu, Jiang, Han, Jiang, Liu; ICLR 2026): **still-abstract**. Full text could not be fetched: OpenReview PDF/forum sit behind a browser check (also via curl), no arXiv version found by search, a liner.com summary gives only high-level text. What is known: PixFusion = per-pixel fusion guided by Gaussian-smoothed std maps; ChannelFusion = per-channel hybrid with a consistency-ratio gate (hard selection vs soft fusion); KV-Injection with text-derived masks; builds on MaxFusion lineage (variance-based). A "correlation threshold delta" appears in its ablation list (RQ4). No evidence of projection, but not excluded. TODO: a human with browser access should read its Sec. 3 and confirm.
- Result: **none of the two verified papers uses a projection; third unverified.** Searching for PCGrad-style removal on conditioning features found nothing (only PCGrad/CAGrad in gradient space; 2607.14398 uses a symmetric projection on constraint gradients, snippet only).

### I1 prior art (init-from-sim, sim-to-real, with control)
All verified-full-text unless marked.
1. **phi-PD / NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation** (Toyota Research Institute, UT Austin, JHU; arXiv 2512.05106, preprint, authors not individually listed here). Replaces Gaussian init noise by noise that keeps the source image's Fourier phase and randomises magnitude; Frequency-Selective Structured noise has a cutoff radius r (larger r = more structure, less photorealism, per their ablation). Trained with the swapped noise; no extra parameters, positioned against ControlNet-Tile and SDEdit baselines. Backbones SD1.5, FLUX-dev, Wan2.2-14B (with a 4-step LightX2V LoRA). CARLA re-rendering improves zero-shot planner transfer to Waymo (WOD-E2E) by about 49-50%. Not tested on Z-Image. States depth/normal inputs may need a prior. **Closest threat for I1**: explicit structure-vs-realism knob, sim2real, few-step Wan.
2. **SimGen: Simulator-conditioned Driving Scene Generation** (Zhou, Simon, Peng, Mo, Zhu, Guo, Zhou; arXiv 2406.09386, NeurIPS 2024 per my memory, unverified). SDEdit-style: the sim's depth+segmentation *condition* latents are noised to intermediate t_s and denoised (frozen); t_s in [0.4,0.65], chosen 0.5; "smaller t_s gives SimCond more control but potentially compromising realism." The second stage (ControlNet-style adapter) starts from pure noise. Differs from I1: the init is from sim condition maps, not the sim RGB; SD-based, many steps.
3. **Bridging Clear and Adverse Driving Conditions: Domain Adaptation with Simulation, Diffusion, and GANs** (Shapiro, Showgan, Mullick; Bosch; arXiv 2508.13592, preprint). CARLA -> ALDM or ControlNet, then pixel-wise blend I_B = w*I_diffusion + (1-w)*I_sim with per-class weights from the segmentation (0.9 sky/buildings, 0.8 road, 0.3 pedestrians, 0.1 vehicles), plus Reinhard colour matching. This is the only opened paper with an explicit realism/fidelity handle that works against near-copy and hallucination. Init method (SDEdit strength) not specified in the body; downstream AP: CARLA+ControlNet blend 29.46 vs CARLA alone 29.89 vs ImgAug 32.51 (ACDC), i.e. the blended diffusion did not help. Useful as honest negative prior.
Other relevant, opened:
- **Cosmos-Transfer1** (NVIDIA, arXiv 2503.14492; opened via HTML): multi-branch ControlNet (edge, vis, depth, seg) with a spatiotemporal control map weighting branches per region; sim2real for robotics and AV. Init from pure noise (no SDEdit mentioned in the excerpt). Reports structure vs diversity trade-off by modality (Vis/Edge: high SSIM, low LPIPS diversity; depth/seg: the reverse). Overlap with F1/F2 (region weights) more than I1. Realism claims qualitative (DOVER proxy), no downstream or human study in the excerpt.
- **A Hybrid Approach for Closing the Sim2real Appearance Gap in Game Engine Synthetic Datasets** (Pasios; arXiv 2605.02291, preprint): FLUX.2-klein-4B prompted as image editor from the RGB frame only (no control, no SDEdit details), then REGEN image-to-image net. FLUX alone CMMD 3.734 -> 2.488, FLUX+REGEN 1.781 on VKITTI2->KITTI; fidelity checked post hoc by mIoU/mAP; hallucination named as a limitation. Shows few-step edit models do sim2real but with no knob.
- **DriveCtrl** (Zhao, Wang, Chen, Donzella, Bashford-Rogers, Debattista; arXiv 2605.15116; HTML abstract/method partly): Wan2.2-Fun-A14B-Control, frozen, timestep-switched LoRA pair, depth control; no SDEdit init; SSIM 0.831; admits residual structure changes.
- **Driving with DINO** (Chen et al.; arXiv 2602.06159; ACM MM 2026; abstract only, **still-abstract**): names the "Consistency-Realism Dilemma"; DINO features with principal-subspace projection to remove synthetic texture. Good framing citation for the realism/fidelity trade-off.
- **RealMaster** (Cohen-Bar, Sobol, Bensadoun, Sheynin, Gafni, Patashnik, Cohen-Or, Zohar; arXiv 2603.23462; abstract only, **still-abstract**): GTA-V to photoreal video via anchor-propagated paired data and IC-LoRA; states "structural precision vs global semantic transformation" tension.
- **Video ControlNet** (arXiv 2305.19193; HTML full): synthetic-to-real video, depth/normal ControlNet, noise from pure Gaussian with initial-noise optimisation, no SDEdit. Authors not checked here.
- **Zhao et al. 2404.09111** (already listed): GAN vs ControlNet from sim label maps.
- **Z-Image diffusers docs** https://huggingface.co/docs/diffusers/main/en/api/pipelines/z_image : **verified**. ZImageImg2ImgPipeline exists (strength default 0.6, example 8 steps, guidance 0; strength 1 ignores the image) and ZImageInpaintPipeline. So init-from-source on Z-Image-Turbo is shipping functionality; pipeline docs show no control-hint combination.
- Few-step theory: "On Distillation of Guided Diffusion Models" (arXiv 2210.03142; snippet only, **still-abstract**) says on a 4-step distilled model 2 steps ~ 50% noise and more noise = better prompt alignment but less input fidelity. TurboEdit (2408.00735, snippet only) notes few-step editing fails via artifacts or too little edit strength. With 8 steps the strength grid is coarse (0.5 = 4 steps, 0.75 = 6 steps).
- In-cabin: no paper found. Only the Aumovio thesis posting REF5866V (job posting), a talk on domain gap for driver monitoring digital twins (auto-sens, not a paper) and a 2024 thermal GAN paper (2403.00196). Absence from ~5 searches total, not exhaustive.

## Verdicts

**F4: novel in exact form (unverified for Cross-ControlNet), but empirically null.** Closest 3: (1) MaxFusion: cosine-thresholded merge, but average or hard-select by std, no removal (verified full text); (2) Cross-ControlNet: std/consistency-ratio gates, still abstract; (3) AtteConDA: learned hard per-position selection, no cosine (verified). Specific difference: directional negative-component removal that keeps the aligned part. Since our measured conflict rate is 0.7% of tokens, nothing publishable on performance; report as a diagnostic negative result and cite MaxFusion's cosine test as motivation.

**I1: done before as a technique, incremental as our combination.** Closest 3: phi-PD (structure-preserving init, r knob, sim2real, few-step Wan), SimGen (SDEdit t_s knob in driving sim2real, but from condition maps), Bosch 2508.13592 (ControlNet + per-class blending to trade realism for fidelity). Difference left: Union-ControlNet hints plus SDEdit-init on an 8-step distilled S3-DiT for in-cabin; plus an honest strength curve with human realism, and the observation that evaluator metrics reward near-copies. Handling of the realism/fidelity trade-off by prior art: knob only (r in phi-PD, t_s in SimGen), post-hoc spatial blending (Bosch), or none (Pasios, Cosmos). Nobody reports an evaluator-gaming analysis.

## Earlier claims changed
- MaxFusion: now verified full text; correlation test = cosine per location, delta 0.7; averages when aligned, selects max-std otherwise (earlier note "partial-high on F4" -> **partial**: it conflict-tests with cosine but never removes a component).
- AtteConDA: now confirmed to have no cosine/projection in its fusion; F4 overlap lowered from partial to none-partial (hard selection only). Its limitation note supports soft hybrid merging.
- Cross-ControlNet: remains abstract-level; the ICLR page is not machine-readable. Earlier "F1 full-ish" stands.
- "No in-cabin sim-to-real diffusion paper": still holds, stronger evidence now (5 searches).
- Z-Image Turbo img2img exists in diffusers (new, affects I1 novelty).

## Threats to novelty
- phi-PD (init-from-source with a structure/realism knob, sim2real result) and SimGen for I1; Bosch blending and Cosmos region weights for fusion/gating ideas.
- Practical competitors: CARLA2Real / Hybrid-Sim2Real (FLUX.2-klein-4B + REGEN, github stefanos50/CARLA2Real), REGEN (2508.17061, snippet only), HyPER-GAN (2603.10604, snippet only).

## Suggested ideas added
- S5. Per-class (or per-region) strength from the sim labels: low strength on objects that the label prompt must keep (seatbelt, hands), higher on background; closest is Bosch's pixel blending but done in latent/noise space on Turbo (cf. Differential Diffusion 2306.00950, snippet only).
- S6. Report the whole strength-vs-realism curve and compare against the near-copy references as an evaluator-robustness analysis.

## TODO / unverified
- Cross-ControlNet full text (needs browser access to OpenReview 89j1hUxOiF).
- UNITY (2606.20971), Gated injection (2603.27666), EasyControl multi-cond, Ctrl-Adapter router: still abstract-level (not in this pass).
- Snippet-only: OminiControl2, Laytrol, SimGen's venue, TASR, 2607.14398, JointDiT/MMDiT, REGEN, HyPER-GAN, TurboEdit, 2210.03142, Differential Diffusion, Driving with DINO, RealMaster, IntrinsicControlNet (ICCV 2025, HTTP 403), Video ControlNet authors.
- Does phi-PD or any paper combine SDEdit-init with ControlNet on a distilled DiT? Not found, but only ~6 targeted searches.
- Z-Image layer/step count and Fun-ControlNet layer spacing: unchanged.
