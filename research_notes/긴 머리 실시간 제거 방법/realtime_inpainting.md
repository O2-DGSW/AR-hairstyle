# Real-time / lightweight filling of the region behind removed long hair (live video)

Context: AR hairstyle try-on, server GPU (RTX 3090/4070 class), ~800x1300 portrait or 720p/1080p, ~30 fps, fill budget of a few ms per frame on the hot path, heavier work possible asynchronously at a few Hz. Head-area erase already works (time-accumulated background plate + skin fill). Open problem: long hair over shoulders/chest. LaMa and MI-GAN produced ghost hair streaks and smudges beside the neck.

Research date: 2026-10-19. About 17 tool calls. Several primary PDFs (CVF open access) returned HTTP 403, so some numbers come from arXiv HTML pages or secondary pages. Each such case is marked.

---

## 1. Online / real-time video inpainting and object removal (latency, causality, temporal consistency, large masks)

### Takeaway
No published video inpainter both meets a few-ms budget and handles large, structured human regions. The best published causal numbers are about 25-40 ms/frame (memory-based DSTT/FuseFormer on an RTX 2080 Ti, low resolution, no code confirmed). Flow-guided SOTA (ProPainter, E2FGVI) costs 250-330 ms/frame at 480p, is offline, and is non-commercial. For this product, video models are useful only as an asynchronous keyframe source, or as a design pattern (memory of past outputs, causal use), not as a per-frame filler.

### Cited Findings
- **Thiry et al., "Towards Online Real-Time Memory-based Video Inpainting Transformers" (CVPRW/NTIRE 2024).** It turns offline inpainting transformers into causal online models by storing each frame's transformer outputs and reusing them, so only the newest frame is computed as the query (attention goes from quadratic to linear in frame count). A "Refined" variant adds a second, slower inpainter on another GPU that re-inpaints past windows and feeds the results back. — [arXiv HTML 2403.16161](https://arxiv.org/html/2403.16161)
- Thiry et al., DAVIS results on an RTX 2080 Ti (resolution not reported; latency = 1000/FPS):
  - DSTT: offline 19.2 fps / 31.25 dB; online 12.4 fps; **memory 39.5 fps (25 ms) / 30.20 dB**; refined 27.1 fps (2 GPUs).
  - FuseFormer: offline 10.8 fps; **memory 31.0 fps (32 ms) / 30.49 dB**.
  - E2FGVI: offline 8.8 fps / 32.08 dB; memory 11.4 fps (88 ms).
  - — [arXiv HTML 2403.16161](https://arxiv.org/html/2403.16161)
- Thiry et al., FuseFormer GPU memory: online 0.53 GB; memory variant 1.28 GB (DAVIS) / 2.26 GB (YouTube-VOS). — [arXiv HTML 2403.16161](https://arxiv.org/html/2403.16161)
- Thiry et al. say "the code and pretrained models will be made available upon acceptance". No code link was found. — [arXiv HTML 2403.16161](https://arxiv.org/html/2403.16161); [CVF page](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Thiry_Towards_Online_Real-Time_Memory-based_Video_Inpainting_Transformers_CVPRW_2024_paper.html)
- **ProPainter (ICCV 2023)** on DAVIS 480p, V100 32 GB: **0.249 s/frame**. In the same setup E2FGVI took 0.332 s/frame and STTN 0.262 s/frame. FuseFormer, FGT and TSAM were excluded from the timing table because they need more than 32 GB or take too long at 480p. — [ProPainter arXiv 2309.03897](https://arxiv.org/pdf/2309.03897)
- **ProPainter license and memory:** NTU S-Lab License 1.0, non-commercial; commercial use needs permission from the author. Estimated VRAM fp32/fp16 for 50 frames: 1280x720 28G/19G; 720x480 11G/7G; 320x240 3G/2G. Options: `--fp16`, `--subvideo_length`, `--neighbor_length`, `--ref_stride`, `--resize_ratio`. There is no official streaming mode; a community "Streaming ProPainter" (github.com/osmr/propainter) is listed. — [ProPainter GitHub](https://github.com/sczhou/ProPainter)
- **E2FGVI (CVPR 2022)** reports 0.16 s/frame in its own paper. A secondary write-up gives 0.12 s/frame at 432x240 on a Titan XP. — [search summary of E2FGVI review](https://liner.com/review/towards-endtoend-framework-for-flowguided-video-inpainting) (secondary source; hardware and resolution differ from ProPainter's table)
- **Software Mansion live-stream inpainting (Feb 2024, industry blog):**
  - Model: a lightweight DSTT-style model with Swin blocks and conditional positional encodings, running at **480x288**. It uses past frames only, small batches, and torch.compile (about 20% faster).
  - Throughput: above 25 fps.
  - Rejected: diffusion as too heavy, and flow-based methods for static scenes.
  - Limitations they state: dynamic backgrounds, low resolution, and hands and faces are hard.
  - GPU and code: not given.
  - — [Software Mansion blog](https://swmansion.com/blog/pushing-the-boundaries-with-real-time-video-inpainting-35cd82c624fb)
- **Fane, "A Real-Time Diminished Reality Approach to Privacy in MR Collaboration" (arXiv Aug 2025):** a modified DSTT plus YOLOv11 and segmentation, above 20 fps at 720p (GPU not stated). It removes objects, not people. — [arXiv 2509.10466](https://arxiv.org/abs/2509.10466)
- The same paper's search snippet says DSTT "is not prepared for ONNX/TensorRT export". — [search snippet, arXiv 2509.10466](https://arxiv.org/pdf/2509.10466)
- **InstantViR (CVPR 2026):**
  - What it is: a one-step student distilled from a video-diffusion teacher. It handles streaming random inpainting among other inverse problems.
  - Speed: above 35 fps on an A100, "up to 100x" faster than iterative video-diffusion priors.
  - — [CVPR 2026 poster page](https://cvpr.thecvf.com/virtual/2026/poster/37756)
- **US Patent 12,131,448 B2 (Oct 2024):** claims causal removal that inpaints the current frame and blends it with features from earlier frames by weighted averaging. This is a patent claim, not a benchmark. — [USPTO gazette](https://patentsgazette.uspto.gov/week44/OG/html/1527-5/US12131448-20241029.html)

### Inferences
- Even the fastest causal transformer (memory DSTT, about 25 ms on a 2080 Ti at an unstated, probably low resolution) is several times over the "few ms" budget. It is also trained on generic object or random masks (DAVIS/YouTube-VOS), so it would continue whatever texture surrounds the hole, which here means hair. The ghost-hair failure seen with LaMa and MI-GAN would likely reappear.
- Flow-guided propagation (ProPainter, E2FGVI, FGT) only helps when the occluded pixels become visible in some frame. Hair lying on the chest of a mostly static user may never move off, so propagation has nothing to copy. The work then falls to the hallucination stage, which is exactly where ghost hair appears.
- The useful idea from this literature is architectural, and it matches the existing background plate: compute an expensive fill rarely (keyframe or asynchronous), store it in a body-anchored memory, and warp or reuse it per frame. The "Refined" variant in Thiry et al. (slow second model refreshing past outputs) is the same pattern as "async heavy fill at a few Hz plus cheap per-frame reuse".
- ProPainter's license (non-commercial) rules it out for a product without negotiation. Other repo licenses were not verified here (see Gaps).

### Gaps
- FGT and DSTT per-frame latency at 720p/1080p on an RTX 3090/4090: not found.
- Whether the Thiry et al. code was ever released: not confirmed.
- No 2025-2026 paper was found that reports a causal video inpainter running under 10 ms per frame at 720p on a consumer GPU.
- Licenses of E2FGVI, FGT, DSTT and STTN were not verified in this session.

---

## 2. Fast image inpainters (MI-GAN, LaMa variants, AOT-GAN, MAT, distilled diffusion, TensorRT) on large masks over bodies and clothes

### Takeaway
At 512x512 on a modern GPU, LaMa (27M params) runs in about 13 ms and MAT in about 28 ms. MI-GAN is about 10x smaller and cheaper again and ships an ONNX pipeline under MIT. That puts a 512-crop GAN fill within a few-ms to ~10 ms budget with fp16/TensorRT. Quality is the problem, not speed: these models are trained on Places, FFHQ and random masks, so they copy surrounding texture. They do not know the shoulder silhouette and continue hair into the hole. Distilled diffusion is still 50 ms or more per call and only suits asynchronous keyframes.

### Cited Findings
- **Latency and parameters at 512x512 on an RTX 4090** (runtime not stated, so possibly plain PyTorch). — [IN2 paper, arXiv 2401.10537](https://arxiv.org/html/2401.10537)

  | Method | Params | Latency |
  |---|---|---|
  | LaMa | 27.3M | 13 ms |
  | MAT | 61.6M | 28 ms |
  | CoordFill | 31.6M | 10 ms |
  | IN2 | 15.5M | 20 ms |

- **MI-GAN (ICCV 2023, Picsart):**
  - Size and cost: "approximately one order of magnitude computationally cheaper and smaller" than SOTA inpainters. It is built with adversarial training, re-parametrization, and distillation from CoModGAN. — [CVF paper page](https://openaccess.thecvf.com/content/ICCV2023/html/Sargsyan_MI-GAN_A_Simple_Baseline_for_Image_Inpainting_on_Mobile_Devices_ICCV_2023_paper.html)
  - Code and weights: code is MIT. Weights are under a separate LICENSE-WEIGHTS file (not read). Models come at 256 (FFHQ, Places2) and 512 (Places2). — [MI-GAN GitHub](https://github.com/Picsart-AI-Research/MI-GAN)
  - Export: ONNX pipeline (`migan_pipeline_v2.onnx` on Hugging Face). Paper timings are for model inference only, not the ONNX pipeline. — [MI-GAN GitHub](https://github.com/Picsart-AI-Research/MI-GAN)
  - Large masks: the README says that for high-resolution inputs with a large area, the best results come from "small, incremental brush strokes". — [MI-GAN GitHub](https://github.com/Picsart-AI-Research/MI-GAN)
  - Benchmarks vs LaMa (supplement, as summarised by search): LaMa scores higher on Places2 PSNR (17.95 vs 16.12). The summarised large-mask FID figures were garbled, so they are not reported here. — [MI-GAN supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Sargsyan_MI-GAN_A_Simple_ICCV_2023_supplemental.pdf) (search summary only; PDF returned 403)
  - Model files, per a third-party Flutter package: MI-GAN 80 MB vs LaMa 209 MB, described as "7x smaller". — [seika package docs](https://pub.dev/packages/seika) (secondary)
- **LaMa deployment artefacts:**
  - Fixed-size 512x512 big-LaMa ONNX export, stated to work in TensorRT (no timings given). — [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)
  - Qualcomm ships LaMa-Dilated at 512x512 for NPUs. — [Qualcomm AI Hub](https://aihub.qualcomm.com/compute/models/lama_dilated)
- **RETHINED (arXiv Mar 2025):**
  - Method: a lightweight CNN recovers structure, then a "resolution-agnostic patch replacement" step copies real texture.
  - Speed claim: ≤30 ms on mobile-class devices, about 100x faster than SOTA, at similar quality.
  - Released: the DF8K-Inpainting dataset is announced. Code status unverified.
  - — [arXiv 2503.14757](https://arxiv.org/abs/2503.14757)
- **Guided PatchMatch with auto-curation (Adobe, 2022):** uses a deep inpainter at low resolution plus guided PatchMatch to reach camera resolution. It is evidence for the "net for structure + patch copy for texture" hybrid. — [arXiv 2208.03552](https://arxiv.org/pdf/2208.03552) (title and abstract level only)
- **Few-step diffusion inpainting:**
  - InverFill (Qualcomm, Mar 2026): adds about 0.06 s of inversion overhead on an A100 on top of a few-step sampler. It notes that naively applying few-step T2I models to inpainting gives poor harmonisation at the seam. — [arXiv 2603.23463](https://arxiv.org/pdf/2603.23463)
  - SANA-Sprint (Mar 2025): one-step T2I via continuous-time consistency distillation. It is not inpainting-specific. — [arXiv 2503.09641](https://arxiv.org/abs/2503.09641)

### Inferences
- **Speed is solvable.** A 512 crop around the shoulders and chest with MI-GAN or LaMa in fp16/TensorRT should land at single-digit ms on a 3090/4070-class GPU. This is extrapolated from 13 ms for LaMa in unspecified-runtime PyTorch on a 4090, so it must be measured locally. MI-GAN should be several times cheaper.
- **The ghost-hair failure is a training-distribution and context problem, not a capacity problem.** Places2, FFHQ and random-stroke training teach "extend the surrounding texture". When hair surrounds the hole on two sides, hair is the most probable continuation. This matches the observed failure.
- **Mitigations available without new models** (all inferred, none benchmarked in sources):
  - Dilate the hair mask generously so no hair pixels remain in the known region. Strand tips and semi-transparent edges carry most of the hair signal.
  - Replace remaining hair-adjacent context with a neutral or body-colour pre-fill before inference, so the model never sees hair texture.
  - Feed the inpainter only crops where the known region is clothing, skin or background.
- **Fine-tuning is the real fix.** The project already has a forehead inpainter trained with a guidance channel (v4/v5 in memory). The same recipe would apply: fine-tune MI-GAN or the in-house net on portrait and upper-body images with synthetic hair-shaped masks over shoulders and chest. Ground truth would come from short-haired or tied-hair subjects, where the true clothing and shoulders are known. Add a silhouette or parsing guidance channel (see section 3).
- Diffusion (even few-step) belongs only on the asynchronous path. At about 50-100 ms or more per call on an A100-class GPU, it could produce a high-quality "clean body plate" once per pose change or every few seconds, which the cheap path then reuses.

### Gaps
- Exact MI-GAN parameter count, MACs and ms on a desktop GPU: the CVF PDF returned 403. The paper reportedly has these numbers in a bubble chart and tables.
- AOT-GAN latency and params: not found in this session.
- TensorRT fp16 numbers for LaMa or MI-GAN on an RTX 3090/4070: none found. They need a local benchmark.
- Licenses: LaMa is widely reported as Apache-2.0 and MAT as non-commercial, but neither was verified in this session. MI-GAN's weight license was not read.
- RETHINED code availability and license: not confirmed.

---

## 3. Human- and person-aware inpainting: pose- or parsing-guided completion, clothing completion, amodal human de-occlusion, silhouette from keypoints

### Takeaway
Structure-first completion is an established idea: first complete the parsing or silhouette map, then fill pixels conditioned on it. It directly addresses the "smudge beside the neck" failure. Available models are mostly offline: older GANs from 2019-2021, or 2024-2026 diffusion methods taking seconds. None meets the real-time budget, but the design can be copied cheaply: predict or complete a body/clothing parsing map under the hair (from pose keypoints plus visible parsing), then use it as a guidance channel for a fast inpainter or as a constraint for push-pull and patch filling.

### Cited Findings
- **Portrait completion via parsing and pose (TIP 2020):** first recovers a full-body structure (parsing) map with a pose-guided human-parsing network, then fills pixels with a completion network conditioned on that map. — [Tsinghua TIP-2020 paper](https://cg.cs.tsinghua.edu.cn/papers/TIP-2020-portrait.pdf)
- **Prior Based Human Completion (CVPR 2021, Zhao et al.):** human-specific completion using structural priors. — [CVF CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Prior_Based_Human_Completion_CVPR_2021_paper.pdf)
- **LOHC (AAAI, two-stage parsing + image-prior cooperation):**
  - Finding: under large occlusion, completing the parsing map "is not significantly easier" than completing the image itself. They therefore make the parsing and image priors cooperate.
  - Code: https://github.com/ZhaoHengrun/LOHC.
  - — [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/28578/29124)
- **ESP (2024, MPI Perceiving Systems):** pose-conditioned generation of people in scenes. It uses human-parsing masks as tight inpainting guides so the background is untouched. — [ESP project](https://ps.is.mpg.de/publications/esp)
- **CompleteMe (arXiv Apr 2025):** reference-based human image completion, i.e. completing the occluded body using another image of the same person. It notes that amodal methods "often fall short" for varied poses and fine details. — [arXiv 2504.20042](https://arxiv.org/html/2504.20042v1)
- **PHAC: Promptable Human Amodal Completion (CVPR 2026):** point-based prompts are encoded through ControlNet modules for diffusion-based human amodal completion. Code status unknown. — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36807)
- **Contact-aware amodal completion for human-object interaction (ICCV 2025):** training-free multi-regional diffusion inpainting guided by contact masks. Code promised "upon acceptance". — [arXiv 2508.00427](https://arxiv.org/html/2508.00427v1)
- **Hair-specific removal ("bald converters"):**
  - Stable-Hair (Jul 2024): an SD-based bald converter strips hair before transfer. Training data is produced with an LLM writing prompts and SD inpainting creating reference images. — [arXiv 2407.14078](https://arxiv.org/pdf/2407.14078)
  - Stable-Hair v2 (Jul 2025) reuses that converter for multi-view hair transfer. — [arXiv 2507.07591](https://arxiv.org/pdf/2507.07591)
  - AnyBald (WACV 2026): adapts latent-diffusion inpainting with learnable text prompts (no explicit mask at inference), adds an attention regulariser focused on hair, and builds a synthetic hair/no-hair pair pipeline. Evaluated on CelebA and DeepFashion2. — [WACV 2026 poster](https://wacv.thecvf.com/virtual/2026/poster/402); [supplement](https://openaccess.thecvf.com/content/WACV2026/supplemental/Choi_AnyBald_Toward_Realistic_WACV_2026_supplemental.pdf)
  - The search summary of AnyBald's comparisons mentions residual hair and colour inconsistency around the mask boundary in some in-the-wild cases. — [AnyBald supplement](https://openaccess.thecvf.com/content/WACV2026/supplemental/Choi_AnyBald_Toward_Realistic_WACV_2026_supplemental.pdf) (search summary)

### Inferences
- The smudge beside the neck is the missing-structure failure that parsing-first methods were designed for. A cheap real-time version of "structure first" for this product would be:
  - (a) Run a body-pose model (already standard; shoulder and neck keypoints are reliable even under hair).
  - (b) Build a shoulder/neck/torso silhouette under the hair from the keypoints plus the visible body-parsing boundary. Options: a fitted template polygon/spline from neck to shoulder points, or a per-session cached silhouette captured when the hair moved away.
  - (c) Feed that silhouette (and per-region labels: skin / clothing / background) as a guidance channel to the inpainter, or fill each label region separately from same-label exemplars.
  - This extrapolates TIP 2020 and LOHC to a lightweight setting. The LOHC finding warns that predicting the parsing map under big occlusions is itself hard. Pose keypoints and session history are the cheap priors that make it tractable here.
- AnyBald's DeepFashion2 evaluation and its synthetic hair/no-hair pair pipeline are the most relevant recipe for generating training data for shoulder-length removal. Its diffusion runtime makes it an offline data generator or asynchronous-path tool, not a per-frame model.
- Stable-Hair's data pipeline (SD inpainting to make bald references) suggests a way to bootstrap a body-only fine-tuning set: use a slow diffusion inpainter offline to produce clean torso plates, then distil them into a fast GAN with the hair-shaped mask distribution.

### Gaps
- No latency numbers were found for any human-completion model (LOHC, PHAC, CompleteMe, AnyBald, Stable-Hair). All are presumed offline, at seconds per image for the diffusion ones.
- No source found that specifically evaluates removal of long hair over shoulders and chest (as opposed to the head region).
- No source found on estimating a body silhouette under occlusion from 2D keypoints alone in real time. SMPL/DensePose-guided fill was not researched for latency in this session.
- Licenses of LOHC, AnyBald and Stable-Hair were not checked.

---

## 4. Classical / cheap approaches: GPU PatchMatch, exemplar copy, symmetry, push-pull with silhouette, temporal accumulation (keyframe inpaint + warp), session caching

### Takeaway
Cheap methods that copy real pixels can meet the few-ms budget, and by construction they never invent hair if the source regions exclude hair. The strongest pieces from the literature are:
- **Hybrid "small net for structure, patch copy for texture":** RETHINED at ≤30 ms on mobile; Adobe Guided PatchMatch.
- **Online caching of past outputs:** Thiry memory; Software Mansion client-side caching.

For this product, the natural extension of the existing head-area background plate is a body-anchored "clean torso plate". It is filled asynchronously, at keyframes or when hair swings away, warped per frame by shoulder keypoints, and topped up with a same-label exemplar or push-pull fill restricted by a silhouette.

### Cited Findings
- RETHINED combines a lightweight CNN (structure) with a resolution-agnostic patch-replacement step (texture), reaching ≤30 ms on mobile-class hardware. — [arXiv 2503.14757](https://arxiv.org/abs/2503.14757)
- Guided PatchMatch (Adobe): a deep net at low resolution guides PatchMatch to fill at camera resolution. — [arXiv 2208.03552](https://arxiv.org/pdf/2208.03552)
- The memory/caching principle (reuse previous frames' computations or outputs, causal) gives up to 3x throughput over naive online processing in Thiry et al. — [arXiv HTML 2403.16161](https://arxiv.org/html/2403.16161)
- Software Mansion's live-stream system reached real time partly through client-side frame caching and inpainting only masked areas at low resolution, compositing over full-resolution originals. It states it works best on static webcam-like scenes. — [Software Mansion blog](https://swmansion.com/blog/pushing-the-boundaries-with-real-time-video-inpainting-35cd82c624fb)
- The 2019 RGBD real-time occlusion-removal work avoided full video inpainting because inpainting was too slow, relying on recorded and accumulated background instead. — [search result summary, unpaywall 10.1145/3306131.3317025](https://unpaywall.org/10.1145%2F3306131.3317025) (secondary summary)

### Inferences
These are design proposals for heddy, built from the findings above; none were benchmarked in sources.
- **Per-session torso plate.** Whenever the user turns, tosses hair or the hair moves, pixels of shoulder and chest become visible. Accumulate them in a pose-normalised torso UV, for example a rectangle anchored on the two shoulder keypoints and the neck point, with a confidence map. Fill the remaining holes asynchronously with a heavy model (diffusion or fine-tuned GAN) at a few Hz. Per frame, warp with a 2D affine or thin-plate transform from the keypoints: that is a few ms of texture sampling. This mirrors the existing head-area plate.
- **Exemplar copy restricted by label.**
  - Clothing: fill hidden clothing from visible clothing of the same parsing label (the other shoulder, the lower chest). Mirror symmetry across the body midline (neck-to-sternum line) is a cheap prior for shoulders and collars.
  - Skin: fill hidden neck and clavicle skin from the existing skin-fill path.
  - Hair: never use it as a source.
  - A GPU PatchMatch restricted to same-label sources at low resolution (e.g. 1/4 scale), then upsampled with guided blending, should fit a few ms. This is inferred; there are no sourced numbers.
- **Push-pull with silhouette constraints.** The current low-frequency diffusion makes flat blobs because it blurs across the shoulder/background boundary. Running push-pull separately per region, with each region bounded by the estimated silhouette, would keep a sharp shoulder edge. Adding a mean-texture or noise layer sampled from the same region's visible pixels would reduce the "flat" look. Cost: a few pyramid levels, under 1 ms on a GPU.
- **Hybrid quality ladder:**
  1. Per-frame: warped torso plate (cache), plus same-label patch copy, plus silhouette-constrained push-pull for any leftover.
  2. Async: refresh the plate with a fine-tuned fast GAN at a few Hz, and optionally a diffusion pass on pose change.

### Gaps
- No sourced latency numbers were found for GPU PatchMatch at 720p/1080p on an RTX 3090/4070.
- No source found specifically on pose-normalised torso texture caching for AR try-on.
- The quality of symmetry-based clothing completion (asymmetric garments, prints) was not evaluated in any source found.

---

## 5. Stopping the inpainter from re-drawing hair (mask dilation, hiding hair context, guidance channels, body-only fine-tuning)

### Takeaway
The literature that addresses this directly is hair-removal work (Stable-Hair, AnyBald). Both rely on purpose-built training data (synthetic hair/no-hair pairs) and hair-focused attention or regularisation, not on mask tricks. AnyBald still reports residual hair at mask boundaries. For a fast GAN the practical recipe is:
- generous dilation so no strand pixels remain in the known region;
- pre-filling the hole with a hair-free guess so the network sees no hair texture;
- a guidance channel (silhouette or parsing);
- fine-tuning on body-only targets with hair-shaped masks.

### Cited Findings
- Stable-Hair removes all hair first with a dedicated SD-based bald converter. It argues this avoids blending the new style with leftover strands and lowers artifacts. Its paired training data is produced by an LLM and SD inpainting. — [arXiv 2407.14078](https://arxiv.org/pdf/2407.14078)
- AnyBald uses:
  - learnable text prompts for a latent-diffusion inpainter;
  - an attention regularisation loss focused on hair;
  - a synthetic augmentation pipeline that builds diverse hair/no-hair pairs.
  - — [WACV 2026 poster](https://wacv.thecvf.com/virtual/2026/poster/402)
- AnyBald comparisons still show residual hair and colour inconsistency at the mask boundary in some in-the-wild cases. — [AnyBald supplement](https://openaccess.thecvf.com/content/WACV2026/supplemental/Choi_AnyBald_Toward_Realistic_WACV_2026_supplemental.pdf) (search summary)
- Few-step diffusion applied to inpainting shows harmonisation artefacts at the background/fill seam. — [InverFill arXiv 2603.23463](https://arxiv.org/pdf/2603.23463)
- MI-GAN's README recommends incremental small strokes for large areas at high resolution, implicitly admitting degraded large-hole quality. — [MI-GAN GitHub](https://github.com/Picsart-AI-Research/MI-GAN)

### Inferences
- **Why ghost hair appears.** In the current failure, LaMa and MI-GAN see hair on the hole boundary: strand tips, semi-transparent fringe, hair beyond the mask. Fourier/FFC layers in LaMa are known to propagate repetitive texture globally, which plausibly explains the streaks. This is a reasoned hypothesis, not a sourced claim.
- **Mitigations in order of cost:**
  1. Dilate the hair mask more (especially downward along strands) and feather the composite. Also remove hair-coloured, low-alpha pixels outside the matte.
  2. Pre-fill the hole with the silhouette-constrained push-pull or plate guess before calling the inpainter, so the input image holds no hair. Use the inpainter as a refiner (texture and detail) with a softer mask.
  3. Crop the context so the inpainter only sees torso, clothing and background.
  4. Add guidance channels (silhouette, parsing labels, shoulder keypoint heatmaps). This is the same recipe that helped the in-house forehead net (v4).
  5. Fine-tune on body-only targets:
     - Source images: people with tied or short hair, so true shoulders are visible, plus AnyBald/Stable-Hair-style synthetic pairs.
     - Masks: hair-shaped, sampled from real long-hair mattes pasted over the shoulders, so the mask distribution matches deployment.
     - Losses: perceptual and adversarial losses, plus optionally a "hair-likeness" penalty from the hair segmenter run on the output (penalise any predicted hair inside the fill).
- **Segmenter penalty.** The last point turns the existing hair segmenter into a cheap discriminator. A fill that the segmenter flags as hair is by definition a failure, and this can also serve as an automatic quality gate at runtime: reject the fill and fall back to the plate or push-pull.

### Gaps
- No source found that measures the effect of mask dilation or context hiding on ghost-texture continuation for LaMa or MI-GAN.
- No source found that uses a segmentation-based "no hair in the fill" loss (proposed above as an inference).
- AnyBald and Stable-Hair runtime and code licenses were not checked.

---

### Overall budget assessment (inference from all sections)

| Candidate | Est. per-frame cost on 3090/4070 | Fits hot path? | Hair-ghost risk | License |
|---|---|---|---|---|
| Silhouette-constrained push-pull | <1 ms (inferred) | Yes | None (no hair source) | own |
| Pose-warped per-session torso plate | ~1-2 ms (inferred) | Yes | None if plate is clean | own |
| Same-label GPU PatchMatch / exemplar, low-res | few ms (inferred, unsourced) | Likely | None if hair is excluded from sources | own |
| MI-GAN 512 crop (ONNX/TRT) | low single-digit ms (inferred, unverified) | Likely | High unless fine-tuned or guided | MIT code; weights license unread |
| LaMa 512 | 13 ms on 4090 (sourced, runtime unstated) | Marginal | High (observed) | not verified here |
| MAT 512 | 28 ms on 4090 (sourced) | No (async only) | Probably high | not verified here |
| Memory-DSTT/FuseFormer online | 25-32 ms on 2080 Ti, low res (sourced) | No | High (generic training) | code not released? |
| ProPainter / E2FGVI | 250-330 ms at 480p on V100 (sourced) | No | n/a (offline) | ProPainter non-commercial |
| Few-step diffusion / AnyBald-style | ≥60 ms to seconds (partly sourced) | Async keyframes only | Lower with hair-specific training | varies |
