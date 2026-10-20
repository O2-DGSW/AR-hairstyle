# Generative hair removal ("bald conversion") and hairstyle transfer: crop extent, long hair, speed, keyframe propagation, and training data

Scope: HairMapper, Stable-Hair (Bald Converter), HairFastGAN, Barbershop, StyleYourHair, HairCLIP/v2, AnyBald, HairPort/Bald Konverter, HairShifter (video), plus keyframe-propagation and few-step inpainting work. Researched 2026-10-19. Distinguishes author claims from measured numbers where possible.

## Q1. What crop do bald/hair-transfer methods operate on, do they fill shoulders/clothing behind long hair, and how good are they on long hair?

### Takeaway
The StyleGAN family (HairMapper, Barbershop, StyleYourHair, HairCLIP/v2, HairFastGAN) works on FFHQ-aligned face crops. Their own papers list long-to-short hair as a failure case: skin texture comes out wrong and "shadows" of the old hair remain. None of them is designed to rebuild clothing or background below the shoulders. The diffusion-era methods (Stable-Hair's Bald Converter in 2024, AnyBald at WACV 2026, HairPort's Bald Konverter on FLUX.1 Kontext in 2026) claim to need no alignment and to handle half-body or in-the-wild images. None of them publishes a long-hair-over-shoulders evaluation, a resolution, or a crop spec.

### Cited Findings
**HairMapper (CVPR 2022, Wu et al.)**
- It is StyleGAN2-based. It learns a hair-removal direction in latent space, using a "female-male-bald" pipeline to build paired data for women and a separation-boundary-plus-diffusion method for men. The authors claim it generalizes across gender and age. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_HairMapper_Removing_Hair_From_Portraits_Using_GANs_CVPR_2022_paper.html)
- Input must be aligned with DLib and the FFHQ preparation script. The generator is StyleGAN2-ada `stylegan2-ffhq-config-f`, i.e. the FFHQ 1024 face crop. The license is non-commercial only. It is Windows-tested only, and the CUDA extensions are hard to compile. Pretrained models are behind a Google form. A dataset repo, `non-hair-FFHQ`, is linked. The README gives no runtime numbers and says nothing about long hair. — [GitHub oneThousand1000/HairMapper](https://github.com/oneThousand1000/HairMapper)
- The authors bald-converted about 6,000 FFHQ portraits. This is first reported by a secondary Chinese news source — [QbitAI](https://www.qbitai.com/2022/04/33970.html). It is consistent with Stable-Hair's description of NH-FFHQ (about 6k images, made by running FFHQ through HairMapper) — [Stable-Hair arXiv HTML](https://arxiv.org/html/2407.14078v2)

**Stable-Hair (arXiv 2407.14078, July 2024; AAAI 2025)**
- Stage 1 is a "Bald Converter" built on a Latent ControlNet with Stable Diffusion. It is trained on NH-FFHQ, about 6,000 HairMapper-generated bald FFHQ images (from a 20k FFHQ subset): 8,000 steps, batch size 16, lr 5e-5, one H800 GPU. Both stages sample with DDIM at 30 steps, CFG 1.5. — [arXiv HTML v2](https://arxiv.org/html/2407.14078v2)
- The authors claim it works "without requiring image alignment or cropping" and handles half-body shots and cartoon characters. The paper gives no input resolution, no crop specification, and no inference time. The only limitation listed is that hair accessories can be transferred. — [arXiv HTML v2](https://arxiv.org/html/2407.14078v2)
- Implication of the training data: the Bald Converter's supervision comes from HairMapper outputs on FFHQ face crops, so its "teacher" never saw bald shoulders or clothing behind long hair. This is an inference, but it follows directly from the stated data source. — [arXiv HTML v2](https://arxiv.org/html/2407.14078v2)
- GitHub issue #34 asks how to use full-body photos to change hair. Issues #31 and #32 request an improved checkpoint, which suggests the released checkpoint is weaker than the paper's. I read the titles only, not the bodies. — [Stable-Hair issues](https://github.com/Xiaojiu-z/Stable-Hair/issues)

**HairFastGAN (NeurIPS 2024, arXiv 2404.01094)**
- Its Alignment module inpaints with a pretrained SEAN model at **256×256**. It uses FS-space tensors of 16×16, upsampled to 32×32 and 64×64 in post-processing, which inpaints non-matching regions with discriminator and LPIPS losses and a soft-dilated mask (radius-25 kernel). — [arXiv HTML](https://arxiv.org/html/2404.01094v1)
- The authors' stated limitation: "inpaint may not work very well for cases where long hair is replaced by short hair". It gives unrealistic skin texture or shadows of the former hair (their Fig. 6A). SEAN also produces artifacts on weak classes such as ears and "when the hair is not connected to the head". Ponytails and braids are poorly supported, and it sometimes fails to remove hair when the target is bald. — [arXiv HTML](https://arxiv.org/html/2404.01094v1)
- HairShifter notes that the HairCLIPv2 and HairFastGAN baselines rely on face-alignment preprocessing, which causes background inconsistencies. — [HairShifter arXiv HTML](https://arxiv.org/html/2507.12758v1)

**AnyBald (WACV 2026, UNIST 3D Vision & Robotics Lab)**
- It fine-tunes a pretrained latent-diffusion inpainting model, with learnable text prompts instead of an explicit mask at inference. A regularization loss steers attention to hair. A "scalable data augmentation pipeline" synthesizes paired hair/non-hair images for in-the-wild scenes. The authors claim the best FID and CLIP-I on their test set. — [WACV 2026 poster page](https://wacv.thecvf.com/virtual/2026/poster/402); [IEEE Xplore](https://ieeexplore.ieee.org/document/11492067/)
- The project page shows CelebA in-the-wild portraits and **DeepFashion2 full-body images**. It gives no resolution, crop spec, speed, or step count, and has no long-hair or limitations section. Code is "Coming Soon". Bald outputs are fed into DECA and Stable-Hair stage 2. — [Project page](https://vision3d-lab.github.io/anybald)

**HairPort / Bald Konverter (repo says SIGGRAPH '26)**
- The Bald Converter is a LoRA, "in-context adaptation", on FLUX.1-Kontext-dev (a gated base model). LoRA weights `deepmancer/bald_konverter` were released in April 2026. The full HairPort pipeline takes "several minutes per transfer pair", and the repo recommends ≥24 GB VRAM. The repo license is CC BY-NC-ND 4.0, and the FLUX.1 Kontext dev upstream license also applies. — [GitHub deepmancer/HairPort](https://github.com/deepmancer/HairPort)
- The **Baldy dataset** has 6,400 bald/original pairs at 1024×1024, CC BY-NC-ND 4.0. Its pipeline is the most relevant one found for upper-body coverage. Hairstyles from DiffLocks, Hair20K, USC-HairSalon, and CT2Hair are fitted onto **SMPL-X head/body models** and rendered in Blender with varied camera, lighting, and materials. ControlNet++, SDXL, and FLUX Kontext then make the hair/bald pairs photorealistic. Hairstyle IDs carry size tokens XXS–XL, but crop extent and long-hair share are not stated. — [HF dataset card deepmancer/baldy](https://huggingface.co/datasets/deepmancer/baldy)

**Other**
- A 2026 Chinese-language journal paper on identity-preserving portrait hairstyle removal (J. Shandong Univ., Vol. 61, No. 6) exists. I did not read its details. — [SDU journal](https://lxbwk.njournal.sdu.edu.cn/EN/Y2026/V61/I6/95)

### Inferences
- Every StyleGAN method inherits the FFHQ crop, roughly the head plus a little neck. Hair below the shoulders is mostly outside the crop or at its edge, so those methods cannot be used to fill clothing behind long hair. This matches the earlier HairFastGAN experience.
- The diffusion bald converters are the only family with a plausible chance of rebuilding shoulders and clothing behind long hair: they are generative inpainters with a strong image prior and are claimed to work on half-body or full-body images. No paper quantifies this, so it has to be tested on our own long-hair samples.
- Baldy's SMPL-X body + rendered strand hair + diffusion-photorealization pipeline is effectively a published blueprint for generating "upper body with long hair / without" pairs. Its license is NC-ND, though.

### Gaps
- I found no paper reporting quantitative or qualitative results specifically for **hair below the shoulders covering clothing**.
- Exact training resolution and crop for Stable-Hair, AnyBald, and Bald Konverter are not published on the pages I read.
- The AnyBald augmentation pipeline (how pairs are synthesized) is not described on the abstract or project page. The full paper on IEEE Xplore was not accessed.
- I did not access Barbershop's and StyleYourHair's own papers. Their FFHQ-1024 aligned operation is inferred from the shared StyleGAN2-FFHQ backbone, not cited.

## Q2. Speed: per-image latency, and distilled / one-step / real-time variants

### Takeaway
Optimization-based StyleGAN editors take minutes; encoder-based HairFastGAN takes about 0.4–0.8 s per image at its native settings. Diffusion bald converters are 30-step DDIM (Stable-Hair) or FLUX-Kontext-scale. FLUX.1 Kontext dev measures about 0.25–0.6 s **per step** on RTX 5090 or RTX PRO 6000, so several seconds per image. Few-step distilled inpainting (TurboFill, 4 steps) exists. Nothing comes near 30 fps per frame for a generative bald converter. The only real-time hair-video result found (HairShifter, about 20 fps at 512²) gets there by generating once and then animating.

### Cited Findings
- HairFastGAN's own benchmark (CelebA-HQ, median per-image time, models preloaded): HairFast 0.41 s on A100 and 0.78 s on V100; HairCLIP 0.28 / 0.36 s; HairCLIPv2 about 112 / 221 s; StyleYourHair about 84 / 239 s; Barbershop about 213 / 645 s. These are author-measured. — [HairFastGAN arXiv](https://arxiv.org/pdf/2404.01094)
- Stable-Hair uses DDIM with 30 steps per stage. No wall-clock time is published. — [arXiv HTML v2](https://arxiv.org/html/2407.14078v2)
- FLUX.1 Kontext dev transformer latency per diffusion step (NVIDIA-measured): RTX 5090 FP4 273 ms and FP8 358 ms; RTX PRO 6000 Blackwell FP4 254 ms, FP8 317 ms, and BF16 607 ms. Step count and resolution are not given. — [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-flux-1-kontext-for-image-editing-with-low-precision-quantization)
- The Replicate listing says Kontext dev on H100 "typically completes within 60 seconds", and that the figure varies a lot with input. This is a loose upper bound that includes queueing and overhead. — [Replicate](https://replicate.com/lucataco/flux-kontext-dev)
- The full HairPort pipeline (including Bald Konverter) takes "several minutes per transfer pair". — [GitHub HairPort](https://github.com/deepmancer/HairPort)
- TurboFill (CVPR 2025, Adobe) trains an inpainting adapter on the DMD2 few-step distilled T2I model with a 3-step adversarial scheme. With **4 diffusion steps** it is claimed to beat multi-step BrushNet, which needs over 50 steps. — [arXiv 2504.00996](https://arxiv.org/abs/2504.00996v1); [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_TurboFill_Adapting_Few-step_Text-to-image_Model_for_Fast_Image_Inpainting_CVPR_2025_paper.html)
- HairShifter (ACM MM 2025) runs at about 20+ FPS at 512×512 on one RTX 3090 without acceleration. The animation module costs about 1.57 TFLOPs per frame, against about 74.23 TFLOPs for the one-time anchor generation. The authors say competing per-frame methods run at about 1 FPS. These are author-reported. — [HairShifter arXiv HTML](https://arxiv.org/html/2507.12758v1)

### Inferences
- Per-frame generative bald conversion at 30 fps and ≥720p is not feasible with any published model. Even a 4-step distilled SD-class inpainter is probably on the order of 100 ms+ at 1024². That is an estimate, not a measurement.
- Seconds-scale per-session cost is realistic. One Kontext edit at about 20–28 steps × 0.25–0.6 s ≈ 5–17 s on a 5090-class GPU. Stable-Hair at 30 SD steps is likely a few seconds on a modern GPU (unmeasured).

### Gaps
- No published wall-clock times for Stable-Hair's Bald Converter, AnyBald, or Bald Konverter.
- No one-step or turbo-distilled **hair-removal** model was found. TurboFill is generic inpainting, and its exact latency numbers were not extracted.

## Q3. Can a one-off generative "bald + no long hair" keyframe per session be produced, then reused per frame by warping or propagation? Prior art?

### Takeaway
Yes. There is direct prior art for this exact pattern. **HairShifter** generates one expensive "anchor" hair-edited frame and then animates it per frame at about 20 fps at 512² with a lightweight SPADE-style decoder. **Meet-In-Style** (IEEE CG&A 2025) runs diffusion on keyframes, trains a small few-shot patch-based network per session, and then runs at 30 fps. A Snap patent describes real-time temporally stable hairstyle change using previous-prediction conditioning and a predicted warp field and soft mask. Diffusion video editors such as TokenFlow and Rerender propagate keyframes, but they are offline. The main weak point every source flags is long-to-short hair and occluded regions.

### Cited Findings
- **HairShifter**: one static anchor is made with an image hair-transfer module (Stable-Hair fine-tuned on video, about 2k FFHQ images plus about 18k CelebV-HQ frames, more than 60k triplets). The anchor frame is chosen automatically as the source frame whose pose is closest to the reference. A Multi-Scale Gated SPADE decoder then produces each frame from the anchor (hair appearance) and the driving frame (motion and non-hair context). Training data is CelebV-HQ; evaluation used 200 videos × 100 frames. Stated limitations: **long-to-short hair transfer still produces artifacts and relies on consistent video inpainting of the hair region**. There is no secondary hair dynamics, and hand-hair interactions degrade blending. Code is promised. — [arXiv 2507.12758](https://arxiv.org/html/2507.12758v1)
- **Meet-In-Style** (Kunz, Texler, Mould, Sýkora; IEEE CG&A 45(2), 2025) combines diffusion image stylization with few-shot patch-based training to produce a custom image-to-image network. It stylizes at "30 fps on commodity graphics hardware" and was aimed at video conferencing. Training time and resolution are not given on the abstract page. — [CTU DCGI page](https://dcgi.fel.cvut.cz/en/publications/2025/kunz-cga-diffusion/)
- **Texler et al. 2020** (SIGGRAPH / TOG), "Interactive Video Stylization Using Few-Shot Patch-Based Training": an appearance-translation network trained from scratch on a few stylized keyframes, with no pre-training. It supports real-time inference, parallel processing, and random access, and merges multiple keyframes without explicit blending. Code is linked. — [arXiv 2004.14489](https://arxiv.org/abs/2004.14489); [code mirror](https://github.com/zeta1999/few-shot-patch-based-training)
- **Snap patent** (US 2025/0022264, Jan 2025), "Photo-realistic temporally stable hairstyle change in real-time": an encoder-decoder takes the current frame concatenated with the previous prediction. It outputs an output tensor, a **warping field**, and a **soft mask**, and the final result is a convex blend of the warped source and the warped output. It is trained with the "previous frame" sampled 5–15 frames back, or imitated with small affine transforms, because consecutive frames alone fail to train it. It targets about 30 fps. It does not describe how the hair region is masked or filled. — [Patent summary (nweon)](https://patent.nweon.com/39322)
- **TokenFlow** edits keyframes and propagates diffusion features by nearest-neighbor token correspondence. Keyframe sampling cuts per-frame time by about 20%, and DDIM inversion (1000 steps) is the main bottleneck, so it is offline. — [arXiv 2307.10373](https://arxiv.org/pdf/2307.10373)
- AdaFlow (2025) computes correspondences once instead of at every step. This makes long-video editing efficient, but still not real-time. — [arXiv 2502.05433](https://arxiv.org/pdf/2502.05433)
- A 2025 paper combines FRESCO + SDEdit + Ebsynth and Plug-and-Play + TokenFlow pipelines, and reports no speeds. Ebsynth is non-generative example-based patch synthesis. — [arXiv 2512.03905](https://arxiv.org/pdf/2512.03905)

### Inferences
- A practical pattern for heddy, based on the evidence above:
  1. At session start, or when a good frontal frame appears, generate a bald, no-long-hair keyframe in a few seconds with a diffusion bald converter (Stable-Hair stage 1, Bald Konverter, or a fine-tuned 4-step inpainter on an upper-body crop).
  2. Each frame, warp the keyframe's background/shoulder/neck fill into the current hair mask region. Options are face-mesh and pose-driven warping for skin and neck, optical flow, or a HairShifter- or Snap-style small network that predicts the warp and mask from the current frame + previous output + keyframe.
  3. Optionally use multiple keyframes for different yaw angles, as in Texler's multi-keyframe merging.
- Risk: shoulders and clothing move non-rigidly, and the user's real long hair moves and re-occludes. The warped fill is only valid where the keyframe's "behind hair" content still matches the current geometry. Every method flags long-to-short as a weak point.

### Gaps
- No paper was found that does exactly "generative bald keyframe + warped fill for live AR at 30 fps". HairShifter is the closest, and it targets 512² face video, not upper body.
- Meet-In-Style's and Texler's per-session training times and resolutions were not extracted. The Kunz25 PDF exceeded the fetch size limit.

## Q4. Datasets and approaches for training a lightweight specialized "remove long hair from upper-body portrait" network

### Takeaway
Published bald-pair datasets are either FFHQ face crops generated by GAN editing (HairMapper's non-hair-FFHQ / NH-FFHQ, about 6k) or synthetic renders (Baldy: 6,400 pairs at 1024², SMPL-X body + strand hair + diffusion photorealization). AnyBald claims an in-the-wild augmentation pipeline. Licenses are non-commercial (HairMapper NC, Baldy CC BY-NC-ND), so for a product the pipeline idea matters more than the data. The "composite hair onto a hairless upper body" self-supervised approach is consistent with Baldy's design.

### Cited Findings
- non-hair-FFHQ / NH-FFHQ: about 6k bald FFHQ images produced by HairMapper, used as Stable-Hair's Bald Converter training set. Non-commercial. — [HairMapper GitHub](https://github.com/oneThousand1000/HairMapper); [Stable-Hair arXiv](https://arxiv.org/html/2407.14078v2)
- Baldy: 6,400 pairs at 1024×1024. Strand hair from DiffLocks, Hair20K, USC-HairSalon, and CT2Hair is placed on SMPL-X head/body and rendered in Blender, then photorealized with ControlNet++, SDXL, and FLUX Kontext. CC BY-NC-ND 4.0. — [HF baldy](https://huggingface.co/datasets/deepmancer/baldy)
- AnyBald: a "scalable data augmentation pipeline" of paired hair/non-hair images across diverse real-world scenes, including full-body DeepFashion2 examples. Method details and release are pending. — [AnyBald project](https://vision3d-lab.github.io/anybald)
- Stable-Hair and HairShifter build training triplets from CelebV-HQ video (35,666 clips, 15,653 identities) to get pose variation. — [Stable-Hair arXiv](https://arxiv.org/html/2407.14078v2); [HairShifter](https://arxiv.org/html/2507.12758v1)
- Snap's real-time training trick: pair the current frame with a 5–15-frames-earlier frame, or with an affine-jittered copy, so the network learns to use the previous prediction without trivially copying it. — [Patent summary](https://patent.nweon.com/39322)

### Inferences
- A self-supervised pair recipe for the heddy upper-body case, analogous to the FFHQ forehead inpainter:
  - (a) Take upper-body frames of short-haired or bald people, which have real shoulders, neck, and clothing.
  - (b) Paste segmented long-hair layers (alpha mattes) from other people, or render strand grooms via our own 3D groom renderer aligned with face pose, over the shoulders. Randomize length, color, and alpha edge softness.
  - (c) Train a small inpainter to recover (a) given (b) + hair mask. Mask dilation and "ghost hair" alpha fringes must be part of the augmentation, since those cause the LaMa/MI-GAN streaks.
  - This is the 2D-composite version of Baldy's 3D-render pipeline.
- A distillation route: run a slow diffusion bald converter offline on many long-hair upper-body images to make pseudo-pairs, as Stable-Hair did with HairMapper. Then train a fast feed-forward net on those pairs. Quality is bounded by the teacher, and license constraints of the teacher model apply.

### Gaps
- No public commercially licensed upper-body long-hair/no-hair paired dataset was found.
- No public numbers on how well a lightweight feed-forward model trained this way performs on below-shoulder hair. This needs our own experiments.
