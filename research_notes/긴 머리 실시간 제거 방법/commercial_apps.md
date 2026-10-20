# How commercial real-time AR hair try-on products handle the user's existing long hair (short-style / bald effects)

Research date: 2026-10-19. About 17 search/fetch calls. Primary sources were used where possible (Snap developer docs, L'Oréal patent text, Perfect Corp and Banuba product pages, Google paper). Several vendor docs blocked fetching or did not answer, so some gaps remain. These are listed explicitly.

## Q1. Snap Lens Studio, TikTok Effect House, Meta Spark: do they remove long hair, or only recolor / overlay?

### Takeaway
Snap is the only social-AR platform with documented real-time hair removal: a "bald ML model" layer in its Hair Simulation template that "blocks off existing hair". The docs say nothing about how it treats hair below the head. Snap disables it entirely on weak devices or at low FPS. The platforms' general Hair Segmentation tools only recolor or texture the hair, or replace it with an image, and Snap staff confirmed that segmentation alone cannot hide hair. TikTok Effect House reportedly lists a "Hair Eraser" generative effect, but I could not find its documentation.

### Cited Findings
- **Snap Hair Simulation template (Lens Studio 4.55.1 docs, undated page):** "The `Back Orthographic Camera` in the template contains the `ML bald model`… it is used to block off existing hair from users, creating a smooth look when combined with the hair component." — [Snap Lens Studio: Hair Simulation template](https://developers.snap.com/lens-studio/4.55.1/references/templates/face/hair-simulation)
- **Snap: the bald layer is optional and drops out under load.** "the `ML Bald Layer` will be toggled based on `FPS` and `mlPerformanceIndex` of the device". The `OptimizationController` "would turn on fallback mode and disable bald ML on devices determined by its value". `FPSThreshold` means the lens "would turn on fallback mode if fps is lower than the `FPSThreshold`… regardless of type of the device." — [Snap Lens Studio: Hair Simulation template](https://developers.snap.com/lens-studio/4.55.1/references/templates/face/hair-simulation)
- **Snap: removal and rendering are interdependent.** The docs describe "many inter-connecting pieces such as Blur layer, bald ML and head collider". A `HairBlurController` applies "a layer of smoothness catered to hair's direction". Objects not on a separate camera "will be blurred by the HairBlurController script!" — [Snap Lens Studio: Hair Simulation guide](https://developers.snap.com/lens-studio/4.55.1/references/guides/lens-features/tracking/face/face-effects/hair-simulation); [template page](https://developers.snap.com/lens-studio/4.55.1/references/templates/face/hair-simulation)
- **Snap: the docs never mention long hair below the shoulders.** The only length reference is about rendering cost: "some long curly hair might need to have a higher resolution." The page says nothing about shoulders, the torso, or limits of the bald model. — [Snap Lens Studio: Hair Simulation template](https://developers.snap.com/lens-studio/4.55.1/references/templates/face/hair-simulation)
- **Snap Segmentation template (Portrait Hair):** it segments the hair region and replaces it "with an image, tiled image or post effect coloring". This is recolor or replace, not removal. — [Snap Lens Studio: Segmentation template](https://developers.snap.com/lens-studio/references/templates/face/segmentation); [Fritz.ai tutorial on Lens Studio segmentation textures](https://fritz.ai/segmentation-textures-in-lens-studio/)
- **Snap community forum (older thread, date not retrievable because the page returned 403):** a developer asked how to hide the area where hair used to be when combining a bald effect with segmentation. The reply said segmentation textures pull directly from the camera texture, "so the hair will still be included". In other words, segmentation alone does not erase hair. This comes from the search snippet only; I could not fetch the page. — [Lens Studio community post](https://support.lensstudio.snapchat.com/hc/en-us/community/posts/4403351723412/comments/4403351773076)
- **Snapchat's consumer "Bald Character" lens (viral in 2021):** made by Snap, it lets you see what it would look like "if you straight up just buzzed your head". Articles give no technical details and do not say how it handles long hair. — [Distractify, 2021](https://www.distractify.com/p/bald-filter-snapchat)
- **Snap's 2020 facial-hair add/remove lenses** use "the same types of technology of other ML Lenses, such as the Facecraft lenses and age slider". This suggests a GAN-style image-to-image ML Lens, but the article does not name the method. — [Social Samosa, June 2020](https://www.socialsamosa.com/2020/06/snapchat-new-filters-to-add-remove-facial-hair); [search summary of related coverage](https://www.qradio.com/archive/snapchat-saving-people-from-hair-disasters-with-new-filters/)
- **TikTok Effect House:** third-party guides list "Hair Segmentation" under Segmentation, next to Portrait, Hand and Clothing Segmentation. They list "Hair Eraser" under Generative Effects, next to Eyebrow Eraser, Smile and Pucker. — [Minter.io guide](https://minter.io/blog/how-to-create-your-own-tiktok-filters/); [CapCut Effect House guide](https://www.capcut.com/resource/tiktok-effect-house)
- **Effect House Hair Segmentation in practice** is used for recoloring, with adjustable color, softness, radius and blend mode. — [CapCut Effect House guide](https://www.capcut.com/resource/tiktok-effect-house)
- **TikTok's bald filter** came after Snapchat's: "After going viral, Tiktok released a filter similar to Snapchat's Bald character." — [itgeared](https://www.itgeared.com/how-to-get-the-bald-filter-on-tiktok/)
- **Google (YouTube-scale AR) hair ML from 2019 is recolor-only.** "Real-time Hair Segmentation and Recoloring on Mobile GPUs" (CVPR 2019 workshop, arXiv 1907.06740) presents a small segmentation network running at roughly 30 to 100+ FPS on mobile GPUs, plus a recoloring scheme. It does no removal. The authors say it is deployed "in a major AR application" used by millions. — [arXiv 1907.06740](https://arxiv.org/pdf/1907.06740)

### Inferences
- Snap's documented design suggests that real-time bald removal is treated as best-effort and discardable. When the device cannot afford it, the lens falls back to drawing the new hair over the user's real hair. In practice this works for styles that cover the original hair, such as long styles, and fails for short or bald targets.
- The docs never mention how the bald model handles hair over the shoulders, and the hair-simulation templates are full head-of-hair styles. Snap therefore probably never claims clean removal of below-shoulder hair. This is an inference, not a documented statement.
- The Snap docs pair the bald layer with a direction-aware blur on the rendered hair. This hints that soft, blurry hair rendering is used to hide boundary errors between the erased region and the new hair (labeled inference).

### Gaps
- No architecture, resolution or training details for Snap's "ML bald model". I could not find whether it is a GAN image-to-image model and whether its output covers the shoulder and chest region.
- No primary Effect House documentation for "Hair Eraser". The official docs did not come up in search, so its scope (head only, or also long hair) and whether it runs live are unknown.
- Meta Spark (Instagram/Facebook): no documentation found on hair removal or a bald effect in this pass. From prior knowledge, which I did not re-verify here, Meta shut down the Spark AR platform in January 2025, so current Instagram effects are first-party only.
- No review or video description found that documents what the Snap or TikTok bald lenses do with long hair hanging over the shoulders.

## Q2. Perfect Corp (YouCam), ModiFace (L'Oréal), Banuba, Meitu, DeepAR, FaceApp etc.: live video or still photo for hairstyle change and hair removal? How is long hair handled?

### Takeaway
In shipping beauty-tech products, live AR is consistently used for hair color. Hairstyle or length changes, which require removing hair, are offered mostly as still-photo generative features: Perfect Corp's GAN and later GenAI "AI Hairstyle", L'Oréal Style My Hair upload-a-photo, and Perfect Corp's photo-in/image-out API. Banuba is the exception: it claims "live and photo-based" hairstyle try-on but publishes no details on removing existing hair. No vendor documents real-time inpainting of the body behind long hair.

### Cited Findings
- **ModiFace / L'Oréal Professionnel "Style My Hair":** offers "3D real-time haircolor try-on" in the app and in salons, presented as the first professional haircare brand to scale it. — [L'Oréal news](https://www.loreal.com/en/india/news/brands/l-oreal-professionnel-integrates-3-d-real-time-hair-color)
- **Style My Hair, live color:** a face tracker detects the hair, applies virtual color to that area, and the color follows the hair in real time as the head moves. — [L'Oréal China article on Style My Hair](https://www.lorealchina.com/zh/articles/science-and-technology/haircolor-virtual-try-on-loreal-professionnel-style-myhair/)
- **Style My Hair, haircuts:** users "upload a picture and experiment with haircuts, hairstyles, and professional hair colours". Style change is photo-based. — [Fashion Magazine](https://fashionmagazine.com/beauty-grooming/loreal-makes-moves-tech-industry-two-game-changing-beauty-apps/)
- **ModiFace hair model training:** reportedly 220,000 manually annotated facial images were used to train hair shape and structure detection. — [Nuvo Magazine](https://nuvomagazine.com/beauty/how-loreal-is-leading-the-future-of-beauty-with-artificial-intelligence)
- **Perfect Corp, Oct 2022:** GAN-based hairstyle simulation with 12 styles, including short ones such as "pixie cut" and "buzz cut", plus "AR hair color try on". The styles are photo-generated and the color is AR. — [Cosmetics Business](https://www.cosmeticsbusiness.com/buzz-cut-or-bob-perfect-corp-launches-virtual-tech-so-consumers-can-try-on-new-hairstyles-204287); [BusinessWire, 2022-10-13](https://www.businesswire.com/news/home/20221013005644/en)
- **Perfect Corp, Oct 2023:** a generative AI hairstyle update in YouCam Makeup, described as moving beyond GAN. The company claimed users can visualize hairstyles and lengths "in real-time". The press text is ambiguous about whether this means live camera or fast photo generation. — [Mann Publications, 2023-10-29](https://www.mannpublications.com/fashionmannuscript/2023/10/29/perfect-corp-unveils-generative-ai-technology-for-hairstyling-in-youcam-makeup-app-with-virtual-try-on-solution/)
- **YouCam Makeup consumer flow (photo):** "Open the app and tap 'AI Hairstyle'. Choose a photo." The guide even suggests chaining filters on saved images. — [Perfect Corp blog: hair extensions try-on](https://www.perfectcorp.com/consumer/blog/hair/hair-extensions-try-on)
- **Perfect Corp AI Hairstyle API (2026 page):** you POST "the user image file" plus a style_id and get back "the transformed high-res image URL". It has a 150+ style library. "AI will first check if the uploaded photo meets the requirements." I could not read the actual requirements. There is no live-video mode. — [Perfect Corp AI Hairstyle API](https://yce.perfectcorp.com/ai-api/products/ai-hairstyle-api)
- **Perfect Corp Hair and Beard API suite (June 2026):** adds AI Hair Extension try-on and "AI Hair Length Detection for automated measurement from a photo". — [BusinessWire, 2026-06-16](https://secure.businesswire.com/news/home/20260616828286/en/Perfect-Corp.-Launches-Industrys-Most-Comprehensive-AI-Hair-Beard-API-Suite-Combining-Virtual-Try-On-with-Intelligent-Hair-Analysis)
- **Independent 2025 review of YouCam hairstyles:** some styles "can look a bit artificial, especially with more dramatic changes". This comes from the search summary; the review did not detail long-to-short artifacts. — [Perfect Corp blog roundup / review cited in search](https://www.perfectcorp.com/consumer/blog/hair/best-ai-hairstyle-testing-apps)
- **Banuba Virtual Hairstyle Try-On:** "Live and photo-based detection". It uses face tracking plus "hairline and head geometry detection" for placement and "hair region detection" for covered areas. Rendering is described as "AI-based simulation" accounting for light, volume and gravity. It runs on mobile (GLES 3.0+/Metal), desktop and WebGL2, and web store integrations are processed in the cloud. The page does not say whether existing hair is removed. — [Banuba Virtual Hairstyle Try-On](https://www.banuba.com/virtual-hairstyle-try-on)
- **Banuba on its own color product:** "accuracy with hair ends recognition is the pain point of all virtual hair coloring solutions". — [Banuba blog: hair segmentation / recolor](https://banuba.com/blog/hair-segmentation-virtual-hair-color-try-on)
- **Banuba Hair AI page:** describes a "virtual hair length try-on" to see different lengths, with no technical detail. — [Banuba Hair AI](https://www.banuba.com/hair-ai)
- **Other photo apps (BeautyPlus, Budgetpixel etc.)** describe short-hair filters as photo AI edits. BeautyPlus says it maps facial features and overlays the selected style. All of this is marketing copy, not a test. — [BeautyPlus short hair filter](https://www.beautyplus.com/ai-hairstyle/short-hair-filter); [Budgetpixel AI short hair filter](https://budgetpixel.com/features/ai-short-hair-filter)

### Inferences
- The industry pattern suggests the long-hair erasure problem has mostly been sidestepped commercially. Products do it offline on a single photo, where heavy generative inpainting with seconds of latency is acceptable. In live video, they keep the user's hair (recolor) or cover it with a voluminous 3D style.
- Perfect Corp's photo-requirement gate ("check if the uploaded photo meets the requirements") is consistent with restricting inputs. Whether it rejects loose long hair is unknown.

### Gaps
- Meitu (BeautyCam/Meitu Xiuxiu), DeepAR, FaceApp: no primary or review sources found in this pass on whether their short-hair or bald effects run live, or on long-hair artifacts.
- Perfect Corp's actual image requirements (FAQ answer not rendered) and the meaning of "real-time" in the 2023 GenAI hairstyle release.
- No source documents Banuba removing existing long hair in live mode, or what artifacts it produces.

## Q3. Patents describing hair removal or inpainting (especially real-time, especially the body region behind long hair)

### Takeaway
The most relevant patent found is L'Oréal FR3161301B3: a real-time 3D hair try-on on live smartphone or webcam video, filed 2024-04-15 and published 2026-05-15. Its removal step is classical and cheap: dilate the hair mask, inpaint with Telea fast marching, render a skin-tone scalp mesh, and apply Poisson blending, with a GAN given as an alternative. For long or voluminous hair, it explicitly falls back to telling the user to tie a tight ponytail. It does not address hair hanging over the shoulders or chest. Snap's hair patents found are about rendering and reconstruction, not removal.

### Cited Findings
- **L'Oréal FR3161301B3, "3D Virtual Hair Try-On Method and System":** assignee L'Oréal SA, filed 2024-04-15 (FR2024003874), published 2026-05-15. Inventors include Tang Jason Tsz To and Yousaf Hamza. It describes "replacing the user's natural hair with a 3D representation" on live video (smartphone or webcam). — [PatSnap: FR3161301B3](https://eureka.patsnap.com/patent/FR3161301B3)
- **FR3161301B3 removal pipeline (quoted):**
  - [0036] "the user is virtually bald by the hair removal filter 112 before the new hair model is rendered over it"
  - [0037] "The hair segmentation mask is expanded to increase its coverage area"
  - [0038] inpainting with "a known technique (e.g., the fast marching method, Telea, 2003)"
  - [0039–0040] "A scalp mesh is aligned with the face mesh… rendered on the image, with skin material of a skin tone similar to the face… Poisson blending is also applied to smooth the transition"
  - [0041] "a GAN model… may be provided to process the input image and remove the hair."
  - — [PatSnap: FR3161301B3](https://eureka.patsnap.com/patent/FR3161301B3)
- **FR3161301B3 long-hair handling (quoted, [0026]):** users with long hair or high-volume styles that "interfere with a 3D hairstyle to be applied" are "advised to apply a tight ponytail or other tight hairstyle". In this alternative embodiment no baldness filter is used, and users "can refuse and proceed in all cases." — [PatSnap: FR3161301B3](https://eureka.patsnap.com/patent/FR3161301B3)
- **FR3161301B3, body region:** the face mesh can optionally include scalp, neck and shoulders. A head, neck and shoulders approximation serves as a collision mesh ([0028]). The occlusion depth map "is initialized with the depth of the face, neck, and shoulders" ([0059]). The patent text read (about 100k of 112k characters) contains no treatment of erasing hair over the shoulders, back or chest, no background reconstruction, and no frame-rate numbers. — [PatSnap: FR3161301B3](https://eureka.patsnap.com/patent/FR3161301B3)
- **Snap US 12,374,036 B2 "Single image three-dimensional hair reconstruction"** covers reconstruction and rendering, not removal. — [Google Patents US12374036](https://patents.google.com/patent/US12374036)
- **Snap US 12,190,422 B2 "Hair styles system for rendering hair strands based on hair spline data" (Jan 2025)** is a rendering patent. — [USPTO Gazette US12190422](https://patentsgazette.uspto.gov/week01/OG/html/1530-1/US12190422-20250107.html)
- **US12002136B2 "Style try-on with reverse GANs" (hairstyle try-on):** uses GAN inversion to fill missing hair structure during transfer. It is photo or offline GAN work, not real-time body inpainting. — [Google Patents US12002136](https://patents.google.com/patent/US12002136)
- **Academic (non-product) context:** Stable-Hair (AAAI) trains a "Bald Converter" diffusion stage before transfer. AnyBald (WACV 2026) does diffusion-based in-the-wild hair removal. Both are far from a real-time budget. — [Stable-Hair arXiv 2407.14078](https://arxiv.org/html/2407.14078v2); [AnyBald, WACV 2026](https://wacv.thecvf.com/virtual/2026/poster/402)

### Inferences
- The L'Oréal patent is the clearest evidence of what a major vendor considers shippable for live removal. It uses a cheap non-learned inpaint (Telea), a geometric skin proxy (scalp mesh) and Poisson blending around the head. The hard long-hair case is handed to the user ("tie a ponytail") rather than solved algorithmically. This matches our own finding that the region under long hair has no recoverable information.
- Telea fast marching produces smeared or streaky fills on large regions, which is similar to the diffusion blobs we observed. The patent's reliance on it suggests L'Oréal accepted artifacts only for small, near-head regions. This is inference; the patent gives no quality claims.

### Gaps
- I did not find a Snap, Meitu, Perfect Corp or Google patent specifically claiming real-time inpainting of shoulder, neck or clothing regions behind long hair. Full-text USPTO/Espacenet searches (e.g., "hair removal" AND "inpaint" AND assignee) were not done within this budget.
- The last ~12k characters of the FR3161301B3 page were not read, though the claims section may restate the same content.

## Q4. Typical UX guidance given to users (tie hair back, hair behind shoulders, framing)

### Takeaway
The only explicit long-hair UX guidance found in a primary vendor source is L'Oréal's patent: advise "a tight ponytail or other tight hairstyle" and let the user decline. Other vendor-facing guidance found is generic (lighting, single subject, photo validation gates).

### Cited Findings
- **L'Oréal:** users with long or voluminous hair are "advised to apply a tight ponytail or other tight hairstyle"; users "can refuse and proceed in all cases." — [PatSnap: FR3161301B3](https://eureka.patsnap.com/patent/FR3161301B3)
- **Perfect Corp API:** an automatic pre-check that "the uploaded photo meets the requirements" before generation. The requirement text was not retrievable. — [Perfect Corp AI Hairstyle API](https://yce.perfectcorp.com/ai-api/products/ai-hairstyle-api)
- **Generic third-party guidance:** "well-lit photos with one main subject usually give the cleanest long-hair previews". — [Budgetpixel AI long hair filter](https://budgetpixel.com/features/ai-long-hair-filter)
- **Generic capture guidance:** "avoid strong backlight and very low resolution" (not Banuba-specific). — [aiindigo tutorial](https://aiindigo.com/tutorials/getting-started-with-hairstyleai-realistic-virtual-hair-try-ons)
- **Snap:** the Hair Simulation docs give no user-facing guidance about existing hair. Their only mitigations are developer-side: blur the rendered hair and disable bald ML on slow devices. — [Snap Lens Studio: Hair Simulation template](https://developers.snap.com/lens-studio/4.55.1/references/templates/face/hair-simulation)

### Inferences
- For a live product, the evidence-backed options are those used by L'Oréal and Snap: (a) ask the user to tie hair back or move it behind the shoulders, with opt-out, (b) restrict live short or bald styles when long hair is detected, or fall back to overlay, and (c) keep heavy generative removal for a still-photo "snapshot" mode, as Perfect Corp and Style My Hair do. Applying this to heddy is a recommendation, not a documented industry rule.

### Gaps
- No sources found for in-app onboarding text, such as "put your hair behind your shoulders", in Snapchat, TikTok, YouCam, Style My Hair, Meitu or FaceApp. Checking would need hands-on app testing or screenshots, which were out of scope here.
- No product review was found that systematically documents long-hair-over-shoulder artifacts for any live bald or short-hair effect.
