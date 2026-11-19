# Image Forgery Detection

Upload an image and get a verdict (authentic or potentially forged), the type of forgery found, and visual evidence showing where.

Three independent checks run on every image. If any one passes its threshold, the image is reported as potentially forged.

| Check | Method |
| --- | --- |
| Splicing (content pasted from another image) | Error Level Analysis (ELA) + small CNN |
| Copy-move (a region cloned inside the same image) | SIFT keypoint matching + RANSAC |
| AI-generated image | EfficientNet-B0 classifier (transfer learning) |

When splicing or copy-move fires, the app shows the original next to an estimated edited region, with the share of the image it covers. Two segmentation networks (U-Nets, EfficientNet-B0 encoders, 384 px) were trained on the same data and split, and each is used for the edit type it scored best on:

| Flagged check | Edited-region model | Why |
| --- | --- | --- |
| Copy-move | **ELA localizer**: reads the ELA image | Best on cloned regions (F1 0.69 vs 0.60) and small edits |
| Splicing only | **Noise localizer**: reads the RGB image plus SRM noise residuals, no ELA | Best on pasted content (F1 0.87 vs 0.85) and 5x fewer false marks on authentic images |

When copy-move fires, the matched keypoints on each side of the copy are also outlined as two approximate regions. The copy-move check (SIFT) takes priority because it rarely fires on authentic images, while the splicing classifier also fires on many copy-move edits.

An optional **Extra forensic layers** section adds four classic views of the full-resolution image. They use no trained model and never change the verdict:

| Layer | What it shows | When it helps |
| --- | --- | --- |
| Noise level | Fine grain per area, relative to the whole image | Pasted or smoothed patches; flat areas are naturally dark |
| JPEG block grid | Areas carrying a second, shifted 8×8 JPEG grid | Pastes from another JPEG, only if the final image was saved losslessly (PNG, TIFF, BMP). In synthetic tests: ~80% of the paste found, no false marks on 60 authentic photos; a final JPEG save erases the trace |
| Camera sensor pattern (CFA) | Areas missing the colour-filter interpolation pattern | Camera originals only; resized, web-compressed or generated images have no pattern. In a synthetic test: 99% of a paste marked, under 1% elsewhere |
| Frequency spectrum | Repeating patterns in the fine detail | Resizing, upscaling, some AI generators; also triggered by fabrics, grids and screens |

## Run the app

Python 3.12 is recommended. On Windows, tick "Add Python to PATH" when installing.

```
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app opens at http://localhost:8501.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Models, the detectors, region estimation and page flow |
| `forensics.py` | The extra forensic layers (noise, JPEG grid, CFA, spectrum) |
| `design.py` | HTML pieces and page text |
| `style.css` | Colours, fonts and layout |
| `.streamlit/config.toml` | Forces light mode |
| `config.json` | Thresholds for the three checks, plus the mask thresholds of the two localizers (`mask_t_noise`, `mask_t_ela`, chosen on validation) |
| `splicing_model.keras` | Splicing classifier (ELA + CNN) |
| `ai_model_v2.keras` | AI-generated image classifier |
| `localizer_ela.keras` | Edited-region model on the ELA image, used for copy-move |
| `localizer_noise.keras` | Edited-region model on RGB + noise residuals, used for splicing |
| `splice_unet_v2.keras` | Previous edited-region model; no longer used by the app |
| `training/train_localizer.py` | Prepares CASIA, trains and evaluates both localizers (`prep`, `train`, `eval`, `compare`) |
| `training/train_localizer_colab.ipynb` | The same script as a Colab notebook (GPU training) |
| `Image_forgery_detection.ipynb` | Colab notebook: data, training, evaluation of the three checks |

## Data

- **CASIA v2.0** (Kaggle: `divg07/casia-20-image-tampering-detection-dataset`): authentic and tampered images with ground-truth masks. Used for the splicing classifier, the copy-move evaluation and the U-Net.
- **A Hugging Face dataset of real and AI images** from several modern generators, plus CASIA authentic images, for the AI classifier.

## Limitations

- Scores are statistical indicators, not proof.
- CASIA authentic and tampered images can differ in file format and compression, so part of the measured accuracy may come from that rather than from the edits.
- ELA works best on JPEG and is weakened by heavy compression, resizing and screenshots.
- The edited-region maps are estimates. On the same 782 CASIA test edits:

  | | ELA localizer | Noise localizer |
  | --- | --- | --- |
  | Mean F1 / median F1 / mean IoU | 0.74 / 0.91 / 0.68 | 0.69 / 0.90 / 0.63 |
  | F1 on splicing / copy-move | 0.85 / 0.69 | 0.87 / 0.60 |
  | Edits missed entirely (all / under 1% of the image) | 12% / 28% | 15% / 36% |
  | Area falsely marked on authentic images | 12.4% | 2.6% |

  The splicing gap (0.87 vs 0.85) is small and may be chance. These two were trained with a new split; the previous model's figures (F1 0.70) came from a different split, so they are not directly comparable.
- The edited-region model is chosen from which check fired, and a copy-move edit that SIFT misses but the splicing classifier catches is mapped by the noise localizer, which is weaker on copies.
