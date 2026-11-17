# Image Forgery Detection

Upload an image and get a verdict (authentic or potentially forged), the type of forgery found, and visual evidence showing where.

Three independent checks run on every image. If any one passes its threshold, the image is reported as potentially forged.

| Check | Method |
| --- | --- |
| Splicing (content pasted from another image) | Error Level Analysis (ELA) + small CNN |
| Copy-move (a region cloned inside the same image) | SIFT keypoint matching + RANSAC |
| AI-generated image | EfficientNet-B0 classifier (transfer learning) |

When a check fires, the app also shows an estimated edited region:

- **Splicing:** a U-Net takes the ELA image and predicts a mask of the edited pixels.
- **Copy-move:** the matched keypoints on each side of the copy are outlined as two approximate regions.

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
| `design.py` | HTML pieces and page text |
| `style.css` | Colours, fonts and layout |
| `.streamlit/config.toml` | Forces light mode |
| `config.json` | Thresholds for the three checks, plus the U-Net mask threshold (`mask_t`) |
| `splicing_model.keras` | Splicing classifier (ELA + CNN) |
| `ai_model_v2.keras` | AI-generated image classifier |
| `splice_unet.keras` | U-Net for pixel-level splicing localization |
| `Image_forgery_detection.ipynb` | Colab notebook: data, training, evaluation |

## Data

- **CASIA v2.0** (Kaggle: `divg07/casia-20-image-tampering-detection-dataset`): authentic and tampered images with ground-truth masks. Used for the splicing classifier, the copy-move evaluation and the U-Net.
- **A Hugging Face dataset of real and AI images** from several modern generators, plus CASIA authentic images, for the AI classifier.

## Limitations

- Scores are statistical indicators, not proof.
- CASIA authentic and tampered images can differ in file format and compression, so part of the measured accuracy may come from that rather than from the edits.
- ELA works best on JPEG and is weakened by heavy compression, resizing and screenshots.
- The edited-region maps are estimates. On CASIA test images the U-Net reached a mean pixel IoU of 0.49 and F1 of 0.57 (median F1 0.72), and it missed about one edit in five entirely, more often the smallest ones.
