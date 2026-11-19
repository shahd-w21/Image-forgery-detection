import html, io, json, os
import numpy as np, cv2, streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

from design import (load_css, join_names, human_size, check_html, meta_html, caption_html,
                    HERO, EXPLAIN, FINE, LAYERS_INTRO)
import forensics

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load():
    m = lambda f, **kw: tf.keras.models.load_model(os.path.join(HERE, f), **kw)
    splice = m("splicing_model.keras")
    ai = m("ai_model_v2.keras")
    # Edited-region localizers, each used for the edit type it scored best on (CASIA v2 test set):
    #   noise: RGB + SRM noise residuals, best for splicing (F1 0.87 vs 0.85) and 5x fewer false marks
    #   ela:   ELA image, best for copy-move (F1 0.69 vs 0.60) and small edits
    loc = {"noise": m("localizer_noise.keras", compile=False), "ela": m("localizer_ela.keras", compile=False)}
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    return splice, ai, loc, cfg


splice_model, ai_model, localizers, cfg = load()


# ---------- Detector 1: splicing (ELA + CNN) ----------
def ela_diff(img, quality=90):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality); buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    return np.abs(np.asarray(img, dtype=np.int16) - np.asarray(resaved, dtype=np.int16))


def ela_input(img, scale=15, size=128):
    d = np.clip(ela_diff(img) * scale, 0, 255).astype(np.uint8)
    return cv2.resize(d, (size, size))


# ---------- Detector 2: copy-move (SIFT) ----------
def copy_move_score(rgb, min_dist=40, ratio=0.7, max_side=800):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    s = min(1.0, max_side / max(gray.shape))
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s)
    kp, des = cv2.SIFT_create(nfeatures=3000).detectAndCompute(gray, None)
    if des is None or len(kp) < 10:
        return 0, [], s
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des, des, k=4)
    pairs = []
    for m in knn:
        m = [x for x in m if x.trainIdx != x.queryIdx]
        if len(m) < 2:
            continue
        if m[0].distance < ratio * m[1].distance:
            p1 = np.array(kp[m[0].queryIdx].pt); p2 = np.array(kp[m[0].trainIdx].pt)
            if np.linalg.norm(p1 - p2) > min_dist:
                pairs.append((p1, p2))
    if len(pairs) < 4:
        return 0, [], s
    src = np.float32([p[0] for p in pairs]); dst = np.float32([p[1] for p in pairs])
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    if inl is None:
        return 0, [], s
    good = [p for p, k in zip(pairs, inl.ravel()) if k]
    return len(good), good, s


# ---------- Detector 3: AI-generated (EfficientNet-B0, 224x224) ----------
def ai_score(img):
    x = np.asarray(img.convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.float32)[None]
    return float(ai_model.predict(x, verbose=0)[0][0])


# ---------- Evidence images ----------
def evidence(img, rgb, good, max_side=800):
    h, w = rgb.shape[:2]
    k = min(1.0, max_side / max(h, w))          # same scale the copy-move detector used
    size = (max(1, int(w * k)), max(1, int(h * k)))
    orig = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if k < 1 else rgb.copy()

    d = cv2.GaussianBlur(ela_diff(img).sum(axis=2).astype(np.float32), (0, 0), 3)
    heat = np.clip(d / 60.0 * 255, 0, 255).astype(np.uint8)          # fixed scale, comparable across images
    heat = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, size, interpolation=cv2.INTER_AREA)

    cm = orig.copy()
    for p1, p2 in good:
        a = tuple(p1.astype(int)); b = tuple(p2.astype(int))
        cv2.line(cm, a, b, (214, 58, 108), 2)
        cv2.circle(cm, a, 5, (255, 255, 255), -1)
        cv2.circle(cm, b, 5, (255, 255, 255), -1)
    return orig, heat, cm


# ---------- Pixel-level localisation ----------
def edit_mask(img, rgb, kind, size, thr, min_frac=0.0005):
    """Segmentation network -> mask of the estimated edited pixels, resized to `size` = (w, h).
    kind 'noise' sees the RGB image (it computes the noise residuals itself); kind 'ela' sees the ELA
    image. Both work at 384x384, exactly as trained. The prediction is averaged with the mirrored one."""
    if kind == "noise":
        x = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        x = ela_input(img, size=384).astype(np.float32)
    batch = np.stack([x, x[:, ::-1]])
    pr = localizers[kind].predict(batch, verbose=0)[..., 0]
    p = (pr[0] + pr[1][:, ::-1]) / 2
    m = (cv2.resize(p, size, interpolation=cv2.INTER_LINEAR) > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)      # drop tiny specks
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_frac * m.size:
            m[lab == i] = 0
    return m * 255


def copy_regions(good, shape):
    """Approximate cloned regions: filled hull of the matched keypoints on each side of the copy."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if len(good) < 3:
        return mask
    p1 = np.array([p[0] for p in good]); p2 = np.array([p[1] for p in good])
    v = p2 - p1
    ref = v[0] / (np.linalg.norm(v[0]) + 1e-9)
    flip = (v @ ref) < 0                                         # each pair appears in both directions
    a = np.where(flip[:, None], p2, p1); b = np.where(flip[:, None], p1, p2)
    for pts in (a, b):
        pts = np.unique(pts.astype(np.int32), axis=0)
        if len(pts) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def cutout(rgb, mask, pad=0.08):
    """The edited pixels alone: everything else made transparent, cropped to the region with a margin."""
    m = mask > 0
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    h, w = m.shape
    py, px = int((ys.max() - ys.min() + 1) * pad) + 4, int((xs.max() - xs.min() + 1) * pad) + 4
    y0, y1 = max(0, ys.min() - py), min(h, ys.max() + py + 1)
    x0, x1 = max(0, xs.min() - px), min(w, xs.max() + px + 1)
    rgba = np.dstack([rgb, (m * 255).astype(np.uint8)])
    return rgba[y0:y1, x0:x1]


def tint(rgb, mask, color, alpha=0.45):
    out = rgb.astype(np.float32)
    m = mask > 0
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


# ---------- One cached analysis per uploaded file ----------
@st.cache_data(show_spinner=False, max_entries=8)
def analyse(data: bytes):
    im = Image.open(io.BytesIO(data))
    fmt = im.format or "Unknown"
    img = ImageOps.exif_transpose(im).convert("RGB")      # respects phone-photo rotation
    rgb = np.asarray(img)
    p_splice = float(splice_model.predict(ela_input(img)[None], verbose=0)[0][0])
    n_copy, good, _ = copy_move_score(rgb)
    p_ai = ai_score(img)
    orig, heat, cm = evidence(img, rgb, good)
    size = (orig.shape[1], orig.shape[0])

    # Pick the localizer for the edit type that was flagged. The copy-move check (SIFT) is the more
    # specific of the two, so when it fires the edit is most likely a clone -> ELA localizer.
    kind = "ela" if n_copy > cfg["copy_move_t"] else ("noise" if p_splice > cfg["splice_t"] else None)
    out = dict(p_splice=p_splice, n_copy=int(n_copy), p_ai=p_ai, orig=orig, heat=heat, cm=cm,
               size=img.size, fmt=fmt, loc_kind=kind)
    if kind:
        em = edit_mask(img, rgb, kind, size, cfg[f"mask_t_{kind}"])
        buf = io.BytesIO(); Image.fromarray(em).save(buf, "PNG")
        out.update(edit_ov=tint(orig, em, (214, 58, 108)), edit_pct=float((em > 0).mean() * 100),
                   edit_cut=cutout(orig, em), edit_mask_png=buf.getvalue())
    if kind == "ela":
        cmk = copy_regions(good, orig.shape)
        out.update(copy_ov=tint(orig, cmk, (200, 138, 40)), copy_pct=float((cmk > 0).mean() * 100))
    return out


@st.cache_data(show_spinner=False, max_entries=4)
def extra_layers(data: bytes):
    """The optional forensic layers, computed on the full-resolution image only when asked for."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    rgb = np.asarray(img)
    h, w = rgb.shape[:2]
    k = min(1.0, 800 / max(h, w))
    size = (max(1, int(w * k)), max(1, int(h * k)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if k < 1 else rgb.copy()
    noise_img, noise_pct = forensics.noise_map(rgb, size)
    grid_img, grid_pct, grid_note = forensics.jpeg_grid(rgb, small)
    cfa_img, cfa_pct, cfa_note = forensics.cfa_map(rgb, small)
    return dict(noise=noise_img, noise_pct=noise_pct, grid=grid_img, grid_pct=grid_pct, grid_note=grid_note,
                fft=forensics.spectrum(rgb), cfa=cfa_img, cfa_pct=cfa_pct, cfa_note=cfa_note)


# ---------- Page ----------
st.set_page_config(page_title="Image Forgery Detection", layout="wide")
st.markdown(load_css(), unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

up = st.file_uploader("Image to examine", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
                      label_visibility="collapsed")

if up is None:
    st.markdown(EXPLAIN, unsafe_allow_html=True)
    st.markdown(FINE, unsafe_allow_html=True)
    st.stop()

with st.spinner("Examining the image"):
    r = analyse(up.getvalue())

p_splice, n_copy, p_ai = r["p_splice"], r["n_copy"], r["p_ai"]
f_splice = p_splice > cfg["splice_t"]
f_copy = n_copy > cfg["copy_move_t"]
f_ai = p_ai > cfg["ai_t"]

flagged = []
if f_splice: flagged.append("splicing")
if f_copy: flagged.append("copy-move")
if f_ai: flagged.append("AI generation")

# verdict
if flagged:
    title, state = "Potentially forged.", "flag"
    sub = f"Flagged by {len(flagged)} of 3 checks: {join_names(flagged)}."
else:
    title, state = "No signs of forgery.", "clear"
    sub = "All three checks stayed below their thresholds."
st.markdown(
    f'<div class="verdict"><div class="bar {state}"></div>'
    f'<div><div class="verdict-title">{title}</div><div class="verdict-sub">{sub}</div></div></div>',
    unsafe_allow_html=True)

# file details
w_px, h_px = r["size"]
st.markdown(meta_html([
    ("File", html.escape(up.name)),
    ("Dimensions", f"{w_px} × {h_px} px"),
    ("Format", html.escape(r["fmt"])),
    ("File size", human_size(up.size)),
]), unsafe_allow_html=True)

# the three checks, each on a gauge with its threshold marked and a shaded flag zone
copy_max = max(2 * cfg["copy_move_t"], n_copy * 1.15, 1)
rows = (
    check_html("Splicing", "Content pasted in from another image", "Error-level analysis and a CNN",
               f"{p_splice:.2f}", min(p_splice, 1) * 100, cfg["splice_t"] * 100,
               f"{cfg['splice_t']:.2f}", "1", f"threshold {cfg['splice_t']:.2f}", f_splice)
    + check_html("Copy-move", "A region duplicated inside the same image", "SIFT keypoint matching",
                 f"{n_copy}", n_copy / copy_max * 100, cfg["copy_move_t"] / copy_max * 100,
                 f"{cfg['copy_move_t']}", f"{copy_max:.0f}", f"threshold {cfg['copy_move_t']} matches", f_copy)
    + check_html("AI generation", "An image made or altered by a generative model", "EfficientNet-B0 classifier",
                 f"{p_ai:.2f}", min(p_ai, 1) * 100, cfg["ai_t"] * 100,
                 f"{cfg['ai_t']:.2f}", "1", f"threshold {cfg['ai_t']:.2f}", f_ai)
)
st.markdown('<div class="section-title">The three checks</div>'
            f'<div class="checks">{rows}</div>', unsafe_allow_html=True)

# downloadable report
report = {
    "file": up.name, "format": r["fmt"], "dimensions_px": [w_px, h_px],
    "verdict": "potentially forged" if flagged else "no signs of forgery",
    "flagged_checks": flagged,
    "scores": {"splicing": round(p_splice, 3), "copy_move_matches": n_copy, "ai_generation": round(p_ai, 3)},
    "thresholds": {"splicing": cfg["splice_t"], "copy_move_matches": cfg["copy_move_t"], "ai_generation": cfg["ai_t"]},
    "edited_region": {
        "localizer": {"ela": "ELA segmentation network (copy-move)",
                      "noise": "noise segmentation network (splicing)"}.get(r["loc_kind"]),
        "estimated_area_pct": round(r["edit_pct"], 2) if r["loc_kind"] else None,
        "copy_move_matched_area_pct": round(r["copy_pct"], 2) if r["loc_kind"] == "ela" else None,
    },
}
st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
st.download_button("Save report", data=json.dumps(report, indent=2),
                   file_name="forgery_report.json", mime="application/json")

# evidence prints
st.markdown('<div class="section-title">Evidence</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.image(r["orig"])
    st.markdown(caption_html("Submitted image", "As uploaded, with phone rotation corrected."),
                unsafe_allow_html=True)
with c2:
    st.image(r["heat"])
    st.markdown(caption_html("Compression map",
                             "Brighter areas respond differently to re-saving. A compact patch with sharp edges "
                             "is the sign to look for. Edges and fine texture also glow, so read it together "
                             "with the splicing score."), unsafe_allow_html=True)
with c3:
    st.image(r["cm"])
    st.markdown(caption_html("Duplicate matches",
                             "Lines join regions that match each other. No lines means no duplicated region "
                             "was found."), unsafe_allow_html=True)

# estimated edited region, only when splicing or copy-move fired, with the original beside it
if r["loc_kind"]:
    st.markdown('<div class="section-title">Where the edit is</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="large")
    with d1:
        st.image(r["orig"])
        st.markdown(caption_html("Original", "For comparison with the marked region."), unsafe_allow_html=True)
    with d2:
        st.image(r["edit_ov"])
        how = ("Copy-move was flagged, so this map comes from the network that reads the compression "
               "pattern, which did best on cloned regions in our tests."
               if r["loc_kind"] == "ela" else
               "Splicing was flagged, so this map comes from the network that reads the sensor noise, "
               "which did best on pasted content in our tests.")
        note = (f"About {r['edit_pct']:.1f}% of the image. {how} An estimate, not an exact outline."
                if r["edit_pct"] > 0 else
                f"No clear region found, so the edit may be small or hard to see. {how}")
        st.markdown(caption_html("Edited region", note), unsafe_allow_html=True)
    with d3:
        if r["edit_cut"] is not None:
            st.image(r["edit_cut"])
            st.markdown(caption_html("Edited pixels alone",
                                     "Only the pixels the network marked as edited, cut out of the image and "
                                     "enlarged. Everything else is removed."), unsafe_allow_html=True)
            st.download_button("Save the mask (PNG)", data=r["edit_mask_png"],
                               file_name="edited_region_mask.png", mime="image/png")
    if r["loc_kind"] == "ela":
        f1, _, _ = st.columns(3, gap="large")
        with f1:
            st.image(r["copy_ov"])
            note = (f"Shaded areas cover the matching keypoints on both sides of the copy (the original and "
                    f"the duplicate), about {r['copy_pct']:.1f}% of the image. Only one side is the edit."
                    if r["copy_pct"] > 0 else "Too few matches to outline a region.")
            st.markdown(caption_html("Matching areas (copy-move)", note), unsafe_allow_html=True)

# optional forensic layers: for inspection only, they never change the verdict
st.markdown('<div class="section-title">Extra forensic layers</div>', unsafe_allow_html=True)
st.markdown(LAYERS_INTRO, unsafe_allow_html=True)
if st.toggle("Show the extra layers"):
    with st.spinner("Running the extra layers"):
        x = extra_layers(up.getvalue())
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.image(x["noise"])
        st.markdown(caption_html(
            "Noise level",
            "How much fine grain each area has compared with the image as a whole: dark is less, bright is more. "
            "A patch that stands out from similar surroundings may come from another camera or have been smoothed. "
            "Flat areas such as sky are naturally darker."), unsafe_allow_html=True)
    with e2:
        st.image(x["grid"])
        st.markdown(caption_html("JPEG block grid", x["grid_note"]), unsafe_allow_html=True)
    e3, e4 = st.columns(2, gap="large")
    with e3:
        st.image(x["cfa"])
        st.markdown(caption_html("Camera sensor pattern (CFA)", x["cfa_note"]), unsafe_allow_html=True)
    with e4:
        st.image(x["fft"])
        st.markdown(caption_html(
            "Frequency spectrum",
            "The fine detail of the image's centre, broken into repeating patterns. A plain glow fading from "
            "the centre is normal. Regular bright dots away from the centre point to resizing, upscaling or some "
            "AI generators, but fabrics, grids and screens produce them too."), unsafe_allow_html=True)

st.markdown(FINE, unsafe_allow_html=True)