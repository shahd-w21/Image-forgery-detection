import html, io, json, os
import numpy as np, cv2, streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf

from design import (load_css, join_names, human_size, check_html, meta_html, caption_html,
                    HERO, EXPLAIN, FINE)

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load():
    splice = tf.keras.models.load_model(os.path.join(HERE, "splicing_model.keras"))
    ai = tf.keras.models.load_model(os.path.join(HERE, "ai_model_v2.keras"))
    unet = tf.keras.models.load_model(os.path.join(HERE, "splice_unet_v2.keras"), compile=False)
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    return splice, ai, unet, cfg


splice_model, ai_model, unet_model, cfg = load()


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
def splice_mask(img, size, thr, min_frac=0.0005):
    """U-Net (EfficientNet-B0 encoder) on a 384x384 ELA image -> mask of the estimated edited pixels,
    resized to `size` = (w, h). The prediction is averaged with that of the mirrored image."""
    x = ela_input(img, size=384).astype(np.float32)
    batch = np.stack([x, x[:, ::-1]])
    pr = unet_model.predict(batch, verbose=0)[..., 0]
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
    sm = splice_mask(img, (orig.shape[1], orig.shape[0]), cfg.get("mask_t", 0.3))
    cmk = copy_regions(good, orig.shape)
    return dict(p_splice=p_splice, n_copy=int(n_copy), p_ai=p_ai,
                orig=orig, heat=heat, cm=cm, size=img.size, fmt=fmt,
                splice_ov=tint(orig, sm, (214, 58, 108)), splice_pct=float((sm > 0).mean() * 100),
                copy_ov=tint(orig, cmk, (200, 138, 40)), copy_pct=float((cmk > 0).mean() * 100))


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
    "estimated_edited_area_pct": {
        "splicing": round(r["splice_pct"], 2) if f_splice else None,
        "copy_move": round(r["copy_pct"], 2) if f_copy else None,
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

# estimated edited regions, only for the checks that fired
if f_splice or f_copy:
    st.markdown('<div class="section-title">Where the edit is</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="large")
    with d1:
        if f_splice:
            st.image(r["splice_ov"])
            note = (f"Estimated by a segmentation network: about {r['splice_pct']:.1f}% of the image. "
                    "An estimate, not an exact outline." if r["splice_pct"] > 0 else
                    "The network found no clear region, so the edit may be small or hard to see.")
            st.markdown(caption_html("Edited region (splicing)", note), unsafe_allow_html=True)
    with d2:
        if f_copy:
            st.image(r["copy_ov"])
            note = (f"Shaded areas cover the matched keypoints on both sides of the copy, about "
                    f"{r['copy_pct']:.1f}% of the image. This is an approximate region, not an outline."
                    if r["copy_pct"] > 0 else "Too few matches to outline a region.")
            st.markdown(caption_html("Cloned regions (copy-move)", note), unsafe_allow_html=True)

st.markdown(FINE, unsafe_allow_html=True)