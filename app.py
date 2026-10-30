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
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    return splice, ai, cfg


splice_model, ai_model, cfg = load()


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
    return dict(p_splice=p_splice, n_copy=int(n_copy), p_ai=p_ai,
                orig=orig, heat=heat, cm=cm, size=img.size, fmt=fmt)


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

st.markdown(FINE, unsafe_allow_html=True)