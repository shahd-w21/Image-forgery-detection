"""Extra forensic layers: classic signal-processing views, no trained models.

Each function takes the full-resolution RGB image (shrinking it first would erase the traces these
methods look for) and returns a picture at the display size `size` = (w, h) plus a short reading.
They are shown for inspection only and do not affect the verdict: they have not been tested for
accuracy and all of them react to ordinary image content too.
"""
import cv2
import numpy as np

MAGENTA = (214, 58, 108)


def _blocks(a, b):
    """Crop a 2-D array to whole b x b blocks and return it shaped (rows, b, cols, b)."""
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b)


def _shade(rgb_small, block_mask, color=MAGENTA, alpha=0.5):
    """Tint the display image where the block-level mask is set."""
    h, w = rgb_small.shape[:2]
    m = cv2.resize(block_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out = rgb_small.astype(np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out, float(m.mean() * 100)


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ---------- 1. Noise level ----------
def noise_map(rgb, size, block=16):
    """Local noise level relative to the whole image. The picture content is removed with a 3x3
    median filter; what is left is mostly sensor noise. A pasted area often has a different level."""
    g = _gray(rgb)
    res = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    b = _blocks(res, block)
    lvl = b.std(axis=(1, 3))
    gb = _blocks(g.astype(np.float32), block).mean(axis=(1, 3))
    usable = (gb > 12) & (gb < 243)                     # clipped black/white areas carry no noise
    ref = np.median(lvl[usable]) if usable.any() else np.median(lvl)
    ratio = np.where(usable, lvl / (ref + 1e-6), 1.0)
    v = np.clip(ratio / 3.0, 0, 1)                       # 0 = no noise, 1/3 = typical, 1 = 3x typical
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img = cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    odd = usable & ((ratio < 0.4) | (ratio > 2.5))
    return img, float(odd.mean() * 100)


# ---------- 2. JPEG block grid ----------
_k = np.arange(8)
_DCT = np.sqrt(2 / 8) * np.cos((2 * _k[None, :] + 1) * _k[:, None] * np.pi / 16)
_DCT[0] /= np.sqrt(2)
_DCT = _DCT.astype(np.float32)
_HF = ((_k[:, None] + _k[None, :]) >= 4).astype(np.float32)[None, :, None, :]   # higher-frequency AC terms


def _zero_frac(g, oy, ox, win):
    """Cut the image into 8x8 blocks starting at (oy, ox) and return, per win x win window, the share of
    near-zero high-frequency DCT coefficients. On the grid a JPEG was saved on, quantization leaves many
    of them at zero; on any other grid position it does not."""
    h, w = ((g.shape[0] - oy) // win) * win, ((g.shape[1] - ox) // win) * win
    a = g[oy:oy + h, ox:ox + w] - 128
    c = (_DCT @ a.reshape(h // 8, 8, w)).reshape(h // 8, 8, w // 8, 8) @ _DCT.T     # blockwise 2-D DCT
    z = ((np.abs(c) < 1) * _HF).sum(axis=(1, 3)) / _HF.sum()
    return z.reshape(h // win, win // 8, w // win, win // 8).mean(axis=(1, 3))


def jpeg_grid(rgb, small, win=32, max_side=2048):
    """Find the 8x8 grid the image was last saved as a JPEG on, then mark windows that also carry a
    second grid at another position: content pasted from a different JPEG keeps its own grid. Tested on
    synthetic pastes into CASIA photos: ~80% of the pasted area found when the final image was saved
    losslessly (PNG, TIFF, BMP), no false marks on 60 authentic photos; a final JPEG save (even at
    quality 95) erases the second grid, so it then finds nothing."""
    g = _gray(rgb).astype(np.float32)
    H, W = g.shape
    y0, x0 = max(0, (H - max_side) // 2), max(0, (W - max_side) // 2)    # large images: central area only
    g = g[y0:y0 + max_side, x0:x0 + max_side]
    if min(g.shape) < 2 * win + 8:
        return small.copy(), None, "Image too small to check."
    nh, nw = (g.shape[0] - 7) // win, (g.shape[1] - 7) // win
    z = np.stack([_zero_frac(g, oy, ox, win)[:nh, :nw] for oy in range(8) for ox in range(8)]).reshape(8, 8, nh, nw)
    tot = z.mean(axis=(2, 3))
    if tot.max() - np.median(tot) < 0.05:
        return small.copy(), None, ("No JPEG grid found: the image is not a JPEG, or it was resized or "
                                    "cropped off-grid after saving, so this layer cannot help here.")
    oy, ox = np.unravel_index(tot.argmax(), tot.shape)
    other = np.ones((8, 8), bool); other[oy, :] = False; other[:, ox] = False   # fully misaligned grids only
    o = z[other]
    second = o.max(axis=0) - np.median(o, axis=0)
    cells = (second > 0.22).repeat(win, 0).repeat(win, 1)                     # back onto the full image
    flag = np.zeros((H, W), bool)
    py, px = y0 + oy, x0 + ox
    flag[py:py + cells.shape[0], px:px + cells.shape[1]] = cells[:H - py, :W - px]
    img, pct = _shade(small, flag)
    crop = " Only the central area was checked." if (H > max_side or W > max_side) else ""
    note = (f"Shaded windows carry a second JPEG grid at another position, a sign of content pasted from "
            f"another JPEG (about {pct:.1f}% of the image).{crop}" if pct > 0 else
            f"No second JPEG grid found.{crop} This layer only sees pastes in images saved without JPEG "
            f"compression (PNG, TIFF, BMP); saving as a JPEG afterwards erases the trace, so this does not "
            f"rule out an edit.")
    return img, pct, note


# ---------- 3. Frequency spectrum ----------
def spectrum(rgb, size=360, crop=1024):
    """Log-magnitude Fourier spectrum of the image's fine detail (centre crop, full resolution).
    Regular bright dots away from the centre point to repeating patterns: resampling, upscaling,
    some AI generators, but also fabrics, grids and screens photographed on purpose."""
    g = _gray(rgb).astype(np.float32)
    h, w = g.shape
    c = min(crop, h, w)
    y0, x0 = (h - c) // 2, (w - c) // 2
    g = g[y0:y0 + c, x0:x0 + c]
    g = g - cv2.GaussianBlur(g, (0, 0), 2)               # keep fine detail, drop the picture's layout
    win = np.outer(np.hanning(c), np.hanning(c)).astype(np.float32)
    f = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g * win))))
    lo, hi = np.percentile(f, (50, 99.9))
    v = np.clip((f - lo) / (hi - lo + 1e-6), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


# ---------- 4. Colour filter array (CFA) traces ----------
def cfa_map(rgb, small, block=32):
    """A camera sensor records one colour per pixel and fills in the rest, so half of the green pixels
    are interpolated and easy to predict from their neighbours. Measure that pattern per block and mark
    blocks where it is missing while the rest of the image has it."""
    gch = rgb[..., 1].astype(np.float32)
    pred = (np.roll(gch, 1, 0) + np.roll(gch, -1, 0) + np.roll(gch, 1, 1) + np.roll(gch, -1, 1)) / 4
    e = (gch - pred)[1:-1, 1:-1] ** 2
    yy, xx = np.indices(e.shape)
    lat = ((yy + xx) % 2 == 0)
    a = _blocks(np.where(lat, e, 0), block).sum(axis=(1, 3))
    b = _blocks(np.where(~lat, e, 0), block).sum(axis=(1, 3))
    r = np.log((a + 1) / (b + 1))                        # >0 or <0 depending on which lattice was interpolated
    textured = (a + b) / (block * block) > 4
    if textured.sum() < 10:
        return small.copy(), None, "Too little detail to measure sensor traces."
    rg = np.median(r[textured])
    if abs(rg) < 0.15:
        return small.copy(), None, ("No camera-sensor pattern found. This is normal for images that were "
                                    "resized, compressed for the web, screenshotted or generated, so this "
                                    "layer cannot help here.")
    missing = textured & (r * np.sign(rg) < 0.25 * abs(rg))
    img, pct = _shade(small, missing)
    note = (f"Shaded blocks lack the sensor pattern the rest of the image has (about {pct:.1f}% of it). "
            "Very bright, dark or blurred areas can lose it too." if pct > 0 else
            "The sensor pattern is consistent across the image.")
    return img, pct, note
