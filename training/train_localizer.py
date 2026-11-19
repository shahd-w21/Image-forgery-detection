"""Train and evaluate pixel-level edit localizers on CASIA v2.

Two models, same decoder, same loss, same data, same split, so the only difference is what they see:
  ela    - one EfficientNet-B0 encoder on the ELA image (the recipe of splice_unet_v2.keras)
  noise  - two EfficientNet-B0 encoders: one on the RGB image, one on SRM noise residuals
           (fixed high-pass filters that strip the content and keep the sensor noise). No ELA.

Train on a GPU (the Colab notebook train_localizer_colab.ipynb runs these steps):
  python train_localizer.py prep    --data <CASIA2 root> --cache <dir>
  python train_localizer.py train   --cache <dir> --mode noise --out models/
  python train_localizer.py eval    --cache <dir> --model models/unet_noise.keras --mode noise
  python train_localizer.py compare --cache <dir> --models "ela=ela:a.keras,noise=noise:b.keras"
The split is derived from file names, so every machine gets the same train/val/test images.
The saved .keras files use only built-in layers, so the app loads them with TensorFlow as usual.
The app uses them as localizer_ela.keras and localizer_noise.keras (re-saved without optimizer state).
"""
import argparse, io, json, os, random, sys

import cv2
import numpy as np
from PIL import Image

SIZE = 384
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------- data
def ela_384(img, quality=90, scale=15):
    """Identical to app.py's ela_input(img, size=384)."""
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=quality); buf.seek(0)
    re = Image.open(buf).convert("RGB")
    d = np.abs(np.asarray(img, np.int16) - np.asarray(re, np.int16))
    return cv2.resize(np.clip(d * scale, 0, 255).astype(np.uint8), (SIZE, SIZE))


def find_dirs(root):
    au = tp = gt = None
    for d, subs, files in os.walk(root):
        b = os.path.basename(d).lower()
        n = sum(f.lower().endswith(EXTS) for f in files)
        if n < 100:
            continue
        if "groundtruth" in b.replace(" ", "") or b in ("gt", "masks", "mask"):
            gt = gt or d
        elif b == "au":
            au = au or d
        elif b == "tp":
            tp = tp or d
    return au, tp, gt


def mask_for(tp_name, gt_index):
    stem = os.path.splitext(tp_name)[0]
    return gt_index.get(stem + "_gt") or gt_index.get(stem)


def split_of(name, seed=42):
    """Deterministic 70/15/15 split from the file name alone (no listdir-order dependence)."""
    r = random.Random(f"{seed}:{name}").random()
    return "train" if r < 0.70 else ("val" if r < 0.85 else "test")


def prep(args):
    au, tp, gt = find_dirs(args.data)
    print("Au:", au, "\nTp:", tp, "\nGT:", gt)
    if not (tp and gt):
        sys.exit("Could not find the Tp and ground-truth folders.")
    gt_index = {os.path.splitext(f)[0]: os.path.join(gt, f) for f in os.listdir(gt) if f.lower().endswith(EXTS)}

    items = []
    for f in sorted(os.listdir(tp)):
        if f.lower().endswith(EXTS) and (m := mask_for(f, gt_index)):
            items.append((os.path.join(tp, f), m, 1))
    au_files = sorted(f for f in os.listdir(au) if f.lower().endswith(EXTS)) if au else []
    au_test = [f for f in au_files if split_of(f) == "test"]
    items += [(os.path.join(au, f), None, 0) for f in au_test]      # authentic images: test only (false alarms)
    print(f"tampered with masks: {sum(i[2] for i in items)} | authentic (test): {len(au_test)}")

    os.makedirs(args.cache, exist_ok=True)
    n = len(items)
    mm = lambda k, shape: np.lib.format.open_memmap(os.path.join(args.cache, k + ".npy"), "w+", np.uint8, shape)
    rgb, ela, msk = mm("rgb", (n, SIZE, SIZE, 3)), mm("ela", (n, SIZE, SIZE, 3)), mm("mask", (n, SIZE, SIZE))
    meta, keep = [], []
    for i, (p, m, lab) in enumerate(items):
        try:
            img = Image.open(p).convert("RGB")
            rgb[i] = cv2.resize(np.asarray(img), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            ela[i] = ela_384(img)
            if m:
                g = np.asarray(Image.open(m).convert("L"))
                if g.shape[:2] != img.size[::-1]:
                    raise ValueError("mask size differs from image size")
                msk[i] = (cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
            else:
                msk[i] = 0
            name = os.path.basename(p)
            meta.append(dict(i=i, name=name, tampered=lab,
                             kind=("copy-move" if name.startswith("Tp_S") else "splice") if lab else "authentic",
                             split=split_of(name) if lab else "test",
                             area=float(msk[i].mean())))
        except Exception as e:
            print("skip", p, e)
        if i % 500 == 0:
            print(f"{i}/{n}", flush=True)
    for a in (rgb, ela, msk):
        a.flush()
    json.dump(meta, open(os.path.join(args.cache, "meta.json"), "w"))
    from collections import Counter
    print(Counter((m["split"], m["kind"]) for m in meta))


# ---------------------------------------------------------------- model
import keras
from keras import layers, ops

SKIPS = ["block6a_expand_activation", "block4a_expand_activation",
         "block3a_expand_activation", "block2a_expand_activation"]


def encoder(name):
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                             input_shape=(SIZE, SIZE, 3))
    return keras.Model(base.input, [base.get_layer("top_activation").output] +
                       [base.get_layer(s).output for s in SKIPS], name=f"{name}_encoder")


def srm_layer():
    """Three standard SRM high-pass kernels (as in RGB-N), each applied to the channel mean. Frozen."""
    k1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4
    k2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12
    k3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2
    w = np.zeros((5, 5, 3, 3), np.float32)
    for j, k in enumerate((k1, k2, k3)):
        for c in range(3):
            w[:, :, c, j] = k / 3
    conv = layers.Conv2D(3, 5, padding="same", use_bias=False, trainable=False, name="srm")
    return conv, w


def decoder(top, skips):
    x = top
    for f, s in zip((256, 128, 64, 32), skips):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, s])
        for _ in range(2):
            x = layers.Activation("relu")(layers.BatchNormalization()(layers.Conv2D(f, 3, padding="same")(x)))
    x = layers.UpSampling2D()(x)
    for _ in range(2):
        x = layers.Activation("relu")(layers.BatchNormalization()(layers.Conv2D(16, 3, padding="same")(x)))
    return layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)


def build(mode):
    inp = layers.Input((SIZE, SIZE, 3))
    if mode == "ela":
        top, *skips = encoder("ela")(inp)
    else:
        conv, w = srm_layer()
        res = conv(inp)
        conv.set_weights([w])
        res = layers.Rescaling(4.0, offset=127.5, name="srm_scale")(res)   # residuals ~[-32, 32] -> ~[0, 255]
        rt, *rs = encoder("rgb")(inp)
        nt, *ns = encoder("noise")(res)
        top = layers.Concatenate()([rt, nt])
        skips = [layers.Concatenate()([a, b]) for a, b in zip(rs, ns)]
    return keras.Model(inp, decoder(top, skips), name=f"unet_{mode}")


def loss_fn(y, p):
    """BCE + Dice + extra BCE weight on a band around the true mask edge (sharper outlines)."""
    p = ops.clip(p, 1e-6, 1 - 1e-6)
    bce = -(y * ops.log(p) + (1 - y) * ops.log(1 - p))
    dil = ops.max_pool(y, 7, 1, "same")
    ero = -ops.max_pool(-y, 7, 1, "same")
    w = 1.0 + 4.0 * (dil - ero)
    inter = ops.sum(y * p, axis=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (ops.sum(y, axis=(1, 2, 3)) + ops.sum(p, axis=(1, 2, 3)) + 1)
    return ops.mean(bce * w, axis=(1, 2, 3)) + dice


# ---------------------------------------------------------------- training
class Batches(keras.utils.PyDataset):
    def __init__(self, cache, mode, idx, bs, train, **kw):
        super().__init__(**kw)
        self.x = np.load(os.path.join(cache, ("ela" if mode == "ela" else "rgb") + ".npy"), mmap_mode="r")
        self.m = np.load(os.path.join(cache, "mask.npy"), mmap_mode="r")
        self.idx, self.bs, self.train = np.array(idx), bs, train
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def on_epoch_end(self):
        if self.train:
            np.random.shuffle(self.idx)

    def __getitem__(self, b):
        ids = np.sort(self.idx[b * self.bs:(b + 1) * self.bs])
        x = self.x[ids].astype("float32")
        y = self.m[ids].astype("float32")[..., None]
        if self.train:
            flip = np.random.rand(len(ids)) < 0.5            # flips only: resizing/blurring would erase the traces
            x[flip] = x[flip, :, ::-1]; y[flip] = y[flip, :, ::-1]
        return x, y


def ids(meta, split, tampered=1):
    return [m["i"] for m in meta if m["split"] == split and m["tampered"] == tampered]


def train(args):
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    keras.utils.set_random_seed(42)
    model = build(args.mode)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss=loss_fn)
    tr = Batches(args.cache, args.mode, ids(meta, "train"), args.bs, True)
    va = Batches(args.cache, args.mode, ids(meta, "val"), args.bs, False)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"unet_{args.mode}.keras")
    cbs = [keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
           keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
           keras.callbacks.CSVLogger(os.path.join(args.out, f"log_{args.mode}.csv"))]
    model.fit(tr, validation_data=va, epochs=args.epochs, callbacks=cbs, verbose=2)
    print("saved", path)


# ---------------------------------------------------------------- evaluation
def clean(m, min_frac=0.0005):
    """Same speck removal as app.py's splice_mask."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_frac * m.size:
            m[lab == i] = 0
    return m


def predict(model, x, idx=None, bs=8):
    """Flip-averaged probabilities (as in the app), returned as float16 to keep memory low."""
    idx = np.arange(len(x)) if idx is None else np.asarray(idx)
    out = []
    for b in range(0, len(idx), bs):
        xb = np.asarray(x[idx[b:b + bs]], dtype="float32")
        p = model.predict(np.concatenate([xb, xb[:, :, ::-1]]), verbose=0)[..., 0]
        k = len(xb)
        out.append(((p[:k] + p[k:, :, ::-1]) / 2).astype(np.float16))
    return np.concatenate(out)


def scores(pred, gt):
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return f1, iou


GRID = [0.2, 0.3, 0.4, 0.5, 0.6]


def val_f1(pv, gv, t):
    return float(np.mean([scores(clean((p > t).astype(np.uint8)).astype(bool), g)[0] for p, g in zip(pv, gv)]))


def model_probs(path, mode, cache, vi, ti):
    model = keras.models.load_model(path, compile=False)
    X = np.load(os.path.join(cache, ("ela" if mode == "ela" else "rgb") + ".npy"), mmap_mode="r")
    pv, pt = predict(model, X, vi), predict(model, X, ti)
    del model; keras.backend.clear_session()
    return pv, pt


def evaluate(args):
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    M = np.load(os.path.join(args.cache, "mask.npy"), mmap_mode="r")
    vi = ids(meta, "val"); test = [m for m in meta if m["split"] == "test"]
    pv, P = model_probs(args.model, args.mode, args.cache, vi, [m["i"] for m in test])
    gv = M[vi].astype(bool)
    t = max(GRID, key=lambda t: val_f1(pv, gv, t))                   # chosen on validation, frozen for test
    res = summarize(test, M, P, t)
    res = {"model": args.model, "mode": args.mode, **res}
    print(json.dumps(res, indent=2))
    if args.report:
        json.dump(res, open(args.report, "w"), indent=2)


def compare(args):
    """Score several models and their pairwise blends on the same test images.
    --models label=mode:path,label=mode:path,...   Blend weight and threshold are chosen on validation."""
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    M = np.load(os.path.join(args.cache, "mask.npy"), mmap_mode="r")
    vi = ids(meta, "val"); test = [m for m in meta if m["split"] == "test"]; ti = [m["i"] for m in test]
    gv = M[vi].astype(bool)
    probs = {}
    for spec in args.models.split(","):
        label, rest = spec.split("=", 1); mode, path = rest.split(":", 1)
        print("predicting", label, flush=True)
        probs[label] = model_probs(path, mode, args.cache, vi, ti)

    out = {}
    for label, (pv, pt) in probs.items():
        t = max(GRID, key=lambda t: val_f1(pv, gv, t))
        out[label] = {"threshold": t, **summarize(test, M, pt, t)}
        print(label, json.dumps(out[label]), flush=True)
    labels = list(probs)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            (av, at), (bv, bt) = probs[a], probs[b]
            best = max(((w, t) for w in (0.3, 0.5, 0.7) for t in GRID),
                       key=lambda wt: val_f1(wt[0] * av.astype(np.float32) + (1 - wt[0]) * bv, gv, wt[1]))
            w, t = best
            pt = (w * at.astype(np.float32) + (1 - w) * bt).astype(np.float16)
            key = f"{w:.1f}*{a} + {1 - w:.1f}*{b}"
            out[key] = {"threshold": t, **summarize(test, M, pt, t)}
            print(key, json.dumps(out[key]), flush=True)
    if args.report:
        json.dump(out, open(args.report, "w"), indent=2)


def summarize(test, M, P, t):
    rows = []
    for m, p in zip(test, P):
        pr = clean((p > t).astype(np.uint8)).astype(bool)
        g = M[m["i"]].astype(bool)
        if m["tampered"]:
            f1, iou = scores(pr, g)
            rows.append(dict(kind=m["kind"], area=m["area"], f1=f1, iou=iou, missed=bool((pr & g).sum() == 0)))
        else:
            rows.append(dict(kind="authentic", fp_area=float(pr.mean()), any_fp=bool(pr.any())))

    T = [r for r in rows if r["kind"] != "authentic"]; A = [r for r in rows if r["kind"] == "authentic"]
    f = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else float("nan")
    return {
        "threshold": t, "n_test_tampered": len(T), "n_test_authentic": len(A),
        "mean_f1": f(T, "f1"), "median_f1": float(np.median([r["f1"] for r in T])), "mean_iou": f(T, "iou"),
        "f1_splice": f([r for r in T if r["kind"] == "splice"], "f1"),
        "f1_copy_move": f([r for r in T if r["kind"] == "copy-move"], "f1"),
        "missed": f(T, "missed"),
        "missed_small_lt1pct": f([r for r in T if r["area"] < 0.01], "missed"),
        "f1_small_lt1pct": f([r for r in T if r["area"] < 0.01], "f1"),
        "authentic_any_false_region": f(A, "any_fp"), "authentic_mean_false_area": f(A, "fp_area"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval", "compare"])
    ap.add_argument("--models", help="compare: label=mode:path,label=mode:path,...")
    ap.add_argument("--data"); ap.add_argument("--cache", required=True)
    ap.add_argument("--mode", choices=["ela", "noise"], default="noise")
    ap.add_argument("--out", default="models"); ap.add_argument("--model"); ap.add_argument("--report")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    {"prep": prep, "train": train, "eval": evaluate, "compare": compare}[a.cmd](a)
