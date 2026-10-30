"""Presentation layer for the forgery-detection app.

style.css  - all colours, type and layout (edit this to restyle the page)
design.py  - the HTML snippets and page copy that use those styles
app.py     - detection logic and page flow only
"""
import html, os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_css():
    with open(os.path.join(HERE, "style.css"), encoding="utf-8") as f:
        return f"<style>{f.read()}</style>"


# ---------- Small HTML helpers ----------
def join_names(items):
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def human_size(n):
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / (1024 * 1024):.1f} MB"


def check_html(name, what, method, value, fill_pct, tick_pct, tick_label, scale_max, thr_text, flagged):
    state = "flag" if flagged else "clear"
    label = "Flagged" if flagged else "Clear"
    aria = html.escape(f"{name}: {value}, {thr_text}", quote=True)
    return (
        f'<div class="check">'
        f'<div><div class="check-name">{name}</div><div class="check-what">{what}</div>'
        f'<div class="check-method">{method}</div></div>'
        f'<div class="gauge-wrap"><div class="gauge" role="img" aria-label="{aria}">'
        f'<div class="zone" style="--t:{tick_pct:.1f}%"></div>'
        f'<div class="fill {state}" style="--w:{fill_pct:.1f}%"></div>'
        f'<div class="tick" style="--t:{tick_pct:.1f}%" data-v="{tick_label}"></div></div>'
        f'<div class="scale"><span>0</span><span>{scale_max}</span></div></div>'
        f'<div class="reading"><div class="num">{value}</div>'
        f'<div class="state {state}">{label}</div><div class="thr">{thr_text}</div></div>'
        f'</div>'
    )


def meta_html(items):
    cells = "".join(f'<div><span>{k}</span><b>{v}</b></div>' for k, v in items)
    return f'<div class="meta">{cells}</div>'


def caption_html(title, text):
    return f'<div class="cap"><b>{title}</b><span>{text}</span></div>'


# ---------- Page copy ----------
HERO = """
<div class="mark"><span class="bars" aria-hidden="true"><i></i><i></i><i></i><i></i></span>Image Forgery Detection</div>
<div class="hero-title">Was this image altered?</div>
<p class="hero-sub">Upload a photo. Three checks look for pasted objects, cloned regions, and AI generation, and show where the evidence is.</p>
"""

EXPLAIN = """
<div class="explain">
<div><div class="check-name">Splicing</div><p>Looks for regions saved with a different compression history than the rest of the image, a sign that something was pasted in from another source.</p><p class="method">Error-level analysis and a CNN</p></div>
<div><div class="check-name">Copy-move</div><p>Finds areas that appear twice inside the same image, such as an object cloned to hide something or to add more of it.</p><p class="method">SIFT keypoint matching</p></div>
<div><div class="check-name">AI generation</div><p>Estimates whether the image came from a generative model such as Midjourney, DALL·E, or Stable Diffusion.</p><p class="method">EfficientNet-B0 classifier</p></div>
</div>
"""

FINE = """
<div class="fine">Scores are statistical indicators, not proof. Heavy compression, resizing, and screenshots can hide forgery clues. The AI check misses about one in four images from the generators it was trained on, so newer generators may pass undetected.</div>
"""