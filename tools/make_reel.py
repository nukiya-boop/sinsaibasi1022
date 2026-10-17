"""心斎橋 禅園 すき焼きPR リール動画（1080x1920 / 30秒 / 外国人向け・和牛推し）を生成する。

usage: python3 tools/make_reel.py <fonts_dir> <output.mp4> [bgm_source.mp4]
"""
import math
import subprocess
import sys

import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, DUR = 1080, 1920, 30, 30.0
NF = int(DUR * FPS)
FONTS = sys.argv[1]
OUT = sys.argv[2]
BGM = sys.argv[3] if len(sys.argv) > 3 else None
RESERVE_URL = "https://www.tablecheck.com/shops/zen-en-shinsaibashi/reserve"

GOLD = (214, 178, 112)
WHITE = (250, 246, 238)


def font(name, size):
    return ImageFont.truetype(f"{FONTS}/{name}.ttf", size)


F_KICK = font("CormorantGaramond-normal-600", 38)
F_TITLE = "CormorantGaramond-italic-500"
F_JP = font("ShipporiMincho-normal-500", 36)


# ---------- easing ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_io(p):
    p = clamp(p)
    return p * p * (3 - 2 * p)


def ease_out(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_in_out_expo(p):
    p = clamp(p)
    if p < 0.5:
        return (2 ** (20 * p - 10)) / 2
    return (2 - 2 ** (-20 * p + 10)) / 2


# ---------- shots ----------
BASE_SCALE = 1.25  # 余白を持たせてズーム・パンできるようにする


def load_base(path, cx=0.5, cy=0.5):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]
    if iw / ih > W / H:
        cw, ch = int(ih * W / H), ih
    else:
        cw, ch = iw, int(iw * H / W)
    x0 = int(clamp(cx * iw - cw / 2, 0, iw - cw))
    y0 = int(clamp(cy * ih - ch / 2, 0, ih - ch))
    crop = img[y0:y0 + ch, x0:x0 + cw]
    base = cv2.resize(crop, (int(W * BASE_SCALE), int(H * BASE_SCALE)), interpolation=cv2.INTER_AREA)
    # 暖色寄りのグレーディング
    f = base.astype(np.float32) / 255.0
    f = np.clip((f - 0.5) * 1.06 + 0.5, 0, 1)
    f[..., 0] *= 1.03
    f[..., 2] *= 0.96
    return np.clip(f, 0, 1)


def kenburns(base, zoom, px=0.0, py=0.0, extra=1.0, dy=0.0):
    """zoom=1 で base 全体を表示。px/py は -1..1 で余白内のパン。extra は遷移用追加ズーム。"""
    bh, bw = base.shape[:2]
    a = zoom * extra * W / bw
    vis_w, vis_h = W / (zoom * W / bw), H / (zoom * W / bw)
    cx = bw / 2 + px * (bw - vis_w) / 2
    cy = bh / 2 + py * (bh - vis_h) / 2
    M = np.float32([[a, 0, W / 2 - a * cx], [0, a, H / 2 - a * cy + dy]])
    return cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def fast_blur(img, sigma):
    if sigma < 0.5:
        return img
    s = 4
    small = cv2.resize(img, (W // s, H // s), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / s)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = (1 - 0.38 * (((XX - W / 2) / (W * 0.75)) ** 2 + ((YY - H / 2) / (H * 0.7)) ** 2))[..., None]
VIGNETTE = np.clip(VIGNETTE, 0.55, 1)


def grad_band(y_center, height, strength):
    """テロップ可読性のための柔らかい暗幕"""
    d = np.abs(YY - y_center) / (height / 2)
    g = np.clip(1 - d, 0, 1) ** 1.6 * strength
    return (1 - g)[..., None]


# ---------- transitions ----------
def tr_zoom(A_fn, B_fn, p):
    e = ease_in_out_expo(p)
    A = A_fn(extra=1 + 0.35 * e)
    B = B_fn(extra=1.25 - 0.25 * e)
    blur = 22 * math.sin(math.pi * p)
    mix = (1 - e) * A + e * B
    return fast_blur(mix, blur)


def tr_shoji(A_fn, B_fn, p):
    """障子が開くように縦スリットで切り替え"""
    A, B = A_fn(), B_fn()
    n = 6
    sw = W / n
    mask = np.zeros((H, W), np.float32)
    for i in range(n):
        q = ease_in_out_expo(clamp((p - i * 0.06) / (1 - 0.06 * (n - 1))))
        x0 = int(i * sw)
        x1 = int((i + 1) * sw)
        if i % 2 == 0:
            mask[: int(H * q), x0:x1] = 1
        else:
            mask[H - int(H * q):, x0:x1] = 1
    mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
    out = A * (1 - mask) + B * mask
    # スリットの金の縁
    edge = np.abs(cv2.Sobel(mask[..., 0], cv2.CV_32F, 0, 1, ksize=3))
    edge = np.clip(edge * 3, 0, 1)[..., None]
    gold = np.array(GOLD, np.float32) / 255
    return out * (1 - edge) + gold * edge


def tr_lightleak(A_fn, B_fn, p):
    A, B = A_fn(extra=1 + 0.05 * p), B_fn(extra=1.05 - 0.05 * p)
    e = ease_io(p)
    mix = A * (1 - e) + B * e
    k = math.sin(math.pi * p) ** 1.5
    cx = W * (0.2 + 0.6 * p)
    leak = np.exp(-(((XX - cx) / (W * 0.55)) ** 2 + ((YY - H * 0.35) / (H * 0.5)) ** 2))[..., None]
    col = np.array([1.0, 0.62, 0.28], np.float32)
    return np.clip(mix + leak * col * 0.45 * k, 0, 1)


def tr_diagonal(A_fn, B_fn, p):
    A, B = A_fn(), B_fn(extra=1.08 - 0.08 * ease_out(p))
    e = ease_in_out_expo(p)
    proj = (XX * 0.5 + YY * 0.866) / (W * 0.5 + H * 0.866)
    t = -0.1 + 1.2 * e
    mask = np.clip((t - proj) / 0.04 + 0.5, 0, 1)[..., None]
    out = A * (1 - mask) + B * mask
    line = np.exp(-((proj - t) / 0.004) ** 2)[..., None] * (0 < p < 1)
    gold = np.array(GOLD, np.float32) / 255
    return out * (1 - line) + gold * line


def tr_whip(A_fn, B_fn, p):
    e = ease_in_out_expo(p)
    A = A_fn(dy=-e * H)
    B = B_fn(dy=(1 - e) * H)
    off = int(round((1 - e) * H))
    out = A.copy()
    out[off:] = B[off:]
    k = int(1 + 120 * math.sin(math.pi * p))
    return cv2.blur(out, (1, k)) if k > 2 else out


def tr_iris(A_fn, B_fn, p):
    A, B = A_fn(), B_fn(extra=1.12 - 0.12 * ease_out(p))
    e = ease_in_out_expo(p)
    r = e * math.hypot(W, H) * 0.55
    dist = np.sqrt((XX - W / 2) ** 2 + (YY - H * 0.48) ** 2)
    mask = np.clip((r - dist) / 6 + 0.5, 0, 1)[..., None]
    out = A * (1 - mask) + B * mask
    ring = np.exp(-((dist - r) / 3.0) ** 2)[..., None] * (0 < p < 1)
    gold = np.array(GOLD, np.float32) / 255
    return out * (1 - ring) + gold * ring


def tr_defocus(A_fn, B_fn, p):
    e = ease_io(p)
    A = fast_blur(A_fn(), 30 * e)
    B = fast_blur(B_fn(), 30 * (1 - e))
    return A * (1 - e) + B * e


# ---------- telop ----------
def tracked(draw, xy, text, fnt, fill, tracking):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += fnt.getlength(ch) + tracking


def tracked_width(text, fnt, tracking):
    return sum(fnt.getlength(c) for c in text) + tracking * (len(text) - 1)


def render_telop(t, tel):
    """t: テロップ開始からの経過秒。RGBA の PIL 画像を返す。"""
    dur = tel["dur"]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    out_a = 1 - ease_io((t - (dur - 0.35)) / 0.35)
    lift = -14 * ease_io((t - (dur - 0.35)) / 0.35)
    y = tel["y"] + lift

    # kicker（字間が詰まりながらフェードイン）
    pk = ease_out(t / 0.7)
    tr = 10 + 14 * (1 - pk)
    kw = tracked_width(tel["kicker"], F_KICK, tr)
    tracked(d, ((W - kw) / 2, y), tel["kicker"], F_KICK, GOLD + (int(255 * pk * out_a),), tr)

    # title（1文字ずつ浮き上がる）
    size = 108
    ft = font(F_TITLE, size)
    while ft.getlength(tel["title"]) > 900:
        size -= 4
        ft = font(F_TITLE, size)
    tw = ft.getlength(tel["title"])
    x = (W - tw) / 2
    ty = y + 56
    for i, ch in enumerate(tel["title"]):
        pc = ease_out((t - 0.25 - i * 0.035) / 0.55)
        if pc > 0:
            d.text((x, ty + 26 * (1 - pc)), ch, font=ft, fill=WHITE + (int(255 * pc * out_a),))
        x += ft.getlength(ch)

    # 金の細線
    pl = ease_in_out_expo((t - 0.6) / 0.7)
    lw = 280 * pl
    ly = ty + size + 34
    if lw > 1:
        d.line([(W / 2 - lw / 2, ly), (W / 2 + lw / 2, ly)], fill=GOLD + (int(230 * out_a),), width=2)
        if pl > 0.95:
            r = 5
            d.polygon([(W / 2, ly - r), (W / 2 + r, ly), (W / 2, ly + r), (W / 2 - r, ly)],
                      fill=GOLD + (int(255 * out_a),))

    # 日本語サブ
    pj = ease_out((t - 0.9) / 0.6)
    jw = tracked_width(tel["jp"], F_JP, 6)
    tracked(d, ((W - jw) / 2, ly + 30 + 10 * (1 - pj)), tel["jp"], F_JP,
            WHITE + (int(225 * pj * out_a),), 6)

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(layer.getchannel("A").point(lambda v: v * 0.85))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    return Image.alpha_composite(shadow, layer)


def composite(frame, layer):
    arr = np.asarray(layer, dtype=np.float32) / 255.0
    a = arr[..., 3:4]
    return frame * (1 - a) + arr[..., :3] * a


# ---------- end card ----------
def make_qr():
    q = qrcode.QRCode(border=2, box_size=10, error_correction=qrcode.constants.ERROR_CORRECT_M)
    q.add_data(RESERVE_URL)
    q.make(fit=True)
    img = q.make_image(fill_color=(20, 16, 12), back_color=(250, 246, 238)).convert("RGBA")
    return img.resize((280, 280), Image.NEAREST)


QR = None


def render_endcard(t):
    global QR
    if QR is None:
        QR = make_qr()
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    def a(start, dur=0.6):
        return ease_out((t - start) / dur)

    def center(text, fnt, y, fill, alpha, tracking=0):
        w = tracked_width(text, fnt, tracking)
        tracked(d, ((W - w) / 2, y), text, fnt, fill + (int(255 * alpha),), tracking)

    k = a(0.2)
    center("OSAKA  ·  SHINSAIBASHI", F_KICK, 300, GOLD, k, 10)
    big = font("ShipporiMincho-normal-700", 190)
    p = a(0.4, 0.9)
    for i, ch in enumerate("禅園"):
        pc = ease_out((t - 0.4 - i * 0.18) / 0.8)
        x = W / 2 - 200 + i * 210
        d.text((x, 370 + 20 * (1 - pc)), ch, font=big, fill=WHITE + (int(255 * pc),))
    center("ZEN-EN", font("CormorantGaramond-normal-500", 54), 610, WHITE, p, 22)
    pl = ease_in_out_expo((t - 0.9) / 0.7)
    lw = 360 * pl
    if lw > 1:
        d.line([(W / 2 - lw / 2, 712), (W / 2 + lw / 2, 712)], fill=GOLD + (230,), width=2)
    center("Wagyu Sukiyaki & Kaiseki", font(F_TITLE, 66), 740, WHITE, a(1.1))
    info = font("CormorantGaramond-normal-500", 38)
    for i, line in enumerate([
        "1 min walk from Shinsaibashi Station",
        "Hotel Nikko Osaka Bldg. B2F",
        "Private Rooms  ·  English Menu Available",
    ]):
        center(line, info, 860 + i * 56, WHITE, a(1.4 + i * 0.15) * 0.92)

    pq = a(2.0, 0.7)
    if pq > 0:
        card = Image.new("RGBA", (330, 330), (250, 246, 238, 255))
        card.paste(QR, (25, 25))
        card.putalpha(int(255 * pq))
        layer.alpha_composite(card, (int((W - 330) / 2), int(1060 + 20 * (1 - pq))))
        dd = ImageDraw.Draw(layer)
        dd.rectangle([(W - 346) / 2, 1052 + 20 * (1 - pq), (W + 346) / 2, 1398 + 20 * (1 - pq)],
                     outline=GOLD + (int(255 * pq),), width=2)
    center("RESERVE ONLINE  ·  24/7", F_KICK, 1430, GOLD, a(2.3), 8)
    center("ご予約はこちらから", F_JP, 1482, WHITE, a(2.4) * 0.9, 6)
    return layer


# ---------- timeline ----------
IMG = "images/"
shots = [
    # file, cx, cy, zoom0, zoom1, px0, px1, py0, py1
    dict(f="320A9442.jpg", cx=0.5, cy=0.55, z=(1.18, 1.0), px=(0, 0), py=(0.3, 0.1)),
    dict(f="320A9457.jpg", cx=0.56, cy=0.5, z=(1.0, 1.14), px=(-0.2, 0.2), py=(0, 0)),
    dict(f="320A9515.jpg", cx=0.72, cy=0.5, z=(1.12, 1.0), px=(0, 0), py=(-0.5, 0.0)),
    dict(f="240619_0054.jpg", cx=0.42, cy=0.5, z=(1.0, 1.12), px=(0.1, -0.1), py=(-0.3, 0.1)),
    dict(f="320A9451.jpg", cx=0.56, cy=0.5, z=(1.15, 1.02), px=(0.3, -0.1), py=(0, 0)),
    dict(f="240619_0056.jpg", cx=0.5, cy=0.5, z=(1.0, 1.12), px=(0, 0), py=(0.4, 0.0)),
    dict(f="320A9534.jpg", cx=0.52, cy=0.5, z=(1.02, 1.16), px=(0, 0), py=(-0.4, -0.2)),
    dict(f="320A9483.jpg", cx=0.5, cy=0.5, z=(1.15, 1.0), px=(0, 0), py=(0, 0)),
]
# 各ショットの [開始, 終了]。重なり部分がトランジション
TD = 0.7
starts = [0.0, 3.6, 7.2, 10.8, 14.4, 18.0, 21.6, 25.0]
ends = [s + TD for s in starts[1:]] + [DUR]
transitions = [tr_zoom, tr_shoji, tr_lightleak, tr_diagonal, tr_whip, tr_iris, tr_defocus]

telops = [
    dict(kicker="OSAKA  ·  SHINSAIBASHI", title="Wagyu Sukiyaki", jp="心斎橋 禅園 ・ 名物すき焼き", y=330),
    dict(kicker="THE FINEST JAPANESE BEEF", title="A4 Saga Wagyu", jp="佐賀県産 A4ランク黒毛和牛", y=1180),
    dict(kicker="RICH  ·  TENDER  ·  JUICY", title="Exquisite Marbling", jp="とろける霜降り", y=1180),
    dict(kicker="OUR SIGNATURE BROTH", title="House-made Warishita", jp="特製割下で", y=1180),
    dict(kicker="AT YOUR TABLE", title="Simmered to Perfection", jp="お客様の目の前で丁寧に", y=1180),
    dict(kicker="SERVED AT ITS PEAK", title="Crafted Just for You", jp="最高の食べ頃でご提供", y=1180),
    dict(kicker="THE AUTHENTIC WAY", title="Dipped in Silky Egg", jp="溶き卵にくぐらせて", y=1180),
]
for i, tel in enumerate(telops):
    tel["start"] = starts[i] + (0.35 if i == 0 else TD * 0.6)
    tel["dur"] = ends[i] - TD * 0.5 - tel["start"]

bases = [load_base(IMG + s["f"], s["cx"], s["cy"]) for s in shots]


def shot_fn(i, t):
    s = shots[i]
    p = ease_io((t - starts[i]) / (ends[i] - starts[i]))
    z = s["z"][0] + (s["z"][1] - s["z"][0]) * p
    px = s["px"][0] + (s["px"][1] - s["px"][0]) * p
    py = s["py"][0] + (s["py"][1] - s["py"][0]) * p

    def f(extra=1.0, dy=0.0):
        img = kenburns(bases[i], z, px, py, extra, dy)
        if i == len(shots) - 1:  # エンドカードは暗くしてぼかす
            k = ease_io((t - starts[i] - 0.3) / 1.2)
            img = fast_blur(img, 10 * k) * (1 - 0.62 * k)
        return img
    return f


def frame_at(t):
    active = [i for i in range(len(shots)) if starts[i] <= t < ends[i]]
    if len(active) == 2:
        i = active[0]
        p = (t - starts[i + 1]) / TD
        img = transitions[i](shot_fn(i, t), shot_fn(i + 1, t), p)
    else:
        img = shot_fn(active[0], t)()
    img = img * VIGNETTE
    for i, tel in enumerate(telops):
        lt = t - tel["start"]
        if 0 <= lt <= tel["dur"]:
            fade = ease_io(lt / 0.5) * (1 - ease_io((lt - tel["dur"] + 0.4) / 0.4))
            img = img * (1 - (1 - grad_band(tel["y"] + 120, 720, 0.68)) * fade)
            img = composite(img, render_telop(lt, tel))
    if t >= starts[-1] + 0.6:
        img = composite(img, render_endcard(t - starts[-1] - 0.6))
    # 冒頭フェードイン・末尾フェードアウト
    img = img * ease_io(t / 0.6) * (1 - ease_io((t - (DUR - 0.5)) / 0.5))
    # 微細なフィルムグレイン
    img = img + np.random.normal(0, 0.012, (H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def main():
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-"]
    if BGM:
        cmd += ["-i", BGM, "-map", "0:v", "-map", "1:a",
                "-af", f"atrim=0:{DUR},afade=t=in:d=0.5,afade=t=out:st={DUR - 2}:d=2",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "22", "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-t", str(DUR), OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(NF):
        proc.stdin.write(frame_at(n / FPS).tobytes())
        if n % 90 == 0:
            print(f"{n}/{NF}", flush=True)
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[4] == "--preview":
        for t in [float(x) for x in sys.argv[5].split(",")]:
            Image.fromarray(frame_at(t)).save(f"{OUT}_{t:05.2f}.jpg", quality=85)
    else:
        main()
