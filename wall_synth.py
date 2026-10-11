"""
Synthetic wall photos with exact box labels, for training the obstacle detector and for the random-wall
demo. Only numpy and OpenCV, so the app can import it.

The six classes follow obstacle_detector.CLASS_NAMES: window 0, door 1, ac_unit 2, meter_panel 3, pipe 4, grill 5.
Images are made to look like plastered, painted or brick walls with a mix of Indian-style fixtures
(window grills, outdoor AC units, meter boxes, drain pipes). They are not photographs: a model trained
only on them will not work on real photos, they are one part of the training mix.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CLASS_NAMES = ["window", "door", "ac_unit", "meter_panel", "pipe", "grill"]
Box = Tuple[int, float, float, float, float]  # class id, x1, y1, x2, y2 in pixels

WALL_COLOURS = [
    (205, 220, 235), (190, 215, 230), (225, 225, 220), (170, 200, 220), (215, 200, 185), (190, 190, 200),
    (160, 175, 215), (200, 230, 235), (235, 235, 235), (150, 170, 190), (185, 205, 160), (200, 185, 215),
]


def _noise(rng, h, w, scale):
    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _wall_texture(rng, h, w):
    base = np.array(WALL_COLOURS[rng.integers(len(WALL_COLOURS))], np.float32)
    img = np.ones((h, w, 3), np.float32) * base
    tex = 0.5 * _noise(rng, h, w, 4) + 0.3 * _noise(rng, h, w, 16) + 0.2 * _noise(rng, h, w, 64)
    img *= (0.88 + 0.24 * tex)[..., None]
    if rng.random() < 0.55:  # rain streaks
        for _ in range(rng.integers(3, 14)):
            x = int(rng.integers(0, w))
            width = int(rng.integers(3, max(4, w // 25)))
            streak = np.clip(_noise(rng, h, 1, 40), 0.3, 1.0)
            img[:, x:x + width] *= (1 - 0.18 * streak)[:, :, None]
    if rng.random() < 0.2:  # brick courses
        step = int(rng.integers(10, 26))
        for y in range(0, h, step):
            cv2.line(img, (0, y), (w, y), (base * 0.6).tolist(), 1)
            offset = (y // step % 2) * step
            for x in range(-offset, w, step * 2):
                cv2.line(img, (x, y), (x, min(h, y + step)), (base * 0.6).tolist(), 1)
    return img


def _shade(img, x1, y1, x2, y2, amount):
    """Soft shadow under and to one side of a fixture."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    dx, dy = int(0.015 * w), int(0.012 * w)
    cv2.rectangle(mask, (int(x1 + dx), int(y1 + dy)), (int(x2 + dx), int(y2 + dy)), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, 0.006 * w))
    img *= (1 - amount * mask)[..., None]


def _rect(img, x1, y1, x2, y2, colour, thickness=-1):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, thickness)


def _gradient_fill(img, x1, y1, x2, y2, c1, c2, vertical=True):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return
    n = (y2 - y1) if vertical else (x2 - x1)
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None] if vertical else np.linspace(0, 1, n, dtype=np.float32)[None, :]
    c1, c2 = np.array(c1, np.float32), np.array(c2, np.float32)
    patch = c1 * (1 - t[..., None]) + c2 * t[..., None]
    img[y1:y2, x1:x2] = np.broadcast_to(patch, (y2 - y1, x2 - x1, 3)) if vertical else np.broadcast_to(patch, (y2 - y1, x2 - x1, 3))


def _draw_grill(img, rng, x1, y1, x2, y2):
    colour = [(30, 30, 30), (40, 40, 120), (60, 60, 60), (20, 60, 20)][rng.integers(4)]
    xi1, yi1, xi2, yi2 = max(0, int(x1)), max(0, int(y1)), min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    if xi2 - xi1 < 6 or yi2 - yi1 < 6:
        return
    roi = img[yi1:yi2, xi1:xi2].copy()  # drawing on a copy keeps diagonal bars inside the box
    h, w = roi.shape[:2]
    thick = max(1, int(w * 0.02))
    cv2.rectangle(roi, (0, 0), (w - 1, h - 1), colour, thick * 2)
    style = rng.integers(3)
    if style == 0:  # vertical bars
        for x in np.linspace(0, w, int(rng.integers(5, 10))):
            cv2.line(roi, (int(x), 0), (int(x), h), colour, thick)
    elif style == 1:  # square lattice
        for x in np.linspace(0, w, int(rng.integers(4, 8))):
            cv2.line(roi, (int(x), 0), (int(x), h), colour, thick)
        for y in np.linspace(0, h, int(rng.integers(3, 7))):
            cv2.line(roi, (0, int(y)), (w, int(y)), colour, thick)
    else:  # diagonal lattice
        n = int(rng.integers(4, 8))
        for k in range(-n, n + 1):
            off = k * w / n
            cv2.line(roi, (int(off), 0), (int(off + h), h), colour, thick)
            cv2.line(roi, (int(off), h), (int(off + h), 0), colour, thick)
    img[yi1:yi2, xi1:xi2] = roi


def _draw_window(img, rng, x1, y1, x2, y2):
    frame = [(235, 235, 235), (60, 90, 130), (90, 120, 160), (40, 40, 40), (190, 200, 205)][rng.integers(5)]
    w, h = x2 - x1, y2 - y1
    _shade(img, x1, y1, x2, y2, 0.35)
    _rect(img, x1, y1, x2, y2, frame)
    border = max(2, int(0.07 * w))
    glass1 = [(170, 150, 110), (120, 100, 80), (90, 110, 130), (60, 70, 60)][rng.integers(4)]
    glass2 = tuple(int(c * rng.uniform(0.4, 0.8)) for c in glass1)
    _gradient_fill(img, x1 + border, y1 + border, x2 - border, y2 - border, glass1, glass2)
    cols, rows = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    for c in range(1, cols):
        x = x1 + c * w / cols
        cv2.line(img, (int(x), int(y1)), (int(x), int(y2)), frame, max(2, border // 2))
    for r in range(1, rows):
        y = y1 + r * h / rows
        cv2.line(img, (int(x1), int(y)), (int(x2), int(y)), frame, max(2, border // 2))
    if rng.random() < 0.5:  # curtain or blind slice
        _rect(img, x1 + border, y1 + border, x2 - border, y1 + border + (h - 2 * border) * rng.uniform(0.1, 0.5), (200, 205, 215))
    _rect(img, x1 - 0.03 * w, y2, x2 + 0.03 * w, y2 + max(3, 0.06 * w), tuple(int(c * 0.8) for c in frame))  # sill


def _draw_door(img, rng, x1, y1, x2, y2):
    wood = [(50, 80, 130), (70, 110, 160), (40, 60, 90), (150, 150, 155), (60, 100, 70), (30, 50, 120)][rng.integers(6)]
    w, h = x2 - x1, y2 - y1
    _shade(img, x1, y1, x2, y2, 0.35)
    _rect(img, x1 - 0.05 * w, y1 - 0.03 * h, x2 + 0.05 * w, y2, tuple(int(c * 0.7) for c in wood))  # frame
    _rect(img, x1, y1, x2, y2, wood)
    for fx in (0.12, 0.55):  # panels
        for fy in (0.08, 0.55):
            _rect(img, x1 + fx * w, y1 + fy * h, x1 + (fx + 0.33) * w, y1 + (fy + 0.37) * h, tuple(int(c * 0.82) for c in wood), 2)
    cv2.circle(img, (int(x2 - 0.15 * w), int(y1 + 0.52 * h)), max(2, int(0.04 * w)), (80, 190, 220), -1)  # handle
    step = max(3, 0.03 * h)
    _rect(img, x1 - 0.1 * w, y2, x2 + 0.1 * w, y2 + step, (150, 150, 150))


def _draw_ac(img, rng, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    body = [(225, 225, 225), (205, 210, 210), (240, 240, 235), (190, 195, 200)][rng.integers(4)]
    _shade(img, x1, y1, x2, y2, 0.45)
    _rect(img, x1, y1, x2, y2, body)
    _rect(img, x1, y1, x2, y2, tuple(int(c * 0.6) for c in body), 2)
    if w / max(h, 1) > 1.25:  # window AC: front vents
        for k in range(1, 8):
            y = y1 + k * h / 8
            cv2.line(img, (int(x1 + 0.08 * w), int(y)), (int(x2 - 0.08 * w), int(y)), (110, 110, 110), 1)
        _rect(img, x1 + 0.7 * w, y1 + 0.08 * h, x1 + 0.95 * w, y1 + 0.3 * h, (90, 90, 90))
    else:  # outdoor condenser: round fan grille
        centre = (int(x1 + 0.5 * w), int(y1 + 0.5 * h))
        radius = int(min(w, h) * 0.38)
        cv2.circle(img, centre, radius, (70, 70, 75), -1)
        for k in range(1, 5):
            cv2.circle(img, centre, int(radius * k / 5), (140, 140, 145), 1)
        for ang in np.linspace(0, np.pi, 6, endpoint=False):
            dx, dy = radius * np.cos(ang), radius * np.sin(ang)
            cv2.line(img, (int(centre[0] - dx), int(centre[1] - dy)), (int(centre[0] + dx), int(centre[1] + dy)), (150, 150, 155), 1)
        cv2.circle(img, centre, max(2, radius // 6), (50, 50, 55), -1)
        _rect(img, x1 - 0.05 * w, y2, x2 + 0.05 * w, y2 + max(3, 0.05 * h), (70, 70, 70))  # bracket


def _draw_meter(img, rng, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    body = [(150, 150, 155), (110, 120, 130), (170, 170, 165), (90, 100, 90)][rng.integers(4)]
    _shade(img, x1, y1, x2, y2, 0.4)
    _rect(img, x1, y1, x2, y2, body)
    _rect(img, x1, y1, x2, y2, tuple(int(c * 0.6) for c in body), 2)
    _rect(img, x1 + 0.1 * w, y1 + 0.12 * h, x2 - 0.1 * w, y1 + 0.72 * h, (60, 60, 60))  # window
    _rect(img, x1 + 0.2 * w, y1 + 0.25 * h, x2 - 0.2 * w, y1 + 0.45 * h, (200, 230, 210))  # display
    for k in range(4):
        cv2.line(img, (int(x1 + (0.25 + k * 0.14) * w), int(y1 + 0.28 * h)), (int(x1 + (0.25 + k * 0.14) * w), int(y1 + 0.42 * h)), (40, 40, 40), 1)
    cv2.circle(img, (int(x2 - 0.12 * w), int(y2 - 0.1 * h)), max(2, int(0.05 * w)), (30, 30, 30), -1)  # lock


def _draw_pipe(img, rng, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    colour = np.array([(190, 190, 190), (60, 60, 60), (60, 60, 150), (40, 130, 200), (120, 120, 120)][rng.integers(5)], np.float32)
    horizontal = w > h
    length = int(w if horizontal else h)
    thick = int(h if horizontal else w)
    for i in range(max(thick, 1)):
        t = abs((i + 0.5) / max(thick, 1) - 0.4) * 2
        shade = 1.1 - 0.5 * t
        c = np.clip(colour * shade, 0, 255).tolist()
        if horizontal:
            cv2.line(img, (int(x1), int(y1 + i)), (int(x2), int(y1 + i)), c, 1)
        else:
            cv2.line(img, (int(x1 + i), int(y1)), (int(x1 + i), int(y2)), c, 1)
    spacing = max(40, thick * 8)
    for k in range(1, max(2, length // spacing)):  # clamps
        pos = k * spacing
        band = max(2, thick // 2)
        if horizontal:
            _rect(img, x1 + pos, y1 - 1, x1 + pos + band, y2 + 1, tuple(int(c * 0.6) for c in colour))
        else:
            _rect(img, x1 - 1, y1 + pos, x2 + 1, y1 + pos + band, tuple(int(c * 0.6) for c in colour))


def _free(boxes, x1, y1, x2, y2, margin):
    for bx1, by1, bx2, by2 in boxes:
        if x1 < bx2 + margin and x2 > bx1 - margin and y1 < by2 + margin and y2 > by1 - margin:
            return False
    return True


def generate_wall(
    seed: int,
    width: int = 960,
    height: Optional[int] = None,
    density: float = 1.0,
    counts: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, List[Box]]:
    """
    Returns (image_bgr uint8, boxes). density scales how many fixtures are placed; counts forces
    exact numbers per class name, for example {"window": 4, "door": 1}.
    """
    rng = np.random.default_rng(seed)
    height = height or int(width * rng.uniform(0.55, 0.9))
    img = _wall_texture(rng, height, width)
    boxes: List[Box] = []
    placed: List[Tuple[float, float, float, float]] = []

    def n_of(name, low, high, clean_chance=0.0):
        if counts is not None:
            return int(counts.get(name, 0))
        if rng.random() < clean_chance:
            return 0
        return int(round(rng.integers(low, high + 1) * density))

    def place(w, h, bottom=None, tries=40, margin=0.02):
        x_low, x_high = 0.02 * width, width - w - 0.02 * width
        y_low, y_high = 0.04 * height, height - h - 0.04 * height
        if x_high <= x_low or (bottom is None and y_high <= y_low):
            return None
        for _ in range(tries):
            x1 = rng.uniform(x_low, x_high)
            y1 = bottom - h if bottom is not None else rng.uniform(y_low, y_high)
            if y1 < 0:
                continue
            if _free(placed, x1, y1, x1 + w, y1 + h, margin * width):
                return x1, y1
        return None

    clean = counts is None and rng.random() < 0.07

    for _ in range(0 if clean else n_of("window", 0, 8)):
        w = rng.uniform(0.08, 0.2) * width
        h = w * rng.uniform(0.8, 1.8)
        spot = place(w, h)
        if spot is None:
            continue
        x1, y1 = spot
        _draw_window(img, rng, x1, y1, x1 + w, y1 + h)
        boxes.append((0, x1, y1, x1 + w, y1 + h))
        placed.append((x1, y1, x1 + w, y1 + h + 0.06 * w))
        if rng.random() < 0.4:
            pad = rng.uniform(0, 0.03) * w
            _draw_grill(img, rng, x1 - pad, y1 - pad, x1 + w + pad, y1 + h + pad)
            boxes.append((5, x1 - pad, y1 - pad, x1 + w + pad, y1 + h + pad))

    for _ in range(0 if clean else n_of("door", 0, 2)):
        w = rng.uniform(0.1, 0.17) * width
        h = min(rng.uniform(0.38, 0.62) * height, w * 2.6)
        spot = place(w, h, bottom=height - rng.uniform(0.0, 0.05) * height)
        if spot is None:
            continue
        x1, y1 = spot
        _draw_door(img, rng, x1, y1, x1 + w, y1 + h)
        boxes.append((1, x1, y1, x1 + w, y1 + h))
        placed.append((x1 - 0.05 * w, y1 - 0.03 * h, x1 + w * 1.05, y1 + h))
        if rng.random() < 0.15:
            _draw_grill(img, rng, x1, y1, x1 + w, y1 + h)
            boxes.append((5, x1, y1, x1 + w, y1 + h))

    for _ in range(0 if clean else n_of("ac_unit", 0, 3)):
        if rng.random() < 0.55:
            w = rng.uniform(0.06, 0.12) * width
            h = w * rng.uniform(0.85, 1.15)
        else:
            w = rng.uniform(0.08, 0.15) * width
            h = w * rng.uniform(0.5, 0.75)
        spot = place(w, h)
        if spot is None:
            continue
        x1, y1 = spot
        _draw_ac(img, rng, x1, y1, x1 + w, y1 + h)
        boxes.append((2, x1, y1, x1 + w, y1 + h))
        placed.append((x1, y1, x1 + w, y1 + h + 0.05 * h))

    for _ in range(0 if clean else n_of("meter_panel", 0, 2)):
        w = rng.uniform(0.045, 0.1) * width
        h = w * rng.uniform(1.1, 1.6)
        spot = place(w, h)
        if spot is None:
            continue
        x1, y1 = spot
        _draw_meter(img, rng, x1, y1, x1 + w, y1 + h)
        boxes.append((3, x1, y1, x1 + w, y1 + h))
        placed.append((x1, y1, x1 + w, y1 + h))

    for _ in range(0 if clean else n_of("pipe", 0, 3)):
        thick = rng.uniform(0.018, 0.04) * width
        if rng.random() < 0.75:  # vertical downpipe
            length = rng.uniform(0.4, 0.95) * height
            w, h = thick, length
        else:
            length = rng.uniform(0.25, 0.7) * width
            w, h = length, thick
        spot = place(w, h, margin=0.005)
        if spot is None:
            continue
        x1, y1 = spot
        _draw_pipe(img, rng, x1, y1, x1 + w, y1 + h)
        boxes.append((4, x1, y1, x1 + w, y1 + h))
        placed.append((x1, y1, x1 + w, y1 + h))

    # distractors that carry no label: wires, a sign board, a plant, damp patches, a shadow band
    for _ in range(int(rng.integers(0, 4))):
        p1 = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        p2 = (int(np.clip(p1[0] + rng.uniform(-0.5, 0.5) * width, 0, width)), int(np.clip(p1[1] + rng.uniform(-0.3, 0.3) * height, 0, height)))
        cv2.line(img, p1, p2, (35, 35, 35), 1)
    if rng.random() < 0.3:
        spot = place(rng.uniform(0.1, 0.2) * width, rng.uniform(0.04, 0.07) * width)
        if spot is not None:
            sx, sy = spot
            _rect(img, sx, sy, sx + 0.15 * width, sy + 0.05 * width, tuple(int(c) for c in rng.integers(40, 230, 3)))
    if rng.random() < 0.5:
        shadow = np.zeros((height, width), np.float32)
        pts = np.array([[0, 0], [int(rng.uniform(0.2, 0.8) * width), 0], [int(rng.uniform(0, 0.6) * width), height], [0, height]])
        cv2.fillPoly(shadow, [pts], 1.0)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 0.03 * width)
        img *= (1 - rng.uniform(0.15, 0.4) * shadow)[..., None]

    img = np.clip(img, 0, 255).astype(np.uint8)

    # a slight perspective, then blur, brightness and JPEG artefacts
    if rng.random() < 0.7:
        m = 0.035 * min(width, height) * rng.uniform(0.2, 1.0)
        src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst = src + rng.uniform(-m, m, src.shape).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        warped = []
        for cls, bx1, by1, bx2, by2 in boxes:
            corners = np.float32([[[bx1, by1]], [[bx2, by1]], [[bx2, by2]], [[bx1, by2]]])
            moved = cv2.perspectiveTransform(corners, matrix)[:, 0]
            warped.append((cls, float(moved[:, 0].min()), float(moved[:, 1].min()), float(moved[:, 0].max()), float(moved[:, 1].max())))
        boxes = warped
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    gain, bias = rng.uniform(0.7, 1.25), rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * gain + bias + rng.normal(0, rng.uniform(0, 5), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 92))])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    clipped = []
    for cls, bx1, by1, bx2, by2 in boxes:
        bx1, bx2 = float(np.clip(bx1, 0, width)), float(np.clip(bx2, 0, width))
        by1, by2 = float(np.clip(by1, 0, height)), float(np.clip(by2, 0, height))
        if bx2 - bx1 >= 4 and by2 - by1 >= 4:
            clipped.append((cls, bx1, by1, bx2, by2))
    return img, clipped


def to_yolo_lines(boxes: List[Box], width: int, height: int) -> List[str]:
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        lines.append(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines
