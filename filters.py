"""
filters.py - GestureVision AI  [ADVANCED v2.0]
-----------------------------------------------
12 camera filters, all accepting BGR numpy arrays.
Each filter is optimised for real-time use.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. Normal
# ─────────────────────────────────────────────────────────────
def apply_normal(frame):
    """Raw frame — no processing."""
    return frame.copy()


# ─────────────────────────────────────────────────────────────
# 2. Grayscale / B&W
# ─────────────────────────────────────────────────────────────
def apply_grayscale(frame):
    """High-contrast black & white conversion."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Slight contrast boost
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────
# 3. Gaussian Blur (Dreamy)
# ─────────────────────────────────────────────────────────────
def apply_blur(frame, k=25):
    """Soft Gaussian blur for a dreamy / portrait feel."""
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


# ─────────────────────────────────────────────────────────────
# 4. Edge Detection (Canny + Neon Green)
# ─────────────────────────────────────────────────────────────
def apply_edges(frame):
    """Canny edges with neon-green tint on black canvas."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blur, 40, 140)
    result = np.zeros((*edges.shape, 3), dtype=np.uint8)
    result[:, :, 1] = edges           # green
    result[:, :, 0] = edges // 4      # slight blue → cyan
    return result


# ─────────────────────────────────────────────────────────────
# 5. Neon Glow
# ─────────────────────────────────────────────────────────────
def apply_neon(frame):
    """
    Electric neon glow effect:
      • Edges extracted → colourised in magenta/cyan
      • Blended back onto darkened original
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur1  = cv2.GaussianBlur(gray, (7, 7), 2)
    edges  = cv2.Canny(blur1, 30, 100)

    # Wide glow halo
    glow_k = 21
    glow   = cv2.GaussianBlur(edges, (glow_k, glow_k), 0)

    # Colour channels: cyan + magenta split
    glow_bgr = np.zeros_like(frame)
    glow_bgr[:, :, 0] = glow          # blue
    glow_bgr[:, :, 2] = glow          # red  → magenta
    glow_bgr[:, :, 1] = glow // 3     # slight green

    # Darken original, then add glow
    dark = (frame.astype(np.float32) * 0.30).astype(np.uint8)
    result = cv2.add(dark, glow_bgr)
    return result


# Cache for vignette masks to avoid re-generating them per-frame
_VIGNETTE_CACHE = {}

# ─────────────────────────────────────────────────────────────
# 6. Sepia Tone
# ─────────────────────────────────────────────────────────────
def apply_sepia(frame):
    """Warm vintage sepia tone using a highly-optimized C++ matrix transform."""
    matrix = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]])
    return cv2.transform(frame, matrix)


# ─────────────────────────────────────────────────────────────
# 7. Cartoon / Comic Effect
# ─────────────────────────────────────────────────────────────
def apply_cartoon(frame):
    """
    Cartoon / cel-shading effect:
      1. Fast edge preserving filter to flatten colours
      2. Canny edges overlaid as black outlines
    """
    color = cv2.edgePreservingFilter(frame, flags=1, sigma_s=10, sigma_r=0.25)

    # Edge mask
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=2
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine: colour × edges
    cartoon = cv2.bitwise_and(color, edges_bgr)
    return cartoon


# ─────────────────────────────────────────────────────────────
# 8. Vintage Vignette
# ─────────────────────────────────────────────────────────────
def apply_vignette(frame):
    """
    Vintage effect:
      • Warm tint
      • Strong pre-calculated oval vignette (dark edges) cached for speed
    """
    h, w = frame.shape[:2]
    key = (h, w)
    if key not in _VIGNETTE_CACHE:
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((X - cx) / (w * 0.55))**2 + ((Y - cy) / (h * 0.55))**2)
        vmask = np.clip(1.0 - dist, 0.0, 1.0) ** 1.8
        vmask = np.stack([vmask] * 3, axis=-1)
        _VIGNETTE_CACHE[key] = vmask
    else:
        vmask = _VIGNETTE_CACHE[key]

    warm = frame.copy()
    warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.15, 0, 255)  # boost red
    warm[:, :, 0] = np.clip(warm[:, :, 0] * 0.85, 0, 255)  # reduce blue

    result = (warm * vmask).astype(np.uint8)
    return result


# ─────────────────────────────────────────────────────────────
# 9. Negative / Invert
# ─────────────────────────────────────────────────────────────
def apply_negative(frame):
    """Simple colour inversion — each pixel becomes 255 - value."""
    return cv2.bitwise_not(frame)


# ─────────────────────────────────────────────────────────────
# 10. HDR / CLAHE (Local Contrast Enhancement)
# ─────────────────────────────────────────────────────────────
def apply_hdr(frame):
    """
    HDR-like local contrast enhancement using CLAHE on L channel (LAB).
    Optimized to boost saturation using fast cv2.multiply instead of float32 casts.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab2 = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.4)
    result = cv2.merge([h, s, v])
    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────────────────────
# 11. Pencil Sketch
# ─────────────────────────────────────────────────────────────
def apply_sketch(frame):
    """
    Pencil-sketch effect using a super fast divider fallback method.
    Avoids cv2.pencilSketch which runs slow bilateral filters.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv   = cv2.bitwise_not(gray)
    blur  = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────
# 12. Bokeh Background Blur
# ─────────────────────────────────────────────────────────────
def apply_background_blur(frame, blur_k=45):
    """Simulated portrait / bokeh mode with elliptical subject mask."""
    if blur_k % 2 == 0:
        blur_k += 1
    h, w = frame.shape[:2]
    # Downsample for faster blur
    small = cv2.resize(frame, (w//2, h//2))
    blurred_small = cv2.GaussianBlur(small, (15, 15), 0)
    blurred = cv2.resize(blurred_small, (w, h))
    
    mask = np.zeros((h, w, 1), dtype=np.uint8)
    cv2.ellipse(mask, (w//2, h//2), (w//3, h//2 - 40), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    alpha = mask.astype(np.float32) / 255.0
    out = (frame * alpha + blurred * (1.0 - alpha)).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────
# 13. Watercolor
# ─────────────────────────────────────────────────────────────
def apply_watercolor(frame):
    """Soft painterly watercolor effect optimized via pyramidal downsampling."""
    small = cv2.pyrDown(frame)
    small = cv2.edgePreservingFilter(small, flags=1, sigma_s=10, sigma_r=0.25)
    res = cv2.pyrUp(small)
    if res.shape != frame.shape:
        res = cv2.resize(res, (frame.shape[1], frame.shape[0]))
    gray  = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 80)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.subtract(res, edges // 3)


# ─────────────────────────────────────────────────────────────
# 14. Oil Paint
# ─────────────────────────────────────────────────────────────
def apply_oil_paint(frame):
    """Approximates oil painting with fast edge preserving filter."""
    res = cv2.edgePreservingFilter(frame, flags=1, sigma_s=15, sigma_r=0.3)
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.4)
    res = cv2.merge([h, s, v])
    return cv2.cvtColor(res, cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────────────────────
# 15. Thermal / Heatmap
# ─────────────────────────────────────────────────────────────
def apply_thermal(frame):
    """Thermal camera simulation using COLORMAP_JET on grayscale."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Slight blur to soften heat distribution
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thermal = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    return thermal


# ─────────────────────────────────────────────────────────────
# 16. Matrix Rain (Green Code Effect)
# ─────────────────────────────────────────────────────────────
def apply_matrix(frame):
    """Matrix digital rain — green-tinted with scanlines."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    green  = np.zeros_like(frame)
    green[:,:,1] = gray   # only green channel
    # Scanline darkening every other row
    green[::2, :] = (green[::2, :].astype(np.float32) * 0.4).astype(np.uint8)
    # Edge highlight
    edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0), 50, 150)
    green[:,:,1] = np.clip(green[:,:,1].astype(np.int16)+edges.astype(np.int16)//2, 0, 255).astype(np.uint8)
    return green


# ─────────────────────────────────────────────────────────────
# 17. Pixelate
# ─────────────────────────────────────────────────────────────
def apply_pixelate(frame, pixel_size=16):
    """8-bit pixel art effect by downscaling then upscaling."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w//pixel_size, h//pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ─────────────────────────────────────────────────────────────
# 18. Emboss / Relief
# ─────────────────────────────────────────────────────────────
def apply_emboss(frame):
    """3D relief emboss effect using a directional kernel."""
    kernel = np.array([[-2,-1,0],
                       [-1, 1,1],
                       [ 0, 1,2]], dtype=np.float32)
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    embossed = cv2.filter2D(gray, -1, kernel) + 128
    embossed = np.clip(embossed, 0, 255).astype(np.uint8)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────
# 19. Vintage Retro Warm 80s Tint
# ─────────────────────────────────────────────────────────────
def apply_retro(frame):
    """Gorgeous warm retro golden 80s film matrix tint."""
    matrix = np.array([[0.18, 0.48, 0.22],
                       [0.15, 0.52, 0.28],
                       [0.22, 0.62, 0.38]])
    return cv2.transform(frame, matrix)


# ─────────────────────────────────────────────────────────────
# 20. Synthwave Retro Sunset Gradient
# ─────────────────────────────────────────────────────────────
def apply_synthwave(frame):
    """Synthwave retro sunset orange/pink duo-tone colorizer."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create custom lookup table
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    for i in range(256):
        r_blend = i / 255.0
        # Blend from deep cyberpunk purple to warm orange-yellow sunset
        lut[0, i, 0] = int(210 * (1 - r_blend) + 20 * r_blend)    # Blue
        lut[0, i, 1] = int(50 * (1 - r_blend) + 120 * r_blend)    # Green
        lut[0, i, 2] = int(240 * (1 - r_blend) + 255 * r_blend)   # Red
    # Apply custom LUT to 3-channel grayscale for a beautiful synthwave sunset gradient
    gray_3d = cv2.merge([gray, gray, gray])
    return cv2.LUT(gray_3d, lut)


# ─────────────────────────────────────────────────────────────
# 21. Pop Art Posterized Color Bands
# ─────────────────────────────────────────────────────────────
def apply_posterize(frame, levels=4):
    """Retro Pop Art posterized color bands quantization."""
    factor = 256 // levels
    return (frame // factor) * factor


# ─────────────────────────────────────────────────────────────
# 22. Cyber Glitch Procedural Scanline chromatic aberration
# ─────────────────────────────────────────────────────────────
def apply_glitch(frame):
    """Real-time chromatic aberration RGB split & horizontal glitch scanlines."""
    h, w = frame.shape[:2]
    glitched = frame.copy()
    
    # 1. Randomly apply horizontal shift slices
    for _ in range(3):
        sy = np.random.randint(0, h - 20)
        sh = np.random.randint(5, 20)
        shift = np.random.randint(-15, 15)
        glitched[sy:sy+sh, :] = np.roll(glitched[sy:sy+sh, :], shift, axis=1)
        
    # 2. Chromatic aberration RGB split channel shift
    b, g, r = cv2.split(glitched)
    r = np.roll(r, 4, axis=1)
    b = np.roll(b, -4, axis=1)
    return cv2.merge([b, g, r])


# ─────────────────────────────────────────────────────────────
# Filter crossfade transition
# ─────────────────────────────────────────────────────────────
def crossfade(old_frame, new_frame, t):
    t = max(0.0, min(1.0, t))
    return cv2.addWeighted(old_frame, 1-t, new_frame, t, 0)


# ─────────────────────────────────────────────────────────────
# Dispatcher — 22 filters total
# ─────────────────────────────────────────────────────────────
_FILTER_MAP = {
    "normal":     apply_normal,
    "grayscale":  apply_grayscale,
    "blur":       apply_blur,
    "edges":      apply_edges,
    "neon":       apply_neon,
    "sepia":      apply_sepia,
    "cartoon":    apply_cartoon,
    "vignette":   apply_vignette,
    "negative":   apply_negative,
    "hdr":        apply_hdr,
    "sketch":     apply_sketch,
    "bokeh":      apply_background_blur,
    "watercolor": apply_watercolor,
    "oil_paint":  apply_oil_paint,
    "thermal":    apply_thermal,
    "matrix":     apply_matrix,
    "pixelate":   apply_pixelate,
    "emboss":     apply_emboss,
    "retro":      apply_retro,
    "synthwave":  apply_synthwave,
    "posterize":  apply_posterize,
    "glitch":     apply_glitch,
}


# ─────────────────────────────────────────────────────────────
# 19. Neural Spectrum Tuner (11.7M+ dynamic procedural filters)
# ─────────────────────────────────────────────────────────────
def apply_neural_spectrum(frame, ix=0.5, iy=0.5):
    """
    Real-time procedural filter engine that generates over 11.7 million 
    unique filters dynamically controlled by a single finger's 2D position.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    # 1. Hue Shift based on X position (0 to 180 degrees)
    hsv[:, :, 0] = (hsv[:, :, 0] + ix * 180.0) % 180.0
    # 2. Saturation scale based on Y position (0.1 to 3.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.1 + iy * 2.9), 0, 255)
    # 3. Brightness/Value scale based on Y position (0.1 to 2.5)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.1 + (1.0 - iy) * 2.4), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


_FILTER_MAP["neural_spectrum"] = apply_neural_spectrum


def apply_filter(frame, filter_name, ix=0.5, iy=0.5):
    if filter_name == "neural_spectrum":
        return apply_neural_spectrum(frame, ix, iy)
    return _FILTER_MAP.get(filter_name, apply_normal)(frame)


def get_filter_names():
    return list(_FILTER_MAP.keys())

