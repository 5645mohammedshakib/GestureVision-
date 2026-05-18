"""
color_detector.py - GestureVision AI
--------------------------------------
Real-time RGB dominant color detection from webcam frame.
Shows R, G, B channel averages and dominant color name + colored bar.
"""

import cv2
import numpy as np

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX

# Extended color name mapping using dominant HSV hue
def _bgr_to_color_name(b_mean, g_mean, r_mean):
    """Returns dominant color name from BGR channel averages."""
    mx = max(r_mean, g_mean, b_mean)
    if mx < 40:
        return "Black",       (20,  20,  20)
    if mx > 200 and min(r_mean,g_mean,b_mean) > 160:
        return "White",       (240, 240, 240)

    # Approximate HSV
    arr = np.array([[[int(b_mean), int(g_mean), int(r_mean)]]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if s < 40:
        return "Gray",        (130, 130, 130)

    if   h <  10: return "Red",     (50,  50,  220)
    elif h <  25: return "Orange",  (30, 130,  230)
    elif h <  35: return "Yellow",  (20, 220,  220)
    elif h <  85: return "Green",   (50, 210,   80)
    elif h < 130: return "Blue",    (220, 100,  40)
    elif h < 160: return "Purple",  (200,  50, 180)
    elif h < 175: return "Pink",    (160,  80, 220)
    else:         return "Red",     (50,  50,  220)


def draw_color_panel(frame, x, y):
    """
    Draws a compact RGB analysis panel at position (x, y).
    Shows: R/G/B bars, dominant color name, colored swatch.
    Returns the dominant color name.
    """
    H, W = frame.shape[:2]

    # Sample center region of frame (avoid edges)
    roi = frame[H//4:3*H//4, W//4:3*W//4]
    b_mean = float(np.mean(roi[:, :, 0]))
    g_mean = float(np.mean(roi[:, :, 1]))
    r_mean = float(np.mean(roi[:, :, 2]))
    total  = b_mean + g_mean + r_mean + 1e-5

    color_name, display_col = _bgr_to_color_name(b_mean, g_mean, r_mean)

    pw, ph = 170, 120
    # Glassmorphism panel
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x+pw, y+ph), (8, 6, 14), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x, y), (x+pw, y+ph), (80, 200, 60), 1, cv2.LINE_AA)

    # Header
    cv2.putText(frame, "COLOR ANALYSIS", (x+8, y+14),
                FONT, 0.30, (0, 220, 180), 1, cv2.LINE_AA)
    cv2.line(frame, (x+4, y+18), (x+pw-4, y+18), (50, 100, 60), 1)

    bar_x = x + 8
    bar_w = pw - 52

    # R bar
    r_w = int(bar_w * r_mean / 255)
    cv2.rectangle(frame, (bar_x, y+24), (bar_x+bar_w, y+34), (30,30,30), -1)
    cv2.rectangle(frame, (bar_x, y+24), (bar_x+r_w,   y+34), (50, 50, 220), -1)
    cv2.putText(frame, f"R {int(r_mean):3d}", (bar_x+bar_w+4, y+33),
                FONT, 0.28, (80, 80, 220), 1)

    # G bar
    g_w = int(bar_w * g_mean / 255)
    cv2.rectangle(frame, (bar_x, y+38), (bar_x+bar_w, y+48), (30,30,30), -1)
    cv2.rectangle(frame, (bar_x, y+38), (bar_x+g_w,   y+48), (50, 200, 50), -1)
    cv2.putText(frame, f"G {int(g_mean):3d}", (bar_x+bar_w+4, y+47),
                FONT, 0.28, (50, 200, 50), 1)

    # B bar
    b_w = int(bar_w * b_mean / 255)
    cv2.rectangle(frame, (bar_x, y+52), (bar_x+bar_w, y+62), (30,30,30), -1)
    cv2.rectangle(frame, (bar_x, y+52), (bar_x+b_w,   y+62), (220, 80, 50), -1)
    cv2.putText(frame, f"B {int(b_mean):3d}", (bar_x+bar_w+4, y+61),
                FONT, 0.28, (200, 100, 50), 1)

    # Color swatch
    swatch_bgr = (int(b_mean), int(g_mean), int(r_mean))
    cv2.rectangle(frame, (bar_x, y+68), (bar_x+30, y+84), swatch_bgr, -1)
    cv2.rectangle(frame, (bar_x, y+68), (bar_x+30, y+84), (100,100,100), 1)

    # Dominant color name
    cv2.putText(frame, color_name, (bar_x+36, y+80),
                FONTB, 0.52, display_col, 1, cv2.LINE_AA)

    # Hex value
    hex_val = f"#{int(r_mean):02X}{int(g_mean):02X}{int(b_mean):02X}"
    cv2.putText(frame, hex_val, (bar_x, y+100),
                FONT, 0.30, (120, 120, 130), 1, cv2.LINE_AA)

    # Percentage bars
    r_pct = int(r_mean/total*100); g_pct = int(g_mean/total*100); b_pct = int(b_mean/total*100)
    cv2.putText(frame, f"R:{r_pct}% G:{g_pct}% B:{b_pct}%",
                (bar_x+60, y+100), FONT, 0.27, (90, 90, 100), 1, cv2.LINE_AA)

    return color_name
