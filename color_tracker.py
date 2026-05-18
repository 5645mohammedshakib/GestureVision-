"""
color_tracker.py - GestureVision AI
--------------------------------------
Real-time HSV color object tracking + air drawing.
Tracks 4 colored objects (orange, purple, green, blue)
and paints their movement on the camera frame.
Toggle with D key.
"""

import cv2
import numpy as np
from collections import deque

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX

# HSV color ranges [H_lo, S_lo, V_lo, H_hi, S_hi, V_hi]
COLOR_RANGES = [
    [5,  107, 80,  19, 255, 255],   # Orange
    [133, 56,  0, 159, 255, 255],   # Purple
    [57,  76,  0, 100, 255, 255],   # Green
    [90,  48,  0, 118, 255, 255],   # Blue
]

# BGR colors for drawing
DRAW_COLORS = [
    (51,  153, 255),   # Orange
    (255,   0, 255),   # Purple
    (0,   255,   0),   # Green
    (255,   0,   0),   # Blue
]

COLOR_NAMES = ["Orange", "Purple", "Green", "Blue"]


class ColorTracker:
    """
    Tracks colored objects via HSV masks and draws their trails
    on the live camera feed.
    """
    def __init__(self, w=1280, h=720, trail_len=120):
        self.w = w
        self.h = h
        # Separate trail deque per color
        self._trails = [deque(maxlen=trail_len) for _ in COLOR_RANGES]
        self._canvas  = np.zeros((h, w, 3), dtype=np.uint8)
        self.active   = False

    def clear(self):
        self._canvas[:] = 0
        for t in self._trails:
            t.clear()

    def process(self, frame):
        """
        Detect each color, update trails, draw on frame.
        Returns modified frame + list of detected (name, cx, cy) tuples.
        """
        hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected = []

        for idx, (rng, col, name) in enumerate(
                zip(COLOR_RANGES, DRAW_COLORS, COLOR_NAMES)):
            lower = np.array(rng[:3])
            upper = np.array(rng[3:])
            mask  = cv2.inRange(hsv, lower, upper)
            # Morphological clean
            mask  = cv2.erode(mask,  None, iterations=1)
            mask  = cv2.dilate(mask, None, iterations=2)

            cx, cy = self._get_center(mask)

            if cx > 0 and cy > 0:
                detected.append((name, cx, cy))
                self._trails[idx].append((cx, cy))
                # Draw live dot indicator on frame
                cv2.circle(frame, (cx, cy), 14, col, cv2.FILLED)
                cv2.circle(frame, (cx, cy), 14, (255,255,255), 2)
                cv2.putText(frame, name, (cx+18, cy+6),
                            FONTB, 0.42, col, 1, cv2.LINE_AA)
            else:
                self._trails[idx].append(None)

            # Draw trail on canvas
            trail = list(self._trails[idx])
            for i in range(1, len(trail)):
                if trail[i] is None or trail[i-1] is None:
                    continue
                alpha  = i / len(trail)
                thick  = max(1, int(5 * alpha))
                col_a  = tuple(int(c * alpha) for c in col)
                cv2.line(self._canvas, trail[i-1], trail[i],
                         col_a, thick, cv2.LINE_AA)

        # Fade canvas over time (ghost trail effect)
        self._canvas = (self._canvas.astype(np.float32) * 0.96).astype(np.uint8)

        # Blend canvas onto frame
        mask_gray = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_bin)
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(self._canvas, self._canvas, mask=mask_bin)
        frame[:] = cv2.add(bg, fg)

        return frame, detected

    def _get_center(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                return x + w // 2, y
        return 0, 0

    def draw_legend(self, frame):
        """Draw a small legend showing which color tracks what."""
        x, y = 14, 100
        panel_h = len(COLOR_RANGES) * 22 + 24

        ov = frame.copy()
        cv2.rectangle(ov, (x-4, y-4), (x+160, y+panel_h), (8,6,14), -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x-4, y-4), (x+160, y+panel_h), (80,200,60), 1)
        cv2.putText(frame, "COLOR TRACKER", (x, y+12),
                    FONT, 0.32, (0,220,180), 1, cv2.LINE_AA)

        for i, (col, name) in enumerate(zip(DRAW_COLORS, COLOR_NAMES)):
            ry = y + 24 + i * 22
            cv2.circle(frame, (x+8, ry), 7, col, -1)
            cv2.putText(frame, f"{name} object", (x+20, ry+5),
                        FONT, 0.33, col, 1, cv2.LINE_AA)
