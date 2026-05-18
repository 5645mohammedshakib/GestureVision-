"""
colorizer.py - GestureVision AI
---------------------------------
B&W → Color using OpenCV DNN (Caffe colorization model).
Auto-downloads model files if not present.
Falls back to sepia if model unavailable.
"""

import cv2
import numpy as np
import os
import urllib.request

MODEL_DIR   = "models"
PROTO_FILE  = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL_FILE  = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
HULL_FILE   = os.path.join(MODEL_DIR, "pts_in_hull.npy")

PROTO_URL = "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt"
HULL_URL  = "https://raw.githubusercontent.com/richzhang/colorization/master/colorization/resources/pts_in_hull.npy"
MODEL_URL = "https://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel"


def _try_download(url, path, label):
    try:
        print(f"[Colorizer] Downloading {label}...")
        urllib.request.urlretrieve(url, path)
        print(f"[Colorizer] {label} ready.")
        return True
    except Exception as e:
        print(f"[Colorizer] Could not download {label}: {e}")
        return False


class Colorizer:
    """
    Converts grayscale webcam frames to colorized output using the
    Zhang et al. (2016) Caffe deep colorization model.
    Falls back to sepia if model files are unavailable.
    """
    def __init__(self):
        self._net    = None
        self._ready  = False
        self._load_model()

    def _load_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Download files if missing
        if not os.path.exists(PROTO_FILE):
            _try_download(PROTO_URL, PROTO_FILE, "prototxt")
        if not os.path.exists(HULL_FILE):
            _try_download(HULL_URL,  HULL_FILE,  "hull points")
        # Note: caffemodel is 125MB — skip auto-download, inform user
        if not os.path.exists(MODEL_FILE):
            print("[Colorizer] caffemodel missing (125MB).")
            print(f"  Download from: {MODEL_URL}")
            print(f"  Save to: {MODEL_FILE}")
            return

        try:
            net    = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)
            kernel = np.load(HULL_FILE)

            class8 = net.getLayerId("class8_ab")
            conv8  = net.getLayerId("conv8_313_rh")
            pts    = kernel.transpose().reshape(2, 313, 1, 1)
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs  = [np.full([1, 313], 2.606, dtype="float32")]

            self._net   = net
            self._ready = True
            print("[Colorizer] Model ready — B&W colorization active.")
        except Exception as e:
            print(f"[Colorizer] Model load error: {e}")

    @property
    def is_ready(self):
        return self._ready

    def colorize(self, frame):
        """
        Colorizes the frame using the deep learning model.
        If model unavailable, returns sepia fallback.
        """
        if not self._ready or self._net is None:
            return self._sepia_fallback(frame)

        try:
            scaled  = frame.astype("float32") / 255.0
            lab     = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            resized = cv2.resize(lab, (224, 224))
            L       = cv2.split(resized)[0]
            L      -= 50

            self._net.setInput(cv2.dnn.blobFromImage(L))
            ab = self._net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))

            L_orig = cv2.split(lab)[0]
            colorized = np.concatenate(
                (L_orig[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            return (colorized * 255).astype("uint8")
        except Exception:
            return self._sepia_fallback(frame)

    @staticmethod
    def _sepia_fallback(frame):
        """Sepia tone as graceful fallback."""
        f = frame.astype(np.float32)
        r = f[:,:,2]*0.393 + f[:,:,1]*0.769 + f[:,:,0]*0.189
        g = f[:,:,2]*0.349 + f[:,:,1]*0.686 + f[:,:,0]*0.168
        b = f[:,:,2]*0.272 + f[:,:,1]*0.534 + f[:,:,0]*0.131
        return np.stack([
            np.clip(b,0,255), np.clip(g,0,255), np.clip(r,0,255)
        ], axis=-1).astype(np.uint8)
