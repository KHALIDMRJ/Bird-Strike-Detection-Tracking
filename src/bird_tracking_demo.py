"""
Bird Strike Detection Demo
Author: Khalid Morjane

Academic demo version.
Core optimization logic is intentionally simplified.
"""

import cv2
import numpy as np

VIDEO_PATH = "sample_video.mp4"
OUTPUT_PATH = "tracked_birds_demo.mp4"

def detect_birds(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred,
        95,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

    return detections

# Advanced merging, filtering and tracking logic is not included
# in this public academic demo version.