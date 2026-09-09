# feature_extraction/detect_candles.py
import sys
import os
import cv2
import numpy as np

# Reuse Phase 3's preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "preprocessing"))
from cv_preprocessing import (
    load_image, resize_image, to_grayscale,
    apply_threshold, remove_noise, apply_morphology, find_contours
)

BODY_WIDTH_RATIO = 0.5  # rows wider than this fraction of the box's max width = body


def get_sorted_boxes(contours):
    """
    Converts contours into bounding boxes (x, y, w, h) and sorts them
    left-to-right, so box[0] = earliest candle in the window,
    box[-1] = most recent (usually the target/pattern candle).
    """
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[0])
    return boxes


def classify_body_rows(binary_img, box):
    """
    Scans each row inside a candle's bounding box and measures the
    horizontal white-pixel width. Wide rows = body. Narrow rows = wick.
    Returns the pixel y-range (top, bottom) belonging to the body.
    """
    x, y, w, h = box
    crop = binary_img[y:y + h, x:x + w]

    row_widths = []
    for row in crop:
        white_pixels = np.where(row > 0)[0]
        if len(white_pixels) == 0:
            row_widths.append(0)
        else:
            row_widths.append(white_pixels[-1] - white_pixels[0] + 1)

    max_width = max(row_widths) if row_widths else 0
    threshold = max_width * BODY_WIDTH_RATIO
    body_rows = [i for i, rw in enumerate(row_widths) if rw >= threshold]

    if not body_rows:
        return y, y + h - 1  # degenerate fallback

    return y + min(body_rows), y + max(body_rows)


def get_candle_color(color_img, box, body_top_px, body_bottom_px):
    """
    Crops the ORIGINAL color image to the body region and checks whether
    green or red dominates. OpenCV uses BGR order: channel 1 = Green, 2 = Red.
    Returns a plain Python bool (not numpy.bool_) to avoid identity-comparison bugs.
    """
    x, y, w, h = box
    body_crop = color_img[body_top_px:body_bottom_px + 1, x:x + w]
    if body_crop.size == 0:
        return None

    mean_color = body_crop.reshape(-1, 3).mean(axis=0)  # [B, G, R]
    return bool(mean_color[1] > mean_color[2])  # True = bullish (green), False = bearish (red)


def detect_candles(image_path):
    """
    Full detection pipeline for one chart image:
    load -> preprocess -> find contours -> analyze each candle.
    Returns a list of dicts, one per candle, left-to-right.
    """
    original = load_image(image_path)
    resized = resize_image(original, size=(300, 300))
    gray = to_grayscale(resized)
    binary = apply_threshold(gray, thresh_val=200)
    cleaned = remove_noise(binary, kernel_size=3)
    morphed = apply_morphology(cleaned, operation="close", kernel_size=3)

    contours = find_contours(morphed)
    boxes = get_sorted_boxes(contours)

    candles = []
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        wick_top_px, wick_bottom_px = y, y + h - 1

        body_top_px, body_bottom_px = classify_body_rows(morphed, box)
        is_bullish = get_candle_color(resized, box, body_top_px, body_bottom_px)

        candles.append({
            "index": idx,
            "x": x, "y": y, "w": w, "h": h,
            "wick_top_px": wick_top_px,
            "wick_bottom_px": wick_bottom_px,
            "body_top_px": body_top_px,
            "body_bottom_px": body_bottom_px,
            "upper_wick_px": body_top_px - wick_top_px,
            "lower_wick_px": wick_bottom_px - body_bottom_px,
            "body_height_px": body_bottom_px - body_top_px,
            "total_range_px": wick_bottom_px - wick_top_px,
            "is_bullish": is_bullish,
        })

    return candles


def draw_detection(image_path, candles):
    """
    Visualizes detection results: blue rectangle = detected body,
    yellow lines/dots = detected wick segments. Body is drawn first,
    wick drawn on top, so thin/zero-length wicks are never hidden
    under the body's border.
    """
    original = load_image(image_path)
    resized = resize_image(original, size=(300, 300))
    output = resized.copy()

    for c in candles:
        x, w = c["x"], c["w"]
        cx = x + w // 2

        # body rectangle (blue) — drawn first
        cv2.rectangle(output, (x, c["body_top_px"]), (x + w, c["body_bottom_px"]), (255, 0, 0), 1)

        # wick lines (yellow) — drawn second, on top
        cv2.line(output, (cx, c["wick_top_px"]), (cx, c["body_top_px"]), (0, 255, 255), 1)
        cv2.line(output, (cx, c["body_bottom_px"]), (cx, c["wick_bottom_px"]), (0, 255, 255), 1)

        # small markers at wick endpoints — ensures near-zero wicks are still visible
        cv2.circle(output, (cx, c["body_top_px"]), 1, (0, 255, 255), -1)
        cv2.circle(output, (cx, c["body_bottom_px"]), 1, (0, 255, 255), -1)

    return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    IMAGE_PATH = "../dataset/images/Hammer/AAPL_147.png"  # change as needed
    candles = detect_candles(IMAGE_PATH)

    print(f"Detected {len(candles)} candles:\n")
    for c in candles:
        if c["is_bullish"] is None:
            direction = "Unknown"
        elif c["is_bullish"]:
            direction = "Bullish"
        else:
            direction = "Bearish"
        print(f"Candle {c['index']}: body={c['body_height_px']}px  "
              f"upper_wick={c['upper_wick_px']}px  lower_wick={c['lower_wick_px']}px  "
              f"range={c['total_range_px']}px  {direction}")

    output = draw_detection(IMAGE_PATH, candles)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(5, 5))
    plt.imshow(output_rgb)
    plt.title("Detected bodies (blue) and wicks (yellow)")
    plt.axis("off")
    plt.savefig("detection_test_output.png")
    print("\nSaved visualization to detection_test_output.png")