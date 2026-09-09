# preprocessing/test_preprocess.py
import matplotlib.pyplot as plt
import cv2
from cv_preprocessing import (
    load_image, resize_image, to_grayscale,
    apply_threshold, remove_noise, apply_morphology,
    detect_edges, find_contours, draw_contours
)

IMAGE_PATH = "../dataset/images/Hammer/AAPL_141.png"

def main():
    original = load_image(IMAGE_PATH)
    resized = resize_image(original, size=(300, 300))
    gray = to_grayscale(resized)
    binary = apply_threshold(gray, thresh_val=200)
    cleaned = remove_noise(binary, kernel_size=3)
    morphed = apply_morphology(cleaned, operation="close", kernel_size=3)
    edges = detect_edges(gray)

    contours = find_contours(morphed)
    contoured = draw_contours(resized, contours)

    print(f"Number of contours (candles) detected: {len(contours)}")

    original_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    contoured_rgb = cv2.cvtColor(contoured, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    titles = ["Original", "Edges (Canny)", "Morphology (close)", f"Contours: {len(contours)} found"]
    images = [original_rgb, edges, morphed, contoured_rgb]
    cmaps = [None, "gray", "gray", None]

    for ax, title, im, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_test_output.png")
    print("Saved comparison to preprocessing_test_output.png")

if __name__ == "__main__":
    main()