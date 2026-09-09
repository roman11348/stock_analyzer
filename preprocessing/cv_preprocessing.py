# preprocessing/cv_preprocessing.py
import cv2
import numpy as np

def load_image(path: str):
    """
    Reads an image from disk using OpenCV.
    Returns the image as a NumPy array in BGR color order,
    or raises an error if the file couldn't be read.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def resize_image(img, size=(300, 300)):
    """
    Resizes an image to a fixed size (width, height).
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def to_grayscale(img):
    """
    Converts a BGR color image to a single-channel grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_threshold(gray_img, thresh_val=200, mode=cv2.THRESH_BINARY_INV):
    """
    Converts a grayscale image into a binary (black/white) image.
    Candle pixels (dark) become white; background (light) becomes black.
    """
    _, binary = cv2.threshold(gray_img, thresh_val, 255, mode)
    return binary


def remove_noise(binary_img, kernel_size=3):
    """
    Removes small noise specks using a median blur.
    """
    return cv2.medianBlur(binary_img, kernel_size)


def apply_morphology(binary_img, operation="close", kernel_size=3):
    """
    Applies a morphological operation to clean up a binary image.
    operation: "open" removes small noise specks,
               "close" fills small gaps/holes inside shapes.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == "open":
        return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("operation must be 'open' or 'close'")


def detect_edges(gray_img, low_thresh=50, high_thresh=150):
    """
    Detects edges using the Canny algorithm.
    low_thresh/high_thresh control sensitivity — pixels with gradient strength
    above high_thresh are definite edges; between low and high are edges only
    if connected to a definite edge (hysteresis).
    """
    return cv2.Canny(gray_img, low_thresh, high_thresh)

def find_contours(binary_img):
    """
    Finds contours (outlines of connected white regions) in a binary image.
    Returns a list of contours, each a NumPy array of boundary points.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(color_img, contours):
    """
    Draws detected contours onto a copy of the original color image, for visualization.
    """
    output = color_img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output