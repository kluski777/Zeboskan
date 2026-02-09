import numpy as np
import matplotlib.pyplot as plt
import cv2

def apply_gamma_correction(image, gamma, draw = False):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                    for i in range(256)]).astype("uint8")
    improved_images = []

    image_enhanced = cv2.LUT(image, table)
    improved_images.append(image_enhanced)

    return improved_images

def best_preprocessing(image, draw = False):
    gamma = 2.0

    clip_limit = 4
    tile_grid_size = 8

    d = 4
    sigma_color = 50
    sigma_space = 50

    # 1. Gamma correction - brightens the teeth
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    image_filtered = cv2.LUT(image, table)

    # 2. CLAHE – enhances local contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    image_filtered = clahe.apply(image_filtered)

    # 3. Bilateral filter - removes noise, preserves edges
    image_filtered = cv2.bilateralFilter(image_filtered, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    return image_filtered


def preprocess_dental_xray(image):
    preprocessed_img = image.copy()

    # 1. Bilateral Filtration - removes noise, preserves edges
    denoised = cv2.bilateralFilter(preprocessed_img, 9, 80, 80)

    # 2. Morphological Top-Hat Transformation (White-Hat)
    kernel_size = (100, 70)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)

    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Improves local contrast on the image after Top-Hat
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(tophat)

    return enhanced