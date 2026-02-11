from skimage.segmentation import active_contour, random_walker
from skimage.graph import rag_mean_color, cut_threshold
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage.filters import gaussian, sobel
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from collections import deque
from scipy import ndimage
import numpy as np
import cv2


def get_teeth_positions(image_metadata_record: dict) -> list[np.ndarray]:
    teeth = []
    if 'objects' in image_metadata_record:
        for tooth_pos in image_metadata_record['objects']:
            teeth.append(np.array(tooth_pos['points']['exterior']))
    return teeth



#! NIE UZYWAC SLICA JEST ZA WOLNY
def segment_teeth_slic(preprocessed_img, n_segments=30, compactness=5, threshold=0.1):
    """SLIC superpixels → RAG merge → same output format as watershed."""
    # Binarize to mask out background
    _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # SLIC needs float [0,1] input
    img_float = preprocessed_img.astype(float) / 255.0
    slic_labels = slic( img_float, n_segments=n_segments, compactness=compactness,
                        start_label=1, channel_axis=None)

    # Zero out background superpixels
    slic_labels[binary == 0] = 0

    # Merge similar adjacent superpixels via Region Adjacency Graph
    rag = rag_mean_color(img_float, slic_labels, mode='similarity')
    markers = cut_threshold(slic_labels, rag, thresh=threshold)

    # Relabel starting from 2 (to match watershed convention: 0=unknown, 1=bg)
    unique = np.unique(markers)
    remap = {old: new for new, old in enumerate(unique, start=2)}
    remap[0] = 0
    markers = np.vectorize(remap.get)(markers).astype(np.int32)

    # Same output format as segment_teeth_watershed
    teeth_mask = np.zeros_like(preprocessed_img)
    teeth_mask[markers > 1] = 255
    contours, _ = cv2.findContours(teeth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return teeth_mask, np.array(contours, dtype=object), markers


#! FELZENSZWALB DAJE DZIWNE WYNIKI
def segment_teeth_felzenszwalb(preprocessed_img, scale=10, sigma=1.5, min_size=50, downscale=2):
    h, w = preprocessed_img.shape
    small = cv2.resize(preprocessed_img, (w // downscale, h // downscale))
    
    labels = felzenszwalb(small.astype(float) / 255.0, scale=scale, sigma=sigma,
                        min_size=min_size, channel_axis=None)
    labels = cv2.resize(labels.astype(float), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labels[binary == 0] = 0

    teeth_mask = np.zeros_like(preprocessed_img)
    teeth_mask[labels > 0] = 255
    contours, _ = cv2.findContours(teeth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return teeth_mask, np.array(contours, dtype=object), labels


def segment_teeth_watershed(preprocessed_img, threshold = 0.1):
    _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    sure_bg = cv2.dilate(binary, kernel_close, iterations=1)

    # Step 4: Distance transform to find sure foreground (inside teeth)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, threshold * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 5: Unknown region (between fg and bg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 6: Create markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Step 7: Apply watershed
    color_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_img, markers)

    # Step 8: Create final tooth mask and contours
    teeth_mask = np.zeros_like(preprocessed_img)
    teeth_mask[markers > 5] = 255

    contours, _ = cv2.findContours(teeth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return teeth_mask, np.array(contours, dtype=object), markers

# ============================================================================
# HELPERS
# ============================================================================

def draw_annotations(image, metadata, color=(0, 255, 0), thickness=4):
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for obj in metadata['objects']:
        pts = np.array(obj['points']['exterior'], dtype=np.int32)
        cv2.polylines(img_color, [pts], isClosed=True, color=color, thickness=thickness)
    return img_color

def _make_markers(shape, centroids, radius=10):
    """Create marker image from centroids dict. Returns markers, tooth_to_label mapping, next_label."""
    markers = np.zeros(shape, dtype=np.int32)
    h, w = shape
    mapping = {}
    label = 1
    for tid, (cx, cy) in centroids.items():
        cx, cy = int(cx), int(cy)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(markers, (cx, cy), radius, label, -1)
            mapping[tid] = label
            label += 1
    return markers, mapping, label


def _bg_mask(img, kernel_size=15, iterations=2):
    """Eroded background mask from Otsu thresholding."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(255 - binary, kernel, iterations=iterations), binary


def _extract_contours(label_map, mapping):
    """Extract largest contour per tooth from a label map."""
    contours = {}
    for tid, lbl in mapping.items():
        mask = (label_map == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours[tid] = max(cnts, key=cv2.contourArea)
    return contours


def _debug_panels(panels, figsize=(18, 6)):
    """Show debug panels: list of (image, title, cmap) tuples."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def _contour_overlay(img, contours):
    """Draw green contours on grayscale image, return BGR overlay."""
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours.values():
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
    return overlay


def segment_with_voronoi(preprocessed_img, centroids, refine=True, min_area=800, max_area=16000, debug=False):
    """
    Podział Voronoi na podstawie centroidów + opcjonalne ograniczenie do zębów.

    Zalety:
    - Bardzo szybka
    - Zawsze daje wynik dla każdego centroidu
    - Naturalny podział przestrzeni
    """
    h, w = preprocessed_img.shape

    # 1. Przygotuj punkty dla Voronoi (dodaj punkty brzegowe)
    points = []
    tooth_ids = []

    for tid, (cx, cy) in centroids.items():
        if 0 <= cx < w and 0 <= cy < h:
            points.append([cx, cy])
            tooth_ids.append(tid)

    if len(points) < 3:
        return {}, np.zeros((h, w), dtype=np.int32)

    # Dodaj punkty na rogach (żeby Voronoi był skończony)
    margin = 50
    corner_points = [
        [-margin, -margin], [w + margin, -margin],
        [-margin, h + margin], [w + margin, h + margin],
        [w/2, -margin], [w/2, h + margin],
        [-margin, h/2], [w + margin, h/2]
    ]
    all_points = np.array(points + corner_points)

    # 2. Oblicz Voronoi
    vor = Voronoi(all_points)

    # 3. Utwórz mapę regionów
    # Dla każdego piksela znajdź najbliższy centroid
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    pixel_coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Oblicz odległości do każdego centroidu
    centroids_arr = np.array(points)

    # Znajdź najbliższy centroid dla każdego piksela
    from scipy.spatial.distance import cdist
    distances = cdist(pixel_coords, centroids_arr)
    nearest_idx = np.argmin(distances, axis=1)

    voronoi_map = nearest_idx.reshape(h, w)

    # 4. Opcjonalne: ogranicz do obszaru zębów (binaryzacja)
    if refine:
        _, binary = cv2.threshold(preprocessed_img, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Maskuj regiony Voronoi
        voronoi_map_refined = voronoi_map.copy()
        voronoi_map_refined[binary == 0] = -1  # tło
    else:
        voronoi_map_refined = voronoi_map

    # 5. Wyodrębnij kontury
    teeth_contours = {}

    for i, tid in enumerate(tooth_ids):
        tooth_mask = (voronoi_map_refined == i).astype(np.uint8) * 255

        contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            valid_contours = []

            cx, cy = int(centroids[tid][0]), int(centroids[tid][1])

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    if cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                        valid_contours.append(cnt)

            if valid_contours:
                best_cnt = max(valid_contours, key=cv2.contourArea)
                teeth_contours[tid] = best_cnt
            else:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > max_area:
                    # print(f"Ząb {tid} obcięty odległością od centroidu: {area:.0f} → {max_area}")

                    # Tworzymy maskę tylko z największego konturu
                    mask_temp = np.zeros_like(tooth_mask)
                    cv2.drawContours(mask_temp, [largest], -1, 255, -1)

                    # Obliczamy odległość euklidesową od centroidu (ręcznie, bo distanceTransform jest od krawędzi)
                    yy, xx = np.mgrid[0:h, 0:w]
                    dist_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)

                    # Tworzymy nową maskę – tylko piksele bliżej niż max_dist
                    max_dist = np.sqrt(max_area / np.pi) * 1.5  # promień koła + zapas

                    tooth_mask = np.where(
                        (mask_temp > 0) & (dist_from_center <= max_dist),
                        255,
                        0
                    ).astype(np.uint8)

                    # Ponownie znajdujemy kontury po obcięciu
                    contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        teeth_contours[tid] = max(contours, key=cv2.contourArea)
                    else:
                        # Jeśli po obcięciu nic nie zostało – zostawiamy oryginalny (rzadkie)
                        teeth_contours[tid] = largest

    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(voronoi_map, cmap='nipy_spectral')
        axes[0].set_title('Voronoi Partition')

        if refine:
            axes[1].imshow(voronoi_map_refined, cmap='nipy_spectral')
            axes[1].set_title('Voronoi + Binary Mask')
        else:
            axes[1].imshow(preprocessed_img, cmap='gray')
            axes[1].set_title('Original')

        overlay = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
        for tid, cnt in teeth_contours.items():
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)
        axes[2].imshow(overlay)
        axes[2].set_title(f'Voronoi Contours: {len(teeth_contours)} teeth')

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return teeth_contours, voronoi_map_refined

def segment_with_mean_shift(preprocessed_img, centroids, sp=15, sr=30, debug=False):
    """
    Mean Shift clustering/segmentation.

    Zalety:
    - Automatycznie znajduje liczbę klastrów
    - Dobre dla regionów o podobnej teksturze
    """
    h, w = preprocessed_img.shape
    img_color = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    shifted = cv2.pyrMeanShiftFiltering(img_color, sp=sp, sr=sr)
    shifted_gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(shifted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    teeth_contours, _ = segment_with_voronoi(shifted_gray, centroids, refine=True, debug=False, max_area=10000)

    if debug:
        vis = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, list(teeth_contours.values()), -1, (0, 255, 0), 2)
        plt.figure(figsize=(10,8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f'v1 - Twoja wersja: {len(teeth_contours)} zębów')
        plt.axis('off')
        plt.show()

    return teeth_contours, shifted_gray