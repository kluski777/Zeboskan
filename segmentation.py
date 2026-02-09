from skimage.segmentation import active_contour, random_walker
from skimage.graph import rag_mean_color, cut_threshold
from skimage.segmentation import slic, mark_boundaries
from skimage.segmentation import felzenszwalb
from skimage.filters import gaussian, sobel
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



#! SLIC i FELZENSZWALB NIE DZIALAJA WCALE
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


#! SLIC i FELZENSZWALB NIE DZIALAJA WCALE
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


# ============================================================================
# METODA 1: MARKER-BASED WATERSHED
# ============================================================================

def segment_with_marker_watershed(preprocessed_img, centroids, marker_radius=10, debug=False):
    """Watershed z centroidami jako markerami."""
    markers, mapping, next_lbl = _make_markers(preprocessed_img.shape, centroids, marker_radius)
    bg, _ = _bg_mask(preprocessed_img)
    markers[bg > 0] = next_lbl

    color_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    result = markers.copy()
    cv2.watershed(color_img, result)
    contours = _extract_contours(result, mapping)

    if debug:
        _debug_panels([
            (markers, 'Initial Markers', 'nipy_spectral'),
            (result, 'After Watershed', 'nipy_spectral'),
            (_contour_overlay(preprocessed_img, contours), f'Contours: {len(contours)} teeth', None),
        ])
    return contours, result


# ============================================================================
# METODA 2: REGION GROWING
# ============================================================================

def _region_grow_single(img, seed, threshold=15, max_iter=10000):
    """Region growing z pojedynczego seeda."""
    h, w = img.shape
    sx, sy = int(seed[0]), int(seed[1])
    if not (0 <= sx < w and 0 <= sy < h):
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    nb = img[max(0, sy-3):min(h, sy+4), max(0, sx-3):min(w, sx+4)]
    ref = np.mean(nb)

    queue = deque([(sx, sy)])
    visited = {(sx, sy)}
    for _ in range(max_iter):
        if not queue:
            break
        x, y = queue.popleft()
        if abs(float(img[y, x]) - ref) <= threshold:
            mask[y, x] = 255
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return mask


def segment_with_region_growing(preprocessed_img, centroids, threshold=20, debug=False):
    """Segmentacja przez region growing z każdego centroidu."""
    h, w = preprocessed_img.shape
    all_masks = np.zeros((h, w), dtype=np.int32)
    contours = {}
    smoothed = cv2.GaussianBlur(preprocessed_img, (5, 5), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for tid, (cx, cy) in centroids.items():
        mask = _region_grow_single(smoothed, (cx, cy), threshold)
        if mask is None or mask.sum() < 100:
            continue
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0:
                contours[tid] = cnt
                all_masks[mask > 0] = tid
                break

    if debug:
        _debug_panels([
            (all_masks, 'Region Growing Masks', 'nipy_spectral'),
            (_contour_overlay(preprocessed_img, contours), f'Region Growing: {len(contours)} teeth', None),
        ], figsize=(14, 6))
    return contours, all_masks


# ============================================================================
# METODA 3: GRABCUT
# ============================================================================

def segment_with_grabcut(preprocessed_img, centroids, bbox_scale=1.5, debug=False):
    """GrabCut z bounding boxami wokół centroidów."""
    color_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR) if preprocessed_img.ndim == 2 else preprocessed_img.copy()
    h, w = preprocessed_img.shape[:2]
    contours = {}
    half_w, half_h = int(w / 16 * bbox_scale / 2), int(h / 4 * bbox_scale / 2)

    for tid, (cx, cy) in centroids.items():
        cx, cy = int(cx), int(cy)
        x1, y1 = max(0, cx-half_w), max(0, cy-half_h)
        x2, y2 = min(w, cx+half_w), min(h, cy+half_h)
        rect = (x1, y1, x2-x1, y2-y1)
        if rect[2] < 20 or rect[3] < 20:
            continue
        try:
            mask = np.zeros((h, w), np.uint8)
            cv2.grabCut(color_img, mask, rect, np.zeros((1,65), np.float64),
                        np.zeros((1,65), np.float64), 5, cv2.GC_INIT_WITH_RECT)
            result = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
            cnts, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                best = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(best) > 500:
                    contours[tid] = best
        except cv2.error:
            continue

    if debug:
        _debug_panels([
            (_contour_overlay(preprocessed_img, contours), f'GrabCut: {len(contours)} teeth', None),
        ], figsize=(12, 8))
    return contours


# ============================================================================
# METODA 4: ACTIVE CONTOURS (SNAKES)
# ============================================================================

def segment_with_active_contours(preprocessed_img, centroids, initial_radius=30,
                                alpha=0.01, beta=0.1, gamma=0.01, debug=False):
    """Active Contours (Snakes) inicjalizowane okręgami wokół centroidów."""
    h, w = preprocessed_img.shape
    img_norm = gaussian(preprocessed_img.astype(float), sigma=2)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
    edge_map = sobel(img_norm)
    contours = {}

    s = np.linspace(0, 2 * np.pi, 100)
    for tid, (cx, cy) in centroids.items():
        init_x, init_y = cx + initial_radius * np.cos(s), cy + initial_radius * np.sin(s)
        if init_x.min() < 0 or init_x.max() >= w or init_y.min() < 0 or init_y.max() >= h:
            continue
        try:
            snake = active_contour(edge_map, np.column_stack([init_x, init_y]),
                                    alpha=alpha, beta=beta, gamma=gamma,
                                    max_num_iter=500, convergence=0.1)
            cnt = snake.astype(np.int32).reshape(-1, 1, 2)
            area = cv2.contourArea(cnt)
            if 500 < area < 50000:
                contours[tid] = cnt
        except Exception:
            continue

    if debug:
        _debug_panels([
            (edge_map, 'Edge Map (Sobel)', 'gray'),
            (_contour_overlay(preprocessed_img, contours), f'Active Contours: {len(contours)} teeth', None),
        ], figsize=(14, 6))
    return contours


# ============================================================================
# METODA 5: RANDOM WALKER
# ============================================================================

def segment_with_random_walker(preprocessed_img, centroids, marker_radius=8, beta=130, debug=False):
    """Random Walker segmentation - probabilistyczna metoda."""
    markers, mapping, next_lbl = _make_markers(preprocessed_img.shape, centroids, marker_radius)
    bg, _ = _bg_mask(preprocessed_img, kernel_size=20)
    markers[bg > 0] = next_lbl

    try:
        labels = random_walker(preprocessed_img.astype(float) / 255.0, markers, beta=beta, mode='bf')
    except Exception as e:
        print(f"Random Walker error: {e}")
        return {}, markers

    contours = _extract_contours(labels, mapping)

    if debug:
        _debug_panels([
            (markers, 'Markers (seeds)', 'nipy_spectral'),
            (labels, 'Random Walker Result', 'nipy_spectral'),
            (_contour_overlay(preprocessed_img, contours), f'Contours: {len(contours)} teeth', None),
        ])
    return contours, labels


# ============================================================================
# METODA 6: DISTANCE TRANSFORM + WATERSHED
# ============================================================================

def segment_with_distance_watershed(preprocessed_img, centroids, debug=False):
    """Ulepszona watershed wykorzystująca distance transform i centroidy."""
    h, w = preprocessed_img.shape
    _, binary = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Markery z centroidów — promień proporcjonalny do distance transform
    markers = np.zeros((h, w), dtype=np.int32)
    mapping = {}
    label = 1
    for tid, (cx, cy) in centroids.items():
        cx, cy = int(cx), int(cy)
        if 0 <= cx < w and 0 <= cy < h:
            radius = max(5, int(dist[cy, cx] * 0.5))
            cv2.circle(markers, (cx, cy), radius, label, -1)
            mapping[tid] = label
            label += 1
    markers[binary == 0] = label

    # Watershed na odwróconej distance transform
    dist_inv = cv2.normalize(dist.max() - dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    result = markers.copy()
    cv2.watershed(cv2.cvtColor(dist_inv, cv2.COLOR_GRAY2BGR), result)
    contours = _extract_contours(result, mapping)

    if debug:
        panels = [
            (binary, 'Binary', 'gray'),
            (dist, 'Distance Transform', 'hot'),
            (markers, 'Initial Markers', 'nipy_spectral'),
            (dist_inv, 'Inverted Distance', 'gray'),
            (result, 'After Watershed', 'nipy_spectral'),
            (_contour_overlay(preprocessed_img, contours), f'Result: {len(contours)} teeth', None),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, (img, title, cmap) in zip(axes.flat, panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    return contours, result


# ============================================================================
# PORÓWNANIE WSZYSTKICH METOD
# ============================================================================

def compare_all_segmentation_methods(preprocessed_img, centroids, gt_metadata=None):
    """Porównuje wszystkie metody segmentacji na jednym obrazie."""
    methods = {
        'Marker Watershed': lambda: segment_with_marker_watershed(preprocessed_img, centroids),
        'Region Growing': lambda: segment_with_region_growing(preprocessed_img, centroids),
        'GrabCut': lambda: segment_with_grabcut(preprocessed_img, centroids),
        'Active Contours': lambda: segment_with_active_contours(preprocessed_img, centroids),
        'Random Walker': lambda: segment_with_random_walker(preprocessed_img, centroids),
        'Distance Watershed': lambda: segment_with_distance_watershed(preprocessed_img, centroids),
    }

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Oryginalny obraz z centroidami
    overlay = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
    for tid, (cx, cy) in centroids.items():
        cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.putText(overlay, str(tid), (int(cx)+5, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    axes[0].imshow(overlay)
    axes[0].set_title(f'Centroids ({len(centroids)} teeth)')
    axes[0].axis('off')

    start = 1
    if gt_metadata:
        axes[1].imshow(draw_annotations(preprocessed_img.copy(), gt_metadata, color=(255,0,0), thickness=2))
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        start = 2

    results = {}
    for i, (name, method) in enumerate(methods.items()):
        ax = axes[start + i]
        try:
            res = method()
            contours = res[0] if isinstance(res, tuple) else res
            results[name] = contours
            ax.imshow(_contour_overlay(preprocessed_img, contours))
            ax.set_title(f'{name}\n({len(contours)} teeth)')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name}\n(FAILED)')
            results[name] = {}
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    return results